import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

/**
 * Screen-space line integral convolution (LIC) of a tangent vector field on a
 * mesh: the field's streamlines as streaks over the mesh's own colouring.
 *
 * Three passes per frame, all at the panel's pixel size:
 * 1. the scene as usual (coloured, lit), into a colour target;
 * 2. the mesh alone with each fragment's velocity projected to the screen,
 *    into a velocity target: unit screen direction (rg), speed relative to the
 *    largest (b) and coverage (a);
 * 3. a full-screen pass that averages a fixed screen-space noise texture along
 *    the streamline through each covered pixel, forward and back, and
 *    multiplies the colour by the result.
 *
 * The noise lives on the screen, not on the surface, so the streaks do not
 * move with the surface as it turns; only the field's direction is carried.
 * Where the speed is near zero the direction means nothing, so the streaks
 * fade out there.
 */

/** Pixels each way along the streamline (the kernel is twice this long). */
const HALF_LENGTH = 24;

const velVertex = /* glsl */ `
attribute vec3 velocity;
uniform float invVmax;
varying vec4 vClip;
varying vec4 vDClip;
varying float vSpeed;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vec4 wv = modelMatrix * vec4(velocity, 0.0);
  // Clip coordinates are linear in the world point, so interpolating the
  // point's and the velocity's clip images is exact for the interpolated
  // point and velocity; the screen direction follows per fragment.
  vClip = projectionMatrix * viewMatrix * wp;
  vDClip = projectionMatrix * viewMatrix * wv;
  vSpeed = length(velocity) * invVmax;
  gl_Position = vClip;
}
`;

const velFragment = /* glsl */ `
uniform vec2 resolution;
varying vec4 vClip;
varying vec4 vDClip;
varying float vSpeed;
void main() {
  // d(xy/w) = (dxy w - xy dw) / w^2; w > 0 in front of the camera, so the
  // 1/w^2 only scales. Then NDC to pixels.
  vec2 d = (vDClip.xy * vClip.w - vClip.xy * vDClip.w) * resolution;
  float len = length(d);
  vec2 dir = len > 0.0 ? d / len : vec2(0.0);
  gl_FragColor = vec4(dir, vSpeed, 1.0);
}
`;

const licVertex = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const licFragment = /* glsl */ `
uniform sampler2D tColor;
uniform sampler2D tVel;
uniform sampler2D tNoise;
uniform vec2 resolution;
uniform float noiseSize;
uniform float strength;
varying vec2 vUv;

float noiseAt(vec2 px) {
  return texture2D(tNoise, px / noiseSize).r;
}

// Hann-weighted noise along the streamline from px, one pixel per step, in
// direction sgn; stops where it leaves the surface.
void walk(vec2 px, vec2 dir, float sgn, inout float sum, inout float wsum) {
  vec2 p = px;
  vec2 d = sgn * dir;
  for (int i = 1; i <= ${HALF_LENGTH}; i++) {
    p += d;
    vec4 v = texture2D(tVel, p / resolution);
    if (v.a < 0.5) break;
    if (dot(v.xy, v.xy) > 0.25) d = sgn * v.xy;
    float w = 0.5 + 0.5 * cos(3.14159265 * float(i) / float(${HALF_LENGTH + 1}));
    sum += w * noiseAt(p);
    wsum += w;
  }
}

void main() {
  vec4 base = texture2D(tColor, vUv);
  vec4 v0 = texture2D(tVel, vUv);
  vec4 outColor = base;
  if (v0.a > 0.5 && dot(v0.xy, v0.xy) > 0.25) {
    vec2 px = vUv * resolution;
    float sum = noiseAt(px);
    float wsum = 1.0;
    walk(px, v0.xy, 1.0, sum, wsum);
    walk(px, v0.xy, -1.0, sum, wsum);
    // The mean of n noise values has spread about 1/sqrt(n) of one value's,
    // so stretch it back toward the full range.
    float lic = clamp(0.5 + (sum / wsum - 0.5) * 0.6 * sqrt(wsum), 0.0, 1.0);
    float fade = strength * smoothstep(0.0, 0.05, v0.b);
    outColor.rgb *= mix(1.0, 0.3 + 0.9 * lic, fade);
  }
  gl_FragColor = outColor;
  #include <colorspace_fragment>
}
`;

function noiseTexture(size: number): THREE.DataTexture {
  const data = new Uint8Array(size * size);
  for (let i = 0; i < data.length; i++) data[i] = Math.floor(Math.random() * 256);
  const t = new THREE.DataTexture(data, size, size, THREE.RedFormat, THREE.UnsignedByteType);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

export class LicPass {
  #color: THREE.WebGLRenderTarget;
  #vel: THREE.WebGLRenderTarget;
  #velScene = new THREE.Scene();
  #velMaterial: THREE.ShaderMaterial;
  #quad: FullScreenQuad;
  #licMaterial: THREE.ShaderMaterial;
  #noise = noiseTexture(256);

  /** `geometry` must carry a `velocity` attribute (3 per vertex, world frame). */
  constructor(geometry: THREE.BufferGeometry) {
    // Linear half floats for the colour, which the last pass converts for the
    // screen; multisampled as the on-screen render is.
    this.#color = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    this.#vel = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    });
    this.#velMaterial = new THREE.ShaderMaterial({
      vertexShader: velVertex,
      fragmentShader: velFragment,
      uniforms: { invVmax: { value: 0 }, resolution: { value: new THREE.Vector2(1, 1) } },
      side: THREE.DoubleSide,
    });
    this.#velScene.add(new THREE.Mesh(geometry, this.#velMaterial));
    this.#licMaterial = new THREE.ShaderMaterial({
      vertexShader: licVertex,
      fragmentShader: licFragment,
      uniforms: {
        tColor: { value: this.#color.texture },
        tVel: { value: this.#vel.texture },
        tNoise: { value: this.#noise },
        resolution: { value: new THREE.Vector2(1, 1) },
        noiseSize: { value: 256 },
        strength: { value: 1 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.#quad = new FullScreenQuad(this.#licMaterial);
  }

  /** The largest speed of the field, which the fade is relative to. */
  set vmax(v: number) {
    this.#velMaterial.uniforms.invVmax.value = v > 0 ? 1 / v : 0;
  }

  /**
   * Draw `scene` through `camera` with the streaks, into the renderer's
   * current viewport (`w` x `h` pixels at the bottom-left). The scene must
   * have been laid out (matrices current) by the caller or by this render.
   */
  render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, w: number, h: number): void {
    if (this.#color.width !== w || this.#color.height !== h) {
      this.#color.setSize(w, h);
      this.#vel.setSize(w, h);
    }
    // Noise cells of about one CSS pixel.
    const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;
    this.#licMaterial.uniforms.noiseSize.value = 256 * Math.max(1, dpr);
    this.#licMaterial.uniforms.resolution.value.set(w, h);
    this.#velMaterial.uniforms.resolution.value.set(w / 2, h / 2);

    renderer.setRenderTarget(this.#color);
    renderer.render(scene, camera);

    const clear = renderer.getClearColor(new THREE.Color());
    const alpha = renderer.getClearAlpha();
    renderer.setClearColor(0x000000, 0);
    renderer.setRenderTarget(this.#vel);
    renderer.clear();
    renderer.render(this.#velScene, camera);
    renderer.setClearColor(clear, alpha);

    renderer.setRenderTarget(null);
    renderer.setViewport(0, 0, w, h);
    renderer.setScissor(0, 0, w, h);
    this.#quad.render(renderer);
  }

  dispose(): void {
    this.#color.dispose();
    this.#vel.dispose();
    this.#velMaterial.dispose();
    this.#licMaterial.dispose();
    this.#quad.dispose();
    this.#noise.dispose();
  }
}
