import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

/**
 * Line integral convolution (LIC) of a tangent vector field on a mesh, traced
 * in screen space: the field's streamlines as streaks over the mesh's own
 * colouring.
 *
 * Three passes per frame, all at the panel's pixel size:
 * 1. the scene as usual (coloured, lit), into a colour target;
 * 2. the mesh alone, into a field target: each fragment's velocity projected
 *    to the screen as a unit direction (rg), 1 plus its speed relative to
 *    the largest (b, 0 off the surface), and a noise value (a);
 * 3. a full-screen pass that averages the noise along the screen streamline
 *    through each covered pixel, forward and back, and multiplies the colour
 *    by the result.
 *
 * The noise is a function of each vertex's `ref` position (the point it
 * labels, for the page the unit sphere of the solver's parametrization), not
 * of the screen, so the grain stays on the surface as the view turns and as
 * the surface deforms. Its cells are scaled to about a pixel at the current
 * camera distance, so zooming rescales the grain. Where the speed is near
 * zero the direction means nothing, so the streaks fade out there.
 */

/** Pixels each way along the streamline (the kernel is twice this long). */
const HALF_LENGTH = 24;

const fieldVertex = /* glsl */ `
attribute vec3 velocity;
attribute vec3 ref;
uniform float invVmax;
varying vec4 vClip;
varying vec4 vDClip;
varying vec3 vRef;
varying float vSpeed;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vec4 wv = modelMatrix * vec4(velocity, 0.0);
  // Clip coordinates are linear in the world point, so interpolating the
  // point's and the velocity's clip images is exact for the interpolated
  // point and velocity; the screen direction follows per fragment.
  vClip = projectionMatrix * viewMatrix * wp;
  vDClip = projectionMatrix * viewMatrix * wv;
  vRef = ref;
  vSpeed = length(velocity) * invVmax;
  gl_Position = vClip;
}
`;

const fieldFragment = /* glsl */ `
uniform vec2 resolution;
uniform float cells;
varying vec4 vClip;
varying vec4 vDClip;
varying vec3 vRef;
varying float vSpeed;

// Integer hash of a lattice point to [0, 1).
float hash(ivec3 c) {
  uvec3 p = uvec3(c);
  uint h = p.x * 1597334677u ^ p.y * 3812015801u ^ p.z * 2798796415u;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  h ^= h >> 16u;
  return float(h >> 8u) / 16777216.0;
}

// Value noise: lattice values, trilinearly interpolated.
float noise(vec3 x) {
  vec3 f = floor(x);
  vec3 t = x - f;
  ivec3 c = ivec3(f);
  float n000 = hash(c);
  float n100 = hash(c + ivec3(1, 0, 0));
  float n010 = hash(c + ivec3(0, 1, 0));
  float n110 = hash(c + ivec3(1, 1, 0));
  float n001 = hash(c + ivec3(0, 0, 1));
  float n101 = hash(c + ivec3(1, 0, 1));
  float n011 = hash(c + ivec3(0, 1, 1));
  float n111 = hash(c + ivec3(1, 1, 1));
  return mix(
    mix(mix(n000, n100, t.x), mix(n010, n110, t.x), t.y),
    mix(mix(n001, n101, t.x), mix(n011, n111, t.x), t.y),
    t.z);
}

void main() {
  // d(xy/w) = (dxy w - xy dw) / w^2; w > 0 in front of the camera, so the
  // 1/w^2 only scales. Then NDC to pixels.
  vec2 d = (vDClip.xy * vClip.w - vClip.xy * vDClip.w) * resolution;
  float len = length(d);
  vec2 dir = len > 0.0 ? d / len : vec2(0.0);
  gl_FragColor = vec4(dir, 1.0 + vSpeed, noise(vRef * cells));
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
uniform sampler2D tField;
uniform vec2 resolution;
uniform float strength;
varying vec2 vUv;

// Hann-weighted noise along the streamline from px, one pixel per step, in
// direction sgn; stops where it leaves the surface.
void walk(vec2 px, vec2 dir, float sgn, inout float sum, inout float wsum) {
  vec2 p = px;
  vec2 d = sgn * dir;
  for (int i = 1; i <= ${HALF_LENGTH}; i++) {
    p += d;
    vec4 v = texture2D(tField, p / resolution);
    if (v.b < 0.5) break;
    if (dot(v.xy, v.xy) > 0.25) d = sgn * v.xy;
    float w = 0.5 + 0.5 * cos(3.14159265 * float(i) / float(${HALF_LENGTH + 1}));
    sum += w * v.a;
    wsum += w;
  }
}

void main() {
  vec4 base = texture2D(tColor, vUv);
  vec4 v0 = texture2D(tField, vUv);
  vec4 outColor = base;
  if (v0.b > 0.5 && dot(v0.xy, v0.xy) > 0.25) {
    vec2 px = vUv * resolution;
    float sum = v0.a;
    float wsum = 1.0;
    walk(px, v0.xy, 1.0, sum, wsum);
    walk(px, v0.xy, -1.0, sum, wsum);
    // The mean of n noise values has spread about 1/sqrt(n) of one value's,
    // so stretch it back toward the full range.
    float lic = clamp(0.5 + (sum / wsum - 0.5) * 0.6 * sqrt(wsum), 0.0, 1.0);
    float fade = strength * smoothstep(0.0, 0.05, v0.b - 1.0);
    outColor.rgb *= mix(1.0, 0.3 + 0.9 * lic, fade);
  }
  gl_FragColor = outColor;
  #include <colorspace_fragment>
}
`;

export class LicPass {
  #geometry: THREE.BufferGeometry;
  #color: THREE.WebGLRenderTarget;
  #field: THREE.WebGLRenderTarget;
  #fieldScene = new THREE.Scene();
  #fieldMaterial: THREE.ShaderMaterial;
  #quad: FullScreenQuad;
  #licMaterial: THREE.ShaderMaterial;

  /** `geometry` must carry `velocity` (3 per vertex, world frame) and `ref`
   *  (3 per vertex, the point each vertex labels, which the noise is of). */
  constructor(geometry: THREE.BufferGeometry) {
    this.#geometry = geometry;
    // Linear half floats for the colour, which the last pass converts for the
    // screen; multisampled as the on-screen render is.
    this.#color = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    this.#field = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    });
    this.#fieldMaterial = new THREE.ShaderMaterial({
      vertexShader: fieldVertex,
      fragmentShader: fieldFragment,
      uniforms: {
        invVmax: { value: 0 },
        resolution: { value: new THREE.Vector2(1, 1) },
        cells: { value: 100 },
      },
      side: THREE.DoubleSide,
    });
    this.#fieldScene.add(new THREE.Mesh(geometry, this.#fieldMaterial));
    this.#licMaterial = new THREE.ShaderMaterial({
      vertexShader: licVertex,
      fragmentShader: licFragment,
      uniforms: {
        tColor: { value: this.#color.texture },
        tField: { value: this.#field.texture },
        resolution: { value: new THREE.Vector2(1, 1) },
        strength: { value: 1 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.#quad = new FullScreenQuad(this.#licMaterial);
  }

  /** The largest speed of the field, which the fade is relative to. */
  set vmax(v: number) {
    this.#fieldMaterial.uniforms.invVmax.value = v > 0 ? 1 / v : 0;
  }

  /**
   * Draw `scene` through `camera` with the streaks, into the renderer's
   * current viewport (`w` x `h` pixels at the bottom-left). The scene must
   * have been laid out (matrices current) by the caller or by this render.
   */
  render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, w: number, h: number): void {
    if (this.#color.width !== w || this.#color.height !== h) {
      this.#color.setSize(w, h);
      this.#field.setSize(w, h);
    }
    this.#licMaterial.uniforms.resolution.value.set(w, h);
    this.#fieldMaterial.uniforms.resolution.value.set(w / 2, h / 2);
    // Noise cells of about one CSS pixel where the surface faces the camera:
    // pixels per unit length at the surface's centre, over the device ratio.
    const center = this.#geometry.boundingSphere?.center ?? new THREE.Vector3();
    const dist = Math.max(1e-6, camera.position.distanceTo(center));
    const pxPerUnit = (h * camera.zoom) / (2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;
    this.#fieldMaterial.uniforms.cells.value = pxPerUnit / Math.max(1, dpr);

    renderer.setRenderTarget(this.#color);
    renderer.render(scene, camera);

    const clear = renderer.getClearColor(new THREE.Color());
    const alpha = renderer.getClearAlpha();
    // Off the surface: all zero, so b = 0.
    renderer.setClearColor(0x000000, 0);
    renderer.setRenderTarget(this.#field);
    renderer.clear();
    renderer.render(this.#fieldScene, camera);
    renderer.setClearColor(clear, alpha);

    renderer.setRenderTarget(null);
    renderer.setViewport(0, 0, w, h);
    renderer.setScissor(0, 0, w, h);
    this.#quad.render(renderer);
  }

  dispose(): void {
    this.#color.dispose();
    this.#field.dispose();
    this.#fieldMaterial.dispose();
    this.#licMaterial.dispose();
    this.#quad.dispose();
  }
}
