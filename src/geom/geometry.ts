/**
 * The surface geometry on the GPU: lib/geometry.m, compiled once per grid,
 * and the example surfaces it is run on.
 *
 * The position is carried as spherical-harmonic coefficients on the working
 * band, as in the C++ solver, and zero-padded to the product band (twice the
 * working band) on which lib/geometry.m runs.
 */
import { ShtPlan } from '../sht/sht.ts';
import { DerivPlan } from '../sht/deriv.ts';
import { ShtReference } from '../sht/reference.ts';
import { gaussNodesWeights } from '../sht/gauss.ts';
import { gridForLmax, relayoutCoeffs, type ShtConfig } from '../sht/layout.ts';
import { GpuFunction } from '../mgpu/gpuFunction.ts';
import type { Binding } from '../mgpu/compile.ts';
import { operatorLibs } from '../mgpu/libs.ts';
import geometrySource from '../../lib/geometry.m?raw';

/** lib/geometry.m's outputs, in order. */
export const GEOMETRY_OUTPUTS = [
  'vt1', 'vt2', 'vt3', 'vp1', 'vp2', 'vp3', 'n1', 'n2', 'n3',
  'H', 'K', 'sqrtg', 'dp1', 'p2', 'dq2', 'jinv',
] as const;
export type GeometryOutput = (typeof GEOMETRY_OUTPUTS)[number];

/** The product grid for a working band: band 2 * lmax on SHTns-style Gauss x
 *  power-of-two grid (src/sht/layout.ts gridForLmax). */
export function productConfig(lmax: number): ShtConfig {
  const lmaxF = 2 * lmax;
  const { nlat, nphi } = gridForLmax(lmaxF, 1);
  return { lmax: lmaxF, mmax: lmaxF, nlat, nphi };
}

/** cos(theta) and sin(theta) on the grid, from the f64 Gauss nodes. */
export function gridTrig(cfg: ShtConfig): { ct: Float32Array; st: Float32Array } {
  const { x } = gaussNodesWeights(cfg.nlat);
  const ct = new Float32Array(cfg.nlat * cfg.nphi);
  const st = new Float32Array(cfg.nlat * cfg.nphi);
  for (let i = 0; i < cfg.nlat; i++) {
    ct.fill(x[i], i * cfg.nphi, (i + 1) * cfg.nphi);
    st.fill(Math.sqrt(1 - x[i] * x[i]), i * cfg.nphi, (i + 1) * cfg.nphi);
  }
  return { ct, st };
}

/** cos and sin of theta and phi on the grid, for the unit sphere's frame. */
export function gridFrame(cfg: ShtConfig): { ct: Float32Array; st: Float32Array; cp: Float32Array; sp: Float32Array } {
  const { ct, st } = gridTrig(cfg);
  const cp = new Float32Array(cfg.nlat * cfg.nphi);
  const sp = new Float32Array(cfg.nlat * cfg.nphi);
  forGrid(cfg, (_i, phi, k) => {
    cp[k] = Math.cos(phi);
    sp[k] = Math.sin(phi);
  });
  return { ct, st, cp, sp };
}

/**
 * Grid fields on the product grid as coefficients, as the C++ solver makes
 * them (core/fluid_examples.cpp GridToCoeffs): analysed in f64, truncated to
 * the working band `lmax`, and returned padded back to the product band.
 */
export function gridToCoeffs(cfg: ShtConfig, lmax: number, fields: Float64Array[]): Float32Array[] {
  const ref = new ShtReference(cfg);
  const band = { lmax: cfg.lmax, mmax: cfg.mmax };
  const work = { lmax, mmax: lmax };
  return fields.map((g) => relayoutCoeffs(relayoutCoeffs(new Float32Array(ref.analys(g)), band, work), work, band));
}

/** Calls f(theta index, phi, k) for every grid point, k = ilat * nphi + iphi. */
export function forGrid(cfg: ShtConfig, f: (i: number, phi: number, k: number) => void): void {
  for (let i = 0; i < cfg.nlat; i++) {
    for (let j = 0; j < cfg.nphi; j++) f(i, (2 * Math.PI * j) / cfg.nphi, i * cfg.nphi + j);
  }
}

/**
 * Krause & Voigt's perturbed sphere, r = 1 + r0 cos(phi) sin(3 theta), as the
 * C++ solver builds it: sampled on the product grid and taken through
 * gridToCoeffs. r0 = 0 is the unit sphere.
 */
export function perturbedSphere(cfg: ShtConfig, lmax: number, r0: number): Float32Array[] {
  const { x: ct } = gaussNodesWeights(cfg.nlat);
  const f = [0, 1, 2].map(() => new Float64Array(cfg.nlat * cfg.nphi));
  forGrid(cfg, (i, phi, k) => {
    const st = Math.sqrt(1 - ct[i] * ct[i]);
    const r = 1 + r0 * Math.cos(phi) * Math.sin(3 * Math.acos(ct[i]));
    f[0][k] = r * st * Math.cos(phi);
    f[1][k] = r * st * Math.sin(phi);
    f[2][k] = r * ct[i];
  });
  return gridToCoeffs(cfg, lmax, f);
}

/** lib/geometry.m compiled for one product grid. */
export class SurfaceGeometry {
  readonly cfg: ShtConfig;
  readonly sht: ShtPlan;
  readonly fn: GpuFunction;
  #deriv: DerivPlan;

  private constructor(cfg: ShtConfig, sht: ShtPlan, deriv: DerivPlan, fn: GpuFunction) {
    this.cfg = cfg;
    this.sht = sht;
    this.#deriv = deriv;
    this.fn = fn;
  }

  static async create(device: GPUDevice, cfg: ShtConfig): Promise<SurfaceGeometry> {
    const sht = await ShtPlan.create(device, cfg);
    const deriv = await DerivPlan.create(device, sht);
    const npts = cfg.nlat * cfg.nphi;
    const spectral: Binding = { kind: 'tensor', shape: [2, sht.nlm] };
    const grid: Binding = { kind: 'tensor', shape: [npts, 1] };
    const fn = await GpuFunction.create({
      device,
      sht,
      deriv,
      source: geometrySource,
      name: 'geometry',
      bindings: { X1: spectral, X2: spectral, X3: spectral, ct: grid, st: grid },
      libs: operatorLibs,
      nargout: GEOMETRY_OUTPUTS.length,
    });
    const { ct, st } = gridTrig(cfg);
    fn.upload('ct', ct);
    fn.upload('st', st);
    return new SurfaceGeometry(cfg, sht, deriv, fn);
  }

  /** Runs on product-band coefficients of the position. */
  compute(X: Float32Array[]): void {
    X.forEach((x, c) => this.fn.upload(`X${c + 1}`, x));
    this.fn.run();
  }

  read(name: GeometryOutput): Promise<Float32Array> {
    return this.fn.read(name);
  }

  destroy(): void {
    this.fn.destroy();
    this.#deriv.destroy();
    this.sht.destroy();
  }
}
