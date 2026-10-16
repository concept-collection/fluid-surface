/**
 * Time steps of the fluid solver's static mode (models/static_stokes.m) on a
 * fixed surface: BDF-1, the momentum linearized about the current velocity,
 * solved by GMRES (src/fluid/gmres.ts) on the preconditioned operator.
 *
 * Everything but GMRES's own arithmetic runs on the GPU: lib/geometry.m once
 * for the surface, then setup and rhs once per step and matvec once per
 * iteration. They share named buffers, so the geometry and the per-step
 * tables pass from one function to the next without a readback.
 */
import { ShtPlan } from '../sht/sht.ts';
import { DerivPlan } from '../sht/deriv.ts';
import type { ShtConfig } from '../sht/layout.ts';
import { HostBuffers } from '../mgpu/plan.ts';
import { GpuFunction } from '../mgpu/gpuFunction.ts';
import type { Binding } from '../mgpu/compile.ts';
import { operatorLibs } from '../mgpu/libs.ts';
import { gridTrig, productConfig } from '../geom/geometry.ts';
import { chartWeights, spectralMasks, workingIndices } from './spectra.ts';
import { gmres, type GmresOptions } from './gmres.ts';
import geometrySource from '../../lib/geometry.m?raw';
import modelSource from '../../models/static_stokes.m?raw';

export interface StaticStokesParams {
  re: number;
  /** The bending number; the static mode has no bending, but the C++
   *  preconditioner's velocity symbol includes it. */
  be: number;
  dt: number;
  gmres: GmresOptions;
}

export interface StepInfo {
  iterations: number;
  converged: boolean;
  relResid: number;
}

/** The geometry fields the model reads, each fed into a buffer of its name. */
const GEOMETRY_FED = ['vt1', 'vt2', 'vt3', 'vp1', 'vp2', 'vp3', 'n1', 'n2', 'n3', 'H', 'sqrtg', 'jinv'];

export class StaticStokes {
  readonly lmax: number;
  readonly cfg: ShtConfig;
  readonly prm: StaticStokesParams;
  readonly geometry: GpuFunction;
  readonly setup: GpuFunction;
  readonly rhs: GpuFunction;
  readonly matvec: GpuFunction;
  #sht: ShtPlan;
  #deriv: DerivPlan;
  #host: HostBuffers;
  #idx: Int32Array;
  #nlm2: number;

  private constructor(init: {
    lmax: number; cfg: ShtConfig; prm: StaticStokesParams; sht: ShtPlan; deriv: DerivPlan; host: HostBuffers;
    geometry: GpuFunction; setup: GpuFunction; rhs: GpuFunction; matvec: GpuFunction;
  }) {
    this.lmax = init.lmax;
    this.cfg = init.cfg;
    this.prm = init.prm;
    this.#sht = init.sht;
    this.#deriv = init.deriv;
    this.#host = init.host;
    this.geometry = init.geometry;
    this.setup = init.setup;
    this.rhs = init.rhs;
    this.matvec = init.matvec;
    this.#idx = workingIndices(init.cfg, init.lmax);
    this.#nlm2 = 2 * init.sht.nlm;
  }

  static async create(device: GPUDevice, lmax: number, prm: StaticStokesParams): Promise<StaticStokes> {
    const cfg = productConfig(lmax);
    const sht = await ShtPlan.create(device, cfg);
    const deriv = await DerivPlan.create(device, sht);
    const host = new HostBuffers(device);
    const npts = cfg.nlat * cfg.nphi;
    const spec: Binding = { kind: 'tensor', shape: [2, sht.nlm] };
    const grid: Binding = { kind: 'tensor', shape: [npts, 1] };
    const param: Binding = { kind: 'param' };
    const bindings: Record<string, Binding> = { mass: param, disp: param, re: param, be: param, visc: param };
    for (const n of ['X1', 'X2', 'X3', 'U1', 'U2', 'U3', 'Z1', 'Z2', 'Z3', 'Z4', 'lam', 'keep', 'clean', 'filtf', 'e0', 'pscale', 'vdiv'])
      bindings[n] = spec;
    for (const n of ['ct', 'st', 'wq', 'v1', 'v2', 'v3', ...GEOMETRY_FED]) bindings[n] = grid;
    const common = { device, sht, deriv, bindings, host, libs: operatorLibs };
    const geometry = await GpuFunction.create({
      ...common, source: geometrySource, name: 'geometry', nargout: 16,
      feedback: Object.fromEntries(GEOMETRY_FED.map((n) => [n, n])),
    });
    const setup = await GpuFunction.create({
      ...common, source: modelSource, name: 'setup', nargout: 5,
      feedback: { v1: 'v1', v2: 'v2', v3: 'v3', pscale: 'pscale', vdiv: 'vdiv' },
    });
    const rhs = await GpuFunction.create({ ...common, source: modelSource, name: 'rhs', nargout: 4 });
    const matvec = await GpuFunction.create({ ...common, source: modelSource, name: 'matvec', nargout: 4 });

    const { ct, st } = gridTrig(cfg);
    const masks = spectralMasks(cfg, lmax);
    const up = (name: string, data: Float32Array) => host.get(name) && host.upload(name, data);
    up('ct', ct);
    up('st', st);
    up('wq', chartWeights(cfg));
    for (const [name, data] of Object.entries(masks)) up(name, data);
    return new StaticStokes({ lmax, cfg, prm, sht, deriv, host, geometry, setup, rhs, matvec });
  }

  /** Sets the fixed surface from its position's product-band coefficients. */
  setSurface(X: Float32Array[]): void {
    X.forEach((x, c) => this.#host.upload(`X${c + 1}`, x));
    this.geometry.run();
  }

  /** Packs product-band spectra into GMRES's unknowns. */
  pack(fields: ArrayLike<number>[]): Float64Array {
    const m = this.#idx.length;
    const out = new Float64Array(fields.length * m);
    fields.forEach((f, j) => {
      for (let i = 0; i < m; i++) out[j * m + i] = f[this.#idx[i]];
    });
    return out;
  }

  unpack(x: ArrayLike<number>, nfields: number): Float32Array[] {
    const m = this.#idx.length;
    return Array.from({ length: nfields }, (_, j) => {
      const f = new Float32Array(this.#nlm2);
      for (let i = 0; i < m; i++) f[this.#idx[i]] = x[j * m + i];
      return f;
    });
  }

  /** Prepares one step from the current velocity: the transport velocity and the
   *  preconditioner's symbols (setup), then the right-hand side (rhs). */
  async prepare(U: Float32Array[]): Promise<Float64Array> {
    const mass = 1 / this.prm.dt;
    const params = { mass, disp: this.prm.dt, re: this.prm.re, be: this.prm.be, visc: 2 / this.prm.re };
    U.forEach((u, c) => this.#host.upload(`U${c + 1}`, u));
    for (const f of [this.setup, this.rhs, this.matvec]) f.setParams(params);
    this.setup.run();
    this.rhs.run();
    return this.pack(await this.rhs.readMany(['B1', 'B2', 'B3', 'BP']));
  }

  /** The preconditioned operator on GMRES's unknowns. */
  async apply(z: Float64Array): Promise<Float64Array> {
    this.unpack(z, 4).forEach((f, j) => this.#host.upload(`Z${j + 1}`, f));
    this.matvec.run();
    return this.pack(await this.matvec.readMany(['R1', 'R2', 'R3', 'RP']));
  }

  /** One BDF-1 step from velocity U and pressure P (product-band spectra). */
  async step(U: Float32Array[], P: Float32Array): Promise<{ U: Float32Array[]; P: Float32Array; info: StepInfo }> {
    const b = await this.prepare(U);
    const res = await gmres((z) => this.apply(z), b, this.pack([...U, P]), this.prm.gmres);
    const [U1, U2, U3, P1] = this.unpack(res.x, 4);
    return { U: [U1, U2, U3], P: P1, info: { iterations: res.iterations, converged: res.converged, relResid: res.relResid } };
  }

  destroy(): void {
    for (const f of [this.geometry, this.setup, this.rhs, this.matvec]) f.destroy();
    this.#host.destroy();
    this.#deriv.destroy();
    this.#sht.destroy();
  }
}
