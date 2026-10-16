/**
 * Time steps of the fluid solver on a moving surface, without bending or the
 * volume constraint (models/stokes.m): BDF-1, solved by GMRES
 * (src/fluid/gmres.ts) on the preconditioned operator, after which the surface
 * moves with the fluid's normal velocity.
 *
 * Per step: lib/geometry.m on the current surface, setup (the transport
 * velocity, and the area, which the host reads to build the sphere's
 * per-degree blocks), rhs, GMRES on matvec, and advance, whose new position
 * is copied back into the buffers geometry reads. The position stays on the
 * GPU; the velocity and pressure come back to the host for GMRES.
 */
import { ShtPlan } from '../sht/sht.ts';
import { DerivPlan } from '../sht/deriv.ts';
import type { ShtConfig } from '../sht/layout.ts';
import { HostBuffers } from '../mgpu/plan.ts';
import { GpuFunction } from '../mgpu/gpuFunction.ts';
import type { Binding } from '../mgpu/compile.ts';
import { operatorLibs } from '../mgpu/libs.ts';
import { gridFrame, productConfig } from '../geom/geometry.ts';
import { chartWeights, spectralMasks, workingIndices } from './spectra.ts';
import { sphereBlockTables } from './sphereBlocks.ts';
import { gmres, type GmresOptions } from './gmres.ts';
import type { StepInfo } from './staticStokes.ts';
import geometrySource from '../../lib/geometry.m?raw';
import modelSource from '../../models/stokes.m?raw';

export interface MovingStokesParams {
  re: number;
  dt: number;
  gmres: GmresOptions;
}

const GEOMETRY_FED = ['vt1', 'vt2', 'vt3', 'vp1', 'vp2', 'vp3', 'n1', 'n2', 'n3', 'H', 'sqrtg', 'jinv'];
const TABLES = ['i11', 'i12', 'i13', 'i21', 'i22', 'i23', 'i31', 'i32', 'i33', 'itor'];

export class MovingStokes {
  readonly lmax: number;
  readonly cfg: ShtConfig;
  readonly prm: MovingStokesParams;
  readonly fns: Record<'geometry' | 'setup' | 'rhs' | 'matvec' | 'advance', GpuFunction>;
  /** The area at the last step's start. */
  area = 0;
  #sht: ShtPlan;
  #deriv: DerivPlan;
  #host: HostBuffers;
  #idx: Int32Array;
  #nlm2: number;

  private constructor(lmax: number, cfg: ShtConfig, prm: MovingStokesParams, sht: ShtPlan, deriv: DerivPlan,
    host: HostBuffers, fns: MovingStokes['fns']) {
    this.lmax = lmax;
    this.cfg = cfg;
    this.prm = prm;
    this.#sht = sht;
    this.#deriv = deriv;
    this.#host = host;
    this.fns = fns;
    this.#idx = workingIndices(cfg, lmax);
    this.#nlm2 = 2 * sht.nlm;
  }

  static async create(device: GPUDevice, lmax: number, prm: MovingStokesParams): Promise<MovingStokes> {
    const cfg = productConfig(lmax);
    const sht = await ShtPlan.create(device, cfg);
    const deriv = await DerivPlan.create(device, sht);
    const host = new HostBuffers(device);
    const spec: Binding = { kind: 'tensor', shape: [2, sht.nlm] };
    const grid: Binding = { kind: 'tensor', shape: [cfg.nlat * cfg.nphi, 1] };
    const param: Binding = { kind: 'param' };
    const bindings: Record<string, Binding> = { mass: param, visc: param, dt: param };
    for (const n of ['X1', 'X2', 'X3', 'U1', 'U2', 'U3', 'Z1', 'Z2', 'Z3', 'Z4',
      'keep', 'clean', 'filtf', 'ilam', 'cutv', ...TABLES]) bindings[n] = spec;
    for (const n of ['ct', 'st', 'cp', 'sp', 'wq', 'v1', 'v2', 'v3', ...GEOMETRY_FED]) bindings[n] = grid;
    const make = (source: string, name: string, nargout: number, feedback?: Record<string, string>) =>
      GpuFunction.create({ device, sht, deriv, bindings, host, libs: operatorLibs, source, name, nargout, feedback });
    const fns = {
      geometry: await make(geometrySource, 'geometry', 16, Object.fromEntries(GEOMETRY_FED.map((n) => [n, n]))),
      setup: await make(modelSource, 'setup', 4, { v1: 'v1', v2: 'v2', v3: 'v3' }),
      rhs: await make(modelSource, 'rhs', 4),
      matvec: await make(modelSource, 'matvec', 4),
      advance: await make(modelSource, 'advance', 3, { Y1: 'X1', Y2: 'X2', Y3: 'X3' }),
    };
    const frame = gridFrame(cfg);
    const up = (name: string, data: Float32Array) => host.get(name) && host.upload(name, data);
    for (const [name, data] of Object.entries(frame)) up(name, data);
    up('wq', chartWeights(cfg));
    for (const [name, data] of Object.entries(spectralMasks(cfg, lmax))) up(name, data);
    return new MovingStokes(lmax, cfg, prm, sht, deriv, host, fns);
  }

  /** Sets the surface from its position's product-band coefficients. */
  setSurface(X: Float32Array[]): void {
    X.forEach((x, c) => this.#host.upload(`X${c + 1}`, x));
  }

  async readSurface(): Promise<Float32Array[]> {
    return this.fns.advance.readMany(['X1', 'X2', 'X3']);
  }

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

  /** Builds the step's operator about the current surface and velocity U, and
   *  returns the preconditioned right-hand side. */
  async prepare(U: Float32Array[]): Promise<Float64Array> {
    const { dt, re } = this.prm;
    const f = this.fns;
    U.forEach((u, c) => this.#host.upload(`U${c + 1}`, u));
    f.geometry.run();
    f.setup.run();
    this.area = (await f.setup.read('area'))[0];
    const R = Math.sqrt(this.area / (4 * Math.PI));
    const tables = sphereBlockTables(this.cfg, { mass: 1 / dt, disp: dt, re, be: Infinity, R });
    for (const name of TABLES) this.#host.upload(name, tables[name]);
    const params = { mass: 1 / dt, visc: 2 / re, dt };
    for (const fn of [f.rhs, f.matvec, f.advance]) fn.setParams(params);
    f.rhs.run();
    return this.pack(await f.rhs.readMany(['B1', 'B2', 'B3', 'BP']));
  }

  async apply(z: Float64Array): Promise<Float64Array> {
    this.unpack(z, 4).forEach((f, j) => this.#host.upload(`Z${j + 1}`, f));
    this.fns.matvec.run();
    return this.pack(await this.fns.matvec.readMany(['R1', 'R2', 'R3', 'RP']));
  }

  /** One BDF-1 step from velocity U and pressure P; the surface moves on the GPU. */
  async step(U: Float32Array[], P: Float32Array): Promise<{ U: Float32Array[]; P: Float32Array; info: StepInfo }> {
    const b = await this.prepare(U);
    const res = await gmres((z) => this.apply(z), b, this.pack([...U, P]), this.prm.gmres);
    const [U1, U2, U3, P1] = this.unpack(res.x, 4);
    [U1, U2, U3].forEach((u, c) => this.#host.upload(`U${c + 1}`, u));
    this.fns.advance.run();
    return { U: [U1, U2, U3], P: P1, info: { iterations: res.iterations, converged: res.converged, relResid: res.relResid } };
  }

  destroy(): void {
    for (const f of Object.values(this.fns)) f.destroy();
    this.#host.destroy();
    this.#deriv.destroy();
    this.#sht.destroy();
  }
}
