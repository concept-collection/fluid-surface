/**
 * Time steps of the fluid solver on a moving surface, with bending and
 * optionally the volume constraint (models/stokes.m): BDF-1, solved by GMRES
 * (src/fluid/gmres.ts) on the preconditioned operator, after which the surface
 * moves with the fluid's normal velocity.
 *
 * Per step: lib/geometry.m and lib/posderivs.m on the current surface, setup
 * (the transport velocity, and the area, which the host reads to build the
 * sphere's per-degree blocks), bendlin (the bending force and what its
 * derivative reuses), volblock (with the constraint: the 2 x 2 block, which
 * the host reads), rhs, GMRES on matvec, and advance, whose new position
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
import posderivsSource from '../../lib/posderivs.m?raw';
import modelSource from '../../models/stokes.m?raw';

export interface MovingStokesParams {
  re: number;
  /** Bending number; Infinity turns bending off. */
  be: number;
  /** Constrain the enclosed volume. */
  volume: boolean;
  dt: number;
  gmres: GmresOptions;
}

const GEOMETRY_FED = ['vt1', 'vt2', 'vt3', 'vp1', 'vp2', 'vp3', 'n1', 'n2', 'n3', 'H', 'K', 'sqrtg', 'dp1', 'p2', 'dq2', 'jinv'];
/** The position's derivative fields (lib/posderivs.m), which the bending force's derivative reuses. */
const DERIVS = ['sxt', 'xp', 'sxtt', 'sxtp', 'xpp'].flatMap((n) => [1, 2, 3].map((c) => `${n}${c}`));
const BENDLIN_FED = ['BF1', 'BF2', 'BF3', 'ba', 'bb', 'braw', 'bvc'];
const TABLES = ['i11', 'i12', 'i13', 'i21', 'i22', 'i23', 'i31', 'i32', 'i33', 'w11', 'w12', 'w21', 'w22', 'itor'];
/** Relative determinant below which the volume block is treated as singular (as the C++). */
const SCALAR_BLOCK_TOL = 1e-6;

/** A state: velocity and pressure (product-band spectra) and the volume multiplier. */
export interface FluidState {
  U: Float32Array[];
  P: Float32Array;
  lam: number;
}

export class MovingStokes {
  readonly lmax: number;
  readonly cfg: ShtConfig;
  readonly prm: MovingStokesParams;
  readonly fns: Record<'geometry' | 'posderivs' | 'setup' | 'bendlin' | 'volblock' | 'rhs' | 'matvec' | 'advance' | 'diag' | 'position', GpuFunction>;
  /** The area at the last step's start. */
  area = 0;
  /** The volume constraint's 2 x 2 block at the last step's start, and whether it was used. */
  block: number[] = [];
  useBlock = false;
  #sht: ShtPlan;
  #deriv: DerivPlan;
  #host: HostBuffers;
  #bindings: Record<string, Binding> = {};
  #device!: GPUDevice;
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
    const bindings: Record<string, Binding> = {};
    for (const n of ['mass', 'visc', 'dt', 'disp', 'bscale', 'vol', 'cr00', 'crlam', 'ar00', 'arlam', 'ub', 'R', 'qdiv', 'y00'])
      bindings[n] = param;
    bindings.Z5 = { kind: 'tensor', shape: [1, 1] };
    for (const n of ['X1', 'X2', 'X3', 'U1', 'U2', 'U3', 'Z1', 'Z2', 'Z3', 'Z4', 'BF1', 'BF2', 'BF3', 'NC1', 'NC2', 'NC3',
      'keep', 'clean', 'filtf', 'ilam', 'cutv', 'lam', 'real0', 'e0', ...TABLES]) bindings[n] = spec;
    for (const n of ['ct', 'st', 'cp', 'sp', 'wq', 'v1', 'v2', 'v3', 'ba', 'bb', 'braw', 'bvc', ...GEOMETRY_FED, ...DERIVS])
      bindings[n] = grid;
    const fed = (names: string[]) => Object.fromEntries(names.map((n) => [n, n]));
    const make = (source: string, name: string, nargout: number, feedback?: Record<string, string>) =>
      GpuFunction.create({ device, sht, deriv, bindings, host, libs: operatorLibs, source, name, nargout, feedback });
    const fns = {
      geometry: await make(geometrySource, 'geometry', 16, fed(GEOMETRY_FED)),
      posderivs: await make(posderivsSource, 'posderivs', 15, fed(DERIVS)),
      setup: await make(modelSource, 'setup', 4, { v1: 'v1', v2: 'v2', v3: 'v3' }),
      bendlin: await make(modelSource, 'bendlin', 7, fed(BENDLIN_FED)),
      volblock: await make(modelSource, 'volblock', 7, fed(['NC1', 'NC2', 'NC3'])),
      rhs: await make(modelSource, 'rhs', 5),
      matvec: await make(modelSource, 'matvec', 5),
      advance: await make(modelSource, 'advance', 3, { Y1: 'X1', Y2: 'X2', Y3: 'X3' }),
      diag: await make(modelSource, 'diag', 4),
      position: await make(modelSource, 'position', 3),
    };
    const frame = gridFrame(cfg);
    const up = (name: string, data: Float32Array) => host.get(name) && host.upload(name, data);
    for (const [name, data] of Object.entries(frame)) up(name, data);
    up('wq', chartWeights(cfg));
    for (const [name, data] of Object.entries(spectralMasks(cfg, lmax))) up(name, data);
    const ms = new MovingStokes(lmax, cfg, prm, sht, deriv, host, fns);
    ms.#bindings = bindings;
    ms.#device = device;
    return ms;
  }

  /** Compiles another function against this stepper's buffers, so it can take
   *  the step's geometry and tables by name (for checks). */
  compile(source: string, name: string, nargout: number, extra: Record<string, Binding> = {}): Promise<GpuFunction> {
    return GpuFunction.create({
      device: this.#device, sht: this.#sht, deriv: this.#deriv, host: this.#host, libs: operatorLibs,
      bindings: { ...this.#bindings, ...extra }, source, name, nargout,
    });
  }

  /** Sets the surface from its position's product-band coefficients. */
  setSurface(X: Float32Array[]): void {
    X.forEach((x, c) => this.#host.upload(`X${c + 1}`, x));
  }

  async readSurface(): Promise<Float32Array[]> {
    return this.fns.advance.readMany(['X1', 'X2', 'X3']);
  }

  /** Packs product-band spectra, and the multiplier if given, into GMRES's unknowns. */
  pack(fields: ArrayLike<number>[], lam?: number): Float64Array {
    const m = this.#idx.length;
    const out = new Float64Array(fields.length * m + (lam === undefined ? 0 : 1));
    fields.forEach((f, j) => {
      for (let i = 0; i < m; i++) out[j * m + i] = f[this.#idx[i]];
    });
    if (lam !== undefined) out[fields.length * m] = lam;
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
    const { dt, re, be, volume } = this.prm;
    const f = this.fns;
    U.forEach((u, c) => this.#host.upload(`U${c + 1}`, u));
    f.geometry.run();
    f.posderivs.run();
    f.setup.run();
    this.area = (await f.setup.read('area'))[0];
    const R = Math.sqrt(this.area / (4 * Math.PI));
    const mass = 1 / dt;
    const tables = sphereBlockTables(this.cfg, { mass, disp: dt, re, be, R });
    for (const name of TABLES) this.#host.upload(name, tables[name]);
    const params: Record<string, number> = {
      mass, visc: 2 / re, dt, disp: dt, bscale: Number.isFinite(be) ? -2 / be : 0,
      R, qdiv: mass + 4 / (re * R * R), y00: 1 / (2 * Math.sqrt(Math.PI)),
      vol: 0, cr00: 0, crlam: 1, ar00: 0, arlam: 0, ub: 0,
    };
    if (volume) {
      f.volblock.setParams(params);
      f.volblock.run();
      const M = (await f.volblock.readMany(['m00', 'm01', 'm10', 'm11'])).map((v) => v[0]);
      const [m00, m01, m10, m11] = M;
      const det = m00 * m11 - m01 * m10;
      const scale = Math.abs(m00 * m11) + Math.abs(m01 * m10);
      this.block = M;
      this.useBlock = Math.abs(det) > SCALAR_BLOCK_TOL * scale;
      Object.assign(params, {
        vol: 1,
        ub: this.useBlock ? 1 : 0,
        cr00: this.useBlock ? m10 / det : 0,
        crlam: this.useBlock ? -m00 / det : -mass / this.area,
        ar00: -m11 / det,
        arlam: m01 / det,
      });
    }
    for (const fn of [f.bendlin, f.rhs, f.matvec, f.advance]) fn.setParams(params);
    f.bendlin.run();
    f.rhs.run();
    const [B1, B2, B3, BP, BL] = await f.rhs.readMany(['B1', 'B2', 'B3', 'BP', 'BL']);
    return this.pack([B1, B2, B3, BP], BL[0]);
  }

  async apply(z: Float64Array): Promise<Float64Array> {
    this.unpack(z, 4).forEach((f, j) => this.#host.upload(`Z${j + 1}`, f));
    this.#host.upload('Z5', new Float32Array([z[z.length - 1]]));
    this.fns.matvec.run();
    const [R1, R2, R3, RP, RL] = await this.fns.matvec.readMany(['R1', 'R2', 'R3', 'RP', 'RL']);
    return this.pack([R1, R2, R3, RP], RL[0]);
  }

  /** One BDF-1 step; the surface moves on the GPU. */
  async step(s: FluidState): Promise<{ state: FluidState; info: StepInfo }> {
    const b = await this.prepare(s.U);
    const res = await gmres((z) => this.apply(z), b, this.pack([...s.U, s.P], s.lam), this.prm.gmres);
    const [U1, U2, U3, P1] = this.unpack(res.x, 4);
    const lam = this.prm.volume ? res.x[res.x.length - 1] : 0;
    [U1, U2, U3].forEach((u, c) => this.#host.upload(`U${c + 1}`, u));
    this.fns.advance.run();
    return {
      state: { U: [U1, U2, U3], P: P1, lam },
      info: { iterations: res.iterations, converged: res.converged, relResid: res.relResid },
    };
  }

  /** The surface's position on the grid (x, y, z, each nlat * nphi). */
  async surfaceGrid(): Promise<Float32Array[]> {
    this.fns.position.run();
    return this.fns.position.readMany(['x1', 'x2', 'x3']);
  }

  /** A geometry field of the surface as of the last geometry run (diagnostics
   *  runs it on the current surface). */
  async geometryField(name: 'H' | 'K' | 'sqrtg'): Promise<Float32Array> {
    return this.fns.geometry.read(name);
  }

  /** Energies and integrals of a state on the current surface: kinetic energy,
   *  the integral of H^2, area and enclosed volume. */
  async diagnostics(U: Float32Array[]): Promise<{ ekin: number; h2: number; area: number; volume: number }> {
    U.forEach((u, c) => this.#host.upload(`U${c + 1}`, u));
    this.fns.geometry.run();
    this.fns.diag.run();
    const [ekin, h2, area, volume] = (await this.fns.diag.readMany(['ekin', 'h2', 'area', 'volume'])).map((v) => v[0]);
    return { ekin, h2, area, volume };
  }

  destroy(): void {
    for (const f of Object.values(this.fns)) f.destroy();
    this.#host.destroy();
    this.#deriv.destroy();
    this.#sht.destroy();
  }
}
