/**
 * Restarted GMRES with its Krylov basis on the GPU, around an operator that
 * is recorded into a command encoder (a compiled matvec). The host version
 * (src/fluid/gmres.ts) uploads a vector, runs the matvec and reads the result
 * back every iteration, and orthogonalizes on the CPU; here the basis, the
 * orthogonalization and the solution stay on the GPU, in fp32, as in the C++
 * solver's native GMRES (backend/native/gmres_native.cpp).
 *
 * A vector is the operator's inputs laid end to end (for the fluid solver,
 * four product-band spectra and the multiplier), with a weight per entry
 * that is 1 on the unknowns and 0 elsewhere, so the inner product is that of
 * the host's packed vectors. Each iteration copies basis vector k into the
 * operator's inputs, records the operator, copies its outputs into w, and
 * orthogonalizes w against the basis by classical Gram-Schmidt run twice
 * (all k + 1 inner products in one dispatch, then one update), which keeps
 * the basis orthogonal to rounding as modified Gram-Schmidt does, with two
 * dispatches per pass instead of 2(k + 1).
 *
 * The host keeps the small part: the Hessenberg columns (the two passes'
 * coefficients and the new norm) are read back in batches, and the Givens
 * rotations, the residual estimate and the least-squares solve are done in
 * f64, as in the host version. A batch of iterations is recorded into one
 * submission and read back with one wait; its size is the number of
 * iterations the residual estimate's recent rate predicts are left (at most
 * `maxBatch`), so a solve overshoots its stopping point by little. As in the
 * host version, the solve stops when the residual estimate reaches the
 * tolerance, and the explicit residual is computed once at the end.
 */
import type { GmresOptions, GmresResult } from './gmres.ts';

/** A segment of the operator's input or output: a buffer and how many f32 it holds. */
export interface Segment {
  buffer: GPUBuffer;
  count: number;
}

export interface DeviceOperator {
  /** Where a vector's segments go before the operator runs, in order. */
  inputs: Segment[];
  /** Where its result's segments come from, in the same order and sizes. */
  outputs: Segment[];
  /** Records one application. */
  encode(encoder: GPUCommandEncoder): void;
}

const WG = 256;

type Pipe = 'dots0' | 'dots1' | 'update0' | 'update1' | 'norm' | 'scale' | 'resid' | 'combine';

const shader = (n: number, m1: number) => /* wgsl */ `
struct Prm { k: u32, hcol: u32, hdst: u32, vdst: u32 }
@group(0) @binding(0) var<storage, read_write> V: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> h: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<storage, read> b: array<f32>;
@group(0) @binding(6) var<storage, read> y: array<f32>;
@group(0) @binding(7) var<uniform> prm: Prm;
override PASS: u32 = 0u;
const N: u32 = ${n}u;
const M1: u32 = ${m1}u;
var<workgroup> part: array<f32, ${WG}>;

fn reduce(lid: u32, v: f32) -> f32 {
  part[lid] = v;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
    if (lid < s) { part[lid] = part[lid] + part[lid + s]; }
    workgroupBarrier();
  }
  return part[0];
}

// h[hcol + PASS * M1 + j] = <V_j, w>, one workgroup per j <= k.
@compute @workgroup_size(${WG})
fn dots(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let j = g.x;
  var acc = 0.0;
  for (var i = l.x; i < N; i = i + ${WG}u) { acc = acc + wt[i] * V[j * N + i] * w[i]; }
  let t = reduce(l.x, acc);
  if (l.x == 0u) { h[prm.hcol + PASS * M1 + j] = t; }
}

// w -= sum_j h[hcol + PASS * M1 + j] V_j.
@compute @workgroup_size(${WG})
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  var acc = 0.0;
  for (var j = 0u; j <= prm.k; j = j + 1u) { acc = acc + h[prm.hcol + PASS * M1 + j] * V[j * N + i]; }
  w[i] = w[i] - acc;
}

// h[hdst] = |w|, one workgroup.
@compute @workgroup_size(${WG})
fn norm(@builtin(local_invocation_id) l: vec3<u32>) {
  var acc = 0.0;
  for (var i = l.x; i < N; i = i + ${WG}u) { acc = acc + wt[i] * w[i] * w[i]; }
  let t = reduce(l.x, acc);
  if (l.x == 0u) { h[prm.hdst] = sqrt(t); }
}

// V_vdst = w / h[hdst], zero off the unknowns (and everywhere at a breakdown).
@compute @workgroup_size(${WG})
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let r = h[prm.hdst];
  V[prm.vdst * N + i] = select(0.0, wt[i] * w[i] / r, r > 0.0);
}

// w = b - w: the residual, once w holds A x.
@compute @workgroup_size(${WG})
fn resid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  w[i] = wt[i] * (b[i] - w[i]);
}

// x += sum_{j <= k} y_j V_j.
@compute @workgroup_size(${WG})
fn combine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  var acc = 0.0;
  for (var j = 0u; j <= prm.k; j = j + 1u) { acc = acc + y[j] * V[j * N + i]; }
  x[i] = x[i] + acc;
}
`;

export interface DeviceGmresOptions {
  /** Krylov dimension per cycle. */
  restart: number;
  /** Most iterations recorded into one submission. */
  maxBatch?: number;
}

export class DeviceGmres {
  readonly n: number;
  readonly restart: number;
  #device: GPUDevice;
  #op: DeviceOperator;
  #maxBatch: number;
  /** Floats per Hessenberg column: two passes of M1 coefficients and the norm. */
  #s: number;
  #buf: Record<'V' | 'w' | 'wt' | 'h' | 'x' | 'b' | 'y' | 'prm', GPUBuffer>;
  #pipes: Record<Pipe, GPUComputePipeline>;
  #bind: GPUBindGroup;
  /** Uniform slots: one per iteration k (0..restart-1), then the residual's. */
  #slotBytes: number;
  #wt: Float32Array;

  private constructor(device: GPUDevice, op: DeviceOperator, wt: Float32Array, opt: DeviceGmresOptions) {
    this.#device = device;
    this.#op = op;
    this.n = wt.length;
    this.restart = opt.restart;
    this.#maxBatch = opt.maxBatch ?? 8;
    this.#wt = wt;
    const n = this.n;
    const m1 = this.restart + 1;
    this.#s = 2 * m1 + 1;
    this.#slotBytes = Math.max(256, device.limits.minUniformBufferOffsetAlignment);
    const storage = (label: string, count: number) =>
      device.createBuffer({ label, size: 4 * Math.max(count, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.#buf = {
      V: storage('gmres-V', m1 * n),
      w: storage('gmres-w', n),
      wt: storage('gmres-wt', n),
      h: storage('gmres-h', m1 * this.#s + 1),
      x: storage('gmres-x', n),
      b: storage('gmres-b', n),
      y: storage('gmres-y', m1),
      prm: device.createBuffer({ label: 'gmres-prm', size: (this.restart + 1) * this.#slotBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };
    device.queue.writeBuffer(this.#buf.wt, 0, wt as Float32Array<ArrayBuffer>);
    // Slot k: iteration k's column; slot `restart`: the residual's norm, after the columns.
    const slots = new Uint32Array(((this.restart + 1) * this.#slotBytes) / 4);
    for (let k = 0; k < this.restart; k++) {
      const o = (k * this.#slotBytes) / 4;
      slots.set([k, k * this.#s, k * this.#s + 2 * m1, k + 1], o);
    }
    slots.set([0, 0, m1 * this.#s, 0], (this.restart * this.#slotBytes) / 4);
    device.queue.writeBuffer(this.#buf.prm, 0, slots);

    const module = device.createShaderModule({ label: 'gmres', code: shader(n, m1) });
    const layout = device.createBindGroupLayout({
      label: 'gmres',
      entries: [
        ...[0, 1, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } })),
        ...[2, 5, 6].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' as const } })),
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' as const, hasDynamicOffset: true } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ label: 'gmres', bindGroupLayouts: [layout] });
    const pipe = (entryPoint: string, pass?: number) => device.createComputePipeline({
      label: `gmres-${entryPoint}${pass ?? ''}`, layout: pipelineLayout,
      compute: { module, entryPoint, constants: pass === undefined ? {} : { PASS: pass } },
    });
    this.#pipes = {
      dots0: pipe('dots', 0), dots1: pipe('dots', 1), update0: pipe('update', 0), update1: pipe('update', 1),
      norm: pipe('norm'), scale: pipe('scale'), resid: pipe('resid'), combine: pipe('combine'),
    };
    const B = this.#buf;
    this.#bind = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: B.V } }, { binding: 1, resource: { buffer: B.w } },
        { binding: 2, resource: { buffer: B.wt } }, { binding: 3, resource: { buffer: B.h } },
        { binding: 4, resource: { buffer: B.x } }, { binding: 5, resource: { buffer: B.b } },
        { binding: 6, resource: { buffer: B.y } },
        { binding: 7, resource: { buffer: B.prm, size: 16 } },
      ],
    });
  }

  /** `wt` holds one weight per entry of a vector: 1 on the unknowns, 0 elsewhere. */
  static async create(device: GPUDevice, op: DeviceOperator, wt: Float32Array, opt: DeviceGmresOptions): Promise<DeviceGmres> {
    const total = op.inputs.reduce((s, g) => s + g.count, 0);
    if (total !== wt.length || op.outputs.reduce((s, g) => s + g.count, 0) !== total) {
      throw new Error('DeviceGmres: the inputs, outputs and weights must have the same length');
    }
    device.pushErrorScope('validation');
    const g = new DeviceGmres(device, op, wt, opt);
    const err = await device.popErrorScope();
    if (err) {
      g.destroy();
      throw new Error(`GMRES on the GPU: ${err.message}`);
    }
    return g;
  }

  #dispatch(pass: GPUComputePassEncoder, p: Pipe, slot: number, groups: number): void {
    pass.setPipeline(this.#pipes[p]);
    pass.setBindGroup(0, this.#bind, [slot * this.#slotBytes]);
    pass.dispatchWorkgroups(groups);
  }

  /** Records: the operator applied to the vector at `src` (byte offset) into w. */
  #apply(enc: GPUCommandEncoder, src: GPUBuffer, offset: number): void {
    let o = offset;
    for (const g of this.#op.inputs) {
      enc.copyBufferToBuffer(src, o, g.buffer, 0, 4 * g.count);
      o += 4 * g.count;
    }
    this.#op.encode(enc);
    o = 0;
    for (const g of this.#op.outputs) {
      enc.copyBufferToBuffer(g.buffer, 0, this.#buf.w, o, 4 * g.count);
      o += 4 * g.count;
    }
  }

  /** Records iteration k: w = A V_k, orthogonalized twice, its norm, V_{k+1}. */
  #iteration(enc: GPUCommandEncoder, k: number): void {
    this.#apply(enc, this.#buf.V, 4 * k * this.n);
    const pass = enc.beginComputePass({ label: `gmres-iteration-${k}` });
    const blocks = Math.ceil(this.n / WG);
    this.#dispatch(pass, 'dots0', k, k + 1);
    this.#dispatch(pass, 'update0', k, blocks);
    this.#dispatch(pass, 'dots1', k, k + 1);
    this.#dispatch(pass, 'update1', k, blocks);
    this.#dispatch(pass, 'norm', k, 1);
    this.#dispatch(pass, 'scale', k, blocks);
    pass.end();
  }

  async #read(offset: number, count: number): Promise<Float32Array> {
    const stage = this.#device.createBuffer({ size: 4 * count, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      const enc = this.#device.createCommandEncoder({ label: 'gmres-read' });
      enc.copyBufferToBuffer(this.#buf.h, 4 * offset, stage, 0, 4 * count);
      this.#device.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      return new Float32Array(stage.getMappedRange().slice(0));
    } finally {
      stage.destroy();
    }
  }

  /** Records r = b - A x into w, its norm into the residual slot and V_0 = r / |r|; returns |r|. */
  async #residual(): Promise<number> {
    const enc = this.#device.createCommandEncoder({ label: 'gmres-residual' });
    this.#apply(enc, this.#buf.x, 0);
    const pass = enc.beginComputePass({ label: 'gmres-residual' });
    const blocks = Math.ceil(this.n / WG);
    this.#dispatch(pass, 'resid', this.restart, blocks);
    this.#dispatch(pass, 'norm', this.restart, 1);
    this.#dispatch(pass, 'scale', this.restart, blocks);
    pass.end();
    this.#device.queue.submit([enc.finish()]);
    return (await this.#read((this.restart + 1) * this.#s, 1))[0];
  }

  /** The solution as the vector's entries, zero off the unknowns. */
  async readX(): Promise<Float32Array> {
    const stage = this.#device.createBuffer({ size: 4 * this.n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      const enc = this.#device.createCommandEncoder({ label: 'gmres-read-x' });
      enc.copyBufferToBuffer(this.#buf.x, 0, stage, 0, 4 * this.n);
      this.#device.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const x = new Float32Array(stage.getMappedRange().slice(0));
      for (let i = 0; i < this.n; i++) x[i] *= this.#wt[i];
      return x;
    } finally {
      stage.destroy();
    }
  }

  /**
   * Solves A x = b from x0 (both laid out as the operator's inputs). The
   * solution stays on the GPU; `readX` reads it back. `tol` and `maxCycles`
   * are as in the host version; its `restart` must not exceed this one's.
   */
  async solve(b: Float32Array, x0: Float32Array, opt: GmresOptions): Promise<Omit<GmresResult, 'x'>> {
    const m = Math.min(opt.restart, this.restart);
    this.#device.queue.writeBuffer(this.#buf.b, 0, b as Float32Array<ArrayBuffer>);
    this.#device.queue.writeBuffer(this.#buf.x, 0, x0 as Float32Array<ArrayBuffer>);
    let bnorm = 0;
    for (let i = 0; i < this.n; i++) bnorm += this.#wt[i] * b[i] * b[i];
    bnorm = Math.sqrt(bnorm) || 1;
    const goal = opt.tol * bnorm;
    let iterations = 0;
    let converged = false;
    let rn = await this.#residual();
    const m1 = this.restart + 1;
    for (let cycle = 0; cycle < opt.maxCycles && !converged; cycle++) {
      const beta = rn;
      if (beta <= goal) {
        converged = true;
        break;
      }
      const H: number[][] = [];
      const cs: number[] = [];
      const sn: number[] = [];
      const g = [beta];
      let k = 0;
      let done = false;
      while (k < m && !done) {
        // Iterations left at the recent rate of the residual estimate.
        let batch = this.#maxBatch;
        if (k >= 2) {
          const rate = Math.abs(g[k] / g[k - 2]) ** 0.5;
          if (rate > 0 && rate < 1) batch = Math.ceil(Math.log(goal / Math.abs(g[k])) / Math.log(rate));
        }
        batch = Math.max(1, Math.min(batch, this.#maxBatch, m - k));
        const enc = this.#device.createCommandEncoder({ label: 'gmres-batch' });
        for (let i = 0; i < batch; i++) this.#iteration(enc, k + i);
        this.#device.queue.submit([enc.finish()]);
        const cols = await this.#read(k * this.#s, batch * this.#s);
        for (let i = 0; i < batch && !done; i++, k++) {
          const c = cols.subarray(i * this.#s, (i + 1) * this.#s);
          const h = new Array<number>(k + 2).fill(0);
          for (let j = 0; j <= k; j++) h[j] = c[j] + c[m1 + j];
          const wnorm = c[2 * m1];
          h[k + 1] = wnorm;
          for (let j = 0; j < k; j++) {
            const t = cs[j] * h[j] + sn[j] * h[j + 1];
            h[j + 1] = -sn[j] * h[j] + cs[j] * h[j + 1];
            h[j] = t;
          }
          const rho = Math.hypot(h[k], h[k + 1]);
          cs.push(h[k] / rho);
          sn.push(h[k + 1] / rho);
          g.push(-sn[k] * g[k]);
          g[k] = cs[k] * g[k];
          h[k] = rho;
          h[k + 1] = 0;
          H.push(h);
          iterations++;
          if (Math.abs(g[k + 1]) <= goal || wnorm === 0) {
            converged = true;
            done = true;
          }
        }
      }
      if (k === 0) break;
      // Back-substitute for the basis coefficients and update x on the GPU.
      const y = new Float32Array(k);
      const yd = new Float64Array(k);
      for (let i = k - 1; i >= 0; i--) {
        let s = g[i];
        for (let j = i + 1; j < k; j++) s -= H[j][i] * yd[j];
        yd[i] = s / H[i][i];
        y[i] = yd[i];
      }
      this.#device.queue.writeBuffer(this.#buf.y, 0, y);
      const enc = this.#device.createCommandEncoder({ label: 'gmres-combine' });
      const pass = enc.beginComputePass({ label: 'gmres-combine' });
      this.#dispatch(pass, 'combine', k - 1, Math.ceil(this.n / WG));
      pass.end();
      this.#device.queue.submit([enc.finish()]);
      rn = await this.#residual();
    }
    return { iterations, converged, relResid: rn / bnorm };
  }

  destroy(): void {
    for (const b of Object.values(this.#buf)) b.destroy();
  }
}
