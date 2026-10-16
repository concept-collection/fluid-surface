/**
 * One .m function, compiled and runnable on the GPU.
 *
 * turing-surface compiles a model's `init` and `step` into plans that feed one
 * spectral state (its GpuModel). The fluid solver is made of separate
 * operators instead (a vector analysis, the geometry of a surface, a matvec)
 * that the host calls in its own order, so each is compiled on its own: the
 * .m declares its arguments, the host binds them by name, and the outputs are
 * read back or left on the GPU for the next function.
 */
import type { ShtPlan } from '../sht/sht.ts';
import type { DerivPlan } from '../sht/deriv.ts';
import { HostBuffers, ModelPlan } from './plan.ts';
import { CompiledModel, type Binding } from './compile.ts';
import { inFunction, inFunctionAsync, inModel } from './errors.ts';
import type { LibFile } from './libs.ts';

export interface GpuFunctionOptions {
  device: GPUDevice;
  sht: ShtPlan;
  /** Needed if the function calls dtheta, dphi or their shuffles. */
  deriv?: DerivPlan;
  /** The .m text holding the function. */
  source: string;
  /** The function to compile. */
  name: string;
  /** What the host offers, by argument name. */
  bindings: Record<string, Binding>;
  /** Outputs to keep, from the first. */
  nargout: number;
  /** Shared .m files the function may call. */
  libs?: LibFile[];
  /** Buffers shared with other functions; a fresh pool if omitted. */
  host?: HostBuffers;
  /** Outputs to copy into shared buffers after each call, by output name ->
   *  buffer name, so another function can take them as arguments with no
   *  readback. The buffers are created if the pool lacks them. */
  feedback?: Record<string, string>;
}

export class GpuFunction {
  readonly outputs: string[];
  #device: GPUDevice;
  #plan: ModelPlan;
  #host: HostBuffers;
  #ownsHost: boolean;

  private constructor(device: GPUDevice, plan: ModelPlan, host: HostBuffers, ownsHost: boolean, outputs: string[]) {
    this.#device = device;
    this.#plan = plan;
    this.#host = host;
    this.#ownsHost = ownsHost;
    this.outputs = outputs;
  }

  static async create(o: GpuFunctionOptions): Promise<GpuFunction> {
    const npts = o.sht.cfg.nlat * o.sht.cfg.nphi;
    const nlm = o.sht.nlm;
    const compiled = inModel(
      () => new CompiledModel(o.source, o.bindings, { npts, nlm }, `${o.name}.m`, o.libs ?? []),
    );
    const fn = inFunction(o.name, () => compiled.specialize(o.name, o.nargout));
    compiled.finish();
    const host = o.host ?? new HostBuffers(o.device);
    const feedback = fn.outputs.map((out) => {
      const to = o.feedback?.[out.name];
      if (!to) return null;
      const shape = out.ty.kind === 'Numeric' ? out.ty.shape : undefined;
      if (!shape) throw new Error(`${o.name}: output '${out.name}' has no static shape`);
      host.ensure(to, shape.reduce((x: number, y: number) => x * y, 1));
      return to;
    });
    const plan = await inFunctionAsync(o.name, () =>
      ModelPlan.create(o.device, o.sht, { fn, feedback }, host, o.deriv),
    );
    return new GpuFunction(o.device, plan, host, !o.host, fn.outputs.map((x) => x.name));
  }

  /** Write an argument's buffer. */
  upload(name: string, data: Float32Array): void {
    this.#host.upload(name, data);
  }

  setParams(values: Record<string, number>): void {
    this.#plan.setParams(values);
  }

  /** Record one call into an encoder. */
  encode(encoder: GPUCommandEncoder): void {
    this.#plan.encodeSteps(encoder, 1);
  }

  /** Record and submit one call. */
  run(): void {
    const enc = this.#device.createCommandEncoder({ label: 'gpu-function' });
    this.encode(enc);
    this.#device.queue.submit([enc.finish()]);
  }

  /**
   * GPU time of each operation of one call, in milliseconds, from timestamp
   * queries (each op in its own pass; see Plan.encodeTimed). Needs the
   * device's 'timestamp-query' feature; browsers may round the times
   * (Chrome to 100 us unless its WebGPU developer features are on).
   */
  async timeOps(): Promise<{ kind: string; label: string; ms: number }[]> {
    if (!this.#device.features.has('timestamp-query')) throw new Error('timeOps: no timestamp-query on this device');
    const n = this.#plan.opsLength;
    const querySet = this.#device.createQuerySet({ type: 'timestamp', count: 2 * n });
    const resolve = this.#device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const stage = this.#device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      const enc = this.#device.createCommandEncoder({ label: 'gpu-function-timed' });
      const ops = this.#plan.encodeTimed(enc, querySet);
      enc.resolveQuerySet(querySet, 0, 2 * n, resolve, 0);
      enc.copyBufferToBuffer(resolve, 0, stage, 0, 16 * n);
      this.#device.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const t = new BigInt64Array(stage.getMappedRange().slice(0));
      const out: { kind: string; label: string; ms: number }[] = [];
      ops.forEach((op, j) => {
        if (op) out.push({ ...op, ms: Number(t[2 * j + 1] - t[2 * j]) / 1e6 });
      });
      return out;
    } finally {
      querySet.destroy();
      resolve.destroy();
      stage.destroy();
    }
  }

  /** The buffer holding a named argument or output. */
  buffer(name: string): GPUBuffer | undefined {
    return this.#plan.buffer(name) ?? this.#host.get(name)?.buffer;
  }

  /** Read a named argument or output back to the CPU. */
  async read(name: string): Promise<Float32Array> {
    const buffer = this.buffer(name);
    const count = this.#plan.elementCount(name) ?? this.#host.get(name)?.count;
    if (!buffer || count === undefined) throw new Error(`read: no value named '${name}'`);
    const stage = this.#device.createBuffer({
      label: `gpu-function-read-${name}`,
      size: 4 * count,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
      const enc = this.#device.createCommandEncoder({ label: `gpu-function-read-${name}` });
      enc.copyBufferToBuffer(buffer, 0, stage, 0, 4 * count);
      this.#device.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      return new Float32Array(stage.getMappedRange().slice(0));
    } finally {
      stage.destroy();
    }
  }

  /** Read several named values back with one copy and one wait. */
  async readMany(names: string[]): Promise<Float32Array[]> {
    const parts = names.map((name) => {
      const buffer = this.buffer(name);
      const count = this.#plan.elementCount(name) ?? this.#host.get(name)?.count;
      if (!buffer || count === undefined) throw new Error(`readMany: no value named '${name}'`);
      return { buffer, count };
    });
    const total = parts.reduce((n, p) => n + p.count, 0);
    const stage = this.#device.createBuffer({
      label: 'gpu-function-read-many',
      size: 4 * total,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
      const enc = this.#device.createCommandEncoder({ label: 'gpu-function-read-many' });
      let offset = 0;
      for (const p of parts) {
        enc.copyBufferToBuffer(p.buffer, 0, stage, 4 * offset, 4 * p.count);
        offset += p.count;
      }
      this.#device.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const all = new Float32Array(stage.getMappedRange().slice(0));
      offset = 0;
      return parts.map((p) => {
        const out = all.subarray(offset, offset + p.count);
        offset += p.count;
        return out;
      });
    } finally {
      stage.destroy();
    }
  }

  /** What the function compiled to, one line per GPU operation. */
  describe(): string[] {
    return this.#plan.describe();
  }

  destroy(): void {
    this.#plan.destroy();
    if (this.#ownsHost) this.#host.destroy();
  }
}
