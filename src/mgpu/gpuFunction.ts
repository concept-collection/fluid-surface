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
    const plan = await inFunctionAsync(o.name, () =>
      ModelPlan.create(o.device, o.sht, { fn, feedback: fn.outputs.map(() => null) }, host, o.deriv),
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

  /** What the function compiled to, one line per GPU operation. */
  describe(): string[] {
    return this.#plan.describe();
  }

  destroy(): void {
    this.#plan.destroy();
    if (this.#ownsHost) this.#host.destroy();
  }
}
