/**
 * Where a step of the moving solver (src/fluid/movingStokes.ts) spends its
 * time. A step is the operator's setup and right-hand side (prepare), then
 * GMRES on the host, each iteration uploading one vector, running the
 * compiled matvec and reading the result back (apply), then the surface
 * update. The profile splits the wall time of one step into those parts, and
 * an iteration into the matvec's GPU time and the rest (the upload, the wait
 * for the readback, packing on the host):
 *
 * - the matvec's GPU time is measured by running it `repeats` times back to
 *   back and reading one value, less the time of that read alone (when
 *   recording is the slower of the two, this measures recording instead);
 * - a bare round trip is one upload and one read of the matvec's outputs
 *   with no work between them;
 * - recording is the CPU time of the run() call alone: encoding every op
 *   into a command buffer and submitting it, which WebGPU cannot reuse from
 *   one call to the next;
 * - the per-op breakdown, from timestamp queries when the device has them,
 *   sums the matvec's ops by kind (transforms, coefficient shuffles,
 *   element-wise kernels, reductions).
 */
import type { MovingStokes, FluidState } from './movingStokes.ts';

export interface StepProfile {
  iterations: number;
  converged: boolean;
  /** Wall time of the whole step, ms. */
  step: number;
  /** Wall time of prepare (operator setup, right-hand side), ms. */
  prepare: number;
  /** Mean wall time of one matvec application, as GMRES sees it, ms. */
  apply: number;
  /** Its parts, means in ms: unpacking and uploading the vector, recording
   *  and submitting the matvec, waiting for its result to be read back,
   *  packing the result. */
  phases: { upload: number; record: number; wait: number; pack: number };
  /** Host time in GMRES itself over the step (orthogonalization, updates), ms. */
  gmresHost: number;
  /** GPU time of one matvec, from back-to-back runs, ms. */
  matvecGpu: number;
  /** One upload and readback of the matvec's vectors with nothing run, ms. */
  roundTrip: number;
  /** CPU time to record and submit one matvec (the run() call itself), ms. */
  encode: number;
  /** One matvec run alone, from submit to its result read back, ms. */
  single: number;
  /** GPU time of the matvec's ops by kind, ms, with their counts; empty
   *  without timestamp queries. */
  byKind: { kind: string; count: number; ms: number }[];
}

const now = (): number => performance.now();

/** Profiles one step from `state`; the solver's state advances as in a normal step. */
export async function profileStep(device: GPUDevice, ms: MovingStokes, state: FluidState, repeats = 20): Promise<{ profile: StepProfile; state: FluidState }> {
  // Wrap the instance's prepare and apply, which step() calls through `this`.
  let prepare = 0;
  let apply = 0;
  let applies = 0;
  const proto = Object.getPrototypeOf(ms) as MovingStokes;
  ms.prepare = async (U) => {
    const t = now();
    const b = await proto.prepare.call(ms, U);
    prepare += now() - t;
    return b;
  };
  // The same steps as MovingStokes.apply, timed one by one.
  const phase = { upload: 0, record: 0, wait: 0, pack: 0 };
  const mv0 = ms.fns.matvec;
  ms.apply = async (z) => {
    const t = now();
    ms.unpack(z, 4).forEach((f, j) => mv0.upload(`Z${j + 1}`, f));
    mv0.upload('Z5', new Float32Array([z[z.length - 1]]));
    const t1 = now();
    mv0.run();
    const t2 = now();
    const [R1, R2, R3, RP, RL] = await mv0.readMany(['R1', 'R2', 'R3', 'RP', 'RL']);
    const t3 = now();
    const r = ms.pack([R1, R2, R3, RP], RL[0]);
    const t4 = now();
    phase.upload += t1 - t;
    phase.record += t2 - t1;
    phase.wait += t3 - t2;
    phase.pack += t4 - t3;
    apply += t4 - t;
    applies++;
    return r;
  };
  let res;
  const t0 = now();
  try {
    res = await ms.step(state);
  } finally {
    delete (ms as Partial<MovingStokes>).prepare;
    delete (ms as Partial<MovingStokes>).apply;
  }
  // The surface update is a submit with no wait, so the rest is GMRES's own.
  const step = now() - t0;

  const mv = ms.fns.matvec;
  const outputs = ['R1', 'R2', 'R3', 'RP', 'RL'];
  const median = async (f: () => Promise<number>, n = 5): Promise<number> => {
    const xs: number[] = [];
    for (let i = 0; i < n; i++) xs.push(await f());
    return xs.sort((a, b) => a - b)[n >> 1];
  };
  const readOne = async (): Promise<number> => {
    const t = now();
    await mv.read('RL');
    return now() - t;
  };
  const runs = async (): Promise<number> => {
    const t = now();
    for (let i = 0; i < repeats; i++) mv.run();
    await mv.read('RL');
    return now() - t;
  };
  const z = await mv.readMany(['Z1', 'Z2', 'Z3', 'Z4']);
  const roundTrip = await median(async () => {
    const t = now();
    z.forEach((f, j) => mv.upload(`Z${j + 1}`, f));
    await mv.readMany(outputs);
    return now() - t;
  });
  await runs(); // warm up
  const matvecGpu = (await median(runs) - await median(readOne)) / repeats;
  const encode = await median(async () => {
    const t = now();
    mv.run();
    const dt = now() - t;
    await mv.read('RL');
    return dt;
  });
  const single = await median(async () => {
    const t = now();
    mv.run();
    await mv.read('RL');
    return now() - t;
  });

  const byKind: StepProfile['byKind'] = [];
  if (device.features.has('timestamp-query')) {
    const kindOf = (k: string): string =>
      k.startsWith('synth') || k.startsWith('analys') ? 'transforms' :
      k === 'dthetac' || k === 'dthetat' || k === 'dphic' || k === 'dtheta' || k === 'dphi' ? 'shuffles' :
      k === 'kernel' ? 'element-wise kernels' : k === 'dot' ? 'reductions' : k;
    const ops = await mv.timeOps();
    for (const op of ops) {
      const kind = kindOf(op.kind);
      let row = byKind.find((r) => r.kind === kind);
      if (!row) byKind.push((row = { kind, count: 0, ms: 0 }));
      row.count++;
      row.ms += op.ms;
    }
  }
  return {
    profile: {
      iterations: res.info.iterations, converged: res.info.converged, step, prepare, apply: apply / Math.max(applies, 1),
      phases: Object.fromEntries(Object.entries(phase).map(([k, v]) => [k, v / Math.max(applies, 1)])) as StepProfile['phases'],
      gmresHost: step - prepare - apply, matvecGpu, roundTrip, encode, single, byKind,
    },
    state: res.state,
  };
}

/** A profile as lines of text. */
export function formatProfile(p: StepProfile): string[] {
  const f = (x: number) => x.toFixed(2);
  const its = p.iterations;
  const lines = [
    `step ${f(p.step)} ms: prepare ${f(p.prepare)}, ${its} matvec applications ${f(p.apply * its)} (${f(p.apply)} each), GMRES on the host ${f(p.gmresHost)}`,
    `one application ${f(p.apply)} ms in GMRES: upload ${f(p.phases.upload)}, record and submit ${f(p.phases.record)}, wait for the result ${f(p.phases.wait)}, pack ${f(p.phases.pack)}`,
    `measured apart: matvec on the GPU ${f(p.matvecGpu)} (back-to-back runs), recording ${f(p.encode)}, a bare upload and readback ${f(p.roundTrip)}, one run alone from submit to readback ${f(p.single)}`,
  ];
  if (p.byKind.length) {
    const total = p.byKind.reduce((s, r) => s + r.ms, 0);
    lines.push(`matvec ops, timed one pass each (sum ${f(total)} ms): ` +
      p.byKind.map((r) => `${r.kind} ${r.count} ops ${f(r.ms)} ms`).join(', '));
  }
  return lines;
}
