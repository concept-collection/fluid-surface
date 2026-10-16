/**
 * The moving surface without bending (models/stokes.m, src/fluid/movingStokes.ts),
 * against the C++ solver on its cpu backend in f64 on the same grid, for the
 * killing example: the unit sphere in rigid rotation, whose velocity is a
 * Killing field (test/fixtures/killing-cpu-L12, written by
 * scripts/cpp/step_dump.cpp with bending off).
 *
 * 1. The conversion between Cartesian components and radial, spheroidal and
 *    toroidal potentials (lib/cart2qst.m, lib/qst2cart.m), which the C++ does
 *    with recurrences on the coefficients and this does with transforms.
 * 2. The Stokes-preconditioned operator on a random vector, and the
 *    right-hand side.
 * 3. The state after one step and after twenty.
 *
 * 2 and 3 are repeated with the same rigid rotation started on Krause &
 * Voigt's perturbed sphere (r0 0.2; test/fixtures/rotating-perturbed-cpu-L12),
 * where it is not a Killing field, so the surface deforms.
 */
import { relayoutCoeffs } from '../src/sht/layout.ts';
import { ShtPlan } from '../src/sht/sht.ts';
import { DerivPlan } from '../src/sht/deriv.ts';
import { GpuFunction } from '../src/mgpu/gpuFunction.ts';
import { operatorLibs } from '../src/mgpu/libs.ts';
import type { Binding } from '../src/mgpu/compile.ts';
import { gridFrame, productConfig } from '../src/geom/geometry.ts';
import { MovingStokes } from '../src/fluid/movingStokes.ts';
import { spectralMasks, unpackCpp } from '../src/fluid/spectra.ts';
import { readDump } from './dump.ts';
import { relL2, fmt, type Check, type Log } from './check.ts';

const GMRES = { tol: 1e-6, restart: 200, maxCycles: 4 };

const QST_CHECK = `
function [Q, S, T, V1, V2, V3] = qstcheck(U1, U2, U3, ct, st, cp, sp, ilam, keep)
  [Q, S, T] = cart2qst(U1, U2, U3, ct, st, cp, sp, ilam);
  [V1, V2, V3] = qst2cart(Q .* keep, S .* keep, T .* keep, ct, st, cp, sp);
end
`;

export async function movingChecks(device: GPUDevice, check: Check, log: Log): Promise<void> {
  const dump = readDump(new URL('./fixtures/killing-cpu-L12/', import.meta.url));
  const s = dump.scalars;
  const lmax = s.lmax;
  const cfg = productConfig(lmax);
  const work = { lmax, mmax: lmax };
  const prod = { lmax: cfg.lmax, mmax: cfg.mmax };
  const nlm = ((lmax + 1) * (lmax + 2)) / 2;
  const field = (a: Float64Array, c: number) =>
    relayoutCoeffs(new Float32Array(a.subarray(2 * c * nlm, 2 * (c + 1) * nlm)), work, prod);
  const fields = (name: string, n: number) => Array.from({ length: n }, (_, c) => field(dump.arrays[name], c));
  const masks = spectralMasks(cfg, lmax);
  const keepOnly = (f: Float32Array) => f.map((v, i) => v * masks.keep[i]);

  // ---- 1. the potentials ----------------------------------------------------
  {
    const sht = await ShtPlan.create(device, cfg);
    const deriv = await DerivPlan.create(device, sht);
    const spec: Binding = { kind: 'tensor', shape: [2, sht.nlm] };
    const grid: Binding = { kind: 'tensor', shape: [cfg.nlat * cfg.nphi, 1] };
    const fn = await GpuFunction.create({
      device, sht, deriv, source: QST_CHECK, name: 'qstcheck', nargout: 6, libs: operatorLibs,
      bindings: { U1: spec, U2: spec, U3: spec, ilam: spec, keep: spec, ct: grid, st: grid, cp: grid, sp: grid },
    });
    for (const [k, v] of Object.entries(gridFrame(cfg))) fn.upload(k, v);
    fn.upload('ilam', masks.ilam);
    fn.upload('keep', masks.keep);
    fields('cart', 3).forEach((f, c) => fn.upload(`U${c + 1}`, f));
    fn.run();
    const out = await fn.readMany(['Q', 'S', 'T', 'V1', 'V2', 'V3']);
    const qst = fields('qst', 3);
    const back = fields('cart_back', 3);
    const eQ = ['Q', 'S', 'T'].map((n, i) => `${n} ${fmt(relL2(keepOnly(out[i]), qst[i]))}`);
    const eB = Math.max(...[0, 1, 2].map((c) => relL2(keepOnly(out[3 + c]), back[c])));
    log(`  cart2qst + qst2cart: ${fn.describe().length} GPU operations`);
    check(
      'moving: Cartesian <-> radial/spheroidal/toroidal potentials match the C++ recurrences, rel L2',
      Math.max(...[0, 1, 2].map((i) => relL2(keepOnly(out[i]), qst[i]))) < 1e-5 && eB < 1e-5,
      `${eQ.join(', ')}; back to Cartesian ${fmt(eB)}`,
    );
    fn.destroy();
    deriv.destroy();
    sht.destroy();
  }

  for (const [dir, label] of [['killing-cpu-L12', 'killing example'], ['rotating-perturbed-cpu-L12', 'rotating perturbed sphere']]) {
    await stepChecks(device, check, log, readDump(new URL(`./fixtures/${dir}/`, import.meta.url)), label);
  }
}

async function stepChecks(device: GPUDevice, check: Check, log: Log, dump: ReturnType<typeof readDump>, label: string): Promise<void> {
  const s = dump.scalars;
  const lmax = s.lmax;
  const cfg = productConfig(lmax);
  const work = { lmax, mmax: lmax };
  const prod = { lmax: cfg.lmax, mmax: cfg.mmax };
  const nlm = ((lmax + 1) * (lmax + 2)) / 2;
  const field = (a: Float64Array, c: number) =>
    relayoutCoeffs(new Float32Array(a.subarray(2 * c * nlm, 2 * (c + 1) * nlm)), work, prod);
  const fields = (name: string, n: number) => Array.from({ length: n }, (_, c) => field(dump.arrays[name], c));
  const ms = await MovingStokes.create(device, lmax, { re: s.re, be: s.bending ? s.be : Infinity, dt: s.dt, gmres: GMRES });
  const ops = (n: keyof MovingStokes['fns']) => ms.fns[n].describe().length;
  log(`  stokes.m compiles to: setup ${ops('setup')}, rhs ${ops('rhs')}, matvec ${ops('matvec')}, advance ${ops('advance')} GPU operations`);
  const fromCpp = (packed: Float64Array) =>
    ms.pack(unpackCpp(packed, lmax, 4).map((c) => relayoutCoeffs(new Float32Array(c), work, prod)));
  ms.setSurface(fields('X0', 3));
  const U0 = fields('u0', 3);
  const b = await ms.prepare(U0);
  const eB = relL2(b, fromCpp(dump.arrays.b));
  const eA = relL2(await ms.apply(fromCpp(dump.arrays.z)), fromCpp(dump.arrays.Az));
  check(
    `moving: ${label}: Stokes-preconditioned operator and right-hand side match the C++ (fp32 vs f64), rel L2`,
    eA < 1e-5 && eB < 1e-5,
    `A z ${fmt(eA)}, b ${fmt(eB)}; area ${ms.area.toPrecision(8)} (C++ ${s.area.toPrecision(8)})`,
  );

  let U = U0;
  let P: Float32Array = new Float32Array(U0[0].length);
  let its = 0;
  const t0 = performance.now();
  for (let k = 1; k <= s.nsteps; k++) {
    const out = await ms.step(U, P);
    U = out.U;
    P = out.P;
    its += out.info.iterations;
    if (k === 1 || k === s.nsteps) {
      const tag = k === 1 ? '1' : 'N';
      const X = await ms.readSurface();
      const eX = relL2(ms.pack(X), ms.pack(fields(`X${tag}`, 3)));
      const eU = relL2(ms.pack(U), ms.pack(fields(`u${tag}`, 3)));
      const eP = relL2(ms.pack([P]), ms.pack([field(dump.arrays[`p${tag}`], 0)]));
      check(
        `moving: ${label} after ${k} step${k > 1 ? 's' : ''} (dt ${s.dt}) matches the C++, rel L2`,
        out.info.converged && eX < 1e-5 && eU < 1e-4,
        `X ${fmt(eX)}, u ${fmt(eU)}, p ${fmt(eP)}; the surface has moved by ${fmt(relL2(ms.pack(X), ms.pack(fields('X0', 3))))}; ` +
          `${out.info.iterations} GMRES its this step`,
      );
    }
  }
  const msPerStep = (performance.now() - t0) / s.nsteps;
  log(`  ${label}, ${s.nsteps} steps: ${(its / s.nsteps).toFixed(1)} GMRES its/step (C++ at tol 1e-12: ${s.iterations_mean}), ${msPerStep.toFixed(0)} ms/step`);
  ms.destroy();
}
const JVP_CHECK = `
function [J1, J2, J3] = jvpcheck(Y1, Y2, Y3, sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3, ct, st, H, K, n1, n2, n3, dp1, p2, dq2, jinv, ba, bb, braw, bvc, lam, filtf, real0, keep, bscale)
  [J1, J2, J3] = bendjvp(Y1, Y2, Y3, sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3, ct, st, H, K, n1, n2, n3, dp1, p2, dq2, jinv, ba, bb, braw, bvc, lam, filtf, real0, keep, bscale);
end
`;

/** Bending on the rotating perturbed sphere: the force and its derivative
 *  along a random displacement, then the steps, against the C++ with bending on
 *  (test/fixtures/bending-perturbed-cpu-L12). */
export async function bendingChecks(device: GPUDevice, check: Check, log: Log): Promise<void> {
  const dump = readDump(new URL('./fixtures/bending-perturbed-cpu-L12/', import.meta.url));
  const s = dump.scalars;
  const lmax = s.lmax;
  const cfg = productConfig(lmax);
  const work = { lmax, mmax: lmax };
  const prod = { lmax: cfg.lmax, mmax: cfg.mmax };
  const nlm = ((lmax + 1) * (lmax + 2)) / 2;
  const field = (a: Float64Array, c: number) =>
    relayoutCoeffs(new Float32Array(a.subarray(2 * c * nlm, 2 * (c + 1) * nlm)), work, prod);
  const fields = (name: string, n: number) => Array.from({ length: n }, (_, c) => field(dump.arrays[name], c));

  const ms = await MovingStokes.create(device, lmax, { re: s.re, be: s.be, dt: s.dt, gmres: GMRES });
  ms.setSurface(fields('X0', 3));
  await ms.prepare(fields('u0', 3));
  const bf = await ms.fns.bendlin.readMany(['BF1', 'BF2', 'BF3']);
  const eF = relL2(ms.pack(bf), ms.pack(fields('bforce', 3)));

  const spec: Binding = { kind: 'tensor', shape: [2, ((cfg.lmax + 1) * (cfg.lmax + 2)) / 2] };
  const jvp = await ms.compile(JVP_CHECK, 'jvpcheck', 3, { Y1: spec, Y2: spec, Y3: spec });
  jvp.setParams({ bscale: -2 / s.be });
  fields('y', 3).forEach((f, c) => jvp.upload(`Y${c + 1}`, f));
  jvp.run();
  const jy = await jvp.readMany(['J1', 'J2', 'J3']);
  const eJ = relL2(ms.pack(jy), ms.pack(fields('jy', 3)));
  log(`  bendjvp: ${jvp.describe().length} GPU operations; bendlin ${ms.fns.bendlin.describe().length}`);
  check(
    `bending: force and its derivative along a random displacement match the C++ (Be ${s.be}), rel L2`,
    eF < 1e-5 && eJ < 1e-4,
    `force ${fmt(eF)}, derivative ${fmt(eJ)}`,
  );
  jvp.destroy();
  ms.destroy();
  await stepChecks(device, check, log, dump, `rotating perturbed sphere with bending (Be ${s.be})`);
}
