/**
 * Krause & Voigt's perturbed sphere (r = 1 + r0 cos(phi) sin(3 theta), at rest,
 * with bending and the volume constraint) on desktop WebGPU, BDF-1 with a fixed
 * time step, recording the energies and integrals at every step; optionally
 * compared with a run of the C++ solver's fluid_run with the same settings
 * (its diagnostics.csv, and the position at the final snapshot).
 *
 *   npx vite-node scripts/kv.ts -- --lmax 32 --dt 0.005 --T 0.5 --cpp <fluid_run save_dir>
 *
 * fluid_run -example perturbed_sphere -r0 0.4 -Re 1 -Be 2 -order 1 -lmax 32 -dt 0.005 -T 0.5
 *   -relabel_every 0 -frames 0 -snapshot_times 0.5 -save_dir <dir>
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { installWebGpu } from './nodeWebGpu.ts';
import { requestShtDevice } from '../src/sht/sht.ts';
import { relayoutCoeffs } from '../src/sht/layout.ts';
import { perturbedSphere, productConfig } from '../src/geom/geometry.ts';
import { MovingStokes, type FluidState } from '../src/fluid/movingStokes.ts';
import { readDump } from '../test/dump.ts';

const arg = (name: string, dflt: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const lmax = Number(arg('lmax', '32'));
const dt = Number(arg('dt', '0.005'));
const T = Number(arg('T', '0.5'));
const r0 = Number(arg('r0', '0.4'));
const re = Number(arg('Re', '1'));
const be = Number(arg('Be', '2'));
const cpp = arg('cpp', '');
const out = arg('out', `tmp/kv_L${lmax}.csv`);

await installWebGpu();
const device = await requestShtDevice();
const ms = await MovingStokes.create(device, lmax, {
  re, be, volume: true, dt, gmres: { tol: 1e-6, restart: 200, maxCycles: 4 },
});
const cfg = productConfig(lmax);
ms.setSurface(perturbedSphere(cfg, lmax, r0));
// At rest: zero velocity and pressure, product-band spectra.
const zero = () => new Float32Array((cfg.lmax + 1) * (cfg.lmax + 2));
let state: FluidState = { U: [zero(), zero(), zero()], P: zero(), lam: 0 };

interface Row { step: number; t: number; e_kin: number; e_h: number; area: number; volume: number; its: number; lam: number }
const rows: Row[] = [];
const record = async (step: number, its: number) => {
  const d = await ms.diagnostics(state.U);
  rows.push({ step, t: step * dt, e_kin: d.ekin, e_h: (2 / be) * d.h2, area: d.area, volume: d.volume, its, lam: state.lam });
};
await record(0, 0);
const nsteps = Math.round(T / dt);
const t0 = performance.now();
for (let k = 1; k <= nsteps; k++) {
  const res = await ms.step(state);
  state = res.state;
  await record(k, res.info.iterations);
  if (k % 20 === 0 || k === nsteps) {
    const r = rows[rows.length - 1];
    console.log(`step ${k} t ${r.t.toFixed(3)}: E_h ${r.e_h.toPrecision(8)} area ${r.area.toPrecision(8)} volume ${r.volume.toPrecision(8)} E_kin ${r.e_kin.toExponential(4)} its ${r.its}`);
  }
}
const sPerStep = (performance.now() - t0) / 1000 / nsteps;
const header = 'step,t,e_kin,e_h,area,volume,iterations,lam';
writeFileSync(out, [header, ...rows.map((r) => [r.step, r.t, r.e_kin, r.e_h, r.area, r.volume, r.its, r.lam].join(','))].join('\n') + '\n');
console.log(`wrote ${out}; ${(rows.reduce((s, r) => s + r.its, 0) / nsteps).toFixed(1)} GMRES its/step, ${sPerStep.toFixed(2)} s/step`);

if (cpp) {
  const lines = readFileSync(`${cpp}/diagnostics.csv`, 'utf8').trim().split('\n');
  const cols = lines[0].split(',');
  const c = lines.slice(1).map((l) => Object.fromEntries(l.split(',').map((v, i) => [cols[i], Number(v)])));
  const worst: Record<string, number> = { e_h: 0, area: 0, volume: 0, e_kin: 0 };
  for (const r of rows) {
    const q = c.find((x) => x.step === r.step);
    if (!q) continue;
    for (const k of ['e_h', 'area', 'volume'] as const) worst[k] = Math.max(worst[k], Math.abs(r[k] - q[k]) / Math.abs(q[k]));
    // The kinetic energy starts at zero, so it is compared against its peak.
    worst.e_kin = Math.max(worst.e_kin, Math.abs(r.e_kin - q.e_kin));
  }
  const ekinPeak = Math.max(...c.map((x) => x.e_kin));
  worst.e_kin /= ekinPeak;
  const itsC = c.slice(1).reduce((s, x) => s + x.iterations, 0) / (c.length - 1);
  console.log(`against the C++ over ${nsteps} steps, largest relative difference: ` +
    Object.entries(worst).map(([k, v]) => `${k} ${v.toExponential(1)}`).join(', ') +
    ` (e_kin relative to its peak ${ekinPeak.toExponential(3)}); C++ ${itsC.toFixed(1)} GMRES its/step at tol 1e-8`);
  // fluid_run's diagnostics.csv ends at the step before T; its last state is the snapshot.
  for (const t of [0, 0.1, 0.25, 0.5, 1, 2].filter((t) => t < T - 1e-12)) {
    const step = Math.round(t / dt);
    const r = rows[step];
    const q = c.find((x) => x.step === step);
    if (!q) continue;
    console.log(`  t ${t}: E_h ${r.e_h.toPrecision(7)} (C++ ${q.e_h.toPrecision(7)}), area ${r.area.toPrecision(7)} (${q.area.toPrecision(7)}), volume ${r.volume.toPrecision(7)} (${q.volume.toPrecision(7)}), E_kin ${r.e_kin.toExponential(3)} (${q.e_kin.toExponential(3)})`);
  }
  try {
    const snap = readDump(new URL(`file://${cpp.startsWith('/') ? '' : process.cwd() + '/'}${cpp}/snap_t${T}/`));
    const nlm = ((lmax + 1) * (lmax + 2)) / 2;
    const work = { lmax, mmax: lmax };
    const prod = { lmax: cfg.lmax, mmax: cfg.mmax };
    const Xc = [0, 1, 2].map((k) => relayoutCoeffs(new Float32Array(snap.arrays.state_X.subarray(2 * k * nlm, 2 * (k + 1) * nlm)), work, prod));
    const X = await ms.readSurface();
    const a = ms.pack(X);
    const b = ms.pack(Xc);
    let num = 0;
    let den = 0;
    for (let i = 0; i < a.length; i++) {
      num += (a[i] - b[i]) ** 2;
      den += b[i] ** 2;
    }
    console.log(`  final position against the C++ snapshot: rel L2 ${Math.sqrt(num / den).toExponential(1)}`);
  } catch (e) {
    console.log(`  (no snapshot compared: ${e instanceof Error ? e.message : e})`);
  }
}
process.exit(0);
