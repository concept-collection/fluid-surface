/**
 * lib/sphtor.m, the vector analysis built from scalar transforms, against
 * SHTns' own vector transforms and against exact answers.
 *
 * Three questions, kept apart:
 *
 * 1. Is the formula right? In f64 (src/sht/reference.ts sphtor), against
 *    SHTns 3.7.5 on the same Gauss grid (test/fixtures, written by
 *    scripts/shtns_sphtor_fixture.py). The synthesis convention is checked
 *    first, so a sign error in T cannot hide behind a matching one in the
 *    synthesis. The analysis must agree for random grid values too, not just
 *    band-limited fields: the quadrature sums are the same sums.
 * 2. Is the compiled .m the formula? The numbl-compiled WGSL, in fp32,
 *    against the same fixture.
 * 3. How accurate is it in fp32 as lmax grows? Against exact S, T at
 *    L = 16, 32, 64, where the dividing by sin(theta) could cost precision.
 *
 * Degree lmax itself is excluded throughout and reported apart: S and T there
 * would need the analysis at lmax+1 (see lib/sphtor.m).
 */
import { ShtPlan } from '../src/sht/sht.ts';
import { DerivPlan } from '../src/sht/deriv.ts';
import { ShtReference } from '../src/sht/reference.ts';
import { gridForLmax, lmIndex, nlmCalc, type ShtConfig } from '../src/sht/layout.ts';
import { gaussNodesWeights } from '../src/sht/gauss.ts';
import { GpuFunction } from '../src/mgpu/gpuFunction.ts';
import sphtorSource from '../lib/sphtor.m?raw';
import fixture16 from './fixtures/sphtor-shtns-L16.json';
import { relL2, fmt, type Check, type Log } from './check.ts';

interface FixtureCase {
  name: string;
  S_in?: number[];
  T_in?: number[];
  vt: number[];
  vp: number[];
  S: number[];
  T: number[];
}

interface Fixture {
  lmax: number;
  nlat: number;
  nphi: number;
  ct: number[];
  cases: FixtureCase[];
}

/** Degree of each entry of a 2 x nlm spectral array. */
function degrees(cfg: ShtConfig): Int32Array {
  const deg = new Int32Array(2 * nlmCalc(cfg.lmax, cfg.mmax));
  for (let m = 0; m <= cfg.mmax; m++) {
    for (let l = m; l <= cfg.lmax; l++) {
      const i = lmIndex(cfg.lmax, l, m);
      deg[2 * i] = l;
      deg[2 * i + 1] = l;
    }
  }
  return deg;
}

/** 1/(l(l+1)), 0 at l = 0, in the 2 x nlm layout. */
function inverseEigenvalues(cfg: ShtConfig): Float32Array {
  const deg = degrees(cfg);
  return Float32Array.from(deg, (l) => (l === 0 ? 0 : 1 / (l * (l + 1))));
}

/** sin(theta) on the grid, from the f64 Gauss nodes. */
function sinTheta(cfg: ShtConfig): Float32Array {
  const { x } = gaussNodesWeights(cfg.nlat);
  const st = new Float32Array(cfg.nlat * cfg.nphi);
  for (let i = 0; i < cfg.nlat; i++) {
    st.fill(Math.sqrt(1 - x[i] * x[i]), i * cfg.nphi, (i + 1) * cfg.nphi);
  }
  return st;
}

/** Random S, T stopping at degree lmax-1, so the f64 synthesis is exact. */
function randomPotentials(cfg: ShtConfig, seed: number): { S: Float64Array; T: Float64Array } {
  let s = seed >>> 0;
  const rand = (): number => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32 - 0.5;
  };
  const nlm = nlmCalc(cfg.lmax, cfg.mmax);
  const S = new Float64Array(2 * nlm);
  const T = new Float64Array(2 * nlm);
  for (let m = 0; m <= cfg.mmax; m++) {
    for (let l = Math.max(m, 1); l < cfg.lmax; l++) {
      const i = lmIndex(cfg.lmax, l, m);
      for (const q of [S, T]) {
        q[2 * i] = rand() / (1 + l);
        q[2 * i + 1] = m === 0 ? 0 : rand() / (1 + l);
      }
    }
  }
  return { S, T };
}

async function compileSphtor(device: GPUDevice, cfg: ShtConfig) {
  const sht = await ShtPlan.create(device, cfg);
  const deriv = await DerivPlan.create(device, sht);
  const npts = cfg.nlat * cfg.nphi;
  const fn = await GpuFunction.create({
    device,
    sht,
    deriv,
    source: sphtorSource,
    name: 'sphtor',
    bindings: {
      vt: { kind: 'tensor', shape: [npts, 1] },
      vp: { kind: 'tensor', shape: [npts, 1] },
      st: { kind: 'tensor', shape: [npts, 1] },
      ilam: { kind: 'tensor', shape: [2, sht.nlm] },
    },
    nargout: 2,
  });
  fn.upload('st', sinTheta(cfg));
  fn.upload('ilam', inverseEigenvalues(cfg));
  const run = async (vt: ArrayLike<number>, vp: ArrayLike<number>) => {
    fn.upload('vt', Float32Array.from(vt));
    fn.upload('vp', Float32Array.from(vp));
    fn.run();
    return { S: await fn.read('S'), T: await fn.read('T') };
  };
  const destroy = (): void => {
    fn.destroy();
    deriv.destroy();
    sht.destroy();
  };
  return { fn, run, destroy };
}

export async function sphtorChecks(device: GPUDevice, check: Check, log: Log): Promise<void> {
  const fx = fixture16 as Fixture;
  const cfg16: ShtConfig = { lmax: fx.lmax, mmax: fx.lmax, nlat: fx.nlat, nphi: fx.nphi };
  const deg16 = degrees(cfg16);
  const below = (L: number, deg: Int32Array) => (i: number) => deg[i] < L;
  const top = (L: number, deg: Int32Array) => (i: number) => deg[i] === L;

  // ---- 1. the formula, in f64, against SHTns ------------------------------
  const grid = gridForLmax(fx.lmax, 1);
  const ref16 = new ShtReference(cfg16);
  const ctErr = Math.max(...fx.ct.map((c, i) => Math.abs(c - ref16.ct[i])));
  check(
    'sphtor: the fixture is on our grid',
    grid.nlat === fx.nlat && grid.nphi === fx.nphi && ctErr < 1e-14,
    `${fx.nlat} x ${fx.nphi}, Gauss nodes agree to ${fmt(ctErr)}`,
  );

  const band = fx.cases.find((c) => c.name === 'band')!;
  const syn = ref16.sphtorSynth(band.S_in!, band.T_in!);
  const eSyn = Math.max(relL2(syn.vt, band.vt), relL2(syn.vp, band.vp));
  check(
    'sphtor: f64 vector synthesis matches SHTns (the convention)',
    eSyn < 1e-12,
    `rel L2 ${fmt(eSyn)}`,
  );

  for (const c of fx.cases) {
    const r = ref16.sphtor(c.vt, c.vp);
    const eS = relL2(r.S, c.S, below(fx.lmax, deg16));
    const eT = relL2(r.T, c.T, below(fx.lmax, deg16));
    // The top degree's error, as a fraction of the whole field.
    const eTop = relL2(r.S, c.S, top(fx.lmax, deg16)) * Math.sqrt(
      c.S.reduce((s, v, i) => s + (deg16[i] === fx.lmax ? v * v : 0), 0) /
        c.S.reduce((s, v) => s + v * v, 0),
    );
    check(
      `sphtor: f64 formula matches SHTns' vector analysis, ${c.name}, l < lmax`,
      eS < 1e-12 && eT < 1e-12,
      `S ${fmt(eS)}, T ${fmt(eT)}; at l = lmax, S is off by ${fmt(eTop)} of its norm`,
    );
  }

  // ---- 2. the compiled .m, fp32, against SHTns ----------------------------
  const k16 = await compileSphtor(device, cfg16);
  const ops = k16.fn.describe();
  log(`  sphtor.m compiles to ${ops.length} GPU operations:\n    ${ops.join('\n    ')}`);
  for (const c of fx.cases) {
    const g = await k16.run(c.vt, c.vp);
    const eS = relL2(g.S, c.S, below(fx.lmax, deg16));
    const eT = relL2(g.T, c.T, below(fx.lmax, deg16));
    check(
      `sphtor: compiled sphtor.m (fp32) matches SHTns, ${c.name}, L${fx.lmax}`,
      eS < 1e-5 && eT < 1e-5,
      `S ${fmt(eS)}, T ${fmt(eT)}`,
    );
  }
  k16.destroy();

  // ---- 3. fp32 accuracy against exact answers, L16..64 --------------------
  for (const lmax of [16, 32, 64]) {
    const { nlat, nphi } = gridForLmax(lmax, 1);
    const cfg: ShtConfig = { lmax, mmax: lmax, nlat, nphi };
    const ref = new ShtReference(cfg);
    const deg = degrees(cfg);
    const { S, T } = randomPotentials(cfg, 7 + lmax);
    const { vt, vp } = ref.sphtorSynth(S, T);
    const k = await compileSphtor(device, cfg);
    const g = await k.run(vt, vp);
    const eS = relL2(g.S, S, below(lmax, deg));
    const eT = relL2(g.T, T, below(lmax, deg));
    // The worst degree, to see whether the error gathers anywhere.
    let worst = 0;
    let worstL = 0;
    for (let l = 1; l < lmax; l++) {
      const e = relL2(g.S, S, (i) => deg[i] === l);
      if (e > worst) [worst, worstL] = [e, l];
    }
    check(
      `sphtor: fp32 against exact S, T at L${lmax} (${nlat} x ${nphi})`,
      eS < 1e-5 && eT < 1e-5,
      `S ${fmt(eS)}, T ${fmt(eT)}; worst degree of S: l = ${worstL} at ${fmt(worst)}`,
    );
    k.destroy();
  }
}
