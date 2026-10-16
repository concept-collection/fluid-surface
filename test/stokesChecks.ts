/**
 * The static mode (models/static_stokes.m, src/fluid/staticStokes.ts).
 *
 * 1. One BDF-1 step of static_stokes against the C++ solver on its cpu backend
 *    in f64, on the same grid (test/fixtures/static-cpu-L12, written by
 *    scripts/cpp/static_dump.cpp): the preconditioned operator on a random
 *    vector, the preconditioned right-hand side, and the step's solution.
 * 2. The C++ solver's own check (drivers/fluid_killing.cpp -check): a degree-2
 *    toroidal flow on the fixed unit sphere decays at (l(l+1) - 2) / Re, and
 *    BDF-1 makes that ln(1 + rate dt) / dt; the fitted rate must match it.
 *    Degree 1, the Killing field, must not decay.
 */
import { readDump } from './dump.ts';
import { relayoutCoeffs } from '../src/sht/layout.ts';
import { StaticStokes } from '../src/fluid/staticStokes.ts';
import { staticStokes } from '../src/fluid/examples.ts';
import { unpackCpp } from '../src/fluid/spectra.ts';
import { relL2, fmt, type Check, type Log } from './check.ts';


/** GMRES at the fp32 floor, as the C++ solver's fp32 builds use it
 *  (FLUID_FP32_TOL_FLOOR): below about 7e-7 the true residual stalls. */
const GMRES = { tol: 1e-6, restart: 200, maxCycles: 4 };

export async function stokesChecks(device: GPUDevice, check: Check, log: Log): Promise<void> {
  // ---- 1. one step against the C++ ------------------------------------------
  {
    const dump = readDump(new URL('./fixtures/static-cpu-L12/', import.meta.url));
    const s = dump.scalars;
    const lmax = s.lmax;
    const st = await StaticStokes.create(device, lmax, { re: s.re, be: s.be, dt: s.dt, gmres: GMRES });
    check(
      'stokes: the C++ dump is on our product grid',
      st.cfg.nlat === s.nlat_f && st.cfg.nphi === s.nphi_f && st.cfg.lmax === s.lmax_f,
      `band ${lmax}, product band ${st.cfg.lmax} on ${st.cfg.nlat} x ${st.cfg.nphi}`,
    );
    const work = { lmax, mmax: lmax };
    const prod = { lmax: st.cfg.lmax, mmax: st.cfg.mmax };
    const nlm = ((lmax + 1) * (lmax + 2)) / 2;
    const field = (a: Float64Array, c: number) =>
      relayoutCoeffs(new Float32Array(a.subarray(2 * c * nlm, 2 * (c + 1) * nlm)), work, prod);
    const fromCpp = (packed: Float64Array, nf: number) =>
      st.pack(unpackCpp(packed, lmax, nf).map((c) => relayoutCoeffs(new Float32Array(c), work, prod)));

    const ops = (f: { describe(): string[] }) => f.describe().length;
    log(
      `  static_stokes.m compiles to: setup ${ops(st.setup)}, rhs ${ops(st.rhs)}, ` +
        `matvec ${ops(st.matvec)} GPU operations; geometry ${ops(st.geometry)}`,
    );

    st.setSurface([0, 1, 2].map((c) => field(dump.arrays.X, c)));
    const U0 = [0, 1, 2].map((c) => field(dump.arrays.u0, c));
    const b = await st.prepare(U0);
    const eB = relL2(b, fromCpp(dump.arrays.b, 4));
    const az = await st.apply(fromCpp(dump.arrays.z, 4));
    const eA = relL2(az, fromCpp(dump.arrays.Az, 4));
    check(
      'stokes: preconditioned operator and right-hand side match the C++ (fp32 vs f64), rel L2',
      eA < 1e-5 && eB < 1e-5,
      `A z ${fmt(eA)}, b ${fmt(eB)}`,
    );

    const P0 = new Float32Array(U0[0].length);
    const t0 = performance.now();
    const out = await st.step(U0, P0);
    const ms = performance.now() - t0;
    const eU = relL2(st.pack(out.U), st.pack([0, 1, 2].map((c) => field(dump.arrays.u1, c))));
    check(
      `stokes: one static step matches the C++ (GMRES tol ${GMRES.tol}), rel L2 of u`,
      // At tol 1e-6 the solution can differ from the C++'s (solved to 1e-12) by
      // about 20 times that; tol 1e-7 gives 2.4e-6 but no longer converges.
      out.info.converged && eU < 5e-5,
      `u ${fmt(eU)}; ${out.info.iterations} GMRES iterations (C++ at tol 1e-12: ${s.iterations}), ` +
        `residual ${fmt(out.info.relResid)}, ${ms.toFixed(0)} ms`,
    );
    st.destroy();
  }

  // ---- 2. decay rates on the fixed unit sphere -----------------------------
  for (const degree of [2, 1]) {
    const lmax = 16;
    const dt = 0.01;
    const steps = 100;
    const re = 1;
    const st = await StaticStokes.create(device, lmax, { re, be: 2, dt, gmres: GMRES });
    const ic = staticStokes(st.cfg, lmax, degree);
    st.setSurface(ic.X);
    let U = ic.U;
    let P: Float32Array = new Float32Array(U[0].length);
    const ts: number[] = [];
    const logu: number[] = [];
    const norm = (u: Float32Array[]) => Math.sqrt(st.pack(u).reduce((s, v) => s + v * v, 0));
    ts.push(0);
    logu.push(Math.log(norm(U)));
    let its = 0;
    let worstResid = 0;
    const t0 = performance.now();
    for (let k = 1; k <= steps; k++) {
      const out = await st.step(U, P);
      U = out.U;
      P = out.P;
      its += out.info.iterations;
      worstResid = Math.max(worstResid, out.info.relResid);
      ts.push(k * dt);
      logu.push(Math.log(norm(U)));
    }
    const ms = (performance.now() - t0) / steps;
    // The least-squares slope of log |u|, as the C++ fits it.
    const mt = ts.reduce((a, b) => a + b, 0) / ts.length;
    const ml = logu.reduce((a, b) => a + b, 0) / logu.length;
    let num = 0;
    let den = 0;
    for (let i = 0; i < ts.length; i++) {
      num += (ts[i] - mt) * (logu[i] - ml);
      den += (ts[i] - mt) ** 2;
    }
    const fit = -num / den;
    const rate = (degree * (degree + 1) - 2) / re;
    const bdf1 = Math.log(1 + rate * dt) / dt;
    const detail =
      `fitted ${fit.toPrecision(8)}, BDF-1 exact ${bdf1.toPrecision(8)}; ` +
      `${(its / steps).toFixed(1)} GMRES its/step, worst residual ${fmt(worstResid)}, ${ms.toFixed(0)} ms/step`;
    if (degree === 1) {
      check(`stokes: the Killing field (degree 1) does not decay, L${lmax}, ${steps} steps`, Math.abs(fit) < 1e-4, detail);
    } else {
      check(
        `stokes: degree-${degree} toroidal flow decays at the BDF-1 rate, L${lmax}, ${steps} steps, relative error`,
        Math.abs(fit - bdf1) <= 1e-4 * bdf1,
        `${fmt(Math.abs(fit - bdf1) / bdf1)}; ${detail}`,
      );
    }
    st.destroy();
  }
}
