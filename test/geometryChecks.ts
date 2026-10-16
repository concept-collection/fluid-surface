/**
 * lib/geometry.m, the surface geometry built on the GPU from the position's
 * coefficients.
 *
 * 1. The unit sphere, where every output has a closed form: n = -x, H = K = 1,
 *    sqrt(det g) = sin(theta), and the flux weights are those of the sphere.
 * 2. Krause & Voigt's perturbed sphere against the C++ solver's
 *    FluidGeometryBuild on its cpu backend in f64, on the same grid
 *    (test/fixtures/geometry-cpu-perturbed-L12, written by
 *    scripts/cpp/geometry_dump.cpp).
 */
import { readDump } from './dump.ts';
import { ShtReference } from '../src/sht/reference.ts';
import { gridForLmax, relayoutCoeffs, type ShtConfig } from '../src/sht/layout.ts';
import { gaussNodesWeights } from '../src/sht/gauss.ts';
import { GEOMETRY_OUTPUTS, SurfaceGeometry } from '../src/geom/geometry.ts';
import { relL2, fmt, type Check, type Log } from './check.ts';

const OUTPUTS = GEOMETRY_OUTPUTS;

async function compileGeometry(device: GPUDevice, cfg: ShtConfig) {
  const geo = await SurfaceGeometry.create(device, cfg);
  /** Runs on product-band coefficients and reads every output back. */
  const run = async (X: Float32Array[]): Promise<Record<string, Float32Array>> => {
    geo.compute(X);
    const out: Record<string, Float32Array> = {};
    for (const name of OUTPUTS) out[name] = await geo.read(name);
    return out;
  };
  return { fn: geo.fn, run, destroy: () => geo.destroy() };
}

/** A ref_dump directory (the C++ solver's format): a manifest of f64 arrays. */

export async function geometryChecks(device: GPUDevice, check: Check, log: Log): Promise<void> {
  // ---- 1. the unit sphere ---------------------------------------------------
  {
    const lmax = 16;
    const { nlat, nphi } = gridForLmax(lmax, 1);
    const cfg: ShtConfig = { lmax, mmax: lmax, nlat, nphi };
    const ref = new ShtReference(cfg);
    const npts = nlat * nphi;
    const x = new Float64Array(npts);
    const y = new Float64Array(npts);
    const z = new Float64Array(npts);
    for (let i = 0; i < nlat; i++) {
      for (let j = 0; j < nphi; j++) {
        const phi = (2 * Math.PI * j) / nphi;
        const k = i * nphi + j;
        x[k] = ref.st[i] * Math.cos(phi);
        y[k] = ref.st[i] * Math.sin(phi);
        z[k] = ref.ct[i];
      }
    }
    const X = [x, y, z].map((f) => new Float32Array(ref.analys(f)));
    const g = await compileGeometry(device, cfg);
    // A fusion guard: one kernel per line. If fusion stops short the answers
    // stay right and this count jumps (it was 93 before numbl#66, and 31
    // before numbl fused a call on an expression). A device with fewer than 16
    // storage buffers per stage splits some lines (src/mgpu/split.ts): 33 at 8.
    const ops = g.fn.describe().map((l) => l.trim().split(/\s+/)[0]);
    const count = (k: string): number => ops.filter((o) => o === k).length;
    const bound = device.limits.maxStorageBuffersPerShaderStage >= 16 ? 29 : 33;
    check(
      `geometry: geometry.m fuses to one kernel per line (at most ${bound} kernels with ${device.limits.maxStorageBuffersPerShaderStage} storage buffers per stage)`,
      count('kernel') <= bound,
      `${ops.length} GPU operations: ${count('synth')} syntheses, ` +
        `${count('dthetac') + count('dphic')} shuffles, ${count('kernel')} kernels`,
    );
    const out = await g.run(X);
    const sinT = Float64Array.from({ length: npts }, (_, k) => ref.st[Math.floor(k / nphi)]);
    const maxDiff = (a: ArrayLike<number>, b: (k: number) => number): number => {
      let m = 0;
      for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b(k)));
      return m;
    };
    const errs = {
      n: Math.max(
        maxDiff(out.n1, (k) => -x[k]),
        maxDiff(out.n2, (k) => -y[k]),
        maxDiff(out.n3, (k) => -z[k]),
      ),
      H: maxDiff(out.H, () => 1),
      K: maxDiff(out.K, () => 1),
      sqrtg: maxDiff(out.sqrtg, (k) => sinT[k]),
      weights: Math.max(
        maxDiff(out.dp1, () => 0),
        maxDiff(out.p2, () => 0),
        maxDiff(out.dq2, () => 0),
        maxDiff(out.jinv, () => 1),
      ),
    };
    const worst = Math.max(...Object.values(errs));
    check(
      `geometry: unit sphere has its closed form (L${lmax}, ${nlat} x ${nphi}), max error`,
      worst < 1e-4,
      Object.entries(errs)
        .map(([k, v]) => `${k} ${fmt(v)}`)
        .join(', '),
    );
    g.destroy();
  }

  // ---- 2. the perturbed sphere against the C++ cpu backend -----------------
  {
    const dump = readDump(new URL('./fixtures/geometry-cpu-perturbed-L12/', import.meta.url));
    const lmax = dump.scalars.lmax;
    const lmaxF = dump.scalars.lmax_f;
    const nlat = dump.scalars.nlat_f;
    const nphi = dump.scalars.nphi_f;
    const grid = gridForLmax(lmaxF, 1);
    const { x: nodes } = gaussNodesWeights(nlat);
    const ctErr = Math.max(...Array.from(dump.arrays.ct, (c, i) => Math.abs(c - nodes[i])));
    check(
      'geometry: the C++ dump is on our product grid',
      grid.nlat === nlat && grid.nphi === nphi && ctErr < 1e-14,
      `band ${lmax}, product band ${lmaxF}, ${nlat} x ${nphi}, Gauss nodes agree to ${fmt(ctErr)}`,
    );
    const cfg: ShtConfig = { lmax: lmaxF, mmax: lmaxF, nlat, nphi };
    // The C++ position is on the working band; pad it to the product band.
    const nlm = (lmax + 1) * (lmax + 2) / 2;
    const Xw = dump.arrays.X;
    const X = [0, 1, 2].map((c) =>
      relayoutCoeffs(
        new Float32Array(Xw.subarray(2 * c * nlm, 2 * (c + 1) * nlm)),
        { lmax, mmax: lmax },
        { lmax: lmaxF, mmax: lmaxF },
      ),
    );
    const g = await compileGeometry(device, cfg);
    const out = await g.run(X);
    // The C++ grid is theta-contiguous (ip * nlat + it); ours phi-contiguous.
    const geo = dump.arrays.geometry;
    const ns = nlat * nphi;
    const cpp = (field: number): Float64Array => {
      const f = new Float64Array(ns);
      for (let it = 0; it < nlat; it++) {
        for (let ip = 0; ip < nphi; ip++) f[it * nphi + ip] = geo[field * ns + ip * nlat + it];
      }
      return f;
    };
    const errs = OUTPUTS.map((name, i) => [name, relL2(out[name], cpp(i))] as const);
    const worst = Math.max(...errs.map(([, e]) => e));
    check(
      `geometry: perturbed sphere (r0 ${dump.scalars.r0}) matches the C++ cpu backend, rel L2`,
      worst < 1e-5,
      errs.map(([n, e]) => `${n} ${fmt(e)}`).join(', '),
    );
    g.destroy();
  }
}
