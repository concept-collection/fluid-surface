/**
 * The MATLAB-to-WebGPU compiler's own features, checked on small functions
 * against values computed on the host.
 *
 * Struct arguments: a struct of arrays is passed by the host field by field
 * (each field a buffer named `<argument>.<field>`), read with `s.f`, and
 * passed on to subfunctions, which are expanded into the caller.
 *
 * Vector fields: an npts x 3 array broadcasts against npts x 1 fields, with
 * row sums `sum(x, 2)`, column reads `x(:, k)` and concatenation `[a, b, c]`.
 */
import { ShtPlan } from '../src/sht/sht.ts';
import { GpuFunction } from '../src/mgpu/gpuFunction.ts';
import type { Binding } from '../src/mgpu/compile.ts';
import { relL2, fmt, type Check, type Log } from './check.ts';

const STRUCT_SOURCE = `
function y = f(g, x)
  y = g.a .* x + h(g, x);
end

function z = h(q, x)
  z = q.b .* x .* x;
end
`;

const VECTOR_SOURCE = `
function v = f(u, n, s)
  un = sum(u .* n, 2);
  p = u - un .* n + s;
  v = [p(:, 1) + p(:, 3), un, 2 * p(:, 2)];
end
`;

// Per-component fields assembled into vectors are views, and a vector read
// only by its columns is computed column by column: neither gets a buffer.
const VIEW_SOURCE = `
function [a, b] = g(f1, f2, f3, n1, n2, n3, s)
  f = [f1, f2, f3];
  n = [n1, n2, n3];
  fn = sum(f .* n, 2);
  p = f - fn .* n;
  q = p .* s + n;
  a = sum(p .* n, 2) + q(:, 1);
  b = q(:, 2) - q(:, 3);
end
`;

export async function compileChecks(device: GPUDevice, check: Check, log: Log): Promise<void> {
  const sht = await ShtPlan.create(device, { lmax: 8, mmax: 8, nlat: 10, nphi: 32 });
  const n = 320;
  const vec: Binding = { kind: 'tensor', shape: [n, 1] };
  const fn = await GpuFunction.create({
    device, sht, source: STRUCT_SOURCE, name: 'f', nargout: 1,
    bindings: { g: { kind: 'struct', fields: { a: vec, b: vec } }, x: vec },
  });
  const a = Float32Array.from({ length: n }, (_, i) => Math.sin(i));
  const b = Float32Array.from({ length: n }, (_, i) => Math.cos(0.3 * i));
  const x = Float32Array.from({ length: n }, (_, i) => (i % 17) / 17 - 0.5);
  fn.upload('g.a', a);
  fn.upload('g.b', b);
  fn.upload('x', x);
  fn.run();
  const y = await fn.read('y');
  const want = Float64Array.from({ length: n }, (_, i) => a[i] * x[i] + b[i] * x[i] * x[i]);
  const e = relL2(y, want);
  const kernels = fn.describe().filter((l) => l.startsWith('kernel')).length;
  log(`  struct check compiles to: ${fn.describe().join('; ')}`);
  check(
    'compile: a struct argument is read by field and passed on to a subfunction',
    e < 1e-6 && kernels === 1,
    `rel L2 ${fmt(e)}, ${kernels} kernel`,
  );
  fn.destroy();
  await vectorCheck(device, sht, n, check, log);
  sht.destroy();
}

async function vectorCheck(device: GPUDevice, sht: ShtPlan, n: number, check: Check, log: Log): Promise<void> {
  const vec3: Binding = { kind: 'tensor', shape: [n, 3] };
  const fn = await GpuFunction.create({
    device, sht, source: VECTOR_SOURCE, name: 'f', nargout: 1,
    bindings: { u: vec3, n: vec3, s: { kind: 'tensor', shape: [n, 1] } },
  });
  // Column-major: column c of an n x 3 array is entries c*n .. c*n + n - 1.
  const u = Float32Array.from({ length: 3 * n }, (_, i) => Math.sin(0.7 * i));
  const nn = Float32Array.from({ length: 3 * n }, (_, i) => Math.cos(0.2 * i + 1));
  const s = Float32Array.from({ length: n }, (_, i) => (i % 13) / 13 - 0.5);
  fn.upload('u', u);
  fn.upload('n', nn);
  fn.upload('s', s);
  fn.run();
  const v = await fn.read('v');
  const want = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    let un = 0;
    for (let c = 0; c < 3; c++) un += u[i + c * n] * nn[i + c * n];
    const p = [0, 1, 2].map((c) => u[i + c * n] - un * nn[i + c * n] + s[i]);
    want[i] = p[0] + p[2];
    want[i + n] = un;
    want[i + 2 * n] = 2 * p[1];
  }
  const e = relL2(v, want);
  const kernels = fn.describe().filter((l) => l.startsWith('kernel')).length;
  log(`  vector check compiles to: ${fn.describe().join('; ')}`);
  check(
    'compile: npts x 3 vector fields broadcast against npts x 1, with row sums, columns and concatenation',
    e < 1e-6 && kernels === 2,
    `rel L2 ${fmt(e)}, ${kernels} kernels`,
  );
  fn.destroy();

  const col: Binding = { kind: 'tensor', shape: [n, 1] };
  const g = await GpuFunction.create({
    device, sht, source: VIEW_SOURCE, name: 'g', nargout: 2,
    bindings: { f1: col, f2: col, f3: col, n1: col, n2: col, n3: col, s: col },
  });
  const cols = (x: Float32Array) => [0, 1, 2].map((c) => x.subarray(c * n, (c + 1) * n));
  const [f1, f2, f3] = cols(u);
  const [n1, n2, n3] = cols(nn);
  Object.entries({ f1, f2, f3, n1, n2, n3, s }).forEach(([k, x]) => g.upload(k, Float32Array.from(x)));
  g.run();
  const [a, b] = await g.readMany(['a', 'b']);
  const wantA = new Float64Array(n);
  const wantB = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const f = [f1[i], f2[i], f3[i]];
    const nv = [n1[i], n2[i], n3[i]];
    const fn = f[0] * nv[0] + f[1] * nv[1] + f[2] * nv[2];
    const pv = f.map((x, c) => x - fn * nv[c]);
    const q = pv.map((x, c) => x * s[i] + nv[c]);
    wantA[i] = pv[0] * nv[0] + pv[1] * nv[1] + pv[2] * nv[2] + q[0];
    wantB[i] = q[1] - q[2];
  }
  const ev = Math.max(relL2(a, wantA), relL2(b, wantB));
  const gk = g.describe().filter((l) => l.startsWith('kernel')).length;
  log(`  view check compiles to: ${g.describe().join('; ')}`);
  check(
    'compile: vectors assembled from components, or read only by column, need no buffer of their own',
    ev < 1e-6 && gk === 3,
    `rel L2 ${fmt(ev)}, ${gk} kernels`,
  );
  g.destroy();
}
