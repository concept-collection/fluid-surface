/**
 * The MATLAB-to-WebGPU compiler's own features, checked on small functions
 * against values computed on the host.
 *
 * Struct arguments: a struct of arrays is passed by the host field by field
 * (each field a buffer named `<argument>.<field>`), read with `s.f`, and
 * passed on to subfunctions, which are expanded into the caller.
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
  sht.destroy();
}
