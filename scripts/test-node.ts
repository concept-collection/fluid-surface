/**
 * The suite on desktop WebGPU (Google Dawn), against the real pipeline:
 * MATLAB source -> numbl lowering -> generated WGSL -> GPU. Run through
 * vite-node, which resolves numbl's compiler sources and the `?raw` imports:
 *
 *   npm run test:node
 *
 * `--skip-without-gpu` lets a machine without a GPU say so and move on.
 */
import { requestShtDevice } from '../src/sht/sht.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';
import { compileChecks } from '../test/compileChecks.ts';
import { sphtorChecks } from '../test/sphtorChecks.ts';
import { geometryChecks } from '../test/geometryChecks.ts';
import { stokesChecks } from '../test/stokesChecks.ts';
import { movingChecks, bendingChecks, volumeChecks } from '../test/movingChecks.ts';

let failures = 0;
const check = (name: string, ok: boolean, detail: string): void => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
  if (!ok) failures++;
};
const log = (s: string): void => console.log(s);

const only = process.argv.find((a) => a.startsWith('--only='))?.slice(7);
const skipWithoutGpu = process.argv.includes('--skip-without-gpu');

let runtime: string;
let device: GPUDevice;
try {
  runtime = await installWebGpu();
  device = await requestShtDevice();
} catch (e) {
  const detail = `${errMsg(e)}\n${NO_ADAPTER_HINT}`;
  if (skipWithoutGpu) {
    console.log(`SKIP  no WebGPU available here, so these checks did not run.\n${detail}`);
    process.exit(0);
  }
  console.error(`test-node: ${detail}`);
  process.exit(1);
}
console.log(`fluid-surface tests — ${runtime}\n`);

const suites: [string, (d: GPUDevice, c: typeof check, l: typeof log) => Promise<void>][] = [
  ['compile', compileChecks],
  ['sphtor', sphtorChecks],
  ['geometry', geometryChecks],
  ['stokes', stokesChecks],
  ['moving', movingChecks],
  ['bending', bendingChecks],
  ['volume', volumeChecks],
];
for (const [name, run] of suites) if (!only || only === name) await run(device, check, log);

console.log(failures === 0 ? '\nAll tests passed.' : `\n${failures} failed.`);
process.exit(failures === 0 ? 0 : 1);
