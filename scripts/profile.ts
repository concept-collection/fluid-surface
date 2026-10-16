/**
 * Where a step of Krause & Voigt's perturbed sphere spends its time, on
 * desktop WebGPU (src/fluid/profile.ts). Steps from rest, profiling the last
 * of a few so GMRES's iteration count is typical.
 *
 *   npx vite-node scripts/profile.ts -- --lmax 32 --steps 5
 */
import { installWebGpu } from './nodeWebGpu.ts';
import { requestShtDevice } from '../src/sht/sht.ts';
import { perturbedSphere, productConfig } from '../src/geom/geometry.ts';
import { MovingStokes, type FluidState } from '../src/fluid/movingStokes.ts';
import { formatProfile, profileStep } from '../src/fluid/profile.ts';

const arg = (name: string, dflt: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const lmax = Number(arg('lmax', '32'));
const steps = Number(arg('steps', '5'));
const dt = Number(arg('dt', '0.005'));

await installWebGpu();
const device = await requestShtDevice();
const ms = await MovingStokes.create(device, lmax, {
  re: 1, be: 2, volume: true, dt, gmres: { tol: 1e-6, restart: 200, maxCycles: 4 },
});
const cfg = productConfig(lmax);
ms.setSurface(perturbedSphere(cfg, lmax, 0.4));
const zero = () => new Float32Array((cfg.lmax + 1) * (cfg.lmax + 2));
let state: FluidState = { U: [zero(), zero(), zero()], P: zero(), lam: 0 };
for (let k = 1; k < steps; k++) state = (await ms.step(state)).state;
const { profile } = await profileStep(device, ms, state);
console.log(`band ${lmax}, product band ${cfg.lmax} on ${cfg.nlat} x ${cfg.nphi}, step ${steps}, timestamp queries: ${device.features.has('timestamp-query')}`);
for (const line of formatProfile(profile)) console.log(line);
