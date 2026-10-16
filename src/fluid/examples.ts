/**
 * Initial states of the C++ solver's examples (core/fluid_examples.cpp), as
 * product-band coefficients truncated to the working band.
 */
import { gaussNodesWeights } from '../sht/gauss.ts';
import type { ShtConfig } from '../sht/layout.ts';
import { forGrid, gridToCoeffs, perturbedSphere } from '../geom/geometry.ts';

/** Derivative of the Legendre polynomial of degree l at x, by recurrence. */
function legendreDeriv(l: number, x: number): number {
  const P = [1, x];
  const dP = [0, 1];
  for (let n = 1; n < l; n++) {
    P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1);
    dP[n + 1] = dP[n - 1] + (2 * n + 1) * P[n];
  }
  return dP[l];
}

/**
 * static_stokes: the unit sphere with the toroidal velocity of one degree,
 * u = grad P_l(cos theta) x z-hat rotated into the surface. Degree 1 is a rigid
 * rotation about z, the sphere's Killing field. On the fixed sphere the mode
 * decays at (l(l+1) - 2) / Re, so the Killing field does not decay at all.
 */
export function staticStokes(cfg: ShtConfig, lmax: number, degree: number): { X: Float32Array[]; U: Float32Array[] } {
  const { x: ct } = gaussNodesWeights(cfg.nlat);
  const u = [0, 1, 2].map(() => new Float64Array(cfg.nlat * cfg.nphi));
  forGrid(cfg, (i, phi, k) => {
    const gr = -legendreDeriv(degree, ct[i]) * Math.sqrt(1 - ct[i] * ct[i]);
    u[0][k] = -gr * Math.sin(phi);
    u[1][k] = gr * Math.cos(phi);
  });
  return { X: perturbedSphere(cfg, lmax, 0), U: gridToCoeffs(cfg, lmax, u) };
}
