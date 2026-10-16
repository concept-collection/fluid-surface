/**
 * The Stokes preconditioner's per-degree blocks (core/fluid_precond.cpp
 * UploadSphereTables): the BDF step's system on the sphere of radius R, for
 * the radial and spheroidal velocity potentials and the pressure of one
 * degree, and the toroidal symbol. They are built and inverted here in f64,
 * as the C++ builds them on the host, and spread over the spectral layout for
 * models/stokes.m.
 */
import type { ShtConfig } from '../sht/layout.ts';
import { byDegree } from './spectra.ts';

export interface SphereBlockParams {
  mass: number;
  disp: number;
  re: number;
  /** Bending number; Infinity without bending. */
  be: number;
  /** Radius of the sphere with the surface's area. */
  R: number;
}

/** The 3 x 3 block of degree l, row-major, and the toroidal symbol. */
export function sphereBlock(l: number, p: SphereBlockParams): { k: number[]; tor: number } {
  const nu = 1 / (p.re * p.R * p.R);
  const ir = 1 / p.R;
  const lam = l * (l + 1);
  const l0 = l === 0;
  const bend = Number.isFinite(p.be) ? (p.disp / (p.be * p.R ** 4)) * Math.max(lam * (lam - 2), 0) : 0;
  const k = [
    p.mass + 4 * nu + bend, l0 ? 0 : -2 * lam * nu, -2 * ir,
    l0 ? 0 : -2 * nu, l0 ? 1 : p.mass + (2 * lam - 2) * nu, l0 ? 0 : ir,
    2 * ir, -lam * ir, 0,
  ];
  return { k, tor: l0 ? 1 : p.mass + (lam - 2) * nu };
}

function inverse3(a: number[]): number[] {
  const [a0, a1, a2, a3, a4, a5, a6, a7, a8] = a;
  const c = [
    a4 * a8 - a5 * a7, a2 * a7 - a1 * a8, a1 * a5 - a2 * a4,
    a5 * a6 - a3 * a8, a0 * a8 - a2 * a6, a2 * a3 - a0 * a5,
    a3 * a7 - a4 * a6, a1 * a6 - a0 * a7, a0 * a4 - a1 * a3,
  ];
  const det = a0 * c[0] + a1 * c[3] + a2 * c[6];
  return c.map((x) => x / det);
}

/**
 * The blocks' inverses (i11 ... i33), the inverses of their leading 2 x 2
 * velocity blocks (w11 ... w22, the system without the pressure, for the
 * volume constraint's block), and the inverse toroidal symbol, named as
 * models/stokes.m takes them.
 */
export function sphereBlockTables(cfg: ShtConfig, p: SphereBlockParams): Record<string, Float32Array> {
  const inv: number[][] = [];
  const vinv: number[][] = [];
  const itor: number[] = [];
  for (let l = 0; l <= cfg.lmax; l++) {
    const { k, tor } = sphereBlock(l, p);
    inv.push(inverse3(k));
    const det = k[0] * k[4] - k[1] * k[3];
    vinv.push([k[4] / det, -k[1] / det, -k[3] / det, k[0] / det]);
    itor.push(1 / tor);
  }
  const out: Record<string, Float32Array> = { itor: byDegree(cfg, (l) => itor[l]) };
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) out[`i${r + 1}${c + 1}`] = byDegree(cfg, (l) => inv[l][3 * r + c]);
  }
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) out[`w${r + 1}${c + 1}`] = byDegree(cfg, (l) => vinv[l][2 * r + c]);
  }
  return out;
}
