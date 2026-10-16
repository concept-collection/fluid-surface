/**
 * The host's view of the fluid solver's spectral fields: every spectral field
 * lives on the product band (twice the working band), zero above the working
 * band, so that one compiled function needs only one band. The working band's
 * truncation and the solver's filters are then elementwise masks, and the
 * unknowns GMRES sees are the working band's entries alone.
 */
import { gaussNodesWeights } from '../sht/gauss.ts';
import { lmIndex, nlmCalc, type ShtConfig } from '../sht/layout.ts';

/** The masks and per-degree arrays the .m files take, 2 x nlm on the product band. */
export interface SpectralMasks {
  /** l(l+1). */
  lam: Float32Array;
  /** 1 for l <= lmax, except the imaginary part of m = 0 (truncation plus real m = 0). */
  keep: Float32Array;
  /** As keep, without the working band's top three degrees (l < lmax - 2). */
  clean: Float32Array;
  /** 1 below the product band's top three degrees (l < lmaxF - 2). */
  filtf: Float32Array;
  /** 1 at the real part of l = m = 0. */
  e0: Float32Array;
  /** 1/(l(l+1)), 0 at l = 0. */
  ilam: Float32Array;
  /** As keep, without the working band's top two degrees (l < lmax - 1). */
  cutv: Float32Array;
  /** 1 except the imaginary part of m = 0, at every degree. */
  real0: Float32Array;
}

export function spectralMasks(cfg: ShtConfig, lmax: number): SpectralMasks {
  const n = 2 * nlmCalc(cfg.lmax, cfg.mmax);
  const out: SpectralMasks = {
    lam: new Float32Array(n),
    keep: new Float32Array(n),
    clean: new Float32Array(n),
    filtf: new Float32Array(n),
    e0: new Float32Array(n),
    ilam: new Float32Array(n),
    cutv: new Float32Array(n),
    real0: new Float32Array(n),
  };
  for (let m = 0; m <= cfg.mmax; m++) {
    for (let l = m; l <= cfg.lmax; l++) {
      const i = 2 * lmIndex(cfg.lmax, l, m);
      for (const k of [i, i + 1]) {
        const realPart = k === i || m > 0;
        out.lam[k] = l * (l + 1);
        out.keep[k] = l <= lmax && realPart ? 1 : 0;
        out.clean[k] = l < lmax - 2 && realPart ? 1 : 0;
        out.filtf[k] = l < cfg.lmax - 2 ? 1 : 0;
        out.ilam[k] = l > 0 ? 1 / (l * (l + 1)) : 0;
        out.cutv[k] = l < lmax - 1 && realPart ? 1 : 0;
        out.real0[k] = realPart ? 1 : 0;
      }
    }
  }
  out.e0[0] = 1;
  return out;
}

/** A per-degree table (index l) spread over the 2 x nlm layout. */
export function byDegree(cfg: ShtConfig, table: (l: number) => number): Float32Array {
  const out = new Float32Array(2 * nlmCalc(cfg.lmax, cfg.mmax));
  for (let m = 0; m <= cfg.mmax; m++) {
    for (let l = m; l <= cfg.lmax; l++) {
      const i = 2 * lmIndex(cfg.lmax, l, m);
      out[i] = out[i + 1] = table(l);
    }
  }
  return out;
}

/** Quadrature weights for integrals over the (theta, phi) chart on the grid, as
 *  the C++ solver's FluidIntegral: Gauss weight / sin(theta) * 2 pi / nphi. */
export function chartWeights(cfg: ShtConfig): Float32Array {
  const { x, w } = gaussNodesWeights(cfg.nlat);
  const out = new Float32Array(cfg.nlat * cfg.nphi);
  for (let i = 0; i < cfg.nlat; i++) {
    out.fill((w[i] / Math.sqrt(1 - x[i] * x[i])) * ((2 * Math.PI) / cfg.nphi), i * cfg.nphi, (i + 1) * cfg.nphi);
  }
  return out;
}

/** Indices, into a product-band spectral array, of the working band's real
 *  unknowns: the real part at m = 0, real and imaginary parts at m > 0, for
 *  l <= lmax. (lmax + 1)^2 of them, the C++ solver's real embedding. */
export function workingIndices(cfg: ShtConfig, lmax: number): Int32Array {
  const idx: number[] = [];
  for (let m = 0; m <= lmax; m++) {
    for (let l = m; l <= lmax; l++) {
      const i = 2 * lmIndex(cfg.lmax, l, m);
      idx.push(i);
      if (m > 0) idx.push(i + 1);
    }
  }
  return Int32Array.from(idx);
}

/** The C++ solver's packed vector (FluidOpPack, n fields) as working-band
 *  spectra in our layout: per field, m = 0 real parts, then the m > 0 real
 *  parts, then their imaginary parts, in SHTns' LM order. */
export function unpackCpp(xr: ArrayLike<number>, lmax: number, nfields: number): Float64Array[] {
  const nlm = nlmCalc(lmax, lmax);
  const n0 = lmax + 1;
  const nr = (lmax + 1) * (lmax + 1);
  const npos = nlm - n0;
  return Array.from({ length: nfields }, (_, f) => {
    const c = new Float64Array(2 * nlm);
    for (let k = 0; k < nlm; k++) {
      if (k < n0) {
        c[2 * k] = xr[f * nr + k];
      } else {
        c[2 * k] = xr[f * nr + n0 + (k - n0)];
        c[2 * k + 1] = xr[f * nr + n0 + npos + (k - n0)];
      }
    }
    return c;
  });
}
