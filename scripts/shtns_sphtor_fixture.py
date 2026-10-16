"""Writes test/fixtures/sphtor-shtns-L<lmax>.json: SHTns' own vector transforms
on the same Gauss grid the app uses, for test/sphtorChecks.ts.

Two cases:
  band   - random S, T below lmax, synthesized by SHTns (anchors the convention)
  noise  - random grid values, analysed by SHTns (the quadrature sums must
           agree for any grid data, not just band-limited fields)

Needs the shtns Python package (pip install shtns, with FFTW installed):

    python scripts/shtns_sphtor_fixture.py -lmax 16
"""

import argparse
import json
import math

import numpy as np
import shtns


def grid_for_lmax(lmax: int, pdeg: int) -> tuple[int, int]:
    # src/sht/layout.ts gridForLmax
    min_lat = max(lmax + 1, ((pdeg + 1) * lmax + 1) / 2)
    nlat = 2 * math.ceil(min_lat / 2)
    nphi = 1
    while nphi < (pdeg + 1) * lmax + 1:
        nphi *= 2
    return nlat, nphi


def lm_index(lmax: int, l: int, m: int) -> int:
    # src/sht/layout.ts lmIndex
    return m * (lmax + 1) - (m * (m - 1)) // 2 + (l - m)


def to_ours(sh, lmax: int, q) -> list[float]:
    nlm = (lmax + 1) * (lmax + 2) // 2
    out = np.zeros(2 * nlm)
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            k = lm_index(lmax, l, m)
            c = q[sh.idx(l, m)]
            out[2 * k] = c.real
            out[2 * k + 1] = c.imag
    return out.tolist()


def grid_to_ours(v) -> list[float]:
    # v is indexed [ilat, iphi]; ours is spat[ilat * nphi + iphi]
    return np.ascontiguousarray(np.asarray(v)).ravel().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-lmax", type=int, default=16)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-out", type=str, default=None)
    args = parser.parse_args()
    lmax = args.lmax
    nlat, nphi = grid_for_lmax(lmax, 1)
    sh = shtns.sht(lmax, lmax)
    sh.set_grid(nlat=nlat, nphi=nphi, flags=shtns.sht_gauss)
    rng = np.random.default_rng(args.seed)

    cases = []

    S = sh.spec_array()
    T = sh.spec_array()
    for m in range(lmax + 1):
        for l in range(max(m, 1), lmax):
            for q in (S, T):
                re, im = rng.standard_normal(2) / (1 + l)
                q[sh.idx(l, m)] = complex(re, 0.0 if m == 0 else im)
    vt, vp = sh.synth(S, T)
    S2, T2 = sh.analys(vt, vp)
    cases.append(
        {
            "name": "band",
            "S_in": to_ours(sh, lmax, S),
            "T_in": to_ours(sh, lmax, T),
            "vt": grid_to_ours(vt),
            "vp": grid_to_ours(vp),
            "S": to_ours(sh, lmax, S2),
            "T": to_ours(sh, lmax, T2),
        }
    )

    vt = sh.spat_array()
    vp = sh.spat_array()
    vt[:] = rng.standard_normal(vt.shape)
    vp[:] = rng.standard_normal(vp.shape)
    S2, T2 = sh.analys(vt, vp)
    cases.append(
        {
            "name": "noise",
            "vt": grid_to_ours(vt),
            "vp": grid_to_ours(vp),
            "S": to_ours(sh, lmax, S2),
            "T": to_ours(sh, lmax, T2),
        }
    )

    out = args.out or f"test/fixtures/sphtor-shtns-L{lmax}.json"
    with open(out, "w") as f:
        json.dump(
            {
                "shtns": shtns.__version__ if hasattr(shtns, "__version__") else "unknown",
                "lmax": lmax,
                "nlat": nlat,
                "nphi": nphi,
                "ct": sh.cos_theta.tolist(),
                "cases": cases,
            },
            f,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
