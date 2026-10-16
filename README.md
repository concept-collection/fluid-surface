# fluid-surface

Fluid deformable surfaces in the browser: incompressible surface Navier-Stokes with a bending (Helfrich) force on a closed surface that moves with the fluid, following Krause & Voigt (2023), and later its two-phase version (Bachini et al. 2023). The method is spectral in spherical harmonics, the math is written in MATLAB, and [numbl](https://github.com/flatironinstitute/numbl) compiles it to WebGPU compute shaders.

The solver being ported is Dan Fortunato's C++ implementation in meliao/evolving_surface (cpu, CUDA and Metal backends). This is the sibling of [turing-surface](https://github.com/concept-collection/turing-surface), which solves reaction-diffusion on a fixed surface, and it starts from that project's transforms and compiler pipeline. The difference that keeps them apart is that here the geometry changes every step and each step is a saddle-point solve run to a tolerance, where turing-surface's design assumes a fixed surface and a fixed sequence of GPU operations.

**Status:** step 1 of 6, the vector analysis. There is no app yet, only the checks.

## Plan

1. The vector (spheroidal/toroidal) analysis from scalar transforms, against SHTns. Done; see below.
2. The surface geometry (normal, curvatures, area element) on the GPU from the position's coefficients, against the C++ cpu backend.
3. A Killing field on a fixed surface: the matvec, the static preconditioner and GMRES, against the exact BDF-1 decay rate.
4. The Stokes preconditioner, an evolving surface and bending: Krause & Voigt's perturbed sphere at lmax 32, against the C++.
5. Relabeling, then an adaptive time step.
6. Two phases.

As much of each step as possible is MATLAB in [`lib/`](lib/), compiled by numbl. What is written by hand in WGSL or TypeScript is what numbl cannot express yet: the transforms, coefficient-space shuffles and, later, the GMRES control loop.

## Vector analysis from scalar transforms

The fluid solver takes the surface divergence of a tangent field through the spheroidal part of its vector spherical-harmonic analysis. The WebGPU transforms are scalar only, and the C++ webgpu backend aborts at this call. Rather than write a vector Legendre kernel, [`lib/sphtor.m`](lib/sphtor.m) builds the vector analysis from pieces turing-surface already has. In SHTns' convention, `v_theta = dtheta S + dphi T / sin(theta)` and `v_phi = dphi S / sin(theta) - dtheta T`, so projecting onto `grad Y` and `-r x grad Y` gives

```
l(l+1) S = <v_theta, dtheta Y> + <v_phi / sin(theta), dphi Y>
l(l+1) T = <v_theta / sin(theta), dphi Y> - <v_phi, dtheta Y>
```

Since `sin(theta) dtheta Y_l = alpha+ Y_{l+1} + alpha- Y_{l-1}` at every point, `<f, dtheta Y>` is the transpose of turing-surface's `dthetac` shift applied to the analysis of `f / sin(theta)`. The whole operator is then

```matlab
function [S, T] = sphtor(vt, vp, st, ilam)
  [A, B] = analys(vt ./ st, vp ./ st);
  S = (dthetat(A) - dphic(B)) .* ilam;
  T = -(dphic(A) + dthetat(B)) .* ilam;
end
```

which compiles to 10 GPU operations: one batched two-field analysis, four coefficient shuffles and four element-wise kernels. `dthetat` is the one new external operation, and it is the `dthetac` kernel run with transposed coefficient tables. The quadrature sums are SHTns' own, term for term, so the result agrees with SHTns for any grid data and not only for band-limited fields. Gauss nodes never reach the poles, where `sin(theta) = 0`.

The one difference is the top degree. S and T at `l = lmax` would need the analysis at `lmax + 1`, so they are incomplete there (about 8% of the field's norm in the checks below). The fluid solver discards the top three degrees of S, so this does not reach it.

Measured on Dawn (Intel Iris Xe), degrees below lmax, relative L2:

| check | S | T |
|---|---:|---:|
| f64 formula vs SHTns 3.7.5, band-limited field, L16 | 1.5e-15 | 1.3e-15 |
| f64 formula vs SHTns 3.7.5, random grid values, L16 | 1.5e-15 | 1.2e-15 |
| compiled `sphtor.m` (fp32) vs SHTns, band-limited, L16 | 2.7e-7 | 2.6e-7 |
| compiled `sphtor.m` (fp32) vs SHTns, random grid values, L16 | 2.5e-7 | 1.8e-7 |
| fp32 vs exact S, T, L16 (18 x 64 grid) | 2.8e-7 | 2.6e-7 |
| fp32 vs exact S, T, L32 (34 x 128) | 4.6e-7 | 4.1e-7 |
| fp32 vs exact S, T, L64 (66 x 256) | 8.2e-7 | 7.9e-7 |

The fp32 error grows with lmax at the rate of the scalar analysis it is made of, so dividing by `sin(theta)` costs no visible precision. The worst single degree at L64 is 2.1e-6.

## Tests

```bash
npm run test:node
```

Runs [`test/sphtorChecks.ts`](test/sphtorChecks.ts) on desktop WebGPU (Google Dawn, the optional `webgpu` package). The SHTns results it compares against are in [`test/fixtures/`](test/fixtures/), written by [`scripts/shtns_sphtor_fixture.py`](scripts/shtns_sphtor_fixture.py), which needs the `shtns` Python package:

```bash
python scripts/shtns_sphtor_fixture.py -lmax 16
```

## Layout

- `lib/`: the MATLAB operators, compiled by numbl.
- `src/sht/`: the WebGPU spherical-harmonic transforms and coefficient shuffles, vendored from [shtns-webgpu](https://github.com/concept-collection/shtns-webgpu) by way of turing-surface, plus `dthetat`. `reference.ts` is the f64 CPU twin used by the tests.
- `src/mgpu/`: the MATLAB-to-WebGPU pipeline from turing-surface (numbl lowering, planning, WGSL generation). `gpuFunction.ts` compiles one `.m` function into something the host can call.
- `test/`, `scripts/`: the checks and their runner.

## The numbl dependency

numbl is a local `file:../../numbl` dependency, so a sibling checkout of numbl is required, as for turing-surface. We use its compiler internals, reached through the `numbl-src` alias in [`vite.config.ts`](vite.config.ts), and the surface we depend on is written down in [`src/mgpu/numbl.d.ts`](src/mgpu/numbl.d.ts).

## Provenance and license

`src/sht`, `src/mgpu`, `src/tools.ts` and `tools/` were copied from turing-surface at `167fa88`; the first commit here is that copy unchanged. CECILL-2.1, inherited from SHTns through shtns-webgpu.
