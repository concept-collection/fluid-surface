# fluid-surface

Fluid deformable surfaces in the browser: incompressible surface Navier-Stokes with a bending (Helfrich) force on a closed surface that moves with the fluid, following Krause & Voigt (2023), and later its two-phase version (Bachini et al. 2023). The method is spectral in spherical harmonics, the math is written in MATLAB, and [numbl](https://github.com/flatironinstitute/numbl) compiles it to WebGPU compute shaders.

The solver being ported is Dan Fortunato's C++ implementation in meliao/evolving_surface (cpu, CUDA and Metal backends). This is the sibling of [turing-surface](https://github.com/concept-collection/turing-surface), which solves reaction-diffusion on a fixed surface, and it starts from that project's transforms and compiler pipeline. The difference that keeps them apart is that here the geometry changes every step and each step is a saddle-point solve run to a tolerance, where turing-surface's design assumes a fixed surface and a fixed sequence of GPU operations.

**Status:** steps 1 and 2 of 6 (the vector analysis and the surface geometry). There is no app yet, only the checks.

## Plan

1. The vector (spheroidal/toroidal) analysis from scalar transforms, against SHTns. Done; see below.
2. The surface geometry (normal, curvatures, area element) on the GPU from the position's coefficients, against the C++ cpu backend. Done; see below.
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

## Surface geometry

[`lib/geometry.m`](lib/geometry.m) is the C++ solver's `FluidGeometryBuild`: from the coefficients of the position X, the dual tangent vectors, the unit normal, the mean and Gaussian curvatures, the area element and the four weights of the flux-form Laplace-Beltrami operator, 16 grid fields in all. Every derivative is taken in coefficient space and synthesized. The second theta derivative, which the C++ takes from its own recurrence, is written with the first-derivative shift alone, `sin^2(theta) u_thetatheta = (sin(theta) dtheta)^2 u - cos(theta) sin(theta) u_theta`, so no new external operation was needed. The position is zero-padded on the host to the product band (twice the working band) before the call, as in the C++.

It compiles to 63 GPU operations: 15 syntheses in four batches, 15 coefficient shuffles and 33 element-wise kernels, one per line of MATLAB except for four around `sqrt(abs(.))`. Measured on Dawn (Intel Iris Xe):

| check | largest error |
|---|---:|
| unit sphere, L16, against the closed form (n = -x, H = K = 1, sqrt(det g) = sin(theta), sphere weights), max abs | 1.7e-6 (K) |
| Krause & Voigt's perturbed sphere (r0 0.4), working band 12, product band 24 on 26 x 64, against the C++ cpu backend in f64, rel L2 per field | 1.2e-6 (K) |

The C++ values are in [`test/fixtures/geometry-cpu-perturbed-L12/`](test/fixtures/geometry-cpu-perturbed-L12/), in the solver's own `ref_dump` format, written by [`scripts/cpp/geometry_dump.cpp`](scripts/cpp/geometry_dump.cpp). To remake them, build the cpu backend of meliao/evolving_surface (PR #10; without PETSc, and with a serial HDF5 if the system has none) and then the tool against it:

```bash
cmake -S cpp -B cpp/build-cpu -DES_CPU_PETSC=OFF -DES_BUILD_TESTS=OFF -DHDF5_ROOT=<hdf5> -DHDF5_USE_STATIC_LIBRARIES=ON -DSHTNS_GIT_REPOSITORY=https://github.com/danfortunato/shtns-metal.git
```

```bash
scripts/cpp/build.sh <evolving_surface>/cpp <evolving_surface>/cpp/build-cpu
```

```bash
<evolving_surface>/cpp/build-cpu/geometry_dump test/fixtures/geometry-cpu-perturbed-L12 perturbed_sphere 12 2 26 64 0.4
```

## Where numbl needed help

This list records each place the MATLAB could not say what was meant, or the compiler did less than it should, since that is what a numbl paper has to report.

- **A transposed shuffle** (step 1). `dthetat` is a new external operation, though it reuses the `dthetac` kernel.
- **Real square roots** (step 2). `sqrt` of a value whose sign numbl cannot prove is complex, as in MATLAB, and complex arrays do not compile to WGSL. A determinant is nonnegative by Cauchy-Schwarz, which numbl cannot see, so the code says `sqrt(abs(detg))`. MATLAB's `realsqrt` would say it directly; numbl does not have it. The two lines with `sqrt(abs(.))` also compile to two and four kernels, where every other line fuses into one; why is not yet looked into.
- **Fusion stopped at 32 substitutions** (step 2). numbl's inline pass makes one substitution per sweep and gives up after 32 sweeps, which left `geometry.m` at 123 operations instead of 63. [`src/mgpu/compile.ts`](src/mgpu/compile.ts) reruns the pass to its fixed point; the cap belongs in numbl.
- **Bindings per kernel** (step 2). A fused kernel binds every array its expression reads, and some geometry lines read seven, plus the output and the parameters, against WebGPU's default limit of 8 storage buffers. The device now asks for the adapter's limit (16 on this machine). Nothing yet splits a kernel that exceeds it.
- **Vectors are three arrays.** A vector field is written as three scalar fields (`xt1`, `xt2`, `xt3`), and dot and cross products by hand.
- **One band per function.** A function runs on one grid and one band, so the padding to the product band is done by the host. The matvec will need padding and truncation inside the `.m`.

## Tests

```bash
npm run test:node
```

Runs [`test/sphtorChecks.ts`](test/sphtorChecks.ts) and [`test/geometryChecks.ts`](test/geometryChecks.ts) on desktop WebGPU (Google Dawn, the optional `webgpu` package). The SHTns results it compares against are in [`test/fixtures/`](test/fixtures/), written by [`scripts/shtns_sphtor_fixture.py`](scripts/shtns_sphtor_fixture.py), which needs the `shtns` Python package:

```bash
python scripts/shtns_sphtor_fixture.py -lmax 16
```

## Layout

- `lib/`: the MATLAB operators, compiled by numbl.
- `src/sht/`: the WebGPU spherical-harmonic transforms and coefficient shuffles, vendored from [shtns-webgpu](https://github.com/concept-collection/shtns-webgpu) by way of turing-surface, plus `dthetat`. `reference.ts` is the f64 CPU twin used by the tests.
- `src/mgpu/`: the MATLAB-to-WebGPU pipeline from turing-surface (numbl lowering, planning, WGSL generation). `gpuFunction.ts` compiles one `.m` function into something the host can call.
- `test/`, `scripts/`: the checks and their runner; `scripts/cpp/` holds the tool that writes the C++ reference.

## The numbl dependency

numbl is a local `file:../../numbl` dependency, so a sibling checkout of numbl is required, as for turing-surface. We use its compiler internals, reached through the `numbl-src` alias in [`vite.config.ts`](vite.config.ts), and the surface we depend on is written down in [`src/mgpu/numbl.d.ts`](src/mgpu/numbl.d.ts).

## Provenance and license

`src/sht`, `src/mgpu`, `src/tools.ts` and `tools/` were copied from turing-surface at `167fa88`; the first commit here is that copy unchanged. Since then: `dthetat`, the inline pass run to its fixed point, and the storage-buffer limit requested from the adapter. CECILL-2.1, inherited from SHTns through shtns-webgpu.
