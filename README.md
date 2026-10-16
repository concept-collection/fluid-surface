# fluid-surface

Fluid deformable surfaces in the browser: incompressible surface Navier-Stokes with a bending (Helfrich) force on a closed surface that moves with the fluid, following Krause & Voigt (2023), and later its two-phase version (Bachini et al. 2023). The method is spectral in spherical harmonics, the math is written in MATLAB, and [numbl](https://github.com/flatironinstitute/numbl) compiles it to WebGPU compute shaders.

The solver being ported is Dan Fortunato's C++ implementation in meliao/evolving_surface (cpu, CUDA and Metal backends). This is the sibling of [turing-surface](https://github.com/concept-collection/turing-surface), which solves reaction-diffusion on a fixed surface, and it starts from that project's transforms and compiler pipeline. The difference that keeps them apart is that here the geometry changes every step and each step is a saddle-point solve run to a tolerance, where turing-surface's design assumes a fixed surface and a fixed sequence of GPU operations.

**Status:** steps 1 to 4 of 6: the vector analysis, the surface geometry (with a page that shows it), the fluid solver's static mode on a fixed surface, and a moving surface with the bending force and the volume constraint, each checked against the C++, with Krause & Voigt's perturbed sphere at band 32 compared by its energies. The solver is BDF-1 with a fixed time step; BDF-2, an adaptive step, relabeling and two phases are still to come.

## Plan

1. The vector (spheroidal/toroidal) analysis from scalar transforms, against SHTns. Done; see below.
2. The surface geometry (normal, curvatures, area element) on the GPU from the position's coefficients, against the C++ cpu backend. Done; see below.
3. A Killing field on a fixed surface: the matvec, the static preconditioner and GMRES, against the exact BDF-1 decay rate. Done; see below.
4. The moving surface, in three parts: (a) the operator without bending, the Stokes preconditioner and the normal-velocity update, (b) bending, (c) the volume constraint, then Krause & Voigt's perturbed sphere at lmax 32 against the C++. Done; see below.
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

It compiles to 59 GPU operations: 15 syntheses in four batches, 15 coefficient shuffles and 29 element-wise kernels, one per line of MATLAB. Measured on Dawn (Intel Iris Xe):

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

```bash
<evolving_surface>/cpp/build-cpu/static_dump test/fixtures/static-cpu-L12 12 2 26 64 0.01 2 1 2
```

```bash
<evolving_surface>/cpp/build-cpu/step_dump test/fixtures/killing-cpu-L12 killing 12 2 26 64 0.01 1 2 0 20
```

```bash
<evolving_surface>/cpp/build-cpu/step_dump test/fixtures/rotating-perturbed-cpu-L12 killing 12 2 26 64 0.01 1 2 0 20 0.2
```

```bash
<evolving_surface>/cpp/build-cpu/step_dump test/fixtures/bending-perturbed-cpu-L12 killing 12 2 26 64 0.01 1 2 1 20 0.2
```

```bash
<evolving_surface>/cpp/build-cpu/step_dump test/fixtures/perturbed-cpu-L12 perturbed_sphere 12 2 26 64 0.01 1 2 1 20
```

### The page

Live at https://concept-collection.github.io/fluid-surface/ (needs a browser with WebGPU), deployed from `main` by [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml). Locally:

```bash
npm run dev
```

The page runs Krause & Voigt's perturbed sphere live: Play advances the fluid one step per frame (`src/fluid/movingStokes.ts`, with bending and the volume constraint) and draws the surface coloured by H, K or the area element, with the bending and kinetic energies charted over time and the area, volume, multiplier and GMRES iterations listed. The working band, dt and r0 are controls; a band or dt change rebuilds the solver, an r0 change restarts from rest. It also lists `models/stokes.m` beside the GPU operations its matvec compiles to. The rendering is turing-surface's (`src/render/`, adapted from figpack's SphereEmbedding view).

In Chrome on an integrated Intel GPU a step takes about 0.3 s at band 16 (about 20 GMRES iterations), most of it the host GMRES's synchronization per iteration; the same step takes 0.06 to 0.12 s in node. At band 16 the page's energies match the C++ at band 16 to four digits (E_h 21.9227 against 21.9225 at step 32, E_kin 0.3742 against 0.3743), but band 16 under-resolves r0 = 0.4: its kinetic energy peaks at 0.91, against 0.63 at band 32, in both codes.

On the unit sphere, H is 1 to within 5e-7 away from the poles, but within only 4e-5 (band 24) to 9e-5 (band 32) on the latitude ring nearest each pole, against 5e-6 at bands 16 and 48. The x_thetatheta term is `(tt - ct .* sxt) ./ st.^2`, a difference of two small values divided by sin(theta)^2, and fp32 cancellation there is presumably the cause; the C++ formula has the same division. The checks above, at bands 12 and 16, do not see it. Whether it matters to the fluid solver, whose bending force is built on H, is a question for step 4. The flux-form weights `dp1, p2, dq2, jinv` avoid this division by construction.

## The static mode

[`models/static_stokes.m`](models/static_stokes.m) is one BDF-1 step of the C++ solver's static mode (`is_static`): surface Navier-Stokes on a fixed surface, the convection linearized about the current velocity, the normal velocity held at zero. It defines `setup` (the transport velocity and the preconditioner's per-degree symbols, with the mean squared curvature as a GPU reduction), `rhs`, `matvec` (the preconditioned saddle-point operator on velocity and pressure) and the static preconditioner. The operators they use are [`lib/surfgrad.m`](lib/surfgrad.m), the surface gradient, [`lib/tandiv.m`](lib/tandiv.m), the surface divergence through the vector analysis of step 1, and [`lib/deformdiv.m`](lib/deformdiv.m), the divergence of the rate of deformation. GMRES runs on the host in f64 ([`src/fluid/gmres.ts`](src/fluid/gmres.ts)): each iteration uploads one vector, runs the compiled matvec and reads one back. [`src/fluid/staticStokes.ts`](src/fluid/staticStokes.ts) chains the compiled functions; the geometry and the per-step tables pass between them in named GPU buffers, with no readback.

The C++ works on two bands, the working band of the unknowns and a product band twice as large for products, and pads and truncates between them. A compiled function here has one band, so every spectral field is kept on the product band, zero above the working band, and the truncation and the solver's filters are elementwise masks (`keep`, `clean`, `filtf`). GMRES sees only the working band's real unknowns, whose inner product is that of the C++'s real embedding. The cost is coefficient-space work on four times as many coefficients, which is small beside the transforms.

The matvec compiles to 98 GPU operations: 18 syntheses and 15 analyses (batched where they are adjacent), 18 coefficient shuffles and 47 element-wise kernels (74 before it was written with vector fields; see "Where numbl needed help"). Checks, on Dawn (Intel Iris Xe), GMRES at tol 1e-6:

| check | result |
|---|---|
| preconditioned operator on a random vector, against the C++ cpu backend in f64 (static_stokes, band 12, product 24 on 26 x 64), rel L2 | 5.1e-7 |
| preconditioned right-hand side, same, rel L2 | 2.5e-7 |
| one step's velocity, against the C++ solved to 1e-12, rel L2 | 1.8e-5 (7 GMRES iterations) |
| degree-2 toroidal flow on the unit sphere, band 16, dt 0.01, 100 steps: fitted decay rate against BDF-1's exact ln(1 + 4 dt)/dt | 3.9220722 vs 3.9220713, relative error 2.3e-7 (6.2 iterations per step) |
| degree 1, the Killing field (rigid rotation): fitted decay rate | -4.8e-9 |

The one-step difference is set by GMRES's tolerance. GMRES on an fp32 operator cannot go much below 1e-6: at tol 1e-7 the step agrees with the C++ to 2.4e-6, but the true residual stalls near 6.5e-7 and GMRES does not converge. The C++ solver's fp32 builds raise their tolerance to the same floor (`FLUID_FP32_TOL_FLOOR`). A step takes about 33 ms at band 16 in node, most of it the per-iteration synchronization of the host GMRES.

## A moving surface, without bending

[`models/stokes.m`](models/stokes.m) is the C++ solver's step on a moving surface with bending off and no volume constraint: the same momentum as the static mode without its projection, the Stokes preconditioner (the same system on the sphere of the same area, solved exactly degree by degree), and `advance`, which moves the surface with the normal part of the new velocity. [`src/fluid/movingStokes.ts`](src/fluid/movingStokes.ts) runs the step; the position stays on the GPU, written back by `advance` into the buffers `lib/geometry.m` reads.

The preconditioner works on the radial, spheroidal and toroidal potentials Q, S, T of the velocity. The C++ converts to and from them with recurrences on the coefficients for multiplication by x, y and z, which couple degree l to l +- 1 and order m to m +- 1, a shuffle numbl has no operation for. [`lib/cart2qst.m`](lib/cart2qst.m) and [`lib/qst2cart.m`](lib/qst2cart.m) do it on the grid instead: Q is the radial component r . u, and S, T are the vector analysis of step 1 applied to the tangential part. The products are exact on the product grid, so the two agree to rounding; the trade-off is cost, 37 GPU operations per round trip, 10 of them transforms, where the C++ needs none. The 3 x 3 block of each degree is inverted on the host in f64, as the C++ builds its blocks on the host, and the per-degree solve is a 3 x 3 product in `models/stokes.m`.

The matvec compiles to 149 GPU operations. Checks against the C++ cpu backend in f64 (band 12, product 24 on 26 x 64, dt 0.01, Re 1, no bending), GMRES at tol 1e-6, relative L2:

| check | result |
|---|---|
| Cartesian to Q, S, T on a random field, and back | Q 1.9e-7, S 1.2e-7, T 1.9e-7; back 5.1e-7 |
| killing example (the unit sphere in rigid rotation): operator on a random vector, right-hand side | 5.1e-6, 3.0e-7 |
| killing example after 1 and after 20 steps | X 8.2e-8 and 6.0e-8, u 1.7e-7 and 4.6e-7, p 1.5e-6 and 4.0e-6 (3.0 GMRES iterations per step) |
| the same rotation started on the perturbed sphere (r0 0.2): operator, right-hand side | 1.8e-6, 3.1e-7 |
| rotating perturbed sphere after 1 and after 20 steps | X 6.5e-9 and 1.4e-7, u 2.7e-7 and 1.7e-6, p 2.9e-6 and 4.9e-6 (9.6 iterations per step) |

Over the 20 steps the rotating perturbed sphere's position changes by 1.8% and the rotating sphere's by 0.7%: without bending nothing holds the sphere's shape against the rotation, in the C++ as here. A step takes 17 to 34 ms at band 12 in node.

GMRES stops, as the C++'s does, when its own residual estimate reaches the tolerance; the explicit residual is computed once at the end and reported.

## Bending

The Helfrich force is -(2/Be) (Laplacian of H + 2 H (H^2 - K)) n, with the Laplacian in the C++'s flux form ([`lib/fluxlap.m`](lib/fluxlap.m)). The step treats it implicitly, as the C++ does: the surface's displacement over the step is dt times the normal velocity being solved for, so the matvec subtracts the force's derivative along that displacement. [`lib/bendjvp.m`](lib/bendjvp.m) is that derivative (`FluidBendingJvp`): the derivative of the position's derivative fields ([`lib/posderivs.m`](lib/posderivs.m), now shared with `lib/geometry.m`), of the curvatures, normal and Laplacian weights ([`lib/geomtangent.m`](lib/geomtangent.m), `KFGeometryTangent`), of the Laplacian of H, whose weights and metric factor move with the surface, and of the force itself. `bendlin` in `models/stokes.m` computes the force once per step and stores what the derivative reuses. A parameter `bscale = -2/Be` scales both, and 0 turns bending off, as Be = infinity does in the C++.

With bending, the matvec compiles to 255 GPU operations, 99 of them the derivative. Checks against the C++ cpu backend in f64, the rotation of step 4a on the perturbed sphere (r0 0.2) with Be 2, band 12, dt 0.01, GMRES at tol 1e-6, relative L2:

| check | result |
|---|---|
| bending force at the initial surface | 1.5e-6 |
| its derivative along a random displacement | 1.1e-6 |
| operator on a random vector, right-hand side | 2.0e-6, 8.3e-7 |
| after 1 and after 20 steps | X 5.8e-8 and 2.5e-7, u 3.1e-7 and 3.3e-6, p 1.5e-6 and 1.5e-5 (8.5 GMRES iterations per step) |

Over the 20 steps the surface moves by 6.5%. A step takes about 60 ms at band 12 in node.

## The volume constraint

With the constraint, the enclosed volume is held by a multiplier lam on the normal force: the momentum loses lam n, and a new row asks that the flux of the normal velocity through the surface vanish. The multiplier is one more unknown, a 1 x 1 array that the compiled functions take and return. The Stokes preconditioner (as the C++'s) solves for it first from a 2 x 2 block: the sphere velocity's response to the force of a constant pressure and to that of the multiplier, measured by the divergence's constant mode and the flux. `volblock` in `models/stokes.m` computes the block once per step with two velocity-only sphere solves and two reductions each, and the host reads its four numbers and decides, as the C++ does, whether the block is far enough from singular to use. The preconditioner then moves the multiplier's force to the right-hand side and corrects the constant modes of the pressure and the radial velocity. A plan has no branches, so whether the constraint is on and whether the block is used are parameters (`vol`, `ub`); without the constraint the multiplier's row is the identity and it stays zero.

Krause & Voigt's perturbed sphere (r0 0.4, at rest, Re 1, Be 2, volume constrained), band 12, dt 0.01, against the C++, relative L2:

| check | result |
|---|---|
| the 2 x 2 block | 2.1e-7 (used, as in the C++) |
| operator on a random vector with a multiplier, right-hand side | 8.2e-7, 9.5e-7 |
| after 1 and after 20 steps | X 9.1e-8 and 1.7e-7, u 3.8e-6 and 1.1e-5, p 1.6e-6 and 6.2e-7; multiplier 13.16236 (C++ 13.16239) and 9.180041 (9.180042) |

At band 32 (product band 64), the same problem from rest with dt 0.005 for 100 steps, BDF-1, against `fluid_run` on the C++ cpu backend with the same settings (`-order 1 -dt 0.005 -T 0.5 -relabel_every 0`; its product grid is SHTns' choice, ours 66 x 256), with [`scripts/kv.ts`](scripts/kv.ts):

| quantity | largest relative difference over the 100 steps |
|---|---:|
| bending energy E_h (24.11 at rest, 21.70 at t = 0.5) | 6.4e-7 |
| area | 4.8e-7 |
| enclosed volume | 8.3e-7 |
| kinetic energy, relative to its peak (0.63) | 2.0e-5 |
| position at t = 0.5 | 4.9e-7 |

GMRES took 34.6 iterations per step at tol 1e-6 (the C++ 46.1 at 1e-8); a step took 0.29 s here on Dawn with an integrated GPU, against 0.19 s for the C++ on four cores. The volume drifts by 1.8e-4 over the run in both, since the constraint is on the linearized flux.

## Where numbl needed help

This list records each place the MATLAB could not say what was meant, or the compiler did less than it should, since that is what a numbl paper has to report.

- **A transposed shuffle** (step 1). `dthetat` is a new external operation, though it reuses the `dthetac` kernel.
- **Real square roots** (step 2). `sqrt` of a value whose sign numbl cannot prove is complex, as in MATLAB, and complex arrays do not compile to WGSL. A determinant is nonnegative by Cauchy-Schwarz, which numbl cannot see. numbl now has MATLAB's `realsqrt` ([numbl#66](https://github.com/flatironinstitute/numbl/pull/66)), which says the result is real; before it, the code needed `sqrt(abs(detg))`.
- **Fusion stopped at 32 substitutions** (step 2). numbl's inline pass made one substitution per sweep and gave up after 32 sweeps, which left `geometry.m` at 123 operations instead of 61. Fixed in numbl#66: the pass now runs to its fixed point.
- **A call on an expression did not fuse** (step 2). numbl kept every tensor-valued call off its fused path, to exclude shape constructors such as `nan(2,3)`, which also kept `1 ./ realsqrt(ptt .* gpp - ptp.^2)` at three kernels where `realsqrt(detg)` was one. Fixed in [numbl#67](https://github.com/flatironinstitute/numbl/pull/67): a tensor result with tensor arguments of its own shape now fuses.
- **Bindings per kernel** (steps 2 and 3). A fused kernel binds every array its expression reads, plus the output and the parameters, against WebGPU's default limit of 8 storage buffers. The device asks for the adapter's limit (16 on this machine), and [`src/mgpu/split.ts`](src/mgpu/split.ts) splits a statement that still reads too many arrays, hoisting its widest subexpression that fits until the rest does. Each momentum line of the matvec, with the pressure gradient and the viscous divergence folded in, needs one such split.
- **Vectors were three arrays** (fixed after step 4). A vector field was written as three scalar fields, with dot products by hand, and the rate of deformation took 13 scalar lines. The compiler now takes `npts x 3` arrays: element-wise operations broadcast an `npts x 1` field against them as in MATLAB, `sum(x, 2)` is a dot product (unrolled over the columns), `x(:, k)` reads a column at a literal k, and `[a, b, c]` assembles one ([`src/mgpu/wgsl.ts`](src/mgpu/wgsl.ts)). numbl's inline pass folds only same-shape element-wise code, since its C emitter needs that, so [`src/mgpu/fuse.ts`](src/mgpu/fuse.ts) folds the rest after it: broadcasts, row sums, column reads and concatenations. A vector assembled from buffers (`n = [geo.n1, geo.n2, geo.n3]`) is a view, folded into every reader, as is a vector read only by its columns, and a column read of an expression is resolved in the IR (`(a + [d1, d2, d3])(:, 2)` is `a(:, 2) + d2`), so a kernel binds only the buffers it reads. [`lib/surfgrad.m`](lib/surfgrad.m) now returns a vector, [`lib/tandiv.m`](lib/tandiv.m) takes one, and [`lib/deformdiv.m`](lib/deformdiv.m) writes the rate of deformation as the rows of P G P and their symmetric part. The static matvec went from 125 to 98 GPU operations and the moving one from 267 to 245, with the checks unchanged to rounding; the time per step in these small tests did not change measurably, since it is dominated by GMRES's readbacks. The transforms still take one array per component, so the geometry keeps its components (`geo.n1`) and vectors are assembled from them as views. Only a single row of column blocks, row sums along dimension 2 and literal column indices are supported; `lib/geometry.m` and the bending code are still written in components.
- **No structures** (steps 3 to 4, fixed after step 4). The compiler bound only plain arrays and scalars, so every field was its own argument: the moving matvec took 82, and `lib/geomtangent.m` 31. numbl's lowering already had struct types; [`src/mgpu/plan.ts`](src/mgpu/plan.ts) now binds a struct argument field by field, each field a GPU buffer named `<argument>.<field>`, and a field read (`geo.n1`, which numbl's ANF gives its own statement) aliases that buffer, so it costs no kernel and still fuses into the lines that use it. [`src/mgpu/inlineCalls.ts`](src/mgpu/inlineCalls.ts) carries struct arguments into subfunctions. The model now groups its arguments as `geo`, `der`, `bl`, `grid`, `msk` and `tab` (see the head of `models/stokes.m`); the matvec takes 29 arguments, 12 of them scalar parameters, and `lib/bendjvp.m` 9. Only array fields are supported.
- **No branches in a plan.** Bending is turned off by a zero scale (`bscale`), so a run without bending still runs the force's derivative in every matvec, 99 of its 245 GPU operations.
- **`max(tensor, scalar)`** (step 3) was not supported by numbl's lowering. numbl#67 adds the tensor and broadcast forms of two-argument `max` and `min`, and lets a builtin be elementwise for some argument forms only (`max(a, b)` is pointwise, `max(a)` a reduction); `models/static_stokes.m` uses it as the C++ does.
- **Reductions are statements.** `dot` must be its own statement on named arrays; `dot(a, w) / dot(b, w)` in one line does not compile, since the planner sees the grid-sized arguments inside a scalar kernel.
- **One band per function.** A function runs on one grid and one band. Keeping every spectral field on the product band, with masks for truncation and filters (see the static mode above), works around it at some cost.
- **No shifts in the order m** (step 4a). Multiplying by x or y on the sphere couples m to m +- 1, which none of the coefficient shuffles does; the conversion to Q, S, T goes through the grid instead, at the cost of ten transforms.
- **GMRES is hand-written** (step 3), in TypeScript on the host, because its restarts and convergence test need the host in the loop, which a compiled plan (a fixed list of GPU operations) does not have.

## Tests

```bash
npm run test:node
```

Runs [`test/compileChecks.ts`](test/compileChecks.ts) (the compiler's struct arguments and vector fields), [`test/sphtorChecks.ts`](test/sphtorChecks.ts), [`test/geometryChecks.ts`](test/geometryChecks.ts), [`test/stokesChecks.ts`](test/stokesChecks.ts) and [`test/movingChecks.ts`](test/movingChecks.ts) (`-- --only=volume` for one suite) on desktop WebGPU (Google Dawn, the optional `webgpu` package). The SHTns results it compares against are in [`test/fixtures/`](test/fixtures/), written by [`scripts/shtns_sphtor_fixture.py`](scripts/shtns_sphtor_fixture.py), which needs the `shtns` Python package:

```bash
python scripts/shtns_sphtor_fixture.py -lmax 16
```

## Layout

- `lib/`: the MATLAB operators, compiled by numbl.
- `src/sht/`: the WebGPU spherical-harmonic transforms and coefficient shuffles, vendored from [shtns-webgpu](https://github.com/concept-collection/shtns-webgpu) by way of turing-surface, plus `dthetat`. `reference.ts` is the f64 CPU twin used by the tests.
- `src/mgpu/`: the MATLAB-to-WebGPU pipeline from turing-surface (numbl lowering, planning, WGSL generation). `gpuFunction.ts` compiles one `.m` function into something the host can call.
- `models/`: one solver mode per file (`static_stokes.m`, `stokes.m`). `src/fluid/`: the host side, GMRES and the steppers.
- `src/geom/`: the geometry on the GPU and the example surfaces. `src/render/`, `src/main.ts`, `index.html`: the page.
- `test/`, `scripts/`: the checks and their runner, and `scripts/kv.ts`, the Krause & Voigt run; `scripts/cpp/` holds the tool that writes the C++ reference.

## The numbl dependency

numbl is a local `file:../../numbl` dependency, so a sibling checkout of numbl is required, as for turing-surface. It must be at or after `592e9d77` ([numbl#67](https://github.com/flatironinstitute/numbl/pull/67)), for `realsqrt` and the fixed-point inline pass (#66), and for elementwise calls on tensors and two-argument `max` on tensors (#67). We use its compiler internals, reached through the `numbl-src` alias in [`vite.config.ts`](vite.config.ts), and the surface we depend on is written down in [`src/mgpu/numbl.d.ts`](src/mgpu/numbl.d.ts).

## Provenance and license

`src/sht`, `src/mgpu`, `src/tools.ts`, `tools/` and (later) four files of `src/render` were copied from turing-surface at `167fa88`; the first commit here is that copy unchanged. Since then: `dthetat`, `realsqrt` in the WGSL builtin table, the storage-buffer limit requested from the adapter, and the kernel split. CECILL-2.1, inherited from SHTns through shtns-webgpu.
