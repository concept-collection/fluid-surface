% One BDF-1 step of the fluid solver on a moving surface: surface
% Navier-Stokes with the Helfrich bending force and, optionally, a constraint
% on the enclosed volume (a multiplier lam on the normal force), the
% convection linearized about the current velocity and the bending force
% about the current surface, after which the surface moves with the fluid's
% normal velocity. This is the C++ solver's FluidStep with the normal gauge
% (core/fluid_stepper.cpp, core/fluid_op.cpp, core/fluid_bending.cpp),
% preconditioned by the same system on the sphere of the same area
% (core/fluid_precond.cpp, FLUID_PC_STOKES).
%
% Bending is implicit: the step's displacement is dt times the normal
% velocity being solved for, so the matvec subtracts the bending force's
% derivative along it (bendjvp). bscale is -2/Be, and 0 turns bending off.
%
% Spectral fields are on the product band, zero above the working band, with
% the masks of models/static_stokes.m; cutv also drops the working band's
% top two degrees. The sphere's per-degree Stokes block is inverted on the
% host (i11 ... i33, and itor for the toroidal part), as the C++ builds its
% blocks on the host.
%
% Per step the host runs lib/geometry.m and lib/posderivs.m on the surface,
% setup, reads the area to build the blocks, runs bendlin, volblock (with the
% constraint; the host reads its 2 x 2 block) and rhs, GMRES on matvec, then
% advance.

function [v1, v2, v3, area] = setup(U1, U2, U3, n1, n2, n3, sqrtg, wq)
  % the transport velocity: the tangential part of the current velocity
  [u1, u2, u3] = synth(U1, U2, U3);
  un = u1 .* n1 + u2 .* n2 + u3 .* n3;
  v1 = u1 - un .* n1;
  v2 = u2 - un .* n2;
  v3 = u3 - un .* n3;
  area = dot(sqrtg, wq);
end

% The bending force at the current surface, -2/Be (Laplacian of H +
% 2 H (H^2 - K)) n, and what its derivative reuses.
function [BF1, BF2, BF3, ba, bb, braw, bvc] = bendlin(H, K, n1, n2, n3, dp1, p2, dq2, jinv, st, lam, filtf, real0, keep, bscale)
  [L, braw, ba, bb] = fluxlap(analys(H), dp1, p2, dq2, jinv, st, lam, filtf);
  % the Laplacian of H, re-analysed with m = 0 made real, as the C++ does
  lapH = synth(analys(L) .* real0);
  bvc = lapH + 2 * H .* (H .* H - K);
  [F1, F2, F3] = analys(bscale * bvc .* n1, bscale * bvc .* n2, bscale * bvc .* n3);
  BF1 = F1 .* keep;
  BF2 = F2 .* keep;
  BF3 = F3 .* keep;
end

function [B1, B2, B3, BP, BL] = rhs(U1, U2, U3, BF1, BF2, BF3, NC1, NC2, NC3, ct, st, cp, sp, ilam, cutv, clean, e0, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor, mass, vol, cr00, crlam, ar00, arlam, ub, R, qdiv)
  % the history term u / dt and the bending force, and no source for the
  % divergence or the volume
  [B1, B2, B3, BP, BL] = precond(mass * U1 + BF1, mass * U2 + BF2, mass * U3 + BF3, 0 * U1, 0, NC1, NC2, NC3, ct, st, cp, sp, ilam, cutv, clean, e0, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor, vol, cr00, crlam, ar00, arlam, ub, R, qdiv);
end

% The normal's coefficients, and the 2 x 2 block of the volume constraint:
% the sphere velocity's response to the force of a constant pressure and to
% that of the multiplier, measured by the divergence's constant mode and by
% the flux through the surface (FluidPrecondSetup). y00 is Y_0^0.
function [NC1, NC2, NC3, m00, m01, m10, m11] = volblock(H, n1, n2, n3, sqrtg, wq, vt1, vt2, vt3, vp1, vp2, vp3, jinv, ct, st, cp, sp, ilam, filtf, keep, cutv, clean, e0, w11, w12, w21, w22, itor, y00)
  [N1, N2, N3] = analys(n1, n2, n3);
  NC1 = N1 .* keep;
  NC2 = N2 .* keep;
  NC3 = N3 .* keep;
  [P1, P2, P3] = analys(2 * y00 * H .* n1, 2 * y00 * H .* n2, 2 * y00 * H .* n3);
  [a1, a2, a3] = velinv(P1 .* keep, P2 .* keep, P3 .* keep, ct, st, cp, sp, ilam, cutv, clean, w11, w12, w21, w22, itor);
  [b1, b2, b3] = velinv(-NC1, -NC2, -NC3, ct, st, cp, sp, ilam, cutv, clean, w11, w12, w21, w22, itor);
  [m00, m10] = response(a1, a2, a3, n1, n2, n3, sqrtg, wq, vt1, vt2, vt3, vp1, vp2, vp3, jinv, H, st, filtf, e0);
  [m01, m11] = response(b1, b2, b3, n1, n2, n3, sqrtg, wq, vt1, vt2, vt3, vp1, vp2, vp3, jinv, H, st, filtf, e0);
end

% The sphere's velocity system, without the pressure, solved degree by degree.
function [V1, V2, V3] = velinv(F1, F2, F3, ct, st, cp, sp, ilam, cutv, clean, w11, w12, w21, w22, itor)
  [Q, S, T] = cart2qst(F1, F2, F3, ct, st, cp, sp, ilam);
  [U1, U2, U3] = qst2cart((w11 .* Q + w12 .* S) .* cutv, (w21 .* Q + w22 .* S) .* cutv, itor .* T .* cutv, ct, st, cp, sp);
  V1 = U1 .* clean;
  V2 = U2 .* clean;
  V3 = U3 .* clean;
end

% The constant mode of a velocity's surface divergence, and its flux.
function [d00, flux] = response(V1, V2, V3, n1, n2, n3, sqrtg, wq, vt1, vt2, vt3, vp1, vp2, vp3, jinv, H, st, filtf, e0)
  [u1, u2, u3] = synth(V1, V2, V3);
  D = analys(tandiv(u1, u2, u3, n1, n2, n3, vt1, vt2, vt3, vp1, vp2, vp3, jinv, H, st, filtf));
  d00 = dot(D, e0);
  un = (u1 .* n1 + u2 .* n2 + u3 .* n3) .* sqrtg;
  flux = dot(un, wq);
end

function [R1, R2, R3, RP, RL] = matvec(Z1, Z2, Z3, Z4, Z5, NC1, NC2, NC3, sqrtg, wq, e0, vol, cr00, crlam, ar00, arlam, ub, R, qdiv, v1, v2, v3, vt1, vt2, vt3, vp1, vp2, vp3, n1, n2, n3, H, K, dp1, p2, dq2, jinv, sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3, ba, bb, braw, bvc, ct, st, cp, sp, lam, filtf, real0, keep, ilam, cutv, clean, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor, mass, visc, disp, bscale)
  [u1, u2, u3, p] = synth(Z1, Z2, Z3, Z4);
  % velocity gradient G(r, c) = d u_r / d x_c, and the pressure gradient
  [G11, G12, G13] = surfgrad(Z1, vt1, vt2, vt3, vp1, vp2, vp3, st);
  [G21, G22, G23] = surfgrad(Z2, vt1, vt2, vt3, vp1, vp2, vp3, st);
  [G31, G32, G33] = surfgrad(Z3, vt1, vt2, vt3, vp1, vp2, vp3, st);
  [q1, q2, q3] = surfgrad(Z4, vt1, vt2, vt3, vp1, vp2, vp3, st);

  % rate of deformation: the symmetric part of P G P, P = I - n n'
  Gn1 = G11 .* n1 + G12 .* n2 + G13 .* n3;
  Gn2 = G21 .* n1 + G22 .* n2 + G23 .* n3;
  Gn3 = G31 .* n1 + G32 .* n2 + G33 .* n3;
  nG1 = n1 .* G11 + n2 .* G21 + n3 .* G31;
  nG2 = n1 .* G12 + n2 .* G22 + n3 .* G32;
  nG3 = n1 .* G13 + n2 .* G23 + n3 .* G33;
  nGn = n1 .* Gn1 + n2 .* Gn2 + n3 .* Gn3;
  s11 = G11 - n1 .* nG1 - Gn1 .* n1 + n1 .* nGn .* n1;
  s22 = G22 - n2 .* nG2 - Gn2 .* n2 + n2 .* nGn .* n2;
  s33 = G33 - n3 .* nG3 - Gn3 .* n3 + n3 .* nGn .* n3;
  s12 = 0.5 * (G12 - n1 .* nG2 - Gn1 .* n2 + G21 - n2 .* nG1 - Gn2 .* n1) + n1 .* nGn .* n2;
  s13 = 0.5 * (G13 - n1 .* nG3 - Gn1 .* n3 + G31 - n3 .* nG1 - Gn3 .* n1) + n1 .* nGn .* n3;
  s23 = 0.5 * (G23 - n2 .* nG3 - Gn2 .* n3 + G32 - n3 .* nG2 - Gn3 .* n2) + n2 .* nGn .* n3;

  % divergence of the stress, row by row, and of the velocity
  d1 = tandiv(s11, s12, s13, n1, n2, n3, vt1, vt2, vt3, vp1, vp2, vp3, jinv, H, st, filtf);
  d2 = tandiv(s12, s22, s23, n1, n2, n3, vt1, vt2, vt3, vp1, vp2, vp3, jinv, H, st, filtf);
  d3 = tandiv(s13, s23, s33, n1, n2, n3, vt1, vt2, vt3, vp1, vp2, vp3, jinv, H, st, filtf);
  du = tandiv(u1, u2, u3, n1, n2, n3, vt1, vt2, vt3, vp1, vp2, vp3, jinv, H, st, filtf);

  % momentum: inertia, convection by the transport velocity, pressure, viscosity
  lamv = vol * Z5;
  m1 = mass * u1 + G11 .* v1 + G12 .* v2 + G13 .* v3 + q1 + 2 * H .* p .* n1 - visc * d1 - lamv .* n1;
  m2 = mass * u2 + G21 .* v1 + G22 .* v2 + G23 .* v3 + q2 + 2 * H .* p .* n2 - visc * d2 - lamv .* n2;
  m3 = mass * u3 + G31 .* v1 + G32 .* v2 + G33 .* v3 + q3 + 2 * H .* p .* n3 - visc * d3 - lamv .* n3;
  % the normal velocity, whose displacement over the step bends the surface
  un = u1 .* n1 + u2 .* n2 + u3 .* n3;
  [M1, M2, M3, MD, N1, N2, N3] = analys(m1, m2, m3, du, un .* n1, un .* n2, un .* n3);
  % the volume row: the flux of the normal velocity (the identity without the constraint)
  unw = un .* sqrtg;
  flux = dot(unw, wq);
  VR = vol * flux + (1 - vol) * Z5;
  [J1, J2, J3] = bendjvp(disp * N1 .* keep, disp * N2 .* keep, disp * N3 .* keep, sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3, ct, st, H, K, n1, n2, n3, dp1, p2, dq2, jinv, ba, bb, braw, bvc, lam, filtf, real0, keep, bscale);
  [R1, R2, R3, RP, RL] = precond(M1 .* keep - J1, M2 .* keep - J2, M3 .* keep - J3, MD .* keep, VR, NC1, NC2, NC3, ct, st, cp, sp, ilam, cutv, clean, e0, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor, vol, cr00, crlam, ar00, arlam, ub, R, qdiv);
end

function [Y1, Y2, Y3] = advance(U1, U2, U3, X1, X2, X3, n1, n2, n3, clean, dt)
  % the surface moves with the normal part of the new velocity, filtered
  [u1, u2, u3] = synth(U1, U2, U3);
  un = u1 .* n1 + u2 .* n2 + u3 .* n3;
  [W1, W2, W3] = analys(un .* n1, un .* n2, un .* n3);
  Y1 = X1 + dt * W1 .* clean;
  Y2 = X2 + dt * W2 .* clean;
  Y3 = X3 + dt * W3 .* clean;
end

% The Stokes preconditioner: the same system on the sphere of the same area,
% solved exactly, one degree at a time, for the radial, spheroidal and
% toroidal potentials of the velocity and for the pressure. With the volume
% constraint (vol = 1) the multiplier comes first, from the 2 x 2 block (or
% from the sphere's when the block is near singular; the host sets cr00,
% crlam accordingly), its force moves to the right-hand side, and the block
% then fixes the pressure's constant mode, which the degree-zero radial
% momentum row turns into the radial velocity's (KFScalarPairLam,
% KFScalarPairFix; qdiv is that row's symbol, R the sphere's radius).
function [R1, R2, R3, RP, RL] = precond(M1, M2, M3, MP, ML, NC1, NC2, NC3, ct, st, cp, sp, ilam, cutv, clean, e0, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor, vol, cr00, crlam, ar00, arlam, ub, R, qdiv)
  rp0 = dot(MP, e0);
  RL = cr00 * rp0 + crlam * ML;
  lamv = vol * RL;
  [Q, S, T] = cart2qst(M1 + lamv .* NC1, M2 + lamv .* NC2, M3 + lamv .* NC3, ct, st, cp, sp, ilam);
  q = (i11 .* Q + i12 .* S + i13 .* MP) .* cutv;
  s = (i21 .* Q + i22 .* S + i23 .* MP) .* cutv;
  P = (i31 .* Q + i32 .* S + i33 .* MP) .* clean;
  % the constant modes, with the constraint
  p0 = dot(P, e0);
  q0 = dot(q, e0);
  Q0 = dot(Q, e0);
  p00 = ub * (ar00 * rp0 + arlam * ML) + (1 - ub) * p0;
  RP = P + (vol * (p00 - p0)) .* e0;
  qf = q + (vol * ((Q0 + 2 * p00 / R) / qdiv - q0)) .* e0;
  [U1, U2, U3] = qst2cart(qf, s, itor .* T .* cutv, ct, st, cp, sp);
  R1 = U1 .* clean;
  R2 = U2 .* clean;
  R3 = U3 .* clean;
end

% The surface's position on the grid, for drawing it.
function [x1, x2, x3] = position(X1, X2, X3)
  [x1, x2, x3] = synth(X1, X2, X3);
end

% Energies and integrals of the state, for comparing runs: the kinetic energy,
% the integral of H^2 (the bending energy is 2/Be times it), the area and the
% enclosed volume. X is the surface's position, with its geometry.
function [ekin, h2, area, volume] = diag(U1, U2, U3, X1, X2, X3, n1, n2, n3, H, sqrtg, wq)
  [u1, u2, u3, x1, x2, x3] = synth(U1, U2, U3, X1, X2, X3);
  ku = 0.5 * (u1 .* u1 + u2 .* u2 + u3 .* u3) .* sqrtg;
  ekin = dot(ku, wq);
  hh = H .* H .* sqrtg;
  h2 = dot(hh, wq);
  area = dot(sqrtg, wq);
  xn = -(x1 .* n1 + x2 .* n2 + x3 .* n3) .* sqrtg / 3;
  volume = dot(xn, wq);
end
