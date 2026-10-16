% One BDF-1 step of the fluid solver on a moving surface, without the volume
% constraint: surface Navier-Stokes with the Helfrich bending force, the
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
% setup, reads the area to build the blocks, runs bendlin and rhs, GMRES on
% matvec, then advance.

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

function [B1, B2, B3, BP] = rhs(U1, U2, U3, BF1, BF2, BF3, ct, st, cp, sp, ilam, cutv, clean, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor, mass)
  % the history term u / dt and the bending force, and no source for the divergence
  [B1, B2, B3, BP] = precond(mass * U1 + BF1, mass * U2 + BF2, mass * U3 + BF3, 0 * U1, ct, st, cp, sp, ilam, cutv, clean, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor);
end

function [R1, R2, R3, RP] = matvec(Z1, Z2, Z3, Z4, v1, v2, v3, vt1, vt2, vt3, vp1, vp2, vp3, n1, n2, n3, H, K, dp1, p2, dq2, jinv, sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3, ba, bb, braw, bvc, ct, st, cp, sp, lam, filtf, real0, keep, ilam, cutv, clean, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor, mass, visc, disp, bscale)
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
  m1 = mass * u1 + G11 .* v1 + G12 .* v2 + G13 .* v3 + q1 + 2 * H .* p .* n1 - visc * d1;
  m2 = mass * u2 + G21 .* v1 + G22 .* v2 + G23 .* v3 + q2 + 2 * H .* p .* n2 - visc * d2;
  m3 = mass * u3 + G31 .* v1 + G32 .* v2 + G33 .* v3 + q3 + 2 * H .* p .* n3 - visc * d3;
  % the normal velocity, whose displacement over the step bends the surface
  un = u1 .* n1 + u2 .* n2 + u3 .* n3;
  [M1, M2, M3, MD, N1, N2, N3] = analys(m1, m2, m3, du, un .* n1, un .* n2, un .* n3);
  [J1, J2, J3] = bendjvp(disp * N1 .* keep, disp * N2 .* keep, disp * N3 .* keep, sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3, ct, st, H, K, n1, n2, n3, dp1, p2, dq2, jinv, ba, bb, braw, bvc, lam, filtf, real0, keep, bscale);
  [R1, R2, R3, RP] = precond(M1 .* keep - J1, M2 .* keep - J2, M3 .* keep - J3, MD .* keep, ct, st, cp, sp, ilam, cutv, clean, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor);
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
% toroidal potentials of the velocity and for the pressure.
function [R1, R2, R3, RP] = precond(M1, M2, M3, MP, ct, st, cp, sp, ilam, cutv, clean, i11, i12, i13, i21, i22, i23, i31, i32, i33, itor)
  [Q, S, T] = cart2qst(M1, M2, M3, ct, st, cp, sp, ilam);
  q = (i11 .* Q + i12 .* S + i13 .* MP) .* cutv;
  s = (i21 .* Q + i22 .* S + i23 .* MP) .* cutv;
  RP = (i31 .* Q + i32 .* S + i33 .* MP) .* clean;
  [U1, U2, U3] = qst2cart(q, s, itor .* T .* cutv, ct, st, cp, sp);
  R1 = U1 .* clean;
  R2 = U2 .* clean;
  R3 = U3 .* clean;
end
