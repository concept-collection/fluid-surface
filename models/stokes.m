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
% derivative along it (lib/bendjvp.m). bscale is -2/Be, and 0 turns bending
% off.
%
% The volume constraint adds the multiplier, a 1 x 1 unknown Z5, and a row:
% the flux of the normal velocity through the surface. vol = 1 turns it on;
% with vol = 0 the row is the identity, the multiplier's right-hand side is
% zero, and it stays zero. A plan has no branches, so the switches are
% parameters.
%
% The arguments are grouped in structs of arrays:
%   geo   the surface's geometry (lib/geometry.m)
%   der   the position's derivative fields (lib/posderivs.m)
%   bl    what bendlin stores for the bending force's derivative
%   grid  cos and sin of theta (ct, st) and phi (cp, sp) on the grid, and
%         the quadrature weights wq
%   msk   per-coefficient masks and symbols: keep (the working band, m = 0
%         real), clean (also without its top three degrees), cutv (top two),
%         filtf (the product band's top three), e0 (the constant mode), real0
%         (m = 0 real), lam = l(l+1) and ilam = 1/(l(l+1))
%   tab   the sphere's per-degree Stokes blocks, inverted on the host as the
%         C++ builds them there: i11 ... i33, the velocity-only w11 ... w22,
%         and itor for the toroidal part
% Spectral fields are on the product band, zero above the working band.
%
% Per step the host runs lib/geometry.m and lib/posderivs.m on the surface,
% setup, reads the area to build the blocks, runs bendlin, volblock (with the
% constraint; the host reads its 2 x 2 block) and rhs, GMRES on matvec, then
% advance.

function [v1, v2, v3, area] = setup(U1, U2, U3, geo, grid)
  % the transport velocity: the tangential part of the current velocity
  [u1, u2, u3] = synth(U1, U2, U3);
  un = u1 .* geo.n1 + u2 .* geo.n2 + u3 .* geo.n3;
  v1 = u1 - un .* geo.n1;
  v2 = u2 - un .* geo.n2;
  v3 = u3 - un .* geo.n3;
  area = dot(geo.sqrtg, grid.wq);
end

% The bending force at the current surface, -2/Be (Laplacian of H +
% 2 H (H^2 - K)) n, and what its derivative reuses.
function [BF1, BF2, BF3, ba, bb, braw, bvc] = bendlin(geo, grid, msk, bscale)
  [L, braw, ba, bb] = fluxlap(analys(geo.H), geo, grid, msk);
  % the Laplacian of H, re-analysed with m = 0 made real, as the C++ does
  lapH = synth(analys(L) .* msk.real0);
  H = geo.H;
  bvc = lapH + 2 * H .* (H .* H - geo.K);
  [F1, F2, F3] = analys(bscale * bvc .* geo.n1, bscale * bvc .* geo.n2, bscale * bvc .* geo.n3);
  BF1 = F1 .* msk.keep;
  BF2 = F2 .* msk.keep;
  BF3 = F3 .* msk.keep;
end

function [B1, B2, B3, BP, BL] = rhs(U1, U2, U3, BF1, BF2, BF3, NC1, NC2, NC3, grid, msk, tab, mass, vol, cr00, crlam, ar00, arlam, ub, R, qdiv)
  % the history term u / dt and the bending force, and no source for the
  % divergence or the volume
  [B1, B2, B3, BP, BL] = precond(mass * U1 + BF1, mass * U2 + BF2, mass * U3 + BF3, 0 * U1, 0, NC1, NC2, NC3, grid, msk, tab, vol, cr00, crlam, ar00, arlam, ub, R, qdiv);
end

% The normal's coefficients, and the 2 x 2 block of the volume constraint:
% the sphere velocity's response to the force of a constant pressure and to
% that of the multiplier, measured by the divergence's constant mode and by
% the flux through the surface (FluidPrecondSetup). y00 is Y_0^0.
function [NC1, NC2, NC3, m00, m01, m10, m11] = volblock(geo, grid, msk, tab, y00)
  [N1, N2, N3] = analys(geo.n1, geo.n2, geo.n3);
  NC1 = N1 .* msk.keep;
  NC2 = N2 .* msk.keep;
  NC3 = N3 .* msk.keep;
  [P1, P2, P3] = analys(2 * y00 * geo.H .* geo.n1, 2 * y00 * geo.H .* geo.n2, 2 * y00 * geo.H .* geo.n3);
  [a1, a2, a3] = velinv(P1 .* msk.keep, P2 .* msk.keep, P3 .* msk.keep, grid, msk, tab);
  [b1, b2, b3] = velinv(-NC1, -NC2, -NC3, grid, msk, tab);
  [m00, m10] = response(a1, a2, a3, geo, grid, msk);
  [m01, m11] = response(b1, b2, b3, geo, grid, msk);
end

% The sphere's velocity system, without the pressure, solved degree by degree.
function [V1, V2, V3] = velinv(F1, F2, F3, grid, msk, tab)
  [Q, S, T] = cart2qst(F1, F2, F3, grid, msk);
  [U1, U2, U3] = qst2cart((tab.w11 .* Q + tab.w12 .* S) .* msk.cutv, (tab.w21 .* Q + tab.w22 .* S) .* msk.cutv, tab.itor .* T .* msk.cutv, grid);
  V1 = U1 .* msk.clean;
  V2 = U2 .* msk.clean;
  V3 = U3 .* msk.clean;
end

% The constant mode of a velocity's surface divergence, and its flux.
function [d00, flux] = response(V1, V2, V3, geo, grid, msk)
  [u1, u2, u3] = synth(V1, V2, V3);
  D = analys(tandiv(u1, u2, u3, geo, grid, msk));
  d00 = dot(D, msk.e0);
  un = (u1 .* geo.n1 + u2 .* geo.n2 + u3 .* geo.n3) .* geo.sqrtg;
  flux = dot(un, grid.wq);
end

function [R1, R2, R3, RP, RL] = matvec(Z1, Z2, Z3, Z4, Z5, v1, v2, v3, NC1, NC2, NC3, geo, der, bl, grid, msk, tab, mass, visc, disp, bscale, vol, cr00, crlam, ar00, arlam, ub, R, qdiv)
  n1 = geo.n1;
  n2 = geo.n2;
  n3 = geo.n3;
  H = geo.H;
  [u1, u2, u3, p] = synth(Z1, Z2, Z3, Z4);
  % velocity gradient G(r, c) = d u_r / d x_c, and the pressure gradient
  [G11, G12, G13] = surfgrad(Z1, geo, grid);
  [G21, G22, G23] = surfgrad(Z2, geo, grid);
  [G31, G32, G33] = surfgrad(Z3, geo, grid);
  [q1, q2, q3] = surfgrad(Z4, geo, grid);

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
  d1 = tandiv(s11, s12, s13, geo, grid, msk);
  d2 = tandiv(s12, s22, s23, geo, grid, msk);
  d3 = tandiv(s13, s23, s33, geo, grid, msk);
  du = tandiv(u1, u2, u3, geo, grid, msk);

  % momentum: inertia, convection by the transport velocity, pressure,
  % viscosity, and the multiplier's normal force
  lamv = vol * Z5;
  m1 = mass * u1 + G11 .* v1 + G12 .* v2 + G13 .* v3 + q1 + 2 * H .* p .* n1 - visc * d1 - lamv .* n1;
  m2 = mass * u2 + G21 .* v1 + G22 .* v2 + G23 .* v3 + q2 + 2 * H .* p .* n2 - visc * d2 - lamv .* n2;
  m3 = mass * u3 + G31 .* v1 + G32 .* v2 + G33 .* v3 + q3 + 2 * H .* p .* n3 - visc * d3 - lamv .* n3;
  % the normal velocity, whose displacement over the step bends the surface
  un = u1 .* n1 + u2 .* n2 + u3 .* n3;
  [M1, M2, M3, MD, N1, N2, N3] = analys(m1, m2, m3, du, un .* n1, un .* n2, un .* n3);
  % the volume row: the flux of the normal velocity (the identity without the constraint)
  unw = un .* geo.sqrtg;
  flux = dot(unw, grid.wq);
  VR = vol * flux + (1 - vol) * Z5;
  keep = msk.keep;
  [J1, J2, J3] = bendjvp(disp * N1 .* keep, disp * N2 .* keep, disp * N3 .* keep, der, geo, bl, grid, msk, bscale);
  [R1, R2, R3, RP, RL] = precond(M1 .* keep - J1, M2 .* keep - J2, M3 .* keep - J3, MD .* keep, VR, NC1, NC2, NC3, grid, msk, tab, vol, cr00, crlam, ar00, arlam, ub, R, qdiv);
end

function [Y1, Y2, Y3] = advance(U1, U2, U3, X1, X2, X3, geo, msk, dt)
  % the surface moves with the normal part of the new velocity, filtered
  [u1, u2, u3] = synth(U1, U2, U3);
  un = u1 .* geo.n1 + u2 .* geo.n2 + u3 .* geo.n3;
  [W1, W2, W3] = analys(un .* geo.n1, un .* geo.n2, un .* geo.n3);
  Y1 = X1 + dt * W1 .* msk.clean;
  Y2 = X2 + dt * W2 .* msk.clean;
  Y3 = X3 + dt * W3 .* msk.clean;
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
function [R1, R2, R3, RP, RL] = precond(M1, M2, M3, MP, ML, NC1, NC2, NC3, grid, msk, tab, vol, cr00, crlam, ar00, arlam, ub, R, qdiv)
  rp0 = dot(MP, msk.e0);
  RL = cr00 * rp0 + crlam * ML;
  lamv = vol * RL;
  [Q, S, T] = cart2qst(M1 + lamv .* NC1, M2 + lamv .* NC2, M3 + lamv .* NC3, grid, msk);
  q = (tab.i11 .* Q + tab.i12 .* S + tab.i13 .* MP) .* msk.cutv;
  s = (tab.i21 .* Q + tab.i22 .* S + tab.i23 .* MP) .* msk.cutv;
  P = (tab.i31 .* Q + tab.i32 .* S + tab.i33 .* MP) .* msk.clean;
  % the constant modes, with the constraint
  e0 = msk.e0;
  p0 = dot(P, e0);
  q0 = dot(q, e0);
  Q0 = dot(Q, e0);
  p00 = ub * (ar00 * rp0 + arlam * ML) + (1 - ub) * p0;
  RP = P + (vol * (p00 - p0)) .* e0;
  qf = q + (vol * ((Q0 + 2 * p00 / R) / qdiv - q0)) .* e0;
  [U1, U2, U3] = qst2cart(qf, s, tab.itor .* T .* msk.cutv, grid);
  R1 = U1 .* msk.clean;
  R2 = U2 .* msk.clean;
  R3 = U3 .* msk.clean;
end

% The surface's position on the grid, for drawing it.
function [x1, x2, x3] = position(X1, X2, X3)
  [x1, x2, x3] = synth(X1, X2, X3);
end

% Energies and integrals of the state, for comparing runs: the kinetic energy,
% the integral of H^2 (the bending energy is 2/Be times it), the area and the
% enclosed volume. X is the surface's position, with its geometry.
function [ekin, h2, area, volume] = diag(U1, U2, U3, X1, X2, X3, geo, grid)
  [u1, u2, u3, x1, x2, x3] = synth(U1, U2, U3, X1, X2, X3);
  ku = 0.5 * (u1 .* u1 + u2 .* u2 + u3 .* u3) .* geo.sqrtg;
  ekin = dot(ku, grid.wq);
  hh = geo.H .* geo.H .* geo.sqrtg;
  h2 = dot(hh, grid.wq);
  area = dot(geo.sqrtg, grid.wq);
  xn = -(x1 .* geo.n1 + x2 .* geo.n2 + x3 .* geo.n3) .* geo.sqrtg / 3;
  volume = dot(xn, grid.wq);
end
