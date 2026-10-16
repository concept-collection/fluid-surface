% One BDF-1 step of the fluid solver's static mode: surface Navier-Stokes on
% a fixed surface, the momentum linearized about the current velocity, and the
% normal velocity held at zero. This is the C++ solver's FluidStep with
% is_static set (core/fluid_op.cpp, core/fluid_precond.cpp).
%
% Spectral fields are on the product band, zero above the working band. The
% arguments are grouped in structs of arrays, as in models/stokes.m: geo (the
% geometry), grid (cos and sin on the grid, quadrature weights) and msk (the
% masks: keep is 1 for degrees up to the working band and 0 for the imaginary
% part of m = 0, clean also drops the band's top three degrees, filtf the
% product band's top three, e0 picks the constant mode, lam holds l(l+1)).
%
% The host runs setup and rhs once per step, then GMRES on the preconditioned
% operator matvec, whose unknowns are the velocity Z1, Z2, Z3 and the pressure
% Z4.

function [v1, v2, v3, pscale, vdiv] = setup(U1, U2, U3, geo, grid, msk, mass, disp, re, be)
  % the transport velocity: the tangential part of the current velocity
  [u1, u2, u3] = synth(U1, U2, U3);
  un = u1 .* geo.n1 + u2 .* geo.n2 + u3 .* geo.n3;
  v1 = u1 - un .* geo.n1;
  v2 = u2 - un .* geo.n2;
  v3 = u3 - un .* geo.n3;
  % the preconditioner's per-degree symbols, on the sphere of the same mean curvature
  h2 = geo.H .* geo.H .* geo.sqrtg;
  num = dot(h2, grid.wq);
  area = dot(geo.sqrtg, grid.wq);
  meanH2 = num / area;
  lam = msk.lam;
  e0 = msk.e0;
  vdiv = mass + lam / re + (disp / be) * max(lam .* (lam - 2), 0);
  pscale = e0 + (1 - e0) .* (-vdiv ./ (lam + 4 * meanH2));
end

function [B1, B2, B3, BP] = rhs(U1, U2, U3, geo, grid, msk, pscale, vdiv, mass)
  % the history term u / dt, projected onto the tangent plane
  [h1, h2, h3] = synth(mass * U1, mass * U2, mass * U3);
  hn = h1 .* geo.n1 + h2 .* geo.n2 + h3 .* geo.n3;
  [F1, F2, F3] = analys(h1 - hn .* geo.n1, h2 - hn .* geo.n2, h3 - hn .* geo.n3);
  keep = msk.keep;
  [B1, B2, B3, BP] = precond(F1 .* keep, F2 .* keep, F3 .* keep, 0 * U1, geo, grid, msk, pscale, vdiv);
end

function [R1, R2, R3, RP] = matvec(Z1, Z2, Z3, Z4, v1, v2, v3, geo, grid, msk, pscale, vdiv, mass, visc)
  n1 = geo.n1;
  n2 = geo.n2;
  n3 = geo.n3;
  H = geo.H;
  keep = msk.keep;
  e0 = msk.e0;
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

  % momentum: inertia, convection by the transport velocity, pressure, viscosity
  m1 = mass * u1 + G11 .* v1 + G12 .* v2 + G13 .* v3 + q1 + 2 * H .* p .* n1 - visc * d1;
  m2 = mass * u2 + G21 .* v1 + G22 .* v2 + G23 .* v3 + q2 + 2 * H .* p .* n2 - visc * d2;
  m3 = mass * u3 + G31 .* v1 + G32 .* v2 + G33 .* v3 + q3 + 2 * H .* p .* n3 - visc * d3;
  % static mode: the tangential momentum, and the normal velocity itself
  mn = m1 .* n1 + m2 .* n2 + m3 .* n3;
  un = u1 .* n1 + u2 .* n2 + u3 .* n3;
  [M1, M2, M3, MD] = analys(m1 - (mn - un) .* n1, m2 - (mn - un) .* n2, m3 - (mn - un) .* n3, du);
  % the divergence row's constant mode pins the mean pressure instead
  MP = MD .* keep .* (1 - e0) + Z4 .* e0;
  [R1, R2, R3, RP] = precond(M1 .* keep, M2 .* keep, M3 .* keep, MP, geo, grid, msk, pscale, vdiv);
end

% The static preconditioner: the pressure from its per-degree symbol, then the
% velocity from the momentum rows less the pressure's gradient.
function [R1, R2, R3, RP] = precond(M1, M2, M3, MP, geo, grid, msk, pscale, vdiv)
  keep = msk.keep;
  clean = msk.clean;
  RP = pscale .* MP .* clean;
  [g1, g2, g3] = surfgrad(RP, geo, grid);
  [F1, F2, F3] = analys(g1, g2, g3);
  R1 = (M1 - F1 .* keep) ./ vdiv .* clean;
  R2 = (M2 - F2 .* keep) ./ vdiv .* clean;
  R3 = (M3 - F3 .* keep) ./ vdiv .* clean;
end
