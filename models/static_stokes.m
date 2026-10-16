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
  n = [geo.n1, geo.n2, geo.n3];
  v = [v1, v2, v3];
  keep = msk.keep;
  e0 = msk.e0;
  [u1, u2, u3, p] = synth(Z1, Z2, Z3, Z4);
  u = [u1, u2, u3];
  % rows of the velocity gradient G(r, c) = d u_r / d x_c, and the pressure gradient
  g1 = surfgrad(Z1, geo, grid);
  g2 = surfgrad(Z2, geo, grid);
  g3 = surfgrad(Z3, geo, grid);
  q = surfgrad(Z4, geo, grid);
  % divergence of the rate of deformation, row by row, and of the velocity
  [d1, d2, d3] = deformdiv(g1, g2, g3, geo, grid, msk);
  du = tandiv(u, geo, grid, msk);

  % momentum: inertia, convection by the transport velocity, pressure, viscosity
  m = mass * u + [sum(g1 .* v, 2), sum(g2 .* v, 2), sum(g3 .* v, 2)] + q + 2 * geo.H .* p .* n - visc * [d1, d2, d3];
  % static mode: the tangential momentum, and the normal velocity itself
  t = m - sum((m - u) .* n, 2) .* n;
  [M1, M2, M3, MD] = analys(t(:, 1), t(:, 2), t(:, 3), du);
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
  g = surfgrad(RP, geo, grid);
  [F1, F2, F3] = analys(g(:, 1), g(:, 2), g(:, 3));
  R1 = (M1 - F1 .* keep) ./ vdiv .* clean;
  R2 = (M2 - F2 .* keep) ./ vdiv .* clean;
  R3 = (M3 - F3 .* keep) ./ vdiv .* clean;
end
