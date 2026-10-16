% The surface divergence of the rate of deformation, row by row: from the
% rows g1, g2, g3 (npts x 3 each) of the velocity's surface gradient
% G(r, c) = d u_r / d x_c (lib/surfgrad.m), the divergences d1, d2, d3 of the
% rows of the symmetric part of P G P, with P = I - n n' the projection onto
% the tangent plane.

function [d1, d2, d3] = deformdiv(g1, g2, g3, geo, grid, msk)
  n = [geo.n1, geo.n2, geo.n3];
  nG = geo.n1 .* g1 + geo.n2 .* g2 + geo.n3 .* g3;
  Gn = [sum(g1 .* n, 2), sum(g2 .* n, 2), sum(g3 .* n, 2)];
  nGn = sum(Gn .* n, 2);
  % rows of P G P
  k1 = g1 - geo.n1 .* nG - (Gn(:, 1) - geo.n1 .* nGn) .* n;
  k2 = g2 - geo.n2 .* nG - (Gn(:, 2) - geo.n2 .* nGn) .* n;
  k3 = g3 - geo.n3 .* nG - (Gn(:, 3) - geo.n3 .* nGn) .* n;
  % rows of its symmetric part, and their divergence
  d1 = tandiv(0.5 * (k1 + [k1(:, 1), k2(:, 1), k3(:, 1)]), geo, grid, msk);
  d2 = tandiv(0.5 * (k2 + [k1(:, 2), k2(:, 2), k3(:, 2)]), geo, grid, msk);
  d3 = tandiv(0.5 * (k3 + [k1(:, 3), k2(:, 3), k3(:, 3)]), geo, grid, msk);
end
