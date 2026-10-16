% The surface Laplace-Beltrami operator of a field with coefficients U, on
% the grid, in the C++ solver's flux form (FluidLaplacianGrid): the round
% sphere's share -l(l+1) U in coefficient space, plus the divergence of the
% deviation fluxes (dp1 a + p2 b, p2 a + dq2 b) / sin(theta), with
% a = sin(theta) u_theta and b = u_phi, through the vector analysis of
% lib/sphtor.m; then jinv. U and the divergence are filtered to msk.filtf.
%
% Returns lap and what the bending force's derivative reuses: the result
% before jinv (raw) and the grid fields a, b.

function [lap, raw, a, b] = fluxlap(U, geo, grid, msk)
  Uf = U .* msk.filtf;
  [a, b] = synth(dthetac(Uf), dphic(Uf));
  s2 = grid.st .* grid.st;
  [A, B] = analys((geo.dp1 .* a + geo.p2 .* b) ./ s2, (geo.p2 .* a + geo.dq2 .* b) ./ s2);
  raw = synth(-msk.lam .* Uf + (dphic(B) - dthetat(A)) .* msk.filtf);
  lap = geo.jinv .* raw;
end
