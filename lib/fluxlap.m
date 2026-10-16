% The surface Laplace-Beltrami operator of a field with coefficients U, on
% the grid, in the C++ solver's flux form (FluidLaplacianGrid): the round
% sphere's share -l(l+1) U in coefficient space, plus the divergence of the
% deviation fluxes (dp1 a + p2 b, p2 a + dq2 b) / sin(theta), with
% a = sin(theta) u_theta and b = u_phi, through the vector analysis of
% lib/sphtor.m; then jinv. U and the divergence are filtered to filtf.
%
% Returns lap and what the bending force's derivative reuses: the result
% before jinv (raw) and the grid fields a, b.

function [lap, raw, a, b] = fluxlap(U, dp1, p2, dq2, jinv, st, lam, filtf)
  Uf = U .* filtf;
  [a, b] = synth(dthetac(Uf), dphic(Uf));
  s2 = st .* st;
  [A, B] = analys((dp1 .* a + p2 .* b) ./ s2, (p2 .* a + dq2 .* b) ./ s2);
  raw = synth(-lam .* Uf + (dphic(B) - dthetat(A)) .* filtf);
  lap = jinv .* raw;
end
