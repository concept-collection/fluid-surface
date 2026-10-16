% The radial, spheroidal and toroidal potentials Q, S, T of a vector field on
% the unit sphere, from the spectral coefficients U1, U2, U3 of its Cartesian
% components: Q is the radial component r . u, and S, T are those of the
% tangential part (lib/sphtor.m, SHTns' convention). This is the C++ solver's
% FluidCartesianToQst, which uses recurrences on the coefficients for the
% multiplications by x, y, z; here they are products on the grid.
%
% ct, st, cp, sp are cos and sin of theta and phi on the grid; ilam is
% 1/(l(l+1)), 0 at l = 0.

function [Q, S, T] = cart2qst(U1, U2, U3, ct, st, cp, sp, ilam)
  [u1, u2, u3] = synth(U1, U2, U3);
  uh = cp .* u1 + sp .* u2;
  [S, T] = sphtor(ct .* uh - st .* u3, cp .* u2 - sp .* u1, st, ilam);
  Q = analys(st .* uh + ct .* u3);
end
