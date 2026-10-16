% The radial, spheroidal and toroidal potentials Q, S, T of a vector field on
% the unit sphere, from the spectral coefficients U1, U2, U3 of its Cartesian
% components: Q is the radial component r . u, and S, T are those of the
% tangential part (lib/sphtor.m, SHTns' convention). This is the C++ solver's
% FluidCartesianToQst, which uses recurrences on the coefficients for the
% multiplications by x, y, z; here they are products on the grid.

function [Q, S, T] = cart2qst(U1, U2, U3, grid, msk)
  [u1, u2, u3] = synth(U1, U2, U3);
  uh = grid.cp .* u1 + grid.sp .* u2;
  [S, T] = sphtor(grid.ct .* uh - grid.st .* u3, grid.cp .* u2 - grid.sp .* u1, grid.st, msk.ilam);
  Q = analys(grid.st .* uh + grid.ct .* u3);
end
