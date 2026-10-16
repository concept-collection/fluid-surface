% The surface divergence of a vector field f (npts x 3) given on the grid.
% Its tangential part p is pulled back to the sphere, where its divergence is
% -l(l+1) S with S the spheroidal coefficient (lib/sphtor.m):
%
%   div_S f = jinv .* div_sphere(J p) - 2 H (f . n),
%
% with J = 1/jinv the ratio of the surface's area element to the sphere's.
% The divergence is filtered to msk.filtf (degrees below the band's top
% three) before its synthesis. (The C++ solver's KFTanDivW, the sphtor
% analysis and KFTanDivFinish.)

function d = tandiv(f, geo, grid, msk)
  n = [geo.n1, geo.n2, geo.n3];
  fn = sum(f .* n, 2);
  p = f - fn .* n;
  % J times the theta and sin(theta) phi components on the sphere, each over sin(theta)
  wt = sum(p .* [geo.vt1, geo.vt2, geo.vt3], 2) ./ (geo.jinv .* grid.st);
  wp = sum(p .* [geo.vp1, geo.vp2, geo.vp3], 2) ./ geo.jinv;
  [A, B] = analys(wt, wp);
  D = (dphic(B) - dthetat(A)) .* msk.filtf;
  d = geo.jinv .* synth(D) - 2 * geo.H .* fn;
end
