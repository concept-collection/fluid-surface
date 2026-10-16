% The surface divergence of a vector field f = (f1, f2, f3) given on the
% grid. Its tangential part is pulled back to the sphere, where its
% divergence is -l(l+1) S with S the spheroidal coefficient (lib/sphtor.m):
%
%   div_S f = jinv .* div_sphere(J P f) - 2 H (f . n),
%
% with J = 1/jinv the ratio of the surface's area element to the sphere's.
% The divergence is filtered to msk.filtf (degrees below the band's top
% three) before its synthesis. (The C++ solver's KFTanDivW, the sphtor
% analysis and KFTanDivFinish.)

function d = tandiv(f1, f2, f3, geo, grid, msk)
  fn = f1 .* geo.n1 + f2 .* geo.n2 + f3 .* geo.n3;
  p1 = f1 - fn .* geo.n1;
  p2 = f2 - fn .* geo.n2;
  p3 = f3 - fn .* geo.n3;
  % J times the theta and sin(theta) phi components on the sphere, each over sin(theta)
  wt = (p1 .* geo.vt1 + p2 .* geo.vt2 + p3 .* geo.vt3) ./ (geo.jinv .* grid.st);
  wp = (p1 .* geo.vp1 + p2 .* geo.vp2 + p3 .* geo.vp3) ./ geo.jinv;
  [A, B] = analys(wt, wp);
  D = (dphic(B) - dthetat(A)) .* msk.filtf;
  d = geo.jinv .* synth(D) - 2 * geo.H .* fn;
end
