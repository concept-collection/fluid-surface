% The surface divergence of a vector field f = (f1, f2, f3) given on the
% grid. Its tangential part is pulled back to the sphere, where its
% divergence is -l(l+1) S with S the spheroidal coefficient (lib/sphtor.m):
%
%   div_S f = jinv .* div_sphere(J P f) - 2 H (f . n),
%
% with J = 1/jinv the ratio of the surface's area element to the sphere's.
% The divergence is filtered to filtf (degrees below the band's top three)
% before its synthesis. (The C++ solver's KFTanDivW, the sphtor analysis and
% KFTanDivFinish.)

function d = tandiv(f1, f2, f3, n1, n2, n3, vt1, vt2, vt3, vp1, vp2, vp3, jinv, H, st, filtf)
  fn = f1 .* n1 + f2 .* n2 + f3 .* n3;
  p1 = f1 - fn .* n1;
  p2 = f2 - fn .* n2;
  p3 = f3 - fn .* n3;
  % J times the theta and sin(theta) phi components on the sphere, each over sin(theta)
  wt = (p1 .* vt1 + p2 .* vt2 + p3 .* vt3) ./ (jinv .* st);
  wp = (p1 .* vp1 + p2 .* vp2 + p3 .* vp3) ./ jinv;
  [A, B] = analys(wt, wp);
  D = (dphic(B) - dthetat(A)) .* filtf;
  d = jinv .* synth(D) - 2 * H .* fn;
end
