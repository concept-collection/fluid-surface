% The first and second derivatives of a position (or of a displacement) on
% the grid, from the coefficients X1, X2, X3 of its Cartesian components, in
% the sin(theta)-scaled forms that stay bounded at the poles:
%
%   sxt = sin(theta) x_theta,   xp = x_phi,
%   sxtt = sin(theta)^2 x_thetatheta = (sin(theta) dtheta)^2 x - cos(theta) sxt,
%   sxtp = sin(theta) x_thetaphi,   xpp = x_phiphi.
%
% Every derivative is a coefficient shuffle and a synthesis, so the
% coefficients must stop two degrees below the band. The C++ solver's inputs
% to KFGeometry (FluidGeometryBuild); ct is cos(theta) on the grid.

function [sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3] = posderivs(X1, X2, X3, ct)
  T1 = dthetac(X1);
  T2 = dthetac(X2);
  T3 = dthetac(X3);
  P1 = dphic(X1);
  P2 = dphic(X2);
  P3 = dphic(X3);
  [sxt1, sxt2, sxt3, xp1, xp2, xp3] = synth(T1, T2, T3, P1, P2, P3);
  [tt1, tt2, tt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3] = synth(dthetac(T1), dthetac(T2), dthetac(T3), dphic(T1), dphic(T2), dphic(T3), dphic(P1), dphic(P2), dphic(P3));
  sxtt1 = tt1 - ct .* sxt1;
  sxtt2 = tt2 - ct .* sxt2;
  sxtt3 = tt3 - ct .* sxt3;
end
