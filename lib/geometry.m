% The geometry of a surface on the grid, from the spherical-harmonic
% coefficients X1, X2, X3 of its position: the dual tangent vectors
% v^theta = g^{theta j} x_j and v^phi = g^{phi j} x_j, the unit normal n
% (inward on a sphere), the mean and Gaussian curvatures H and K, the area
% element sqrt(det g), and the weights of the flux-form Laplace-Beltrami
% operator (dp1, p2, dq2, jinv; docs/reduced-transforms.md in turing-surface).
% This is the C++ solver's FluidGeometryBuild and KFGeometry.
%
% The coefficients must be zero-padded to a band at least two above their
% content, so that the theta derivatives (lib/posderivs.m) stay in band.
%
% grid.ct and grid.st are cos(theta) and sin(theta) on the grid.
%
% The determinants are nonnegative (Cauchy-Schwarz), but numbl cannot prove
% that, and sqrt of a value of unknown sign is complex in MATLAB, so the roots
% are realsqrt.

function [vt1, vt2, vt3, vp1, vp2, vp3, n1, n2, n3, H, K, sqrtg, dp1, p2, dq2, jinv] = geometry(X1, X2, X3, grid)
  [sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3] = posderivs(X1, X2, X3, grid);
  st = grid.st;

  xt1 = sxt1 ./ st;
  xt2 = sxt2 ./ st;
  xt3 = sxt3 ./ st;
  xtt1 = sxtt1 ./ st.^2;
  xtt2 = sxtt2 ./ st.^2;
  xtt3 = sxtt3 ./ st.^2;
  xtp1 = sxtp1 ./ st;
  xtp2 = sxtp2 ./ st;
  xtp3 = sxtp3 ./ st;

  % the metric and its determinant
  gtt = xt1.^2 + xt2.^2 + xt3.^2;
  gpp = xp1.^2 + xp2.^2 + xp3.^2;
  gtp = xt1 .* xp1 + xt2 .* xp2 + xt3 .* xp3;
  detg = gtt .* gpp - gtp.^2;
  sqrtg = realsqrt(detg);

  % dual tangent vectors
  vt1 = (gpp .* xt1 - gtp .* xp1) ./ detg;
  vt2 = (gpp .* xt2 - gtp .* xp2) ./ detg;
  vt3 = (gpp .* xt3 - gtp .* xp3) ./ detg;
  vp1 = (gtt .* xp1 - gtp .* xt1) ./ detg;
  vp2 = (gtt .* xp2 - gtp .* xt2) ./ detg;
  vp3 = (gtt .* xp3 - gtp .* xt3) ./ detg;

  % unit normal, x_phi cross x_theta
  n1 = (xp2 .* xt3 - xp3 .* xt2) ./ sqrtg;
  n2 = (xp3 .* xt1 - xp1 .* xt3) ./ sqrtg;
  n3 = (xp1 .* xt2 - xp2 .* xt1) ./ sqrtg;

  % second fundamental form and the curvatures
  bpp = xpp1 .* n1 + xpp2 .* n2 + xpp3 .* n3;
  btp = xtp1 .* n1 + xtp2 .* n2 + xtp3 .* n3;
  btt = xtt1 .* n1 + xtt2 .* n2 + xtt3 .* n3;
  H = (gpp .* btt - 2 * gtp .* btp + gtt .* bpp) ./ (2 * detg);
  K = (bpp .* btt - btp.^2) ./ detg;

  % flux-form Laplacian weights, from the sin(theta)-scaled tangent
  ptt = sxt1.^2 + sxt2.^2 + sxt3.^2;
  ptp = sxt1 .* xp1 + sxt2 .* xp2 + sxt3 .* xp3;
  id = 1 ./ realsqrt(ptt .* gpp - ptp.^2);
  dp1 = gpp .* id - 1;
  p2 = -ptp .* id;
  dq2 = ptt .* id - 1;
  jinv = id .* st.^2;
end
