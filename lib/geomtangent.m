% The derivative of the surface geometry when the position x moves by a
% displacement y: of the mean and Gaussian curvatures, the unit normal, and
% the flux-form Laplacian weights (lib/geometry.m). The inputs are the
% derivative fields of x (the struct der, lib/posderivs.m) and of y (the
% d-prefixed arrays); grid.st is sin(theta). The C++ solver's
% KFGeometryTangent.

function [dH, dK, dn1, dn2, dn3, ddp1, dp2, ddq2, djinv] = geomtangent(der, dsxt1, dsxt2, dsxt3, dxp1, dxp2, dxp3, dsxtt1, dsxtt2, dsxtt3, dsxtp1, dsxtp2, dsxtp3, dxpp1, dxpp2, dxpp3, grid)
  sxt1 = der.sxt1; sxt2 = der.sxt2; sxt3 = der.sxt3;
  xp1 = der.xp1; xp2 = der.xp2; xp3 = der.xp3;
  sxtt1 = der.sxtt1; sxtt2 = der.sxtt2; sxtt3 = der.sxtt3;
  sxtp1 = der.sxtp1; sxtp2 = der.sxtp2; sxtp3 = der.sxtp3;
  xpp1 = der.xpp1; xpp2 = der.xpp2; xpp3 = der.xpp3;
  st = grid.st;
  s2 = st .* st;
  xt1 = sxt1 ./ st;
  xt2 = sxt2 ./ st;
  xt3 = sxt3 ./ st;
  dxt1 = dsxt1 ./ st;
  dxt2 = dsxt2 ./ st;
  dxt3 = dsxt3 ./ st;

  % the metric and its derivative
  gtt = xt1 .* xt1 + xt2 .* xt2 + xt3 .* xt3;
  gpp = xp1 .* xp1 + xp2 .* xp2 + xp3 .* xp3;
  gtp = xt1 .* xp1 + xt2 .* xp2 + xt3 .* xp3;
  detg = gtt .* gpp - gtp .* gtp;
  dgtt = 2 * (xt1 .* dxt1 + xt2 .* dxt2 + xt3 .* dxt3);
  dgpp = 2 * (xp1 .* dxp1 + xp2 .* dxp2 + xp3 .* dxp3);
  dgtp = dxt1 .* xp1 + dxt2 .* xp2 + dxt3 .* xp3 + xt1 .* dxp1 + xt2 .* dxp2 + xt3 .* dxp3;
  ddet = dgtt .* gpp + gtt .* dgpp - 2 * gtp .* dgtp;
  sq = realsqrt(detg);
  dsq = ddet ./ (2 * sq);

  % the normal, x_phi cross x_theta over its length, and its derivative
  n1 = (xp2 .* xt3 - xp3 .* xt2) ./ sq;
  n2 = (xp3 .* xt1 - xp1 .* xt3) ./ sq;
  n3 = (xp1 .* xt2 - xp2 .* xt1) ./ sq;
  dn1 = (dxp2 .* xt3 - dxp3 .* xt2 + xp2 .* dxt3 - xp3 .* dxt2 - n1 .* dsq) ./ sq;
  dn2 = (dxp3 .* xt1 - dxp1 .* xt3 + xp3 .* dxt1 - xp1 .* dxt3 - n2 .* dsq) ./ sq;
  dn3 = (dxp1 .* xt2 - dxp2 .* xt1 + xp1 .* dxt2 - xp2 .* dxt1 - n3 .* dsq) ./ sq;

  % the second fundamental form and its derivative
  bpp = xpp1 .* n1 + xpp2 .* n2 + xpp3 .* n3;
  btp = (sxtp1 .* n1 + sxtp2 .* n2 + sxtp3 .* n3) ./ st;
  btt = (sxtt1 .* n1 + sxtt2 .* n2 + sxtt3 .* n3) ./ s2;
  dbpp = dxpp1 .* n1 + dxpp2 .* n2 + dxpp3 .* n3 + xpp1 .* dn1 + xpp2 .* dn2 + xpp3 .* dn3;
  dbtp = (dsxtp1 .* n1 + dsxtp2 .* n2 + dsxtp3 .* n3 + sxtp1 .* dn1 + sxtp2 .* dn2 + sxtp3 .* dn3) ./ st;
  dbtt = (dsxtt1 .* n1 + dsxtt2 .* n2 + dsxtt3 .* n3 + sxtt1 .* dn1 + sxtt2 .* dn2 + sxtt3 .* dn3) ./ s2;

  % the curvatures' derivatives
  num = gpp .* btt - 2 * gtp .* btp + gtt .* bpp;
  dnum = dgpp .* btt + gpp .* dbtt - 2 * (dgtp .* btp + gtp .* dbtp) + dgtt .* bpp + gtt .* dbpp;
  H = num ./ (2 * detg);
  dH = (dnum - 2 * H .* ddet) ./ (2 * detg);
  Kn = bpp .* btt - btp .* btp;
  dK = (dbpp .* btt + bpp .* dbtt - 2 * btp .* dbtp - Kn ./ detg .* ddet) ./ detg;

  % the flux-form weights' derivatives, from the sin(theta)-scaled tangent
  ptt = sxt1 .* sxt1 + sxt2 .* sxt2 + sxt3 .* sxt3;
  ptp = sxt1 .* xp1 + sxt2 .* xp2 + sxt3 .* xp3;
  dptt = 2 * (sxt1 .* dsxt1 + sxt2 .* dsxt2 + sxt3 .* dsxt3);
  dptp = dsxt1 .* xp1 + dsxt2 .* xp2 + dsxt3 .* xp3 + sxt1 .* dxp1 + sxt2 .* dxp2 + sxt3 .* dxp3;
  D2 = ptt .* gpp - ptp .* ptp;
  dD2 = dptt .* gpp + ptt .* dgpp - 2 * ptp .* dptp;
  id = 1 ./ realsqrt(D2);
  did = -0.5 * id .* dD2 ./ D2;
  ddp1 = dgpp .* id + gpp .* did;
  dp2 = -(dptp .* id + ptp .* did);
  ddq2 = dptt .* id + ptt .* did;
  djinv = did .* s2;
end
