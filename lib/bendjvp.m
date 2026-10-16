% The derivative of the Helfrich bending force (models/stokes.m bendlin) when
% the surface moves by Y, which has spectral coefficients on the working band:
% through the geometry's derivative (lib/geomtangent.m), the Laplacian's,
% whose weights and metric factor move too, and the force's own. The inputs
% are the surface's derivative fields (lib/posderivs.m), its geometry
% (lib/geometry.m), and what bendlin stored: the Laplacian's grid fields ba,
% bb, its value before jinv (braw), and the force's normal scalar bvc.
% bscale is -2/Be. The C++ solver's FluidBendingJvp.
function [J1, J2, J3] = bendjvp(Y1, Y2, Y3, sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3, ct, st, H, K, n1, n2, n3, dp1, p2, dq2, jinv, ba, bb, braw, bvc, lam, filtf, real0, keep, bscale)
  [dsxt1, dsxt2, dsxt3, dxp1, dxp2, dxp3, dsxtt1, dsxtt2, dsxtt3, dsxtp1, dsxtp2, dsxtp3, dxpp1, dxpp2, dxpp3] = posderivs(Y1, Y2, Y3, ct);
  [dH, dK, dn1, dn2, dn3, ddp1, dp2d, ddq2, djinv] = geomtangent(sxt1, sxt2, sxt3, xp1, xp2, xp3, sxtt1, sxtt2, sxtt3, sxtp1, sxtp2, sxtp3, xpp1, xpp2, xpp3, dsxt1, dsxt2, dsxt3, dxp1, dxp2, dxp3, dsxtt1, dsxtt2, dsxtt3, dsxtp1, dsxtp2, dsxtp3, dxpp1, dxpp2, dxpp3, st);
  % the Laplacian of H moves with H and with its own weights
  dU = analys(dH) .* filtf;
  [da, db] = synth(dthetac(dU), dphic(dU));
  s2 = st .* st;
  [A, B] = analys((ddp1 .* ba + dp1 .* da + dp2d .* bb + p2 .* db) ./ s2, (dp2d .* ba + p2 .* da + ddq2 .* bb + dq2 .* db) ./ s2);
  draw = synth(-lam .* dU + (dphic(B) - dthetat(A)) .* filtf);
  dlap = synth(analys(draw .* jinv + braw .* djinv) .* real0);
  dvc = dlap + 2 * (dH .* (H .* H - K) + H .* (2 * H .* dH - dK));
  [F1, F2, F3] = analys(bscale * (dvc .* n1 + bvc .* dn1), bscale * (dvc .* n2 + bvc .* dn2), bscale * (dvc .* n3 + bvc .* dn3));
  J1 = F1 .* keep;
  J2 = F2 .* keep;
  J3 = F3 .* keep;
end
