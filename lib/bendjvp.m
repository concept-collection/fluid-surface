% The derivative of the Helfrich bending force (models/stokes.m bendlin) when
% the surface moves by Y, which has spectral coefficients on the working band:
% through the geometry's derivative (lib/geomtangent.m), the Laplacian's,
% whose weights and metric factor move too, and the force's own. The inputs
% are the surface's derivative fields der (lib/posderivs.m), its geometry geo
% (lib/geometry.m), and what bendlin stored in bl: the Laplacian's grid
% fields ba, bb, its value before jinv (braw), and the force's normal scalar
% bvc. bscale is -2/Be. The C++ solver's FluidBendingJvp.
function [J1, J2, J3] = bendjvp(Y1, Y2, Y3, der, geo, bl, grid, msk, bscale)
  [dsxt1, dsxt2, dsxt3, dxp1, dxp2, dxp3, dsxtt1, dsxtt2, dsxtt3, dsxtp1, dsxtp2, dsxtp3, dxpp1, dxpp2, dxpp3] = posderivs(Y1, Y2, Y3, grid);
  [dH, dK, dn1, dn2, dn3, ddp1, dp2, ddq2, djinv] = geomtangent(der, dsxt1, dsxt2, dsxt3, dxp1, dxp2, dxp3, dsxtt1, dsxtt2, dsxtt3, dsxtp1, dsxtp2, dsxtp3, dxpp1, dxpp2, dxpp3, grid);
  H = geo.H;
  K = geo.K;
  % the Laplacian of H moves with H and with its own weights
  dU = analys(dH) .* msk.filtf;
  [da, db] = synth(dthetac(dU), dphic(dU));
  s2 = grid.st .* grid.st;
  [A, B] = analys((ddp1 .* bl.ba + geo.dp1 .* da + dp2 .* bl.bb + geo.p2 .* db) ./ s2, (dp2 .* bl.ba + geo.p2 .* da + ddq2 .* bl.bb + geo.dq2 .* db) ./ s2);
  draw = synth(-msk.lam .* dU + (dphic(B) - dthetat(A)) .* msk.filtf);
  dlap = synth(analys(draw .* geo.jinv + bl.braw .* djinv) .* msk.real0);
  dvc = dlap + 2 * (dH .* (H .* H - K) + H .* (2 * H .* dH - dK));
  [F1, F2, F3] = analys(bscale * (dvc .* geo.n1 + bl.bvc .* dn1), bscale * (dvc .* geo.n2 + bl.bvc .* dn2), bscale * (dvc .* geo.n3 + bl.bvc .* dn3));
  J1 = F1 .* msk.keep;
  J2 = F2 .* msk.keep;
  J3 = F3 .* msk.keep;
end
