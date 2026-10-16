% The Cartesian components' coefficients of the vector field on the unit
% sphere with radial, spheroidal and toroidal potentials Q, S, T (the inverse
% of lib/cart2qst.m):
%
%   u = Q r + v_theta e_theta + v_phi e_phi,
%   v_theta = dtheta S + dphi T / sin(theta),  v_phi = dphi S / sin(theta) - dtheta T.
%
% The C++ solver's FluidQstToCartesian. The coefficients must stop two
% degrees below the band, so that the theta derivatives stay in band.

function [U1, U2, U3] = qst2cart(Q, S, T, grid)
  [q, a, b, c, d] = synth(Q, dthetac(S), dphic(T), dphic(S), dthetac(T));
  vt = (a + b) ./ grid.st;
  vp = (c - d) ./ grid.st;
  w = q .* grid.st + vt .* grid.ct;
  [U1, U2, U3] = analys(w .* grid.cp - vp .* grid.sp, w .* grid.sp + vp .* grid.cp, q .* grid.ct - vt .* grid.st);
end
