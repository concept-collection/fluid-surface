% The surface gradient of a scalar field on the grid, from its spectral
% coefficients F: grad f = v^theta f_theta + v^phi f_phi, with the dual
% tangent vectors of the surface geometry geo (lib/geometry.m).
% sin(theta) f_theta and f_phi come from the coefficient shuffles and a
% synthesis, so the only division is by sin(theta) at the Gauss nodes.
% (The C++ solver's KFGrad.)

function [g1, g2, g3] = surfgrad(F, geo, grid)
  [a, b] = synth(dthetac(F), dphic(F));
  ft = a ./ grid.st;
  g1 = geo.vt1 .* ft + geo.vp1 .* b;
  g2 = geo.vt2 .* ft + geo.vp2 .* b;
  g3 = geo.vt3 .* ft + geo.vp3 .* b;
end
