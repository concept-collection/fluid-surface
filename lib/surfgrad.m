% The surface gradient of a scalar field on the grid, from its spectral
% coefficients F: grad f = v^theta f_theta + v^phi f_phi, with the dual
% tangent vectors v^theta, v^phi of lib/geometry.m. sin(theta) f_theta and
% f_phi come from the coefficient shuffles and a synthesis, so the only
% division is by sin(theta) at the Gauss nodes. (The C++ solver's KFGrad.)

function [g1, g2, g3] = surfgrad(F, vt1, vt2, vt3, vp1, vp2, vp3, st)
  [a, b] = synth(dthetac(F), dphic(F));
  ft = a ./ st;
  g1 = vt1 .* ft + vp1 .* b;
  g2 = vt2 .* ft + vp2 .* b;
  g3 = vt3 .* ft + vp3 .* b;
end
