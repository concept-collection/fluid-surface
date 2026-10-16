% The surface gradient of a scalar field on the grid, from its spectral
% coefficients F: grad f = v^theta f_theta + v^phi f_phi, with the dual
% tangent vectors of the surface geometry geo (lib/geometry.m). The result
% is an npts x 3 vector field. sin(theta) f_theta and f_phi come from the
% coefficient shuffles and a synthesis, so the only division is by
% sin(theta) at the Gauss nodes. (The C++ solver's KFGrad.)

function g = surfgrad(F, geo, grid)
  [a, b] = synth(dthetac(F), dphic(F));
  g = [geo.vt1, geo.vt2, geo.vt3] .* (a ./ grid.st) + [geo.vp1, geo.vp2, geo.vp3] .* b;
end
