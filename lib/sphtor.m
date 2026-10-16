% The spheroidal and toroidal coefficients S, T of a tangent vector field on
% the sphere, from its theta and phi components on the grid, in SHTns'
% convention:
%
%   v_theta = dtheta S + dphi T / sin(theta)
%   v_phi   = dphi S / sin(theta) - dtheta T
%
% Projecting onto grad Y and -r x grad Y gives
%
%   l(l+1) S = <v_theta, dtheta Y> + <v_phi / sin(theta), dphi Y>
%   l(l+1) T = <v_theta / sin(theta), dphi Y> - <v_phi, dtheta Y>
%
% and since sin(theta) dtheta Y_l = alpha^+ Y_{l+1} + alpha^- Y_{l-1} at
% every point, <f, dtheta Y> is the transposed alpha shift (dthetat) of the
% analysis of f / sin(theta). So two scalar analyses and two coefficient
% shuffles give the vector analysis: the quadrature sums are SHTns' own, term
% for term, and the Gauss nodes never reach the poles where sin(theta) = 0.
%
% The one difference is the top degree: S and T at l = lmax would need the
% analysis at lmax+1, so they are incomplete there. The fluid solver discards
% the top degrees of S anyway.
%
% st is sin(theta) on the grid; ilam is 1/(l(l+1)), 0 at l = 0.

function [S, T] = sphtor(vt, vp, st, ilam)
  [A, B] = analys(vt ./ st, vp ./ st);
  S = (dthetat(A) - dphic(B)) .* ilam;
  T = -(dphic(A) + dthetat(B)) .* ilam;
end
