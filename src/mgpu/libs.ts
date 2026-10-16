/**
 * The shared .m library every compiled function can call, as on MATLAB's path:
 * a file's namesake function is visible by its name.
 */
import sphtorSource from '../../lib/sphtor.m?raw';
import surfgradSource from '../../lib/surfgrad.m?raw';
import tandivSource from '../../lib/tandiv.m?raw';
import deformdivSource from '../../lib/deformdiv.m?raw';
import cart2qstSource from '../../lib/cart2qst.m?raw';
import qst2cartSource from '../../lib/qst2cart.m?raw';
import posderivsSource from '../../lib/posderivs.m?raw';
import geomtangentSource from '../../lib/geomtangent.m?raw';
import fluxlapSource from '../../lib/fluxlap.m?raw';
import bendjvpSource from '../../lib/bendjvp.m?raw';

export interface LibFile {
  name: string;
  source: string;
}

export const operatorLibs: LibFile[] = [
  { name: 'sphtor.m', source: sphtorSource },
  { name: 'surfgrad.m', source: surfgradSource },
  { name: 'tandiv.m', source: tandivSource },
  { name: 'deformdiv.m', source: deformdivSource },
  { name: 'cart2qst.m', source: cart2qstSource },
  { name: 'qst2cart.m', source: qst2cartSource },
  { name: 'posderivs.m', source: posderivsSource },
  { name: 'geomtangent.m', source: geomtangentSource },
  { name: 'fluxlap.m', source: fluxlapSource },
  { name: 'bendjvp.m', source: bendjvpSource },
];
