/**
 * The shared .m library every compiled function can call, as on MATLAB's path:
 * a file's namesake function is visible by its name.
 */
import sphtorSource from '../../lib/sphtor.m?raw';
import surfgradSource from '../../lib/surfgrad.m?raw';
import tandivSource from '../../lib/tandiv.m?raw';

export interface LibFile {
  name: string;
  source: string;
}

export const operatorLibs: LibFile[] = [
  { name: 'sphtor.m', source: sphtorSource },
  { name: 'surfgrad.m', source: surfgradSource },
  { name: 'tandiv.m', source: tandivSource },
];
