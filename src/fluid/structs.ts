/**
 * The structs of arrays the .m files take (models/stokes.m), field by field.
 * Each field is a GPU buffer named `<struct>.<field>`, which the host uploads
 * or another compiled function's output feeds.
 */
import type { Binding } from '../mgpu/compile.ts';
import { GEOMETRY_OUTPUTS } from '../geom/geometry.ts';

/** cos and sin of theta and phi on the grid, and the quadrature weights. */
export const GRID_FIELDS = ['ct', 'st', 'cp', 'sp', 'wq'] as const;
/** The surface geometry, lib/geometry.m's outputs. */
export const GEO_FIELDS = GEOMETRY_OUTPUTS;
/** The position's derivative fields, lib/posderivs.m's outputs. */
export const DER_FIELDS = ['sxt', 'xp', 'sxtt', 'sxtp', 'xpp'].flatMap((n) => [1, 2, 3].map((c) => `${n}${c}`));
/** What bendlin stores for the bending force's derivative. */
export const BL_FIELDS = ['ba', 'bb', 'braw', 'bvc'] as const;
/** Masks and per-degree symbols (src/fluid/spectra.ts spectralMasks). */
export const MSK_FIELDS = ['lam', 'keep', 'clean', 'filtf', 'e0', 'ilam', 'cutv', 'real0'] as const;
/** The sphere's per-degree blocks, inverted (src/fluid/sphereBlocks.ts). */
export const TAB_FIELDS = ['i11', 'i12', 'i13', 'i21', 'i22', 'i23', 'i31', 'i32', 'i33', 'w11', 'w12', 'w21', 'w22', 'itor'] as const;

/** A struct binding whose fields all have one shape. */
export const structOf = (fields: readonly string[], field: Binding): Binding => ({
  kind: 'struct',
  fields: Object.fromEntries(fields.map((f) => [f, field])),
});

/** Feedback map sending each named output to `<struct>.<name>`. */
export const into = (struct: string, fields: readonly string[]): Record<string, string> =>
  Object.fromEntries(fields.map((f) => [f, `${struct}.${f}`]));
