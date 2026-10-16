/**
 * Arrows of a tangent vector field on a surface sampled on the Gauss grid
 * (phi-contiguous: point k = ilat * nphi + iphi), as line segments for
 * SphereScene.setLines. The arrows sit on about every `ringStep`-th latitude
 * ring, spaced along each ring about as far apart as the rings, so their
 * density is roughly even rather than crowding at the poles. Each is a shaft
 * and a two-stroke head, lifted slightly off the surface along the outward
 * normal, its length proportional to the vector's, the longest `maxLength`.
 */
export interface ArrowGrid {
  nlat: number;
  nphi: number;
  /** sin(theta) of each ring. */
  st: ArrayLike<number>;
}

export function arrowSegments(
  grid: ArrowGrid,
  x: ArrayLike<number>,
  v: ArrayLike<number>[],
  n: ArrayLike<number>[],
  maxLength: number,
  rings = 16,
): { segments: Float32Array; vmax: number } {
  const { nlat, nphi } = grid;
  const ringStep = Math.max(1, Math.round(nlat / rings));
  const picks: number[] = [];
  for (let i = Math.floor(ringStep / 2); i < nlat; i += ringStep) {
    // As many along the ring as its circumference allows at the ring spacing.
    const count = Math.max(1, Math.round((2 * rings) * grid.st[i]));
    const stride = Math.max(1, Math.round(nphi / count));
    for (let j = (i * 7) % stride; j < nphi; j += stride) picks.push(i * nphi + j);
  }
  let vmax = 0;
  for (const k of picks) vmax = Math.max(vmax, Math.hypot(v[0][k], v[1][k], v[2][k]));
  const out = new Float32Array(picks.length * 18);
  if (vmax === 0) return { segments: new Float32Array(0), vmax };
  const lift = 0.02 * maxLength;
  let o = 0;
  for (const k of picks) {
    const d = [v[0][k], v[1][k], v[2][k]];
    const len = Math.hypot(d[0], d[1], d[2]);
    const L = (maxLength * len) / vmax;
    const nn = [n[0][k], n[1][k], n[2][k]];
    const base = [0, 1, 2].map((c) => x[3 * k + c] + lift * nn[c]);
    const u = len > 0 ? d.map((c) => c / len) : [0, 0, 0];
    const tip = base.map((b, c) => b + L * u[c]);
    // The head's strokes, in the tangent plane: back along the arrow and to either side.
    const side = [nn[1] * u[2] - nn[2] * u[1], nn[2] * u[0] - nn[0] * u[2], nn[0] * u[1] - nn[1] * u[0]];
    const h = 0.3 * L;
    const w = 0.15 * L;
    const left = tip.map((t, c) => t - h * u[c] + w * side[c]);
    const right = tip.map((t, c) => t - h * u[c] - w * side[c]);
    for (const p of [base, tip, tip, left, tip, right]) {
      out[o++] = p[0];
      out[o++] = p[1];
      out[o++] = p[2];
    }
  }
  return { segments: out, vmax };
}
