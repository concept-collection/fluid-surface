/** What every check module reports through: pass/fail with a detail line. */
export type Check = (name: string, ok: boolean, detail: string) => void;
export type Log = (s: string) => void;

/** Relative L2 difference of a against b, over the indices `keep` admits. */
export function relL2(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  keep: (i: number) => boolean = () => true,
): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    if (!keep(i)) continue;
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

export const fmt = (x: number): string => x.toExponential(1);
