/**
 * Restarted GMRES on the host, in f64, around a matvec that runs on the GPU.
 *
 * The C++ solver's native GMRES (backend/native/gmres_native.cpp) forms the
 * Krylov basis on the device and checks convergence on the host in batches.
 * This first version keeps the basis on the host instead: each iteration
 * uploads one vector, runs the compiled matvec and reads one vector back, so
 * it costs one synchronization per iteration. The operator is already left-
 * preconditioned, so the residual it minimizes is the preconditioned one, as
 * in the C++. Orthogonalization is modified Gram-Schmidt, run twice.
 */

export interface GmresOptions {
  /** Stop when the residual is below tol times the right-hand side's norm. */
  tol: number;
  /** Krylov dimension per cycle. */
  restart: number;
  maxCycles: number;
}

export interface GmresResult {
  x: Float64Array;
  iterations: number;
  converged: boolean;
  /** The explicit relative residual |b - A x| / |b| at the end. */
  relResid: number;
}

const dot = (a: Float64Array, b: Float64Array): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

export async function gmres(
  matvec: (x: Float64Array) => Promise<Float64Array>,
  b: Float64Array,
  x0: Float64Array,
  opt: GmresOptions,
): Promise<GmresResult> {
  const n = b.length;
  const bnorm = Math.sqrt(dot(b, b)) || 1;
  const x = Float64Array.from(x0);
  let iterations = 0;
  const residual = async (): Promise<Float64Array> => {
    const ax = await matvec(x);
    return b.map((bi, i) => bi - ax[i]);
  };
  let r = await residual();
  for (let cycle = 0; cycle < opt.maxCycles; cycle++) {
    const beta = Math.sqrt(dot(r, r));
    if (beta <= opt.tol * bnorm) return { x, iterations, converged: true, relResid: beta / bnorm };
    const m = opt.restart;
    const V: Float64Array[] = [r.map((v) => v / beta)];
    const H: number[][] = [];
    const cs: number[] = [];
    const sn: number[] = [];
    const g = [beta];
    let k = 0;
    for (; k < m; k++) {
      const w = await matvec(V[k]);
      iterations++;
      const h = new Array<number>(k + 2).fill(0);
      for (let pass = 0; pass < 2; pass++) {
        for (let j = 0; j <= k; j++) {
          const c = dot(w, V[j]);
          h[j] += c;
          for (let i = 0; i < n; i++) w[i] -= c * V[j][i];
        }
      }
      h[k + 1] = Math.sqrt(dot(w, w));
      // Apply the previous rotations to the new column, then make one more.
      for (let j = 0; j < k; j++) {
        const t = cs[j] * h[j] + sn[j] * h[j + 1];
        h[j + 1] = -sn[j] * h[j] + cs[j] * h[j + 1];
        h[j] = t;
      }
      const rho = Math.hypot(h[k], h[k + 1]);
      cs.push(h[k] / rho);
      sn.push(h[k + 1] / rho);
      g.push(-sn[k] * g[k]);
      g[k] = cs[k] * g[k];
      h[k] = rho;
      h[k + 1] = 0;
      H.push(h);
      const breakdown = !(w.length && Number.isFinite(h[k]) && rho > 0);
      if (Math.abs(g[k + 1]) <= opt.tol * bnorm || breakdown) {
        k++;
        break;
      }
      V.push(w.map((v) => v / Math.sqrt(dot(w, w))));
    }
    // Back-substitute for the coefficients of the basis and update x.
    const y = new Float64Array(k);
    for (let i = k - 1; i >= 0; i--) {
      let s = g[i];
      for (let j = i + 1; j < k; j++) s -= H[j][i] * y[j];
      y[i] = s / H[i][i];
    }
    for (let j = 0; j < k; j++) for (let i = 0; i < n; i++) x[i] += y[j] * V[j][i];
    r = await residual();
  }
  const rn = Math.sqrt(dot(r, r));
  return { x, iterations, converged: rn <= opt.tol * bnorm, relResid: rn / bnorm };
}
