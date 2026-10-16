/**
 * IR expression tree -> one WGSL compute kernel.
 *
 * This is the WebGPU counterpart of numbl's C-side fused emitter
 * (`codegen/emitTensorFused.ts`): for an `Assign` whose right-hand side is
 * element-wise, emit a single kernel that computes one output element per
 * invocation. Because numbl's inline pass (and src/mgpu/fuse.ts) has already
 * folded the ANF temps back together, one source line of MATLAB becomes one
 * kernel.
 *
 * Arrays are column-major and two-dimensional. An operand broadcasts against
 * the output as in MATLAB when each of its dimensions equals the output's or
 * is 1 (an npts x 1 field against an npts x 3 vector field). Beyond the
 * operators and element-wise calls, a kernel takes `sum(x, 2)` (a loop over
 * the columns of each row), `x(:, k)` at a literal k (a column read), and
 * `[a, b, c]` (column blocks).
 *
 * Everything is f32, matching the existing fp32 WebGPU transform backend.
 */
import { getBuiltin } from 'numbl-src/numbl-core/jit/builtins/index.ts';
import { isMultiElement } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import type { IRExpr, Assign } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
import type { NumericType, Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';

/** Raised for a construct the WGSL backend cannot express. Mirrors numbl's
 *  own decline discipline: fail at compile time with a source span, never
 *  silently produce something that computes the wrong thing. */
export class UnsupportedOnGpu extends Error {
  readonly span?: unknown;
  constructor(message: string, span?: unknown) {
    super(message);
    this.name = 'UnsupportedOnGpu';
    this.span = span;
  }
}

const isNumeric = (t: Type): t is NumericType => t.kind === 'Numeric';
const isTensor = (t: Type): boolean => isNumeric(t) && isMultiElement(t);

/** Element-wise binary builtins -> WGSL infix operator. */
const BINARY_OPS: Record<string, string> = {
  plus: '+',
  minus: '-',
  times: '*',
  rdivide: '/',
  // Degenerate to element-wise when at least one side is a scalar; the
  // both-tensor (true matrix) case is rejected below.
  mtimes: '*',
  mrdivide: '/',
};

/** Element-wise unary builtins -> WGSL prefix operator. */
const UNARY_OPS: Record<string, string> = { uminus: '-', uplus: '+' };

/** Element-wise builtin calls -> WGSL builtin of the same arity. */
const CALL_FNS: Record<string, string> = {
  abs: 'abs',
  acos: 'acos',
  asin: 'asin',
  atan: 'atan',
  atan2: 'atan2',
  ceil: 'ceil',
  cos: 'cos',
  cosh: 'cosh',
  exp: 'exp',
  floor: 'floor',
  log: 'log',
  log2: 'log2',
  max: 'max',
  min: 'min',
  round: 'round',
  sign: 'sign',
  sin: 'sin',
  sinh: 'sinh',
  sqrt: 'sqrt',
  // Real by declaration; WGSL's sqrt returns NaN for negative input, where
  // MATLAB's realsqrt errors.
  realsqrt: 'sqrt',
  tan: 'tan',
  tanh: 'tanh',
};

/** WGSL f32 literal. Must always carry a decimal point or exponent, or WGSL
 *  infers AbstractInt and rejects the mixed-type arithmetic. */
function f32Lit(v: number): string {
  if (!Number.isFinite(v)) {
    throw new UnsupportedOnGpu(`cannot emit non-finite literal ${v}`);
  }
  return Number.isInteger(v) && Math.abs(v) < 1e21
    ? `${v}.0`
    : String(v).includes('e')
      ? `${v}f`
      : String(v);
}

/** How a scalar or tensor operand is read inside the kernel. */
export interface KernelInputs {
  /** cName -> storage binding index, for buffer-backed operands. A
   *  multi-element tensor is read at the output's index; a single-element
   *  value (a `dot` result, or a scalar computed from one) is read at [0],
   *  which broadcasts it across the output. */
  tensors: Map<string, number>;
  /** cName -> slot in the params storage buffer, for runtime scalars. */
  params: Map<string, number>;
  /** cName -> defining expression, for scalars the .m computes from
   *  parameters (`us = a + b`). These have no buffer and no param slot; they
   *  become `let` bindings in the prologue of every kernel that reads them. */
  scalars: Map<string, { name: string; expr: IRExpr }>;
}

/** Mutable state while emitting one kernel. */
interface Ctx {
  io: KernelInputs;
  /** `let` lines to emit before the body, in dependency order. */
  prologue: string[];
  /** cName -> WGSL identifier, for scalars already bound in the prologue. */
  bound: Map<string, string>;
  /** Counter for prologue temporaries (row sums, concatenations). */
  temps: number;
}

/**
 * Where an expression is evaluated: the index expressions of the element in
 * a shape rows x cols. The kernel's own scope is the output, with `row` and
 * `col` bound from the invocation index; `sum(x, 2)` evaluates x over a loop
 * of columns, and a concatenation evaluates each block at its own column.
 */
interface Scope {
  rows: number;
  cols: number;
  /** Linear (column-major) index in that shape. */
  lin: string;
  row: string;
  col: string;
  /** The column, when it is known at compile time. */
  colLit?: number;
  /** Inside a loop over columns, where prologue code cannot depend on col. */
  inLoop: boolean;
}

const shape2 = (t: Type): [number, number] | undefined => {
  if (!isNumeric(t) || !t.shape || t.shape.length !== 2) return undefined;
  return [t.shape[0], t.shape[1]];
};

/** Index into an operand of shape r x c read in `scope`, broadcasting a
 *  dimension of 1. Throws if the shapes do not broadcast. */
function operandIndex(r: number, c: number, scope: Scope, name: string, span: unknown): string {
  if (r === scope.rows && c === scope.cols) return scope.lin;
  const okR = r === scope.rows || r === 1;
  const okC = c === scope.cols || c === 1;
  if (!okR || !okC) {
    throw new UnsupportedOnGpu(
      `'${name}' is ${r}x${c}, which does not broadcast against ${scope.rows}x${scope.cols}`,
      span,
    );
  }
  const rowPart = r === 1 ? '0u' : scope.row;
  return c === 1 ? rowPart : `(${rowPart} + ${r}u * ${scope.col})`;
}

/** WGSL identifier for a derived scalar. Avoids a leading underscore, which
 *  WGSL reserves. */
const scalarIdent = (cName: string): string =>
  `s_${cName.replace(/[^A-Za-z0-9_]/g, '_')}`;

/**
 * Bind a .m-derived scalar in the prologue (once), after whatever it depends
 * on, and return its identifier.
 */
function bindScalar(cName: string, ctx: Ctx): string {
  const already = ctx.bound.get(cName);
  if (already) return already;
  const def = ctx.io.scalars.get(cName)!;
  const ident = scalarIdent(cName);
  // Claim the name before emitting the RHS so a (malformed) self-reference
  // cannot recurse forever.
  ctx.bound.set(cName, ident);
  const rhs = emitExpr(def.expr, ctx, SCALAR_SCOPE);
  ctx.prologue.push(`  let ${ident} = ${rhs};`);
  return ident;
}

/** The scope of a derived scalar's definition, which reads no arrays. */
const SCALAR_SCOPE: Scope = { rows: 1, cols: 1, lin: '0u', row: '0u', col: '0u', inLoop: false };

/**
 * Emit the per-element WGSL expression for `e`, evaluated at the element of
 * `scope`.
 */
function emitExpr(e: IRExpr, ctx: Ctx, scope: Scope): string {
  const io = ctx.io;
  switch (e.kind) {
    case 'NumLit':
      return f32Lit(e.value);

    case 'Var': {
      // Buffer-backed operands first: a single-element value in a buffer
      // shadows any compile-time definition the same name had earlier (a
      // scalar seeded `rho = 1` and then updated from a dot result inside
      // the solve loop reads as a buffer from the update on).
      const bound = io.tensors.get(e.cName);
      if (bound !== undefined) {
        if (!isTensor(e.ty)) return `in${bound}[0]`;
        const sh = shape2(e.ty);
        if (!sh) throw new UnsupportedOnGpu(`'${e.name}' has no static two-dimensional shape`, e.span);
        return `in${bound}[${operandIndex(sh[0], sh[1], scope, e.name, e.span)}]`;
      }
      if (isTensor(e.ty)) {
        throw new UnsupportedOnGpu(`no buffer bound for '${e.name}'`, e.span);
      }
      // Scalar: either an exact compile-time value or a runtime parameter.
      if (isNumeric(e.ty) && typeof e.ty.exact === 'number') {
        return f32Lit(e.ty.exact);
      }
      const slot = io.params.get(e.cName);
      if (slot !== undefined) return `prm[${slot}]`;
      if (io.scalars.has(e.cName)) return bindScalar(e.cName, ctx);
      throw new UnsupportedOnGpu(
        `scalar '${e.name}' is not a constant, a parameter, or computed in ` +
          `this model`,
        e.span,
      );
    }

    case 'Binary': {
      if ((e.builtin === 'mtimes' || e.builtin === 'mrdivide') &&
          isTensor(e.left.ty) && isTensor(e.right.ty)) {
        throw new UnsupportedOnGpu(
          `matrix '${e.builtin === 'mtimes' ? '*' : '/'}' is not supported; ` +
            `use the element-wise form ('.${e.builtin === 'mtimes' ? '*' : '/'}')`,
          e.span,
        );
      }
      if (e.builtin === 'power' || e.builtin === 'mpower') {
        return emitPower(e.left, e.right, ctx, scope, e.span);
      }
      const op = BINARY_OPS[e.builtin];
      if (!op) {
        throw new UnsupportedOnGpu(`operator '${e.builtin}' is not supported`, e.span);
      }
      return `(${emitExpr(e.left, ctx, scope)} ${op} ${emitExpr(e.right, ctx, scope)})`;
    }

    case 'Unary': {
      const op = UNARY_OPS[e.builtin];
      if (!op) {
        throw new UnsupportedOnGpu(`unary '${e.builtin}' is not supported`, e.span);
      }
      return `(${op}${emitExpr(e.operand, ctx, scope)})`;
    }

    case 'Call': {
      // A shape constructor used inside an element-wise expression
      // contributes the same constant at every slot, so it needs no buffer.
      if (e.name === 'ones') return '1.0';
      if (e.name === 'zeros') return '0.0';
      if (e.name === 'sum') return emitRowSum(e, ctx, scope);

      const fn = CALL_FNS[e.name];
      const b = getBuiltin(e.name);
      const pointwise = b?.elementwise === true || b?.isElementwiseCall?.(e.args.map((a) => a.ty)) === true;
      if (!fn || !pointwise) {
        // A call numbl resolved to another function in the workspace gets a
        // mangled specialization name; a builtin keeps its source-level name.
        // User-function calls are expanded into the caller before planning
        // (src/mgpu/inlineCalls.ts), so one surviving to kernel emission means
        // the expansion did not reach it — a distinct failure from an
        // unsupported builtin, and worth saying so.
        const isUserFunction = e.cName !== e.name;
        throw new UnsupportedOnGpu(
          isUserFunction
            ? `the call to '${e.name}' was not expanded into the caller — ` +
                `assign its result to a variable on its own line`
            : `'${e.name}' cannot be evaluated element-wise on the GPU`,
          e.span,
        );
      }
      return `${fn}(${e.args.map((a) => emitExpr(a, ctx, scope)).join(', ')})`;
    }

    case 'IndexSlice':
      return emitColumn(e, ctx, scope);

    case 'TensorConcat':
      return emitConcat(e, ctx, scope);

    default:
      throw new UnsupportedOnGpu(`'${e.kind}' is not supported on the GPU`, e.span);
  }
}

/** The literal value of an index expression, or undefined. */
const literal = (e: IRExpr): number | undefined =>
  e.kind === 'NumLit' ? e.value : isNumeric(e.ty) && typeof e.ty.exact === 'number' ? e.ty.exact : undefined;

/** Columns up to which `sum(x, 2)` is unrolled; vector fields have 3. */
const UNROLL_COLS = 8;

/** The scope of column c (from 0) of an array of shape sh, at `row`. */
const columnScope = (sh: [number, number], row: string, c: number): Scope => ({
  rows: sh[0], cols: sh[1], row, col: `${c}u`, colLit: c, lin: sh[0] === 1 ? `${c}u` : `(${row} + ${sh[0] * c}u)`, inLoop: false,
});

/** The row of an array with `rows` rows read in `scope` (broadcasting one row). */
function rowIn(rows: number, scope: Scope, what: string, span: unknown): string {
  if (rows === 1) return '0u';
  if (rows !== scope.rows) {
    throw new UnsupportedOnGpu(`${what} has ${rows} rows but is read in a ${scope.rows}-row expression`, span);
  }
  return scope.row;
}

/**
 * `sum(x, 2)`: the sum along each row. With few columns (a vector field) the
 * sum is unrolled, each term evaluating x at a literal column, so a
 * concatenation or a column read inside x resolves at compile time.
 */
function emitRowSum(e: IRExpr & { kind: 'Call' }, ctx: Ctx, scope: Scope): string {
  const [x, dim] = e.args;
  const sh = x && shape2(x.ty);
  if (e.args.length !== 2 || literal(dim) !== 2 || !sh) {
    throw new UnsupportedOnGpu(`only 'sum(x, 2)' of a two-dimensional array is supported on the GPU`, e.span);
  }
  const row = rowIn(sh[0], scope, `'sum(x, 2)'`, e.span);
  if (sh[1] <= UNROLL_COLS) {
    const terms = Array.from({ length: sh[1] }, (_, c) => emitExpr(x, ctx, columnScope(sh, row, c)));
    return `(${terms.join(' + ')})`;
  }
  const k = ctx.temps++;
  const acc = `rowsum${k}`;
  const c = `c${k}`;
  const inner: Scope = {
    rows: sh[0], cols: sh[1], row, col: c, lin: `(${row} + ${sh[0]}u * ${c})`, inLoop: true,
  };
  const body = emitExpr(x, ctx, inner);
  ctx.prologue.push(
    `  var ${acc}: f32 = 0.0;`,
    `  for (var ${c}: u32 = 0u; ${c} < ${sh[1]}u; ${c} = ${c} + 1u) { ${acc} = ${acc} + ${body}; }`,
  );
  return acc;
}

/** `x(:, k)` at a literal k: x evaluated at column k only. */
function emitColumn(e: IRExpr & { kind: 'IndexSlice' }, ctx: Ctx, scope: Scope): string {
  const sh = shape2(e.base.ty);
  const [r, c] = e.index;
  const k = c?.kind === 'Scalar' ? literal(c.expr) : undefined;
  if (!sh || e.index.length !== 2 || r.kind !== 'Colon' || k === undefined || !Number.isInteger(k)) {
    throw new UnsupportedOnGpu(`only a whole column at a literal index, 'x(:, k)', is supported on the GPU`, e.span);
  }
  if (k < 1 || k > sh[1]) throw new UnsupportedOnGpu(`column ${k} is out of range (1..${sh[1]})`, e.span);
  return emitExpr(e.base, ctx, columnScope(sh, rowIn(sh[0], scope, 'the column', e.span), k - 1));
}

/**
 * `[a, b, c]`: one row of column blocks, each as tall as the result. At a
 * literal column (inside an unrolled row sum or a column read) the block is
 * chosen at compile time; otherwise in the prologue, so a block is read only
 * where it is in range.
 */
function emitConcat(e: IRExpr & { kind: 'TensorConcat' }, ctx: Ctx, scope: Scope): string {
  const sh = shape2(e.ty);
  if (e.cells.length !== 1 || !sh) {
    throw new UnsupportedOnGpu(`only a single row of column blocks, '[a, b, c]', is supported on the GPU`, e.span);
  }
  if (scope.rows !== sh[0] || scope.cols !== sh[1]) {
    throw new UnsupportedOnGpu(`a concatenation must be read at its own shape (${sh[0]}x${sh[1]}); assign it to a variable first`, e.span);
  }
  const cells = e.cells[0];
  const widths = e.cellCols[0];
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== sh[1]) throw new UnsupportedOnGpu(`the blocks are ${total} columns wide, not ${sh[1]}`, e.span);
  const blockScope = (j: number, offset: number): Scope =>
    scope.colLit !== undefined
      ? columnScope([sh[0], widths[j]], scope.row, scope.colLit - offset)
      : {
          rows: sh[0], cols: widths[j], row: scope.row, col: `(${scope.col} - ${offset}u)`,
          lin: `(${scope.row} + ${sh[0]}u * (${scope.col} - ${offset}u))`, inLoop: false,
        };
  if (scope.colLit !== undefined) {
    let offset = 0;
    for (let j = 0; j < cells.length; j++) {
      if (scope.colLit < offset + widths[j]) return emitExpr(cells[j], ctx, blockScope(j, offset));
      offset += widths[j];
    }
  }
  if (scope.inLoop) {
    throw new UnsupportedOnGpu(`a concatenation inside a row sum over more than ${UNROLL_COLS} columns is not supported; assign it to a variable first`, e.span);
  }
  const v = `cat${ctx.temps++}`;
  const lines: string[] = [];
  let offset = 0;
  cells.forEach((cell, j) => {
    const val = emitExpr(cell, ctx, blockScope(j, offset));
    const cond = `${scope.col} < ${offset + widths[j]}u`;
    const last = j === cells.length - 1;
    lines.push(j === 0 ? `  if (${cond}) { ${v} = ${val}; }` : last ? `  else { ${v} = ${val}; }` : `  else if (${cond}) { ${v} = ${val}; }`);
    offset += widths[j];
  });
  ctx.prologue.push(`  var ${v}: f32 = 0.0;`, ...lines);
  return v;
}

/**
 * `x.^k`. WGSL's `pow` is undefined for a negative base, and these fields go
 * negative routinely, so expand small non-negative integer exponents into
 * repeated multiplication — which is also what makes `u.^2` free.
 */
function emitPower(base: IRExpr, exponent: IRExpr, ctx: Ctx, scope: Scope, span: unknown): string {
  const k =
    exponent.kind === 'NumLit'
      ? exponent.value
      : isNumeric(exponent.ty) && typeof exponent.ty.exact === 'number'
        ? exponent.ty.exact
        : undefined;
  const b = emitExpr(base, ctx, scope);
  if (k !== undefined && Number.isInteger(k) && k >= 0 && k <= 8) {
    if (k === 0) return '1.0';
    // bind once so a compound base expression is not re-evaluated k times
    return `pow_i${k}(${b})`;
  }
  if (k !== undefined && Number.isInteger(k) && k < 0 && k >= -8) {
    return `(1.0 / pow_i${-k}(${b}))`;
  }
  throw new UnsupportedOnGpu(
    `'.^' needs a literal integer exponent in [-8, 8] (got ` +
      `${k === undefined ? 'a runtime value' : k}); a negative base makes ` +
      `WGSL's pow() undefined`,
    span,
  );
}

/** Fixed-exponent power helpers, emitted only when used. */
function powHelpers(used: Set<number>): string {
  const out: string[] = [];
  for (const k of [...used].sort((a, b) => a - b)) {
    const body =
      k === 1 ? 'x' : `x${' * x'.repeat(k - 1)}`;
    out.push(`fn pow_i${k}(x: f32) -> f32 { return ${body}; }`);
  }
  return out.join('\n');
}

export const WORKGROUP_SIZE = 64;

export interface Kernel {
  code: string;
  /** Number of output elements. */
  count: number;
  label: string;
}

/**
 * Build the kernel for one element-wise `Assign`. `io` must already map every
 * tensor operand cName to a binding index and every runtime scalar to a
 * params slot; the output is binding 0 and the params buffer is the binding
 * after the last input.
 */
export function buildKernel(
  stmt: Assign,
  io: KernelInputs,
  count: number,
  label: string,
): Kernel {
  if (!isNumeric(stmt.ty)) {
    throw new UnsupportedOnGpu(`'${stmt.name}' is not a numeric array`, stmt.span);
  }
  if (stmt.ty.isComplex) {
    throw new UnsupportedOnGpu(
      `'${stmt.name}' is complex; the GPU backend is real-only (a spectral ` +
        `field is carried as a real 2 x nlm array)`,
      stmt.span,
    );
  }

  const sh = shape2(stmt.ty) ?? [count, 1];
  const scope: Scope = { rows: sh[0], cols: sh[1], lin: 'i', row: 'row', col: 'col', inLoop: false };
  const ctx: Ctx = { io, prologue: [], bound: new Map(), temps: 0 };
  const body = emitExpr(stmt.expr, ctx, scope);

  // pow_iK helpers are discovered during emission; scan the result for them.
  const used = new Set<number>();
  const emitted = [...ctx.prologue, body].join('\n');
  for (const m of emitted.matchAll(/\bpow_i(\d+)\(/g)) used.add(Number(m[1]));

  const decls = [`@group(0) @binding(0) var<storage, read_write> out: array<f32>;`];
  for (const [, slot] of io.tensors) {
    decls.push(
      `@group(0) @binding(${slot + 1}) var<storage, read> in${slot}: array<f32>;`,
    );
  }
  // Params live in a read-only storage buffer rather than a uniform block:
  // uniform arrays would need 16-byte element stride.
  const prmBinding = io.tensors.size + 1;
  decls.push(
    `@group(0) @binding(${prmBinding}) var<storage, read> prm: array<f32>;`,
  );

  const code = `${decls.join('\n')}

${powHelpers(used)}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let row = i % ${sh[0]}u;
  let col = i / ${sh[0]}u;
${ctx.prologue.length ? `${ctx.prologue.join('\n')}\n` : ''}  out[i] = ${body};
}
`;
  return { code, count, label };
}
