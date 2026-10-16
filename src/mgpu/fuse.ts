/**
 * Folds single-use array temps into the statement that reads them, for the
 * expressions numbl's own inline pass leaves alone because the WGSL backend
 * can still compute them in one kernel: broadcasting (an npts x 1 field
 * against an npts x 3 vector field), row sums `sum(x, 2)`, column reads
 * `x(:, k)` and concatenation `[a, b, c]`. numbl's pass requires every
 * operand to have the target's shape, since its C emitter does.
 *
 * The gates follow numbl's (codegen/inlinePass.ts): the producer is a
 * multi-element Assign that is not an output, its readers come later in the
 * same statement list with nothing in between writing the producer's
 * operands or crossing a loop, and each read is in a slot the substitution
 * reaches (an operand, a call argument, a concatenation block, the base of
 * `x(:, k)`). Every statement involved must be one the WGSL emitter computes
 * in a kernel (src/mgpu/wgsl.ts). Unlike numbl's pass, a producer read
 * several times is also folded when that repeats no arithmetic, so a vector
 * field assembled as `n = [n1, n2, n3]`, or read only by its columns, costs
 * no buffer of its own. A producer read once is folded even where the read
 * broadcasts it (an npts x 1 row sum into an npts x 3 expression), so it is
 * evaluated once per column; that repeats a few operations to save a buffer's
 * write and read, the better trade on a GPU.
 */
import { forEachSubExpr, forEachTopLevelExpr } from 'numbl-src/numbl-core/jit/lowering/walk.ts';
import { isMultiElement } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import type { Assign, IRExpr, IRFunc, IRStmt } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
import type { Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import { EXTERNAL_OPS } from './externals.ts';
import { mapVars } from './irwalk.ts';

/** True if the WGSL emitter computes `e` inside an element-wise kernel. */
function inKernel(e: IRExpr): boolean {
  switch (e.kind) {
    case 'NumLit':
    case 'Var':
      return true;
    case 'Binary':
      return inKernel(e.left) && inKernel(e.right);
    case 'Unary':
      return inKernel(e.operand);
    case 'Call':
      // Externals (transforms, dot, ...) are their own dispatches; a call
      // whose cName differs from its name is a user function.
      return !EXTERNAL_OPS.has(e.name) && e.cName === e.name && e.args.every(inKernel);
    case 'IndexSlice':
      return e.index.length === 2 && e.index[0].kind === 'Colon' && e.index[1].kind === 'Scalar' && inKernel(e.base);
    case 'TensorConcat':
      return e.cells.length === 1 && e.cells[0].every(inKernel);
    default:
      return false;
  }
}

/** A rearrangement of buffers with no arithmetic (`[a, b, c]`, `x(:, k)`),
 *  free to repeat in every reader. */
function isView(e: IRExpr): boolean {
  if (e.kind === 'Var' || e.kind === 'NumLit') return true;
  if (e.kind === 'TensorConcat') return e.cells.every((row) => row.every(isView));
  if (e.kind === 'IndexSlice') return isView(e.base);
  return false;
}

const isArrayProducer = (s: IRStmt): s is Assign =>
  s.kind === 'Assign' &&
  s.ty.kind === 'Numeric' && isMultiElement(s.ty) && !s.ty.isComplex &&
  inKernel(s.expr);

/** The literal columns at which `cName` is read as `cName(:, k)` in `e`, or
 *  undefined if it is also read some other way. */
function columnReads(e: IRExpr, cName: string, out: number[] = []): number[] | undefined {
  if (e.kind === 'Var') return e.cName === cName ? undefined : out;
  if (e.kind === 'IndexSlice' && e.base.kind === 'Var' && e.base.cName === cName) {
    const k = e.index[1];
    if (k?.kind !== 'Scalar' || k.expr.kind !== 'NumLit') return undefined;
    out.push(k.expr.value);
    return out;
  }
  let ok = true;
  forEachChild(e, (c) => {
    if (ok && columnReads(c, cName, out) === undefined) ok = false;
  });
  return ok ? out : undefined;
}

/** Every Var read in a statement, through loop bodies. */
function forEachRead(s: IRStmt, visit: (cName: string) => void): void {
  const walk = (e: IRExpr): void =>
    forEachSubExpr(e, (sub) => {
      if (sub.kind === 'Var') visit(sub.cName);
    });
  if (s.kind === 'For') {
    walk(s.start);
    walk(s.end);
    for (const b of s.body) forEachRead(b, visit);
    return;
  }
  forEachTopLevelExpr(s, walk);
}

/** True if `s` assigns any of `names`. */
function writesAny(s: IRStmt, names: ReadonlySet<string>): boolean {
  if (s.kind === 'Assign') return names.has(s.cName);
  if (s.kind === 'MultiAssignCall') return s.outputs.some((o) => o.binding !== null && names.has(o.binding.cName));
  return false;
}

/** The direct sub-expressions the substitution reaches. */
function forEachChild(e: IRExpr, visit: (c: IRExpr) => void): void {
  switch (e.kind) {
    case 'Binary':
      visit(e.left);
      visit(e.right);
      return;
    case 'Unary':
      visit(e.operand);
      return;
    case 'Call':
      e.args.forEach(visit);
      return;
    case 'TensorConcat':
      e.cells.forEach((row) => row.forEach(visit));
      return;
    case 'IndexSlice':
      visit(e.base);
      return;
  }
}

/** Occurrences of `cName` the substitution does not reach. */
function inNonSlot(e: IRExpr, cName: string): boolean {
  switch (e.kind) {
    case 'Binary':
    case 'Unary':
    case 'Call':
    case 'TensorConcat':
    case 'IndexSlice': {
      let found = false;
      forEachChild(e, (c) => {
        if (inNonSlot(c, cName)) found = true;
      });
      return found;
    }
    case 'Var':
    case 'NumLit':
      return false;
    default: {
      let found = false;
      forEachSubExpr(e, (sub) => {
        if (sub.kind === 'Var' && sub.cName === cName) found = true;
      });
      return found;
    }
  }
}

function substitute(e: IRExpr, cName: string, by: IRExpr): IRExpr {
  switch (e.kind) {
    case 'Var':
      // A fresh copy per read: later passes (src/mgpu/split.ts) rewrite
      // expressions in place, which must not reach another reader.
      return e.cName === cName ? mapVars(by, (v) => v) : e;
    case 'Binary':
      return { ...e, left: substitute(e.left, cName, by), right: substitute(e.right, cName, by) };
    case 'Unary':
      return { ...e, operand: substitute(e.operand, cName, by) };
    case 'Call':
      return { ...e, args: e.args.map((a) => substitute(a, cName, by)) };
    case 'TensorConcat':
      return { ...e, cells: e.cells.map((row) => row.map((c) => substitute(c, cName, by))) };
    case 'IndexSlice':
      return { ...e, base: substitute(e.base, cName, by) };
    default:
      return e;
  }
}

/**
 * One fold in `stmts` (or a loop body inside it), or false. A producer read
 * once is folded into its reader. One read several times is folded into all
 * of them when that repeats no arithmetic: it is a view, or every read is a
 * different column `p(:, k)`, each computing only its own column.
 */
function fuseOnce(stmts: IRStmt[], uses: Map<string, number>): boolean {
  for (let i = 0; i < stmts.length; i++) {
    const p = stmts[i];
    if (p.kind === 'For') {
      if (fuseOnce(p.body, uses)) return true;
      continue;
    }
    if (!isArrayProducer(p)) continue;
    const total = uses.get(p.cName) ?? 0;
    if (total === 0) continue;
    const operands = new Set<string>();
    forEachSubExpr(p.expr, (sub) => {
      if (sub.kind === 'Var') operands.add(sub.cName);
    });
    const guarded = new Set([...operands, p.cName]);
    // The readers, up to the last use, with nothing in between overwriting
    // the producer's operands.
    const readers: number[] = [];
    let found = 0;
    for (let j = i + 1; j < stmts.length && found < total; j++) {
      const c = stmts[j];
      let n = 0;
      forEachRead(c, (r) => {
        if (r === p.cName) n++;
      });
      if (n > 0) {
        if (c.kind !== 'Assign' || !inKernel(c.expr) || guarded.has(c.cName) || inNonSlot(c.expr, p.cName)) break;
        readers.push(j);
        found += n;
        continue;
      }
      if (c.kind === 'For' || writesAny(c, guarded)) break;
    }
    if (found !== total) continue;
    if (total > 1 && !isView(p.expr)) {
      const cols = readers.flatMap((j) => columnReads((stmts[j] as Assign).expr, p.cName) ?? [NaN]);
      if (cols.some(Number.isNaN) || new Set(cols).size !== cols.length) continue;
    }
    for (const j of readers) {
      const c = stmts[j] as Assign;
      stmts[j] = { ...c, expr: substitute(c.expr, p.cName, p.expr) };
    }
    stmts.splice(i, 1);
    return true;
  }
  return false;
}

const cols = (t: Type): number | undefined =>
  t.kind !== 'Numeric' ? undefined : !isMultiElement(t) ? 1 : t.shape?.length === 2 ? t.shape[1] : undefined;

/**
 * Column k of `e`, written as an expression that reads only that column:
 * `(a + b)(:, k)` is `a(:, k) + b(:, k)`, a single column broadcasts to every
 * column, and a concatenation gives its block. `slice` is an `x(:, k)` node
 * whose index is reused for the leaves. Undefined where the column cannot be
 * pushed further, in which case the caller keeps `x(:, k)` as it is.
 */
function columnOf(e: IRExpr, k: number, slice: IRExpr & { kind: 'IndexSlice' }): IRExpr | undefined {
  const c = cols(e.ty);
  if (c === 1) return e;
  if (c === undefined || e.ty.kind !== 'Numeric' || !e.ty.shape) return undefined;
  const ty = { ...e.ty, shape: [e.ty.shape[0], 1] } as Type;
  const all = (xs: IRExpr[]): IRExpr[] | undefined => {
    const out = xs.map((x) => columnOf(x, k, slice));
    return out.every((x) => x !== undefined) ? (out as IRExpr[]) : undefined;
  };
  switch (e.kind) {
    case 'Var':
      return { ...slice, base: e, ty } as IRExpr;
    case 'Binary': {
      const lr = all([e.left, e.right]);
      return lr && ({ ...e, left: lr[0], right: lr[1], ty } as IRExpr);
    }
    case 'Unary': {
      const o = columnOf(e.operand, k, slice);
      return o && ({ ...e, operand: o, ty } as IRExpr);
    }
    case 'Call': {
      if (!inKernel(e) || e.name === 'sum') return undefined;
      const args = all(e.args);
      return args && ({ ...e, args, ty } as IRExpr);
    }
    case 'TensorConcat': {
      if (e.cells.length !== 1) return undefined;
      let offset = 0;
      for (let j = 0; j < e.cells[0].length; j++) {
        const w = e.cellCols[0][j];
        if (k <= offset + w) {
          const cell = e.cells[0][j];
          if (w === 1) return cell;
          const kk = k - offset;
          const index = slice.index[1];
          const lit = index.kind === 'Scalar' && index.expr.kind === 'NumLit' ? index.expr : undefined;
          return lit && columnOf(cell, kk, { ...slice, index: [slice.index[0], { kind: 'Scalar', expr: { ...lit, value: kk } }] });
        }
        offset += w;
      }
      return undefined;
    }
    default:
      return undefined;
  }
}

/** Resolves each column read of an expression (not a buffer) in `e`. */
function resolveColumns(e: IRExpr): IRExpr {
  switch (e.kind) {
    case 'Binary':
      return { ...e, left: resolveColumns(e.left), right: resolveColumns(e.right) };
    case 'Unary':
      return { ...e, operand: resolveColumns(e.operand) };
    case 'Call':
      return { ...e, args: e.args.map(resolveColumns) };
    case 'TensorConcat':
      return { ...e, cells: e.cells.map((row) => row.map(resolveColumns)) };
    case 'IndexSlice': {
      const base = resolveColumns(e.base);
      const at = e.index[1];
      if (base.kind === 'Var' || e.index.length !== 2 || e.index[0].kind !== 'Colon' || at?.kind !== 'Scalar' || at.expr.kind !== 'NumLit') {
        return { ...e, base };
      }
      const col = columnOf(base, at.expr.value, e);
      return col ? resolveColumns(col) : { ...e, base };
    }
    default:
      return e;
  }
}

function resolveAll(stmts: IRStmt[]): void {
  stmts.forEach((s, i) => {
    if (s.kind === 'For') resolveAll(s.body);
    else if (s.kind === 'Assign' && inKernel(s.expr)) stmts[i] = { ...s, expr: resolveColumns(s.expr) };
  });
}

/**
 * Folds temps in the function body in place, to a fixed point, then resolves
 * column reads of the folded expressions, so a kernel computing `m(:, 1)` of
 * `m = a + [d1, d2, d3]` reads `a(:, 1)` and `d1` and binds no other buffer.
 */
export function fuseKernels(fn: IRFunc): void {
  for (;;) {
    const uses = new Map<string, number>();
    // Outputs are read by the host: never fold them away.
    for (const c of fn.cOutputs) uses.set(c, 1);
    for (const s of fn.body) forEachRead(s, (n) => uses.set(n, (uses.get(n) ?? 0) + 1));
    if (!fuseOnce(fn.body, uses)) break;
  }
  resolveAll(fn.body);
}
