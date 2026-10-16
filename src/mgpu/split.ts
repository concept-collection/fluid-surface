/**
 * Split element-wise statements that read more arrays than a kernel may bind.
 *
 * numbl's inline pass folds every single-use temp into its consumer, so one
 * line of MATLAB becomes one kernel, which binds every array the line reads.
 * WebGPU caps the storage buffers a kernel binds (maxStorageBuffersPerShader
 * Stage, 8 by default and 8 to 31 in practice), and the output and the
 * parameter buffer take two of them. A long line, such as the fluid momentum
 * with its viscous and pressure terms folded in, can exceed that.
 *
 * This pass hoists subexpressions of such a statement into statements of
 * their own, before it, until every statement fits: each step hoists the
 * subtree that reads the most arrays while fitting by itself, which removes
 * the most bindings from the parent. A statement that fits is left alone, so
 * fusion is undone only where the device requires it.
 */
import { isMultiElement } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import type { Assign, IRExpr, IRStmt } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
import type { Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';

const isTensor = (t: Type): boolean => t.kind === 'Numeric' && isMultiElement(t);

/** Distinct array variables an expression reads. */
function reads(e: IRExpr, out: Set<string>, isArray: (v: IRExpr & { kind: 'Var' }) => boolean): Set<string> {
  switch (e.kind) {
    case 'Var':
      if (isArray(e)) out.add(e.cName);
      break;
    case 'Binary':
      reads(e.left, out, isArray);
      reads(e.right, out, isArray);
      break;
    case 'Unary':
      reads(e.operand, out, isArray);
      break;
    case 'Call':
      for (const a of e.args) reads(a, out, isArray);
      break;
  }
  return out;
}

/** The children of an expression, with a setter for each. */
function children(e: IRExpr): { get: () => IRExpr; set: (x: IRExpr) => void }[] {
  switch (e.kind) {
    case 'Binary':
      return [
        { get: () => e.left, set: (x) => (e.left = x) },
        { get: () => e.right, set: (x) => (e.right = x) },
      ];
    case 'Unary':
      return [{ get: () => e.operand, set: (x) => (e.operand = x) }];
    case 'Call':
      return e.args.map((_, i) => ({ get: () => e.args[i], set: (x: IRExpr) => (e.args[i] = x) }));
    default:
      return [];
  }
}

/**
 * Returns `body` with over-wide element-wise statements split, recursing into
 * loop bodies. `budget` is the most arrays one kernel may read.
 */
export function splitWideKernels(body: IRStmt[], budget: number): IRStmt[] {
  let fresh = 0;
  const isArray = (v: IRExpr & { kind: 'Var' }) => isTensor(v.ty);
  const out: IRStmt[] = [];
  for (const stmt of body) {
    if (stmt.kind === 'For') {
      stmt.body = splitWideKernels(stmt.body, budget);
      out.push(stmt);
      continue;
    }
    if (stmt.kind !== 'Assign' || !isTensor(stmt.ty) || stmt.expr.kind === 'Call') {
      out.push(stmt);
      continue;
    }
    const hoisted: Assign[] = [];
    while (reads(stmt.expr, new Set(), isArray).size > budget) {
      // The widest subtree that fits on its own, below the root.
      let best: { slot: { get: () => IRExpr; set: (x: IRExpr) => void }; n: number } | null = null;
      const visit = (e: IRExpr) => {
        for (const slot of children(e)) {
          const c = slot.get();
          if (c.kind === 'Var' || !isTensor(c.ty)) continue;
          const n = reads(c, new Set(), isArray).size;
          if (n >= 2 && n <= budget && (!best || n > best.n)) best = { slot, n };
          visit(c);
        }
      };
      visit(stmt.expr);
      if (!best) break; // nothing left to hoist; the planner reports the limit
      const { slot } = best as { slot: { get: () => IRExpr; set: (x: IRExpr) => void } };
      const sub = slot.get();
      const name = `${stmt.name}_part${++fresh}`;
      const cName = `${stmt.cName}_part${fresh}`;
      hoisted.push({ kind: 'Assign', name, cName, ty: sub.ty, expr: sub, span: stmt.span } as Assign);
      slot.set({ kind: 'Var', name, cName, ty: sub.ty, span: sub.span } as IRExpr);
    }
    out.push(...hoisted, stmt);
  }
  return out;
}
