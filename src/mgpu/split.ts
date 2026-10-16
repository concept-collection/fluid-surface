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
 *
 * What counts is every buffer a kernel binds: the arrays, and also a
 * single-element value that lives in a buffer (a `dot` result, or a scalar
 * input such as the volume multiplier), which the planner knows; it passes
 * `isBuffer`.
 */
import { isMultiElement } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import type { Assign, IRExpr, IRStmt } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
import type { Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import { childSlots, forEachVar } from './irwalk.ts';

const isTensor = (t: Type): boolean => t.kind === 'Numeric' && isMultiElement(t);

/** Distinct array variables an expression reads. */
function reads(e: IRExpr, out: Set<string>, isArray: (v: IRExpr & { kind: 'Var' }) => boolean): Set<string> {
  forEachVar(e, (v) => {
    if (isArray(v)) out.add(v.cName);
  });
  return out;
}

const children = childSlots;

/**
 * Returns `body` with over-wide element-wise statements split, recursing into
 * loop bodies. `budget` is the most arrays one kernel may read.
 */
export function splitWideKernels(
  body: IRStmt[],
  budget: number,
  isBuffer: (v: IRExpr & { kind: 'Var' }) => boolean = (v) => isTensor(v.ty),
): IRStmt[] {
  let fresh = 0;
  const isArray = isBuffer;
  const out: IRStmt[] = [];
  for (const stmt of body) {
    if (stmt.kind === 'For') {
      stmt.body = splitWideKernels(stmt.body, budget, isBuffer);
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
