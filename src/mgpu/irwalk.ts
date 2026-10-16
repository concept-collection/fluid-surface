/**
 * The sub-expressions of an IR expression that the WGSL backend reads, with a
 * setter for each, so every pass walks the same kinds: operators, calls, a
 * slice's base and literal indices, and a concatenation's cells.
 */
import type { IRExpr } from 'numbl-src/numbl-core/jit/lowering/ir.ts';

export interface ChildSlot {
  get(): IRExpr;
  set(x: IRExpr): void;
}

export function childSlots(e: IRExpr): ChildSlot[] {
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
    case 'IndexSlice':
      return [{ get: () => e.base, set: (x) => (e.base = x) }];
    case 'TensorConcat':
      return e.cells.flatMap((row, r) =>
        row.map((_, c) => ({ get: () => e.cells[r][c], set: (x: IRExpr) => (e.cells[r][c] = x) })),
      );
    default:
      return [];
  }
}

/** Calls `visit` on every Var the expression reads. */
export function forEachVar(e: IRExpr, visit: (v: Extract<IRExpr, { kind: 'Var' }>) => void): void {
  if (e.kind === 'Var') {
    visit(e);
    return;
  }
  for (const slot of childSlots(e)) forEachVar(slot.get(), visit);
}

/** A copy of an expression with each Var replaced by `f(v)` (the rest shared
 *  or shallow-copied; the planner treats expressions as read-only). */
export function mapVars(e: IRExpr, f: (v: Extract<IRExpr, { kind: 'Var' }>) => IRExpr): IRExpr {
  switch (e.kind) {
    case 'Var':
      return f(e);
    case 'Binary':
      return { ...e, left: mapVars(e.left, f), right: mapVars(e.right, f) };
    case 'Unary':
      return { ...e, operand: mapVars(e.operand, f) };
    case 'Call':
      return { ...e, args: e.args.map((a) => mapVars(a, f)) };
    case 'MemberLoad':
      return { ...e, base: mapVars(e.base, f) };
    case 'IndexSlice':
      return { ...e, base: mapVars(e.base, f) };
    case 'TensorConcat':
      return { ...e, cells: e.cells.map((row) => row.map((c) => mapVars(c, f))) };
    default:
      return e;
  }
}
