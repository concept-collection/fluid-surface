/** Reads a ref_dump directory (the C++ solver's format): a manifest of f64 arrays and scalars. */
import { readFileSync } from 'node:fs';

export interface Dump {
  scalars: Record<string, number>;
  arrays: Record<string, Float64Array>;
}

export function readDump(dir: URL): Dump {
  const scalars: Record<string, number> = {};
  const arrays: Record<string, Float64Array> = {};
  for (const line of readFileSync(new URL('manifest.txt', dir), 'utf8').split('\n')) {
    const f = line.trim().split(/\s+/);
    if (f[0] === 'scalar') scalars[f[1]] = Number(f[2]);
    if (f[0] === 'array') {
      const buf = readFileSync(new URL(`${f[1]}.bin`, dir));
      arrays[f[1]] = new Float64Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    }
  }
  return { scalars, arrays };
}
