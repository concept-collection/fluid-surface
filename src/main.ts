/**
 * The page: Krause & Voigt's perturbed sphere, run live. Each frame advances
 * the fluid one BDF-1 step on the GPU (src/fluid/movingStokes.ts, the
 * compiled models/stokes.m), then draws the surface coloured by a geometry
 * field and records the energies.
 *
 * Changing the working band or dt rebuilds the solver; changing r0 restarts
 * the run from rest.
 */
import { requestShtDevice } from './sht/sht.ts';
import { gaussNodesWeights } from './sht/gauss.ts';
import { perturbedSphere, productConfig } from './geom/geometry.ts';
import { MovingStokes, type FluidState } from './fluid/movingStokes.ts';
import { formatProfile, profileStep } from './fluid/profile.ts';
import { SphereScene } from './render/SphereScene.ts';
import { buildTopology, fillColors, fillFieldValues, fillPositions, type SphereMeshTopology } from './render/sphereMesh.ts';
import { colormaps } from './render/colormaps.ts';
import { Colorbar, floorRange } from './render/colorbar.ts';
import { LineChart } from './render/lineChart.ts';
import modelSource from '../models/stokes.m?raw';

const RE = 1;
const BE = 2;
type Field = 'H' | 'K' | 'sqrtg';
const FIELDS: { key: Field; label: string }[] = [
  { key: 'H', label: 'mean curvature H' },
  { key: 'K', label: 'Gaussian curvature K' },
  { key: 'sqrtg', label: 'area element' },
];
const BANDS = [12, 16, 24, 32];
const DTS = [0.0025, 0.005, 0.01, 0.02];

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const playBtn = $<HTMLButtonElement>('play');
const fieldSel = $<HTMLSelectElement>('field');
const bandSel = $<HTMLSelectElement>('band');
const dtSel = $<HTMLSelectElement>('dt');
const r0Input = $<HTMLInputElement>('r0');
const stats = $<HTMLDivElement>('stats');
const err = $<HTMLDivElement>('err');
const sphereBox = $<HTMLDivElement>('sphere');

for (const f of FIELDS) fieldSel.add(new Option(f.label, f.key));
for (const b of BANDS) bandSel.add(new Option(String(b), String(b)));
for (const d of DTS) dtSel.add(new Option(String(d), String(d)));
bandSel.value = '16';
dtSel.value = '0.005';
$<HTMLPreElement>('source').textContent = modelSource;

const colorbar = new Colorbar($<HTMLDivElement>('colorbar'));
const cmap = colormaps.viridis;
const chartEh = new LineChart($<HTMLDivElement>('chart-eh'), 'bending energy E_h');
const chartEkin = new LineChart($<HTMLDivElement>('chart-ekin'), 'kinetic energy E_kin');

let device: GPUDevice;

/** The solver for one band and dt, its mesh, and the run's state. */
interface Run {
  lmax: number;
  dt: number;
  ms: MovingStokes;
  topo: SphereMeshTopology;
  scene: SphereScene;
  positions: Float32Array;
  values: Float32Array;
  colors: Float32Array;
  state: FluidState;
  step: number;
  /** The last frame's numbers, for the stats line. */
  last: { ekin: number; eh: number; area: number; volume: number; its: number; ms: number; converged: boolean };
  field: Float32Array | null;
  volume0: number;
}

let run: Run | null = null;
let playing = false;

async function build(lmax: number, dt: number): Promise<void> {
  stats.textContent = `compiling the solver for band ${lmax}...`;
  const ms = await MovingStokes.create(device, lmax, {
    re: RE, be: BE, volume: true, dt, gmres: { tol: 1e-6, restart: 200, maxCycles: 4 },
  });
  const { nlat, nphi } = ms.cfg;
  const phi = Float64Array.from({ length: nphi }, (_, j) => (2 * Math.PI * j) / nphi);
  const topo = buildTopology(gaussNodesWeights(nlat).x, phi);
  run?.scene.dispose();
  run?.ms.destroy();
  sphereBox.replaceChildren();
  const background = getComputedStyle(document.documentElement).getPropertyValue('--sphere-bg').trim();
  const scene = new SphereScene(sphereBox, topo.numVertices, topo.indices, new Float32Array(topo.sphereRef), background || undefined);
  run = {
    lmax, dt, ms, topo, scene,
    positions: new Float32Array(topo.numVertices * 3),
    values: new Float32Array(topo.numVertices),
    colors: new Float32Array(topo.numVertices * 3),
    state: { U: [], P: new Float32Array(0), lam: 0 },
    step: 0,
    last: { ekin: 0, eh: 0, area: 0, volume: 0, its: 0, ms: 0, converged: true },
    field: null,
    volume0: 0,
  };
  resize();
  const ops = ms.fns.matvec.describe();
  $<HTMLPreElement>('ops').textContent = `${ops.length} GPU operations\n\n${ops.join('\n')}`;
  await reset();
  scene.fitCamera();
}

/** Back to rest on the perturbed sphere of the current r0. */
async function reset(): Promise<void> {
  if (!run) return;
  const r = run;
  const cfg = productConfig(r.lmax);
  r.ms.setSurface(perturbedSphere(cfg, r.lmax, Number(r0Input.value)));
  const zero = () => new Float32Array((cfg.lmax + 1) * (cfg.lmax + 2));
  r.state = { U: [zero(), zero(), zero()], P: zero(), lam: 0 };
  r.step = 0;
  chartEh.clear();
  chartEkin.clear();
  await observe(0, 0, true);
  r.volume0 = r.last.volume;
  draw();
}

/** Records the energies and reads what the frame draws. */
async function observe(its: number, ms: number, converged: boolean): Promise<void> {
  const r = run!;
  const d = await r.ms.diagnostics(r.state.U);
  r.last = { ekin: d.ekin, eh: (2 / BE) * d.h2, area: d.area, volume: d.volume, its, ms, converged };
  const t = r.step * r.dt;
  chartEh.push(t, r.last.eh);
  chartEkin.push(t, r.last.ekin);
  const grid = await r.ms.surfaceGrid();
  const n = r.ms.cfg.nlat * r.ms.cfg.nphi;
  const coords = new Float32Array(3 * n);
  for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) coords[3 * k + c] = grid[c][k];
  fillPositions(r.positions, coords, r.topo, 1);
  r.field = await r.ms.geometryField(fieldSel.value as Field);
}

function draw(): void {
  const r = run;
  if (!r || !r.field) return;
  r.scene.updatePositions(r.positions);
  fillFieldValues(r.values, r.field, r.topo);
  let lo = Infinity;
  let hi = -Infinity;
  for (const x of r.field) {
    lo = Math.min(lo, x);
    hi = Math.max(hi, x);
  }
  const shown = floorRange(lo, hi);
  fillColors(r.colors, r.values, shown.lo, shown.hi, cmap);
  r.scene.updateColors(r.colors);
  colorbar.update(cmap, shown.lo, shown.hi);
  const L = r.last;
  const t = r.step * r.dt;
  const { nlat, nphi, lmax } = r.ms.cfg;
  stats.innerHTML =
    `t <b>${t.toFixed(3)}</b> (step ${r.step}) &middot; E_h <b>${L.eh.toPrecision(6)}</b> &middot; ` +
    `E_kin <b>${L.ekin.toPrecision(4)}</b> &middot; area ${L.area.toPrecision(6)} &middot; ` +
    `volume ${L.volume.toPrecision(6)} (drift ${((L.volume - r.volume0) / r.volume0).toExponential(1)}) &middot; ` +
    `multiplier ${r.state.lam.toPrecision(4)}<br>` +
    `band ${r.lmax}, product band ${lmax} on ${nlat} x ${nphi}, dt ${r.dt} &middot; ` +
    (r.step ? `${L.its} GMRES iterations, ${L.ms.toFixed(0)} ms per step` : 'at rest') +
    (L.converged ? '' : ' &middot; <span class="warn">GMRES did not reach its tolerance</span>');
}

async function advance(): Promise<void> {
  const r = run!;
  const t0 = performance.now();
  const res = await r.ms.step(r.state);
  r.state = res.state;
  r.step++;
  const ms = performance.now() - t0;
  await observe(res.info.iterations, ms, res.info.converged);
  draw();
}

/** Advances one step with its parts timed (src/fluid/profile.ts). */
async function profile(): Promise<void> {
  const r = run!;
  const out = $<HTMLPreElement>('profile-out');
  out.textContent = 'profiling a step...';
  const t0 = performance.now();
  const { profile: p, state } = await profileStep(device, r.ms, r.state);
  r.state = state;
  r.step++;
  await observe(p.iterations, performance.now() - t0, p.converged);
  draw();
  const ts = device.features.has('timestamp-query') ? '' : ' (no timestamp queries on this device, so no per-op times)';
  out.textContent = [`profile of step ${r.step}, band ${r.lmax}${ts}:`, ...formatProfile(p)].join('\n');
}

/** The run loop: one step per frame while playing. */
let busy = false;
async function loop(): Promise<void> {
  if (busy) return;
  busy = true;
  try {
    while (playing && run) {
      await advance();
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  } catch (e) {
    playing = false;
    playBtn.textContent = 'Play';
    err.textContent = e instanceof Error ? e.message : String(e);
  } finally {
    busy = false;
  }
}

/** Runs a change with the loop stopped, then resumes it if it was playing. */
async function paused(task: () => Promise<void>): Promise<void> {
  const was = playing;
  playing = false;
  while (busy) await new Promise((resolve) => setTimeout(resolve, 10));
  try {
    err.textContent = '';
    await task();
  } catch (e) {
    err.textContent = e instanceof Error ? e.message : String(e);
  }
  if (was) {
    playing = true;
    void loop();
  }
}

function resize(): void {
  if (!run) return;
  const b = sphereBox.getBoundingClientRect();
  run.scene.resize(Math.round(b.width), Math.round(b.height));
}

function showR0(): void {
  $<HTMLSpanElement>('r0val').textContent = Number(r0Input.value).toFixed(2);
}

async function start(): Promise<void> {
  try {
    device = await requestShtDevice();
  } catch (e) {
    err.textContent = `WebGPU is required: ${e instanceof Error ? e.message : String(e)}`;
    return;
  }
  showR0();
  const rebuild = () => void paused(() => build(Number(bandSel.value), Number(dtSel.value)));
  await paused(() => build(Number(bandSel.value), Number(dtSel.value)));
  playBtn.addEventListener('click', () => {
    playing = !playing;
    playBtn.textContent = playing ? 'Pause' : 'Play';
    if (playing) void loop();
  });
  $<HTMLButtonElement>('reset').addEventListener('click', () => void paused(reset));
  $<HTMLButtonElement>('profile').addEventListener('click', () => void paused(profile));
  r0Input.addEventListener('input', showR0);
  r0Input.addEventListener('change', () => void paused(reset));
  dtSel.addEventListener('change', rebuild);
  bandSel.addEventListener('change', rebuild);
  fieldSel.addEventListener('change', () =>
    void paused(async () => {
      if (!run) return;
      run.field = await run.ms.geometryField(fieldSel.value as Field);
      draw();
    }),
  );
  $<HTMLButtonElement>('view').addEventListener('click', () => run?.scene.resetCamera());
  new ResizeObserver(resize).observe(sphereBox);
}

void start();
