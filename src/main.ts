/**
 * The page: Krause & Voigt's perturbed sphere, run live. Each frame advances
 * the fluid one BDF-1 step on the GPU (src/fluid/movingStokes.ts, the
 * compiled models/stokes.m), then draws the surface coloured by a geometry
 * field and records the energies.
 *
 * Changing lmax (the working band) or dt rebuilds the solver; changing r0 restarts
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
import { arrowSegments } from './render/arrows.ts';
import modelSource from '../models/stokes.m?raw';
import ref16 from './fluid/reference/fluid_run-perturbed_sphere-r0.4-L16-dt0.005.csv?raw';
import ref32 from './fluid/reference/fluid_run-perturbed_sphere-r0.4-L32-dt0.005.csv?raw';

const RE = 1;
const BE = 2;
type Field = 'H' | 'K' | 'sqrtg' | 'speed' | 'un';
const FIELDS: { key: Field; label: string }[] = [
  { key: 'H', label: 'mean curvature H' },
  { key: 'K', label: 'Gaussian curvature K' },
  { key: 'sqrtg', label: 'area element' },
  { key: 'speed', label: 'speed |u|' },
  { key: 'un', label: 'normal velocity u . n (outward)' },
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
const arrowsBox = $<HTMLInputElement>('arrows');
const licBox = $<HTMLInputElement>('lic');
const refFile = $<HTMLInputElement>('ref-file');
const refNote = $<HTMLSpanElement>('ref-note');
const cmdLine = $<HTMLElement>('cmd');

/** A fluid_run diagnostics.csv: time, kinetic and bending energies by step. */
interface Diagnostics {
  t: number[];
  ekin: number[];
  eh: number[];
}

function parseDiagnostics(csv: string): Diagnostics {
  const [head, ...rows] = csv.trim().split(/\r?\n/);
  const cols = head.split(',').map((c) => c.trim());
  const col = (name: string) => {
    const i = cols.indexOf(name);
    if (i < 0) throw new Error(`the CSV has no '${name}' column (fluid_run's diagnostics.csv has t, e_kin and e_h)`);
    return rows.map((r) => Number(r.split(',')[i]));
  };
  return { t: col('t'), ekin: col('e_kin'), eh: col('e_h') };
}

/** The C++ runs shipped with the page: fluid_run, BDF-1, dt 0.005, r0 0.4, tol 1e-8. */
const BUILT_IN: Record<number, Diagnostics> = { 16: parseDiagnostics(ref16), 32: parseDiagnostics(ref32) };
/** A diagnostics.csv the viewer loaded, which replaces the built-in reference. */
let loaded: { name: string; d: Diagnostics } | null = null;

/** Shows the reference that matches the current settings, if any, and the
 *  fluid_run command that would make one. */
function updateReference(): void {
  const lmax = Number(bandSel.value);
  const dt = Number(dtSel.value);
  const r0 = Number(r0Input.value);
  cmdLine.textContent =
    `fluid_run -example perturbed_sphere -r0 ${r0} -Re ${RE} -Be ${BE} -order 1 -lmax ${lmax} -dt ${dt} ` +
    `-T 0.5 -relabel_every 0 -frames 0 -save_dir out`;
  let ref: Diagnostics | null = null;
  let label = '';
  if (loaded) {
    ref = loaded.d;
    label = `C++ (${loaded.name})`;
    refNote.textContent = `Dashed: ${loaded.name}, loaded here. It is compared as is, so it should come from the command below.`;
  } else if (BUILT_IN[lmax] && dt === 0.005 && Math.abs(r0 - 0.4) < 1e-9) {
    ref = BUILT_IN[lmax];
    label = 'C++ fluid_run';
    refNote.textContent = `Dashed: the C++ solver's fluid_run with the same settings (lmax ${lmax}, dt 0.005, r0 0.4, BDF-1, GMRES tol 1e-8), shipped with the page.`;
  } else {
    refNote.textContent = 'No C++ reference for these settings (the page has lmax 16 and 32 at dt 0.005, r0 0.4). Run the command below and load its diagnostics.csv to compare.';
  }
  chartEh.setReference(ref && { ts: ref.t, ys: ref.eh }, label);
  chartEkin.setReference(ref && { ts: ref.t, ys: ref.ekin }, label);
}

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
  /** sin(theta) of each latitude ring, for spacing the arrows. */
  st: Float64Array;
  /** The surface on the grid (x, y, z per point) and the velocity there. */
  coords: Float32Array;
  vel: { t: Float32Array[]; un: Float32Array; n: Float32Array[] } | null;
}

let run: Run | null = null;
let playing = false;

async function build(lmax: number, dt: number): Promise<void> {
  stats.textContent = `compiling the solver for lmax ${lmax}...`;
  const ms = await MovingStokes.create(device, lmax, {
    re: RE, be: BE, volume: true, dt, gmres: { tol: 1e-6, restart: 200, maxCycles: 4, device: true },
  });
  const { nlat, nphi } = ms.cfg;
  const phi = Float64Array.from({ length: nphi }, (_, j) => (2 * Math.PI * j) / nphi);
  const ct = gaussNodesWeights(nlat).x;
  const topo = buildTopology(ct, phi);
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
    st: Float64Array.from(ct, (c) => Math.sqrt(1 - c * c)),
    coords: new Float32Array(3 * nlat * nphi),
    vel: null,
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
  updateReference();
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
  for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) r.coords[3 * k + c] = grid[c][k];
  fillPositions(r.positions, r.coords, r.topo, 1);
  r.vel = await r.ms.velocityField(r.state.U);
  r.field = await fieldValues(fieldSel.value as Field);
}

/** The values the surface is coloured by. */
async function fieldValues(f: Field): Promise<Float32Array> {
  const r = run!;
  if (f === 'H' || f === 'K' || f === 'sqrtg') return r.ms.geometryField(f);
  const v = r.vel!;
  if (f === 'un') return v.un;
  return v.un.map((un, k) => Math.hypot(v.t[0][k], v.t[1][k], v.t[2][k], un));
}

/** Arrows of the tangential velocity, the longest 0.3 times the surface's radius. */
function drawArrows(): string {
  const r = run!;
  if (!arrowsBox.checked || !r.vel) {
    r.scene.setLines(null);
    return '';
  }
  const size = Math.sqrt(r.last.area / (4 * Math.PI));
  const { nlat, nphi } = r.ms.cfg;
  const { segments, vmax } = arrowSegments({ nlat, nphi, st: r.st }, r.coords, r.vel.t, r.vel.n, 0.3 * size);
  r.scene.setLines(segments);
  return vmax > 0 ? ` &middot; arrows: tangential velocity, longest ${vmax.toPrecision(3)}` : '';
}

/** Streamlines of the tangential velocity over the colouring (src/render/lic.ts). */
function drawStreamlines(): void {
  const r = run!;
  if (!licBox.checked || !r.vel) {
    r.scene.setStreamlines(null);
    return;
  }
  const nv = r.topo.numVertices;
  const comp = new Float32Array(nv);
  const out = new Float32Array(3 * nv);
  for (let c = 0; c < 3; c++) {
    fillFieldValues(comp, r.vel.t[c], r.topo);
    for (let k = 0; k < nv; k++) out[3 * k + c] = comp[k];
  }
  let vmax = 0;
  const [t1, t2, t3] = r.vel.t;
  for (let k = 0; k < t1.length; k++) vmax = Math.max(vmax, Math.hypot(t1[k], t2[k], t3[k]));
  r.scene.setStreamlines(out, vmax);
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
  const arrows = drawArrows();
  drawStreamlines();
  const L = r.last;
  const t = r.step * r.dt;
  const { nlat, nphi, lmax } = r.ms.cfg;
  stats.innerHTML =
    `t <b>${t.toFixed(3)}</b> (step ${r.step}) &middot; E_h <b>${L.eh.toPrecision(6)}</b> &middot; ` +
    `E_kin <b>${L.ekin.toPrecision(4)}</b> &middot; area ${L.area.toPrecision(6)} &middot; ` +
    `volume ${L.volume.toPrecision(6)} (drift ${((L.volume - r.volume0) / r.volume0).toExponential(1)}) &middot; ` +
    `multiplier ${r.state.lam.toPrecision(4)}${arrows}<br>` +
    `lmax ${r.lmax}, products at lmax ${lmax} on ${nlat} x ${nphi}, dt ${r.dt} &middot; ` +
    (r.step ? `${L.its} GMRES iterations, ${stepsPerSecond().toFixed(1)} steps per second` : 'at rest') +
    (L.converged ? '' : ' &middot; <span class="warn">GMRES did not reach its tolerance</span>');
}

async function advance(): Promise<void> {
  const r = run!;
  const t0 = performance.now();
  const res = await r.ms.step(r.state);
  r.state = res.state;
  r.step++;
  const ms = performance.now() - t0;
  if (playing) finished.push(performance.now());
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
  out.textContent = [`profile of step ${r.step}, lmax ${r.lmax}${ts}:`, ...formatProfile(p)].join('\n');
}

/** When the last steps of the current play finished (ms), for the rate. */
const finished: number[] = [];

/** Steps per second: while playing, the rate of the last few seconds, which
 *  includes drawing; otherwise from the last step's own time. */
function stepsPerSecond(): number {
  const now = performance.now();
  while (finished.length > 2 && now - finished[0] > 5000) finished.shift();
  if (playing && finished.length >= 3) return (1000 * (finished.length - 1)) / (finished[finished.length - 1] - finished[0]);
  return run && run.last.ms > 0 ? 1000 / run.last.ms : 0;
}

/** The run loop: one step per frame while playing. */
let busy = false;
async function loop(): Promise<void> {
  if (busy) return;
  busy = true;
  finished.length = 0;
  try {
    while (playing && run) {
      await advance();
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  } catch (e) {
    playing = false;
    playBtn.textContent = 'Play';
    showError(e);
  } finally {
    busy = false;
  }
}

/** The device, for an error report: the adapter and the limits this app leans on. */
function deviceReport(): string {
  if (!device) return '';
  const info = (device as GPUDevice & { adapterInfo?: GPUAdapterInfo }).adapterInfo;
  const L = device.limits;
  const adapter = info ? [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') : 'unknown adapter';
  return `\ndevice: ${adapter}; ${navigator.userAgent}\n` +
    `limits: ${L.maxStorageBuffersPerShaderStage} storage buffers per stage, ` +
    `storage binding ${L.maxStorageBufferBindingSize}, buffer ${L.maxBufferSize}, ` +
    `workgroup storage ${L.maxComputeWorkgroupStorageSize}, invocations ${L.maxComputeInvocationsPerWorkgroup}, ` +
    `features: ${[...device.features].join(', ')}`;
}

/** Shows an error, with the device it happened on. The first GPU validation
 *  error is the one that says what went wrong; later ones follow from it. */
function showError(e: unknown): void {
  if (err.textContent) return;
  err.textContent = (e instanceof Error ? e.message : String(e)) + deviceReport();
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
    showError(e);
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
  device.addEventListener('uncapturederror', (e) => showError(`GPU error: ${(e as GPUUncapturedErrorEvent).error.message}`));
  void device.lost.then((info) => showError(`GPU device lost: ${info.message}`));
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
      run.field = await fieldValues(fieldSel.value as Field);
      draw();
    }),
  );
  $<HTMLButtonElement>('view').addEventListener('click', () => run?.scene.resetCamera());
  arrowsBox.addEventListener('change', draw);
  licBox.addEventListener('change', draw);
  refFile.addEventListener('change', async () => {
    const file = refFile.files?.[0];
    if (!file) return;
    try {
      loaded = { name: file.name, d: parseDiagnostics(await file.text()) };
    } catch (e) {
      loaded = null;
      showError(e);
    }
    updateReference();
  });
  $<HTMLButtonElement>('ref-clear').addEventListener('click', () => {
    loaded = null;
    refFile.value = '';
    updateReference();
  });
  new ResizeObserver(resize).observe(sphereBox);
}

void start();
