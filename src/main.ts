/**
 * The first page: Krause & Voigt's perturbed sphere, coloured by a geometry
 * field that lib/geometry.m computes on the GPU.
 *
 * Moving r0 rebuilds the position's coefficients on the host (f64, as the C++
 * solver does) and reruns the compiled geometry; changing the working band
 * recompiles it for the new product grid.
 */
import { requestShtDevice } from './sht/sht.ts';
import { GEOMETRY_OUTPUTS, SurfaceGeometry, perturbedSphere, productConfig, type GeometryOutput } from './geom/geometry.ts';
import { SphereScene } from './render/SphereScene.ts';
import { buildTopology, fillColors, fillFieldValues, fillPositions, type SphereMeshTopology } from './render/sphereMesh.ts';
import { colormaps } from './render/colormaps.ts';
import { Colorbar, floorRange } from './render/colorbar.ts';
import geometrySource from '../lib/geometry.m?raw';

const FIELDS: { key: GeometryOutput; label: string }[] = [
  { key: 'H', label: 'mean curvature H' },
  { key: 'K', label: 'Gaussian curvature K' },
  { key: 'sqrtg', label: 'area element sqrt(det g)' },
];
const BANDS = [12, 16, 24, 32, 48];

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const fieldSel = $<HTMLSelectElement>('field');
const r0Input = $<HTMLInputElement>('r0');
const r0Val = $<HTMLSpanElement>('r0val');
const bandSel = $<HTMLSelectElement>('band');
const stats = $<HTMLDivElement>('stats');
const err = $<HTMLDivElement>('err');
const sphereBox = $<HTMLDivElement>('sphere');

for (const f of FIELDS) fieldSel.add(new Option(f.label, f.key));
for (const b of BANDS) bandSel.add(new Option(String(b), String(b)));
bandSel.value = '32';
$<HTMLPreElement>('source').textContent = geometrySource;

const colorbar = new Colorbar($<HTMLDivElement>('colorbar'));
const cmap = colormaps.viridis;

interface View {
  lmax: number;
  geo: SurfaceGeometry;
  topo: SphereMeshTopology;
  scene: SphereScene;
  positions: Float32Array;
  values: Float32Array;
  colors: Float32Array;
  /** Fit the camera after the first computation on this grid. */
  needsFit: boolean;
}

let device: GPUDevice;
let view: View | null = null;
/** The geometry's fields from the last run, and the time it took. */
let fields: Partial<Record<GeometryOutput, Float32Array>> = {};
let lastMs = 0;

async function buildView(lmax: number): Promise<void> {
  const cfg = productConfig(lmax);
  const geo = await SurfaceGeometry.create(device, cfg);
  const phi = Float64Array.from({ length: cfg.nphi }, (_, j) => (2 * Math.PI * j) / cfg.nphi);
  const topo = buildTopology(geo.sht.cosTheta, phi);
  view?.scene.dispose();
  view?.geo.destroy();
  sphereBox.replaceChildren();
  const background = getComputedStyle(document.documentElement).getPropertyValue('--sphere-bg').trim();
  const scene = new SphereScene(sphereBox, topo.numVertices, topo.indices, new Float32Array(topo.sphereRef), background || undefined);
  view = {
    lmax,
    geo,
    topo,
    scene,
    positions: new Float32Array(topo.numVertices * 3),
    values: new Float32Array(topo.numVertices),
    colors: new Float32Array(topo.numVertices * 3),
    needsFit: true,
  };
  resize();
  $<HTMLPreElement>('ops').textContent = geo.fn.describe().join('\n');
}

/** Rebuild the surface for the current r0 and rerun the geometry. */
async function compute(): Promise<void> {
  if (!view) return;
  const v = view;
  const r0 = Number(r0Input.value);
  const X = perturbedSphere(v.geo.cfg, v.lmax, r0);
  const t0 = performance.now();
  v.geo.compute(X);
  await device.queue.onSubmittedWorkDone();
  lastMs = performance.now() - t0;
  const next: Partial<Record<GeometryOutput, Float32Array>> = {};
  for (const f of FIELDS) next[f.key] = await v.geo.read(f.key);
  fields = next;

  // The surface itself, synthesized from the same coefficients.
  // One at a time: ShtPlan's CPU convenience shares one readback buffer.
  const grid: Float32Array[] = [];
  for (const x of X) grid.push(await v.geo.sht.synth(x));
  const n = v.geo.cfg.nlat * v.geo.cfg.nphi;
  const coords = new Float32Array(3 * n);
  for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) coords[3 * k + c] = grid[c][k];
  fillPositions(v.positions, coords, v.topo, 1);
  v.scene.updatePositions(v.positions);
  if (v.needsFit) {
    v.scene.fitCamera();
    v.needsFit = false;
  }
  paint();
}

/** Colour the surface by the selected field. */
function paint(): void {
  const v = view;
  const key = fieldSel.value as GeometryOutput;
  const data = fields[key];
  if (!v || !data) return;
  fillFieldValues(v.values, data, v.topo);
  let lo = Infinity;
  let hi = -Infinity;
  for (const x of data) {
    lo = Math.min(lo, x);
    hi = Math.max(hi, x);
  }
  // The colour range is floored so a near-constant field (H on the sphere)
  // does not stretch its roundoff over the whole colormap; the stats line
  // reports the data's own range.
  const shown = floorRange(lo, hi);
  fillColors(v.colors, v.values, shown.lo, shown.hi, cmap);
  v.scene.updateColors(v.colors);
  colorbar.update(cmap, shown.lo, shown.hi);
  const { nlat, nphi, lmax } = v.geo.cfg;
  stats.innerHTML =
    `working band <b>${v.lmax}</b>, product band ${lmax} on ${nlat} x ${nphi} &middot; ` +
    `geometry.m: <b>${v.geo.fn.describe().length}</b> GPU operations, ${GEOMETRY_OUTPUTS.length} fields, ` +
    `<b>${lastMs.toFixed(2)} ms</b> including submit &middot; ` +
    `${FIELDS.find((f) => f.key === key)!.label} in [${lo.toPrecision(7)}, ${hi.toPrecision(7)}]`;
}

function resize(): void {
  if (!view) return;
  const r = sphereBox.getBoundingClientRect();
  view.scene.resize(Math.round(r.width), Math.round(r.height));
}

/**
 * One worker at a time. Input arriving while it runs only sets a flag, so a
 * burst of slider moves costs one more computation, and a band change is
 * never lost behind one.
 */
let busy = false;
let wantBuild = false;
let wantCompute = false;
async function request(build: boolean): Promise<void> {
  wantBuild ||= build;
  wantCompute = true;
  if (busy) return;
  busy = true;
  try {
    while (wantBuild || wantCompute) {
      if (wantBuild) {
        wantBuild = false;
        await buildView(Number(bandSel.value));
      }
      wantCompute = false;
      await compute();
    }
    err.textContent = '';
  } catch (e) {
    err.textContent = e instanceof Error ? e.message : String(e);
  } finally {
    busy = false;
  }
}

function showR0(): void {
  r0Val.textContent = Number(r0Input.value).toFixed(2);
}

async function start(): Promise<void> {
  try {
    device = await requestShtDevice();
  } catch (e) {
    err.textContent = `WebGPU is required: ${e instanceof Error ? e.message : String(e)}`;
    return;
  }
  showR0();
  await request(true);
  r0Input.addEventListener('input', () => {
    showR0();
    void request(false);
  });
  fieldSel.addEventListener('change', paint);
  bandSel.addEventListener('change', () => void request(true));
  $<HTMLButtonElement>('reset').addEventListener('click', () => view?.scene.resetCamera());
  new ResizeObserver(resize).observe(sphereBox);
}

void start();
