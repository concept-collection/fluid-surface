/**
 * A small single-series line chart in SVG, for a quantity over simulated time:
 * a 2px line, recessive axes with the range's ends labelled, the latest value
 * beside the title, and a crosshair with a readout on hover. One series per
 * chart, so quantities of different scale go in separate charts rather than on
 * two y-axes.
 */

const NS = 'http://www.w3.org/2000/svg';
const W = 320;
const H = 120;
const PAD = { left: 52, right: 10, top: 8, bottom: 22 };

const el = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
};

interface Range {
  t0: number;
  t1: number;
  lo: number;
  hi: number;
}

const fmt = (v: number): string => {
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toPrecision(5).replace(/\.?0+$/, '');
};

export class LineChart {
  #ts: number[] = [];
  #ys: number[] = [];
  #svg: SVGSVGElement;
  #path: SVGPathElement;
  #value: HTMLElement;
  #yLabels: SVGTextElement[];
  #xLabels: SVGTextElement[];
  #cross: SVGLineElement;
  #dot: SVGCircleElement;
  #tip: HTMLElement;
  #unit: string;

  constructor(container: HTMLElement, title: string, unit = '') {
    this.#unit = unit;
    container.classList.add('chart');
    const head = document.createElement('div');
    head.className = 'chart-head';
    const name = document.createElement('span');
    name.className = 'chart-title';
    name.textContent = title;
    this.#value = document.createElement('span');
    this.#value.className = 'chart-value';
    head.append(name, this.#value);

    const box = document.createElement('div');
    box.className = 'chart-box';
    this.#svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': title }) as SVGSVGElement;
    const x0 = PAD.left;
    const x1 = W - PAD.right;
    const y0 = H - PAD.bottom;
    this.#svg.append(
      el('line', { x1: x0, x2: x1, y1: y0, y2: y0, class: 'chart-axis' }),
      el('line', { x1: x0, x2: x1, y1: PAD.top, y2: PAD.top, class: 'chart-grid' }),
    );
    this.#yLabels = [PAD.top + 4, y0].map((y) =>
      el('text', { x: x0 - 6, y, 'text-anchor': 'end', class: 'chart-label' }) as SVGTextElement,
    );
    this.#xLabels = [
      el('text', { x: x0, y: H - 6, 'text-anchor': 'start', class: 'chart-label' }) as SVGTextElement,
      el('text', { x: x1, y: H - 6, 'text-anchor': 'end', class: 'chart-label' }) as SVGTextElement,
    ];
    this.#path = el('path', { class: 'chart-line', fill: 'none' }) as SVGPathElement;
    this.#cross = el('line', { y1: PAD.top, y2: y0, class: 'chart-cross', visibility: 'hidden' }) as SVGLineElement;
    this.#dot = el('circle', { r: 4, class: 'chart-dot', visibility: 'hidden' }) as SVGCircleElement;
    // The hit target is the whole plot area, not the line.
    const hit = el('rect', { x: x0, y: PAD.top, width: x1 - x0, height: y0 - PAD.top, fill: 'transparent' });
    this.#svg.append(...this.#yLabels, ...this.#xLabels, this.#path, this.#cross, this.#dot, hit);
    this.#tip = document.createElement('div');
    this.#tip.className = 'chart-tip';
    this.#tip.hidden = true;
    box.append(this.#svg, this.#tip);
    container.append(head, box);

    hit.addEventListener('pointermove', (e) => this.#hover(e as PointerEvent));
    hit.addEventListener('pointerleave', () => this.#hideHover());
  }

  clear(): void {
    this.#ts = [];
    this.#ys = [];
    this.#draw();
  }

  push(t: number, y: number): void {
    this.#ts.push(t);
    this.#ys.push(y);
    this.#draw();
  }

  #range(): Range {
    const t0 = this.#ts[0] ?? 0;
    const t1 = Math.max(this.#ts[this.#ts.length - 1] ?? 1, t0 + 1e-9);
    let lo = Math.min(...this.#ys);
    let hi = Math.max(...this.#ys);
    if (!(hi > lo)) {
      const pad = Math.abs(hi) * 1e-3 || 1e-6;
      lo -= pad;
      hi += pad;
    }
    return { t0, t1, lo, hi };
  }

  #xy(i: number, r: Range): [number, number] {
    const x = PAD.left + ((this.#ts[i] - r.t0) / (r.t1 - r.t0)) * (W - PAD.left - PAD.right);
    const y = H - PAD.bottom - ((this.#ys[i] - r.lo) / (r.hi - r.lo)) * (H - PAD.bottom - PAD.top);
    return [x, y];
  }

  #draw(): void {
    const n = this.#ts.length;
    this.#value.textContent = n ? `${fmt(this.#ys[n - 1])}${this.#unit}` : '';
    if (!n) {
      this.#path.setAttribute('d', '');
      for (const l of [...this.#yLabels, ...this.#xLabels]) l.textContent = '';
      return;
    }
    const r = this.#range();
    let d = '';
    for (let i = 0; i < n; i++) {
      const [x, y] = this.#xy(i, r);
      d += `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
    }
    this.#path.setAttribute('d', d);
    this.#yLabels[0].textContent = fmt(r.hi);
    this.#yLabels[1].textContent = fmt(r.lo);
    this.#xLabels[0].textContent = `t = ${fmt(r.t0)}`;
    this.#xLabels[1].textContent = `${fmt(r.t1)}`;
  }

  #hover(e: PointerEvent): void {
    const n = this.#ts.length;
    if (!n) return;
    const box = this.#svg.getBoundingClientRect();
    const sx = ((e.clientX - box.left) / box.width) * W;
    const r = this.#range();
    const t = r.t0 + ((sx - PAD.left) / (W - PAD.left - PAD.right)) * (r.t1 - r.t0);
    let i = 0;
    for (let k = 1; k < n; k++) if (Math.abs(this.#ts[k] - t) < Math.abs(this.#ts[i] - t)) i = k;
    const [x, y] = this.#xy(i, r);
    for (const [a, v] of [['x1', x], ['x2', x]] as const) this.#cross.setAttribute(a, String(v));
    this.#dot.setAttribute('cx', String(x));
    this.#dot.setAttribute('cy', String(y));
    this.#cross.setAttribute('visibility', 'visible');
    this.#dot.setAttribute('visibility', 'visible');
    this.#tip.hidden = false;
    this.#tip.textContent = `t = ${fmt(this.#ts[i])}: ${fmt(this.#ys[i])}${this.#unit}`;
    const left = (x / W) * box.width;
    this.#tip.style.left = `${Math.min(Math.max(left, 60), box.width - 60)}px`;
  }

  #hideHover(): void {
    this.#cross.setAttribute('visibility', 'hidden');
    this.#dot.setAttribute('visibility', 'hidden');
    this.#tip.hidden = true;
  }
}
