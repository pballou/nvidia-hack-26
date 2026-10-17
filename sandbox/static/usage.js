/* Usage page: the monthly bar chart, drawn on one <canvas> (no DOM element per bar).
   Data: <script id="usage-data"> {months: [{id, short, label, kwh}], selected}.
   Clicking a bar's column calls Acme.act('usage.select_month', {month}); the page reloads
   with that bar highlighted and the details panel filled in.
   Sizes scale with Acme.rootPx(), so the chart follows Settings > Text size.
   Test hook (oracle only): Acme.hooks.usageBars = [{id, x, y}], bar centers in CSS px
   relative to the canvas top-left. */
(function () {
  'use strict';

  const canvas = document.getElementById('usage-chart');
  const dataEl = document.getElementById('usage-data');
  if (!canvas || !dataEl) return;
  const { months, selected } = JSON.parse(dataEl.textContent);
  let hit = { cols: [], base: 0, mx: 0, my: 0, minH: 0 };

  function niceStep(max, count) {
    const raw = max / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
  }

  // Column path: rounded top corners, square at the baseline. open=true leaves the bottom open.
  function columnPath(ctx, x, y, w, base, r, open) {
    r = Math.max(0, Math.min(r, w / 2, base - y));
    ctx.beginPath();
    ctx.moveTo(x, base);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, base);
    if (!open) ctx.closePath();
  }

  function draw() {
    const rem = Acme.rootPx();
    const box = canvas.getBoundingClientRect();
    const W = Math.round(box.width);
    const H = Math.round(box.height);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const css = getComputedStyle(canvas);
    const color = (name) => css.getPropertyValue(name).trim();
    const font = (weight, size) => `${weight} ${Math.round(size * rem)}px ${css.fontFamily}`;
    const hc = document.documentElement.classList.contains('hc');
    const fmt = (v) => v.toLocaleString('en-US');

    const max = Math.max(...months.map((m) => m.kwh));
    const step = niceStep(max, 5);
    const top = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = 0; v <= top; v += step) ticks.push(v);

    ctx.font = font(500, 0.8125);
    const tickW = Math.max(...ticks.map((v) => ctx.measureText(fmt(v)).width));
    const plotL = Math.ceil(tickW + 1.125 * rem);
    const plotR = W - Math.round(0.25 * rem);
    const plotT = Math.round(1.75 * rem);
    const base = H - Math.round(3 * rem);
    const slot = (plotR - plotL) / months.length;
    const barW = Math.round(Math.min(Math.max(2.5 * rem, slot * 0.6), slot - 0.375 * rem));
    const yOf = (v) => Math.round(base - (v / top) * (base - plotT));

    // Gridlines and y-axis ticks.
    ctx.lineWidth = 1;
    ctx.strokeStyle = color('--chart-grid');
    ctx.fillStyle = color('--chart-tick');
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const tickX = plotL - Math.round(0.625 * rem);
    for (const v of ticks) {
      const y = yOf(v) + 0.5;
      if (v > 0) {
        ctx.beginPath();
        ctx.moveTo(plotL, y);
        ctx.lineTo(plotR, y);
        ctx.stroke();
      }
      ctx.fillText(fmt(v), tickX, y);
    }
    ctx.font = font(700, 0.75);
    ctx.textBaseline = 'top';
    ctx.fillText('kWh', tickX, Math.round(0.125 * rem));

    // Bars.
    const cols = months.map((m, i) => {
      const x0 = Math.round(plotL + slot * i + (slot - barW) / 2);
      const y = yOf(m.kwh);
      ctx.fillStyle = color(m.id === selected ? '--chart-bar-selected' : '--chart-bar');
      columnPath(ctx, x0, y, barW, base, 0.25 * rem, false);
      ctx.fill();
      return { id: m.id, x0, x1: x0 + barW, top: y };
    });

    // Selected bar: an outline ring with a small gap around the column.
    const sel = cols.find((c) => c.id === selected);
    if (sel) {
      const gap = Math.round(0.1875 * rem);
      ctx.lineWidth = hc ? 3 : 2;
      ctx.strokeStyle = color('--chart-ring');
      const off = gap + ctx.lineWidth / 2;
      columnPath(ctx, sel.x0 - off, sel.top - off, barW + 2 * off, base, 0.4375 * rem, true);
      ctx.stroke();
    }

    // Baseline.
    ctx.lineWidth = hc ? 2 : 1;
    ctx.strokeStyle = color('--chart-axis');
    ctx.beginPath();
    ctx.moveTo(plotL, base + ctx.lineWidth / 2);
    ctx.lineTo(plotR, base + ctx.lineWidth / 2);
    ctx.stroke();

    // Month labels, plus the year under the first month and under January.
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    months.forEach((m, i) => {
      const x = (cols[i].x0 + cols[i].x1) / 2;
      const isSel = m.id === selected;
      ctx.font = font(isSel ? 800 : 600, 0.875);
      ctx.fillStyle = color(isSel ? '--chart-label' : '--chart-tick');
      ctx.fillText(m.short, x, base + Math.round(0.625 * rem));
      if (i === 0 || m.id.endsWith('-01')) {
        ctx.font = font(500, 0.75);
        ctx.fillStyle = color('--chart-tick');
        ctx.fillText(m.id.slice(0, 4), x, base + Math.round(1.75 * rem));
      }
    });

    hit = {
      cols,
      base,
      mx: Math.max(0, Math.min(0.375 * rem, (slot - barW) / 2 - 2)), // keep a dead zone between bars
      my: 0.5 * rem,
      minH: 1.5 * rem,
    };
    Acme.hooks.usageBars = cols.map((c) => ({ id: c.id, x: Math.round((c.x0 + c.x1) / 2), y: Math.round((c.top + base) / 2) }));
  }

  // The bar whose column (axis to bar top, plus a small margin) contains the point.
  function barAt(e) {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    return hit.cols.find((c) => x >= c.x0 - hit.mx && x <= c.x1 + hit.mx &&
      y >= Math.min(c.top, hit.base - hit.minH) - hit.my && y <= hit.base + hit.my) || null;
  }

  canvas.addEventListener('mousemove', (e) => { canvas.style.cursor = barAt(e) ? 'pointer' : ''; });
  canvas.addEventListener('mouseleave', () => { canvas.style.cursor = ''; });
  canvas.addEventListener('click', (e) => {
    const bar = barAt(e);
    if (bar) Acme.act('usage.select_month', { month: bar.id });
  });
  window.addEventListener('resize', draw);
  draw();
})();
