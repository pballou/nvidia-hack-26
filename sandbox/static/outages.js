/* Outage map page (templates/outages.html).
   One <canvas> draws a stylized town map; there is no DOM element per district. Geometry
   is fixed in a 1000 x 650 design space (600 of map plus a 50-unit legend band), scaled to
   the canvas width. Text sizes follow Acme.rootPx() and html.hc switches the palette, so
   text size and high contrast restyle the map. A click is hit-tested against the district
   polygons and opens the report dialog.
   Deterministic: no timers, no animation, no Date / Math.random. */
(function () {
  'use strict';

  const W = 1000;
  const MAP_H = 600;
  const H = 650;

  // District polygons and label anchors (design units). Neighbors share vertices; the west
  // edges follow the river's east bank and the east edges follow the coast.
  const GEO = {
    northgate: { label: [318, 128], poly: [[149, 0], [470, 0], [490, 125], [440, 262], [181, 262], [154, 190], [179, 95]] },
    hillcrest: { label: [668, 104], poly: [[470, 0], [860, 0], [845, 88], [862, 155], [850, 222], [700, 232], [610, 194], [490, 125]] },
    old_town: { label: [556, 300], poly: [[490, 125], [610, 194], [700, 232], [660, 320], [680, 410], [450, 396], [440, 262]] },
    harbor_point: { label: [792, 316], poly: [[700, 232], [850, 222], [885, 236], [918, 258], [944, 284], [956, 312], [950, 338], [926, 352], [896, 362], [866, 378], [850, 405], [680, 410], [660, 320]] },
    riverside: { label: [316, 430], poly: [[181, 262], [440, 262], [450, 396], [420, 600], [186, 600], [203, 475], [169, 380], [189, 285]] },
    maple_grove: { label: [640, 505], poly: [[450, 396], [680, 410], [850, 405], [845, 455], [870, 540], [852, 600], [420, 600]] },
  };
  const RIVER = [[128, 0], [158, 95], [133, 190], [168, 285], [148, 380], [182, 475], [165, 600]];
  const RIVER_W = 30;
  const COAST = [[860, 0], [845, 88], [862, 155], [850, 222], [885, 236], [918, 258], [944, 284], [956, 312],
    [950, 338], [926, 352], [896, 362], [866, 378], [850, 405], [845, 455], [870, 540], [852, 600]];
  // Main roads; the north-south one runs along district borders.
  const ROADS = [
    [[0, 350], [169, 350], [300, 342], [446, 334], [560, 352], [680, 362], [800, 364], [870, 368]],
    [[470, 0], [490, 125], [440, 262], [450, 396], [420, 600]],
    [[490, 125], [560, 72], [680, 44], [851, 34]],
    [[768, 600], [772, 500], [768, 364]],
  ];
  const PIERS = [[[880, 371], [886, 406]], [[912, 357], [918, 392]]];
  // Per-district texture, clipped to the polygon: a street grid {grid: [degrees, gap]},
  // hill contours {hills: [cx, cy, rx, ry, rings]} or trees {trees: [[x, y], ...]}.
  const TEXTURE = {
    old_town: { grid: [-28, 34] },
    northgate: { grid: [8, 44] },
    riverside: { grid: [-6, 44] },
    hillcrest: { hills: [772, 92, 58, 40, 3] },
    maple_grove: { trees: [[478, 432], [520, 566], [566, 440], [604, 578], [706, 446], [716, 572], [818, 478], [812, 566], [470, 520], [742, 536]] },
  };
  const PARK_TREES = [[28, 30], [70, 22], [50, 78], [96, 64], [22, 128], [74, 140], [40, 196], [88, 210], [24, 262],
    [70, 282], [108, 320], [40, 340], [84, 392], [26, 420], [66, 452], [110, 470], [36, 508], [84, 540], [30, 575], [118, 590]];
  const WAVES = [[905, 104], [962, 168], [978, 236], [935, 446], [975, 548], [900, 572]];
  const HOME = [262, 534];
  const COMPASS = [946, 50];
  const BAY_LABEL = [926, 494];

  // 24x24 glyphs (the same shapes as templates/_icons.html).
  const GLYPH = {
    outage: { path: new Path2D('M13.5 2.5 5 13.5h6l-1.5 8 8.5-11h-6l1.5-8z'), fill: true },
    report: { path: new Path2D('M12 21s-6.5-5.6-6.5-11a6.5 6.5 0 0 1 13 0c0 5.4-6.5 11-6.5 11zM14.5 10a2.5 2.5 0 1 0-5 0 2.5 2.5 0 1 0 5 0z'), fill: true },
    home: { path: new Path2D('M3.5 11 12 4l8.5 7M6 9.5V20h4.5v-5.5h3V20H18V9.5'), fill: false },
  };

  const LIGHT = {
    land: '#edf1e7', park: '#d3e7c9', tree: '#a3cc93', water: '#acd5ef', waterLine: '#5e9cc8',
    road: '#ffffff', roadEdge: '#c3ccd7', border: '#ffffff', label: '#13253f', halo: 'rgba(255, 255, 255, 0.92)',
    hover: 'rgba(255, 255, 255, 0.45)', highlight: '#0f766e', outage: '#d92d20', hatch: 'rgba(217, 45, 32, 0.42)',
    report: '#0f766e', home: '#0b2545', ring: '#ffffff', glyph: '#ffffff', pier: '#f4eee2', pierEdge: '#9c8c74',
    street: 'rgba(255, 255, 255, 0.75)', contour: 'rgba(60, 110, 50, 0.22)', grove: 'rgba(196, 92, 52, 0.3)',
    legendBg: '#ffffff', legendLine: '#dbe2eb', legendText: '#13253f', muted: '#536175',
    district: { northgate: '#e4dcf3', hillcrest: '#d6eacb', old_town: '#f5e1c4', harbor_point: '#d8e2f7', riverside: '#c8e8e1', maple_grove: '#f5d7da' },
  };
  const HC = {
    land: '#000000', park: '#000000', tree: '#2f6b3c', water: '#062a4f', waterLine: '#8fd0ff',
    road: '#8c8c8c', roadEdge: '#000000', border: '#ffffff', label: '#ffffff', halo: '#000000',
    hover: 'rgba(255, 214, 10, 0.22)', highlight: '#ffd60a', outage: '#ff8a80', hatch: 'rgba(255, 138, 128, 0.75)',
    report: '#ffd60a', home: '#ffffff', ring: '#000000', glyph: '#000000', pier: '#000000', pierEdge: '#ffffff',
    street: '#4a4a4a', contour: '#4a4a4a', grove: '#2f6b3c',
    legendBg: '#000000', legendLine: '#ffffff', legendText: '#ffffff', muted: '#ffffff',
    district: {},
  };

  const canvas = document.getElementById('outage-map');
  const dialog = document.getElementById('report-dialog');
  const dataEl = document.getElementById('outage-data');
  if (!canvas || !dialog || !dataEl) return;
  const ctx = canvas.getContext('2d');
  const data = JSON.parse(dataEl.textContent);
  const districts = data.districts.filter((d) => GEO[d.id]);
  const nameOf = Object.fromEntries(data.districts.map((d) => [d.id, d.name]));
  const activeBy = Object.fromEntries(data.active.map((a) => [a.district, a]));
  const reported = new Set(data.reported);

  let s = 1;           // CSS px per design unit
  let hover = null;    // district under the mouse
  let selected = null; // district whose report dialog is open
  let kbd = -1;        // index into districts chosen with the arrow keys

  // ---- geometry --------------------------------------------------------------

  function inside([x, y], poly) {
    let hit = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i];
      const [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  }

  function districtAt(e) {
    const r = canvas.getBoundingClientRect();
    const k = r.width / W;
    const p = [(e.clientX - r.left) / k, (e.clientY - r.top) / k];
    if (p[1] >= MAP_H) return null;
    const d = districts.find((d) => inside(p, GEO[d.id].poly));
    return d ? d.id : null;
  }

  // ---- drawing helpers (design units in, CSS px out) -------------------------

  function trace(pts, close = true) {
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)));
    if (close) ctx.closePath();
  }

  function stroke(color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();
  }

  function hatch(pts, color, gap, width) {
    const xs = pts.map((p) => p[0] * s);
    const ys = pts.map((p) => p[1] * s);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    ctx.save();
    trace(pts);
    ctx.clip();
    ctx.beginPath();
    // 45-degree stripes aligned to the canvas origin, so every hatched area matches.
    for (let d = Math.floor((x0 + y0) / gap) * gap; d <= x1 + y1; d += gap) {
      ctx.moveTo(d - y0, y0);
      ctx.lineTo(d - y1, y1);
    }
    stroke(color, width);
    ctx.restore();
  }

  function texture(id, c) {
    const t = TEXTURE[id];
    if (!t) return;
    const poly = GEO[id].poly;
    ctx.save();
    trace(poly);
    ctx.clip();
    ctx.beginPath();
    if (t.grid) {
      const xs = poly.map((p) => p[0]);
      const ys = poly.map((p) => p[1]);
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
      const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      const R = Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2;
      const [deg, gap] = t.grid;
      for (const a of [deg, deg + 90]) {
        const dx = Math.cos((a * Math.PI) / 180);
        const dy = Math.sin((a * Math.PI) / 180);
        for (let k = -Math.floor(R / gap) * gap; k <= R; k += gap) {
          const px = cx - dy * k;
          const py = cy + dx * k;
          ctx.moveTo((px - dx * R) * s, (py - dy * R) * s);
          ctx.lineTo((px + dx * R) * s, (py + dy * R) * s);
        }
      }
      stroke(c.street, Math.max(1, 2.5 * s));
    } else if (t.hills) {
      const [hx, hy, rx, ry, rings] = t.hills;
      for (let i = 1; i <= rings; i++) {
        ctx.moveTo((hx + (rx * i) / rings) * s, hy * s);
        ctx.ellipse(hx * s, hy * s, ((rx * i) / rings) * s, ((ry * i) / rings) * s, -0.2, 0, Math.PI * 2);
      }
      stroke(c.contour, Math.max(1, 2 * s));
    } else if (t.trees) {
      ctx.fillStyle = c.grove;
      for (const [x, y] of t.trees) {
        ctx.moveTo((x + 8) * s, y * s);
        ctx.arc(x * s, y * s, 8 * s, 0, Math.PI * 2);
      }
      ctx.fill();
    }
    ctx.restore();
  }

  function glyph(kind, cx, cy, size, color) {
    const g = GLYPH[kind];
    ctx.save();
    ctx.translate(cx - size / 2, cy - size / 2);
    ctx.scale(size / 24, size / 24);
    if (g.fill) {
      ctx.fillStyle = color;
      ctx.fill(g.path, 'evenodd');
    } else {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.6;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.stroke(g.path);
    }
    ctx.restore();
  }

  function badge(kind, cx, cy, r, c, rem) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = c[kind];
    ctx.fill();
    stroke(c.ring, Math.max(1.5, 0.125 * rem));
    glyph(kind, cx, cy, r * 1.3, c.glyph);
  }

  function label(text, x, y, c, rem) {
    ctx.lineWidth = 0.3125 * rem;
    ctx.lineJoin = 'round';
    ctx.strokeStyle = c.halo;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = c.label;
    ctx.fillText(text, x, y);
  }

  // ---- the map -----------------------------------------------------------------

  function draw() {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(rect.width * dpr);
    const bh = Math.round(rect.height * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    s = rect.width / W;
    const rem = Acme.rootPx();
    const hc = document.documentElement.classList.contains('hc');
    const c = hc ? HC : LIGHT;
    const font = getComputedStyle(document.body).fontFamily;
    const lit = selected || (kbd >= 0 ? districts[kbd].id : null) || hover;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, W * s, MAP_H * s);
    ctx.clip();

    // Land, the park west of the river, the sea and the river.
    ctx.fillStyle = c.land;
    ctx.fillRect(0, 0, W * s, MAP_H * s);
    trace([[0, 0], ...RIVER.map(([x, y]) => [x - RIVER_W / 2, y]), [0, MAP_H]]);
    ctx.fillStyle = c.park;
    ctx.fill();
    ctx.fillStyle = c.tree;
    for (const [x, y] of PARK_TREES) {
      ctx.beginPath();
      ctx.arc(x * s, y * s, 9 * s, 0, Math.PI * 2);
      ctx.fill();
    }
    trace([...COAST, [W, MAP_H], [W, 0]]);
    ctx.fillStyle = c.water;
    ctx.fill();
    trace(RIVER, false);
    stroke(c.water, RIVER_W * s);
    for (const [x, y] of WAVES) {
      ctx.beginPath();
      ctx.moveTo((x - 14) * s, y * s);
      ctx.quadraticCurveTo((x - 7) * s, (y - 7) * s, x * s, y * s);
      ctx.quadraticCurveTo((x + 7) * s, (y + 7) * s, (x + 14) * s, y * s);
      stroke(c.waterLine, Math.max(1, 2 * s));
    }
    for (const pier of PIERS) {
      trace(pier, false);
      stroke(c.pierEdge, 9 * s);
      trace(pier, false);
      stroke(c.pier, 6 * s);
    }

    // Districts, outage hatching, roads, then borders on top.
    for (const d of districts) {
      trace(GEO[d.id].poly);
      ctx.fillStyle = c.district[d.id] || c.land;
      ctx.fill();
      texture(d.id, c);
      if (d.id === lit) {
        trace(GEO[d.id].poly);
        ctx.fillStyle = c.hover;
        ctx.fill();
      }
      if (activeBy[d.id]) hatch(GEO[d.id].poly, c.hatch, 0.625 * rem, Math.max(1.5, 0.125 * rem));
    }
    for (const road of ROADS) {
      trace(road, false);
      stroke(c.roadEdge, 11 * s);
      trace(road, false);
      stroke(c.road, hc ? Math.max(2, 4 * s) : 7 * s);
    }
    for (const d of districts) {
      trace(GEO[d.id].poly);
      stroke(c.border, hc ? 3 : Math.max(2, 4 * s));
    }
    trace(COAST, false);
    stroke(c.waterLine, Math.max(1.5, 2.5 * s));
    for (const d of districts) {
      if (!activeBy[d.id]) continue;
      trace(GEO[d.id].poly);
      stroke(c.outage, Math.max(2, 0.1875 * rem));
    }
    if (lit) {
      trace(GEO[lit].poly);
      stroke(c.highlight, hc ? 0.3125 * rem : 0.1875 * rem);
    }

    // Markers above each label, then the labels.
    const fontPx = 0.9375 * rem;
    const r = 0.6875 * rem;
    const gap = 0.3125 * rem;
    if (GEO[data.home]) badge('home', HOME[0] * s, HOME[1] * s, r, c, rem);
    for (const d of districts) {
      const kinds = [activeBy[d.id] && 'outage', reported.has(d.id) && 'report'].filter(Boolean);
      const [lx, ly] = GEO[d.id].label.map((v) => v * s);
      let x = lx - ((kinds.length - 1) * (2 * r + gap)) / 2;
      for (const kind of kinds) {
        badge(kind, x, ly - fontPx * 0.6 - gap - r, r, c, rem);
        x += 2 * r + gap;
      }
    }
    ctx.font = `700 ${fontPx}px ${font}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const d of districts) {
      const [lx, ly] = GEO[d.id].label;
      label(nameOf[d.id], lx * s, ly * s, c, rem);
    }

    // Bay name and north arrow.
    ctx.font = `italic 600 ${0.75 * rem}px ${font}`;
    ctx.fillStyle = c.waterLine;
    ctx.fillText('Acme Bay', BAY_LABEL[0] * s, BAY_LABEL[1] * s);
    const [nx, ny] = COMPASS.map((v) => v * s);
    const nr = 0.75 * rem;
    ctx.beginPath();
    ctx.moveTo(nx, ny - nr);
    ctx.lineTo(nx + nr * 0.6, ny + nr);
    ctx.lineTo(nx, ny + nr * 0.45);
    ctx.lineTo(nx - nr * 0.6, ny + nr);
    ctx.closePath();
    ctx.fillStyle = c.label;
    ctx.fill();
    ctx.font = `800 ${0.75 * rem}px ${font}`;
    ctx.fillStyle = c.label;
    ctx.fillText('N', nx, ny - nr - 0.5 * rem);
    ctx.restore();

    legend(c, rem, font);

    Acme.hooks.districts = districts.map((d) => ({
      id: d.id, x: Math.round(GEO[d.id].label[0] * s), y: Math.round(GEO[d.id].label[1] * s),
    }));
  }

  function legend(c, rem, font) {
    const y0 = MAP_H * s;
    const cy = y0 + ((H - MAP_H) * s) / 2;
    const r = 0.5625 * rem;
    ctx.fillStyle = c.legendBg;
    ctx.fillRect(0, y0, W * s, (H - MAP_H) * s);
    ctx.fillStyle = c.legendLine;
    ctx.fillRect(0, y0, W * s, c === HC ? 2 : 1);

    ctx.font = `600 ${0.8125 * rem}px ${font}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    let x = 1.25 * rem;
    const item = (draw, text) => {
      draw(x + r, cy);
      x += 2 * r + 0.5 * rem;
      ctx.fillStyle = c.legendText;
      ctx.fillText(text, x, cy);
      x += ctx.measureText(text).width + 1.5 * rem;
    };
    item((cx, y) => {
      const sq = [[cx - r, y - r], [cx + r, y - r], [cx + r, y + r], [cx - r, y + r]].map(([a, b]) => [a / s, b / s]);
      trace(sq);
      ctx.fillStyle = c.land;
      ctx.fill();
      hatch(sq, c.hatch, 0.375 * rem, Math.max(1.5, 0.125 * rem));
      trace(sq);
      stroke(c.outage, Math.max(2, 0.125 * rem));
      badge('outage', cx + r, y - r, r * 0.7, c, rem);
    }, 'Active outage');
    item((cx, y) => badge('report', cx, y, r, c, rem), 'Your report');
    item((cx, y) => badge('home', cx, y, r, c, rem), 'Your address');

    // Scale bar on the right.
    const right = W * s - 1.25 * rem;
    ctx.textAlign = 'right';
    ctx.fillStyle = c.muted;
    ctx.fillText('1 mi', right, cy);
    const bx1 = right - ctx.measureText('1 mi').width - 0.5 * rem;
    const bx0 = bx1 - 4 * rem;
    const bh = 0.375 * rem;
    ctx.fillStyle = c.legendText;
    ctx.fillRect(bx0, cy - bh / 2, (bx1 - bx0) / 2, bh);
    ctx.strokeStyle = c.legendText;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx0 + 0.5, cy - bh / 2 + 0.5, bx1 - bx0 - 1, bh - 1);
  }

  // ---- report dialog -----------------------------------------------------------

  const title = dialog.querySelector('[data-testid="report-title"]');
  const known = document.getElementById('report-known');
  const clear = document.getElementById('report-clear');
  const types = document.getElementById('report-type');
  const danger = document.getElementById('report-danger');
  const details = dialog.querySelector('[data-testid="report-details"]');
  const error = document.getElementById('report-error');
  let current = null;

  function openReport(id) {
    const name = nameOf[id];
    const a = activeBy[id];
    current = id;
    selected = id;
    draw();
    title.textContent = 'Report an outage in ' + name;
    known.hidden = !a;
    clear.hidden = !!a;
    if (a) {
      known.querySelector('span').textContent =
        `We already know about an outage in ${name} (${a.status.toLowerCase()}, ${a.customers} customers affected). Your report still helps our crews.`;
    } else {
      clear.querySelector('span').textContent = `No known outages in ${name} right now. Your report goes straight to our crews.`;
    }
    Acme.select(types.querySelector('[data-value="no_power"]'));
    details.value = '';
    Acme.setError(error, '');
    Acme.openModal(dialog, {
      onClose: () => {
        selected = null;
        hover = null;
        draw();
      },
    });
  }

  types.addEventListener('change', () => { danger.hidden = types.dataset.value !== 'downed_line'; });

  dialog.querySelector('[data-testid="report-submit"]').addEventListener('click', () => {
    if (!current) return;
    Acme.act('outages.report', { district: current, type: types.dataset.value || 'no_power', details: details.value },
      { onError: (msg) => Acme.setError(error, msg) });
  });

  // ---- canvas input ------------------------------------------------------------

  canvas.addEventListener('click', (e) => {
    const id = districtAt(e);
    if (id) openReport(id);
  });

  canvas.addEventListener('mousemove', (e) => {
    if (selected) return;
    const id = districtAt(e);
    if (id === hover) return;
    hover = id;
    canvas.style.cursor = id ? 'pointer' : '';
    draw();
  });

  canvas.addEventListener('mouseleave', () => {
    if (!hover) return;
    hover = null;
    canvas.style.cursor = '';
    draw();
  });

  canvas.addEventListener('keydown', (e) => {
    const n = districts.length;
    if (['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp'].includes(e.key)) {
      e.preventDefault();
      const step = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : -1;
      kbd = kbd < 0 ? (step > 0 ? 0 : n - 1) : (kbd + step + n) % n;
      draw();
    } else if ((e.key === 'Enter' || e.key === ' ') && kbd >= 0) {
      e.preventDefault();
      openReport(districts[kbd].id);
    }
  });

  canvas.addEventListener('blur', () => {
    if (kbd < 0) return;
    kbd = -1;
    draw();
  });

  window.addEventListener('resize', draw);
  draw();
})();
