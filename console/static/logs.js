/* GUI assist logs page (/logs): every run's trace, event by event. Vanilla JS, no build step, no external resources.
 *
 * Reads the console backend's logs API (console/README.md):
 *   GET /api/logs/runs?limit=200          {runs: [RunSummary, ...]} newest first
 *   GET /api/logs/runs/{run_id}           {summary, console, events}
 *   GET /api/logs/runs/{run_id}/download  the same JSON as an attachment (the "Download JSON" link)
 * Events are the agent's events.jsonl lines (kinds and payloads in agent/trace.py) plus the console's own
 * "voice" (first) and "verdict" (last). Everything from a trace is rendered with textContent, never innerHTML.
 *
 * Honest by construction: a field the trace doesn't have shows "—" (never a guess), and mock runs say "mock".
 * Deep links: /logs?run=<id>&kind=plan,grounding&model=base&q=text. The only timer is the 3 s auto-refresh,
 * and it runs only while the selected run's status is "running".
 */
'use strict';

(() => {
  const HC_KEY = 'gui-assist.high-contrast';  // shared with the console page
  const RUNS_LIMIT = 200;
  const REFRESH_MS = 3000;
  const RUN_ID_RE = /^[A-Za-z0-9._-]{1,80}$/;
  const STAGES = ['voice', 'summary', 'plan', 'ground', 'act'];
  const STAGE_WORDS = {
    voice: 'speech to text', summary: 'screen summary (VLM)', plan: 'planning (Lightning)',
    ground: 'grounding (VLM)', act: 'browser action',
  };
  const KIND_ORDER = ['voice', 'start', 'observation', 'plan', 'tool_call', 'grounding', 'action', 'error', 'done',
    'verdict'];
  const KIND_LABEL = { tool_call: 'tool call' };
  const KIND_LATENCY = { plan: 'plan', grounding: 'ground', action: 'act', observation: 'summary' };
  const STATUS = {
    done: ['Done', 'ok'], max_steps: ['Step limit', 'warn'], stopped: ['Stopped', 'warn'], error: ['Error', 'bad'],
    running: ['Running', 'live'], unknown: ['Unknown', 'muted'],
  };
  const REASON_WORDS = {
    done: 'the agent called done', max_steps: 'it reached the step limit', stopped: 'stopped by the user',
    error: 'it stopped after an error',
  };
  const VERDICT_LABEL = 'verified by sandbox state, not seen by the agent';

  // --- theme first (this file loads in <head>, so the saved choice applies before first paint) -------------------
  const prefs = {
    get(k) { try { return window.localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  (() => {
    const saved = prefs.get(HC_KEY);
    const on = saved !== null ? saved === '1'
      : Boolean(window.matchMedia && window.matchMedia('(prefers-contrast: more)').matches);
    document.documentElement.classList.toggle('hc', on);
  })();

  // --- state ------------------------------------------------------------------------------------------------
  const S = {
    runs: [],
    runsLoaded: false,
    runsError: '',
    selectedId: null,
    detail: null,          // the selected run's {summary, console, events}
    detailFor: null,       // run id the detail belongs to
    detailSeq: 0,          // bumps per request, so a slow old response never overwrites a newer one
    runsSeq: 0,
    filters: { kinds: null, model: '', q: '' },  // kinds: null = all kinds
    expanded: new Set(),   // event indexes with details open
    evEls: [],             // per event index: {li, row, sig, hay, visible}
    activeIdx: null,       // the event row that holds the tab stop
    refreshTimer: 0,       // the only timer: set while the selected run is running (scheduleRefresh)
    lastRefresh: null,
    runsTotal: 0,
    dialogReturn: null,    // the thumbnail that opened the image dialog (focus goes back to it)
  };
  const ui = {};

  // --- small helpers ------------------------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  function h(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else node.setAttribute(k, v === true ? '' : String(v));
    }
    for (const c of children.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      node.append(c instanceof Node ? c : String(c));
    }
    return node;
  }

  const str = (v) => (v === null || v === undefined ? '' : String(v));
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
  const has = (v) => v !== null && v !== undefined && v !== '';
  const oneLine = (s) => str(s).replace(/\s+/g, ' ').trim();
  const clip = (s, n) => (s.length > n ? `${s.slice(0, n).trimEnd()}…` : s);
  const sv = (text) => h('span', { class: 'visually-hidden', text });

  function setText(node, text) {
    if (node && node.textContent !== text) node.textContent = text;
  }

  /** "—" for a value the trace doesn't have (screen readers hear "not recorded"). */
  function na(label = 'not recorded') {
    return h('span', { class: 'na' }, h('span', { 'aria-hidden': 'true', text: '—' }), sv(label));
  }

  const val = (v, fmt = str) => (has(v) ? fmt(v) : na());

  function compactJson(v, max = 200) {
    let out;
    try { out = JSON.stringify(v); } catch { out = String(v); }
    return clip(str(out), max);
  }

  function prettyJson(v) {
    try { return JSON.stringify(v, null, 2); } catch { return String(v); }
  }

  const nf0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

  /** 676 ms, 3.03 s, 24.1 s, 2 min 3 s. */
  function fmtMs(ms) {
    if (!isNum(ms)) return '';
    const a = Math.abs(ms);
    if (a < 1000) return `${nf0.format(ms)} ms`;
    const s = ms / 1000;
    if (a < 10000) return `${s.toFixed(2)} s`;
    if (a < 60000) return `${s.toFixed(1)} s`;
    return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  }

  /** Table cells: like fmtMs, but "2m 26s" past a minute. */
  function fmtMsShort(ms) {
    if (!isNum(ms) || Math.abs(ms) < 60000) return fmtMs(ms);
    const s = Math.round(ms / 1000);
    return `${Math.floor(s / 60)}m ${s % 60}s`;
  }

  /** "+3,440 ms" from the run's first event. */
  function fmtT(ms) {
    if (!isNum(ms)) return '';
    return `${ms < 0 ? '−' : '+'}${nf0.format(Math.abs(ms))} ms`;
  }

  const pad2 = (n) => String(n).padStart(2, '0');
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function fmtClock(ts) {
    const d = new Date(ts * 1000);
    return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  }

  /** "17:50:25" today, "Oct 16 17:50:25" otherwise (local time). */
  function fmtWhen(ts) {
    if (!isNum(ts)) return '';
    const d = new Date(ts * 1000);
    const now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    return sameDay ? fmtClock(ts) : `${MONTHS[d.getMonth()]} ${d.getDate()} ${fmtClock(ts)}`;
  }

  function fmtFull(ts) {
    if (!isNum(ts)) return '';
    const d = new Date(ts * 1000);
    return `${d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}`
      + ` ${fmtClock(ts)}`;
  }

  function isoOf(ts) {
    try { return new Date(ts * 1000).toISOString(); } catch { return ''; }
  }

  /** '/settings?tab=accessibility' from a sandbox URL (the sid query param is noise). */
  function displayUrl(u) {
    if (!u) return '';
    try {
      const url = new URL(u, window.location.href);
      url.searchParams.delete('sid');
      const q = url.searchParams.toString();
      return url.pathname + (q ? `?${q}` : '') + url.hash;
    } catch {
      return str(u);
    }
  }

  function parseArgs(args) {
    if (isObj(args)) return args;
    if (typeof args === 'string') {
      try { const v = JSON.parse(args); return isObj(v) ? v : {}; } catch { return {}; }
    }
    return {};
  }

  /** {data, text}: the parsed JSON and the exact response text (Copy JSON copies the detail's text verbatim,
   *  so it is byte-identical to the Download, which the backend serializes the same way). */
  async function fetchJson(url) {
    const res = await fetch(url, { headers: { Accept: 'application/json' }, cache: 'no-store' });
    let text = '';
    let data = null;
    try { text = await res.text(); data = JSON.parse(text); } catch { data = null; }
    if (!res.ok) {
      const err = new Error(oneLine(isObj(data) && data.error) || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return { data, text };
  }

  async function getJson(url) {
    return (await fetchJson(url)).data;
  }

  let announceFlip = false;
  function announce(text) {  // no timer: alternate a trailing space so the same words are read again
    announceFlip = !announceFlip;
    ui.srStatus.textContent = text + (announceFlip ? '' : ' ');
  }

  const kindLabel = (k) => KIND_LABEL[k] || k || 'event';
  const kindRank = (k) => { const i = KIND_ORDER.indexOf(k); return i < 0 ? KIND_ORDER.length : i; };

  function kindBadge(kind) {
    const k = str(kind) || 'event';
    return h('span', { class: 'kind-badge', 'data-kind': KIND_ORDER.includes(k) ? k : 'other', text: kindLabel(k) });
  }

  function statusChip(status) {
    const [label, tone] = STATUS[status] || [status ? `${status}` : 'Unknown', 'muted'];
    return h('span', { class: `status-chip tone-${tone}`, 'data-status': str(status) || 'unknown', text: label });
  }

  /** ✓ / ✗ / — with words for screen readers. */
  function tick(v, yes, no, none) {
    if (v === true) return h('span', { class: 'tick tick-yes', title: yes }, h('span', { 'aria-hidden': 'true', text: '✓' }), sv(yes));
    if (v === false) return h('span', { class: 'tick tick-no', title: no }, h('span', { 'aria-hidden': 'true', text: '✗' }), sv(no));
    return h('span', { class: 'tick tick-none', title: none }, h('span', { 'aria-hidden': 'true', text: '—' }), sv(none));
  }

  function agentBadge(agent) {
    if (agent === 'mock') return h('span', { class: 'agent-tag agent-mock', title: 'Scripted mock agent: no models ran', text: 'mock' });
    if (agent === 'real') return h('span', { class: 'agent-tag agent-real', title: 'Lightning + the grounder VLM', text: 'real' });
    return h('span', { class: 'agent-tag', text: str(agent) || '?' });
  }

  // --- event words ------------------------------------------------------------------------------------------
  function callWords(call) {
    if (!isObj(call)) return 'an unknown tool call';
    const name = str(call.name);
    const a = parseArgs(call.arguments);
    switch (name) {
      case 'click': return `click “${oneLine(a.target_description) || '(no target)'}”`;
      case 'type_text': {
        const target = oneLine(a.target_description);
        return `type “${str(a.text)}”${target ? ` into “${target}”` : ''}`;
      }
      case 'press_key': return `press ${oneLine(a.key) || '(no key)'}`;
      case 'scroll': return `scroll ${oneLine(a.direction) || '?'}${a.amount ? ` (${oneLine(a.amount)})` : ''}`;
      case 'done': {
        const ok = a.success === true ? 'success' : a.success === false ? 'no success' : 'no success flag';
        return `done (${ok}): ${clip(oneLine(a.summary) || '(no summary)', 160)}`;
      }
      default: return name ? `${name}(${compactJson(call.arguments ?? {}, 100)})` : 'an unnamed tool call';
    }
  }

  function changedWords(c) {
    if (c === true) return ' · screen changed';
    if (c === false) return ' · no change';
    return '';
  }

  /** One line, plain words. Mock events (payload.mock) never claim a model did something. */
  function describe(ev) {
    const p = isObj(ev.payload) ? ev.payload : {};
    const mock = p.mock === true;
    switch (ev.kind) {
      case 'voice': return `Heard: “${oneLine(p.text)}”`;
      case 'start': {
        const where = displayUrl(p.start_url);
        return `Run started${where ? ` at ${where}` : ''}: “${oneLine(p.request)}”`;
      }
      case 'observation': {
        const o = isObj(p.observation) ? p.observation : {};
        const title = oneLine(o.title);
        const where = displayUrl(o.url);
        const s = isObj(p.summary) ? p.summary : {};
        const sum = s.error ? ` · summary failed: ${oneLine(s.error)}` : '';
        const reused = s.reused === true ? ' · description reused (no new summarizer call)' : '';
        const what = mock ? 'Screen (scripted mock summary)' : 'Read the screen';
        return `${what}: ${title || 'untitled page'}${where ? ` (${where})` : ''}${changedWords(o.screen_changed)}${sum}`
          + reused;
      }
      case 'plan': {
        const calls = Array.isArray(p.tool_calls) ? p.tool_calls : [];
        const think = p.thinking === true ? ' · thinking' : '';
        const who = mock ? 'Mock script' : 'Lightning';
        if (!calls.length) {
          const said = oneLine(typeof p.content === 'string' ? p.content : '');
          return `${who} replied without a tool call${said ? `: “${clip(said, 140)}”` : ''}${think}`;
        }
        const more = calls.length > 1 ? ` (+${calls.length - 1} more)` : '';
        return `${who} chose: ${callWords(calls[0])}${more}${think}`;
      }
      case 'tool_call': {
        if (p.valid === false) return `Invalid tool call ${str(p.name) || '?'}: ${oneLine(p.error) || 'rejected by the schema'}`;
        return oneLine(p.description) || `Tool call: ${callWords(p)}`;
      }
      case 'grounding': {
        const target = oneLine(p.target);
        const who = mock ? 'Mock script (pre-measured point)' : 'Grounder';
        if (isNum(p.x) && isNum(p.y)) return `${who} placed “${target}” at (${Math.round(p.x)}, ${Math.round(p.y)})`;
        return `${who} couldn’t place “${target}”${p.error ? `: ${oneLine(p.error)}` : ''}`;
      }
      case 'action': {
        const a = parseArgs(p.arguments);
        if (p.ok === false) return `Action ${str(p.name) || ''} failed: ${oneLine(p.error) || 'unknown error'}`.replace('  ', ' ');
        let what;
        switch (p.name) {
          case 'click':
            what = isObj(p.click) && isNum(p.click.x) ? `Clicked at (${Math.round(p.click.x)}, ${Math.round(p.click.y)})` : 'Clicked';
            break;
          case 'type_text': what = `Typed “${str(a.text)}”`; break;
          case 'press_key': what = `Pressed ${oneLine(a.key) || '?'}`; break;
          case 'scroll': what = `Scrolled ${oneLine(a.direction) || '?'}${a.amount ? ` (${oneLine(a.amount)})` : ''}`; break;
          case 'done': what = 'Finished (the done tool)'; break;
          default: what = `${str(p.name) || 'Action'} done`;
        }
        return what + changedWords(p.screen_changed);
      }
      case 'error': {
        const where = p.where ? ` in ${oneLine(p.where)}` : '';
        const rec = p.recoverable === true ? ' (recoverable)' : p.recoverable === false ? ' (not recoverable)' : '';
        return `Error${where}: ${oneLine(p.message) || 'no message'}${rec}`;
      }
      case 'done': {
        const reason = REASON_WORDS[p.reason] || (p.reason ? `reason ${oneLine(p.reason)}` : 'no reason given');
        const claim = p.success_claimed === true ? 'the agent claims success'
          : p.success_claimed === false ? 'the agent does not claim success' : 'no success claim';
        const steps = isNum(p.steps) ? ` · ${p.steps} step${p.steps === 1 ? '' : 's'}` : '';
        return `Run ended: ${reason} · ${claim}${steps}`;
      }
      case 'verdict': {
        if (p.passed === true) return `✓ Verified: the task is done${p.explanation ? ` (${oneLine(p.explanation)})` : ''}`;
        if (p.passed === false) return `✗ Not verified: the task is not done${p.explanation ? ` (${oneLine(p.explanation)})` : ''}`;
        return `Sandbox check: ${oneLine(p.explanation) || 'no result'}`;
      }
      default:
        return `${str(ev.kind) || 'event'}: ${compactJson(ev.payload === undefined ? ev : ev.payload, 160)}`;
    }
  }

  function eventLatency(ev) {
    const p = isObj(ev.payload) ? ev.payload : {};
    if (ev.kind === 'voice') return isNum(p.latency_ms) ? p.latency_ms : null;
    const key = KIND_LATENCY[ev.kind];
    const t = isObj(ev.timings_ms) ? ev.timings_ms : {};
    return key && isNum(t[key]) ? t[key] : null;
  }

  function eventTone(ev) {
    const p = isObj(ev.payload) ? ev.payload : {};
    if (ev.kind === 'error') return p.recoverable === true ? 'warn' : 'bad';
    if (ev.kind === 'tool_call' && p.valid === false) return 'warn';
    if (ev.kind === 'action' && p.ok === false) return 'bad';
    if (ev.kind === 'action' && p.screen_changed === false) return 'warn';
    if (ev.kind === 'grounding' && (p.error || !isNum(p.x))) return 'warn';
    if (ev.kind === 'verdict') return p.passed === true ? 'ok' : p.passed === false ? 'bad' : '';
    if (ev.kind === 'done') return p.reason === 'done' ? '' : p.reason === 'error' ? 'bad' : 'warn';
    return '';
  }

  // --- init -----------------------------------------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', init);

  function init() {
    for (const id of [
      'hc-toggle', 'runs-section', 'runs-meta', 'refresh-btn', 'refresh-note', 'runs-scroll', 'runs-table', 'runs-body',
      'runs-empty', 'run-placeholder', 'run-section', 'run-heading', 'run-badges', 'copy-btn', 'download-link',
      'copy-status', 'copy-fallback-wrap', 'copy-fallback', 'copy-fallback-close', 'run-error', 'run-summary',
      'events-meta', 'kind-chips', 'model-filter', 'event-search', 'clear-filters', 'event-list', 'events-empty',
      'img-dialog', 'img-dialog-title', 'img-dialog-img', 'img-dialog-close', 'sr-status',
    ]) ui[id.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = $(id);

    ui.hcToggle.setAttribute('aria-pressed', String(document.documentElement.classList.contains('hc')));
    ui.hcToggle.addEventListener('click', () => {
      const on = !document.documentElement.classList.contains('hc');
      document.documentElement.classList.toggle('hc', on);
      ui.hcToggle.setAttribute('aria-pressed', String(on));
      prefs.set(HC_KEY, on ? '1' : '0');
    });
    window.addEventListener('storage', (e) => {  // the console page flipped the theme in another tab
      if (e.key !== HC_KEY) return;
      const on = e.newValue === '1';
      document.documentElement.classList.toggle('hc', on);
      ui.hcToggle.setAttribute('aria-pressed', String(on));
    });

    ui.refreshBtn.addEventListener('click', () => refresh({ user: true }));
    ui.runsBody.addEventListener('keydown', onRunsKey);
    ui.runsBody.addEventListener('click', onRunsClick);

    ui.eventList.addEventListener('keydown', onEventKey);
    ui.eventList.addEventListener('click', onEventClick);
    ui.eventList.addEventListener('focusin', (e) => {
      const row = e.target.closest && e.target.closest('.ev-row');
      if (row) setActiveEvent(Number(row.dataset.idx), false);
    });
    ui.kindChips.addEventListener('change', onKindChange);
    ui.modelFilter.addEventListener('change', () => {
      S.filters.model = ui.modelFilter.value;
      applyFilters();
      writeUrl(false);
    });
    ui.eventSearch.addEventListener('input', () => {
      S.filters.q = ui.eventSearch.value;
      applyFilters();
      writeUrl(false);
    });
    ui.eventSearch.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && ui.eventSearch.value) {  // the native clear, made explicit for every browser
        e.preventDefault();
        ui.eventSearch.value = '';
        S.filters.q = '';
        applyFilters();
        writeUrl(false);
      }
    });
    ui.clearFilters.addEventListener('click', () => {
      S.filters = { kinds: null, model: '', q: '' };
      renderFilters();
      applyFilters();
      writeUrl(false);
      announce('Showing all events.');
    });

    ui.copyBtn.addEventListener('click', copyJson);
    ui.copyFallbackClose.addEventListener('click', () => {
      ui.copyFallbackWrap.hidden = true;
      ui.copyBtn.focus();
    });
    ui.imgDialogClose.addEventListener('click', () => ui.imgDialog.close());
    ui.imgDialog.addEventListener('click', (e) => { if (e.target === ui.imgDialog) ui.imgDialog.close(); });
    ui.imgDialog.addEventListener('close', () => {
      const back = S.dialogReturn;
      S.dialogReturn = null;
      ui.imgDialogImg.removeAttribute('src');
      if (back && back.isConnected) back.focus();
    });

    window.addEventListener('popstate', () => {
      readUrl();
      renderFilters();
      if (S.selectedId && S.selectedId !== S.detailFor) loadDetail(S.selectedId, {});
      else if (!S.selectedId) showPlaceholder();
      else applyFilters();
      markSelectedRow();
    });

    readUrl();
    renderFilters();
    loadRuns().then(() => {
      if (!S.selectedId) showPlaceholder();
    });
    if (S.selectedId) loadDetail(S.selectedId, {});
  }

  // --- URL state --------------------------------------------------------------------------------------------------
  function readUrl() {
    const q = new URLSearchParams(window.location.search);
    const run = q.get('run');
    S.selectedId = run || null;  // an invalid id is shown as "not found", not silently dropped
    const kinds = q.get('kind');
    S.filters.kinds = kinds ? new Set(kinds.split(',').map((s) => s.trim()).filter(Boolean)) : null;
    S.filters.model = q.get('model') || '';
    S.filters.q = q.get('q') || '';
  }

  function writeUrl(push) {
    const q = new URLSearchParams();
    if (S.selectedId) q.set('run', S.selectedId);
    if (S.filters.kinds) q.set('kind', [...S.filters.kinds].sort((a, b) => kindRank(a) - kindRank(b)).join(','));
    if (S.filters.model) q.set('model', S.filters.model);
    if (S.filters.q) q.set('q', S.filters.q);
    const qs = q.toString().replace(/%2C/gi, ',');
    const url = `${window.location.pathname}${qs ? `?${qs}` : ''}`;
    if (url === window.location.pathname + window.location.search) return;
    window.history[push ? 'pushState' : 'replaceState'](null, '', url);
  }

  // --- refresh ----------------------------------------------------------------------------------------------------
  async function refresh({ user = false } = {}) {
    window.clearTimeout(S.refreshTimer);
    S.refreshTimer = 0;
    if (user) {
      ui.refreshBtn.disabled = true;
      ui.refreshBtn.setAttribute('aria-busy', 'true');
    }
    const jobs = [loadRuns()];
    if (S.selectedId) jobs.push(loadDetail(S.selectedId, { refresh: true }));
    await Promise.all(jobs);
    if (user) {
      ui.refreshBtn.disabled = false;
      ui.refreshBtn.removeAttribute('aria-busy');
      if (document.activeElement === document.body) ui.refreshBtn.focus();
      announce(`Refreshed: ${S.runs.length} run${S.runs.length === 1 ? '' : 's'}.`);
    }
  }

  /** The one timer: every 3 s while the selected run is still running. */
  function scheduleRefresh() {
    window.clearTimeout(S.refreshTimer);
    S.refreshTimer = 0;
    const running = S.detail && isObj(S.detail.summary) && S.detail.summary.status === 'running';
    setText(ui.refreshNote, running ? 'This run is still going: refreshing every 3 s.' : refreshedText());
    ui.refreshNote.dataset.live = String(Boolean(running));
    if (running) S.refreshTimer = window.setTimeout(() => refresh(), REFRESH_MS);
  }

  function refreshedText() {
    return S.lastRefresh ? `Updated ${fmtClock(S.lastRefresh / 1000)}` : '';
  }

  // --- runs table ---------------------------------------------------------------------------------------------------
  async function loadRuns() {
    const seq = ++S.runsSeq;
    try {
      const data = await getJson(`/api/logs/runs?limit=${RUNS_LIMIT}`);
      if (seq !== S.runsSeq) return;
      const runs = isObj(data) && Array.isArray(data.runs) ? data.runs.filter((r) => isObj(r) && r.run_id) : [];
      S.runs = runs;
      S.runsTotal = isObj(data) && isNum(data.total) ? data.total : runs.length;
      S.runsError = '';
    } catch (err) {
      if (seq !== S.runsSeq) return;
      S.runsError = err.status ? `Couldn’t load the runs: ${err.message}.`
        : 'Can’t reach the console backend. Is it running?';
    }
    S.runsLoaded = true;
    S.lastRefresh = Date.now();
    renderRuns();
    if (!S.refreshTimer) setText(ui.refreshNote, refreshedText());
  }

  function runHref(id) {
    const q = new URLSearchParams();
    q.set('run', id);
    return `/logs?${q.toString()}`;
  }

  function cell(content, cls) {
    return h('td', { class: cls || null }, content);
  }

  function modelsCell(m) {
    const models = isObj(m) ? m : {};
    const item = (abbr, title, v) => h('span', { class: 'mdl', title: `${title}: ${has(v) ? str(v) : 'not in the trace'}` },
      h('abbr', { title, text: abbr }), ' ', has(v) ? h('span', { class: 'mdl-v', text: str(v) }) : na());
    return h('span', { class: 'models' },
      item('L', 'Lightning (planner)', models.lightning),
      item('G', 'Grounder VLM', models.grounder),
      item('S', 'Screen-summary VLM', models.summary),
      has(models.asr) ? item('ASR', 'Speech recognizer', models.asr) : null);
  }

  function runRow(r) {
    const req = oneLine(r.request);
    const link = h('a', {
      class: 'run-link', href: runHref(r.run_id), 'data-run-id': r.run_id, tabindex: '-1', title: req || r.run_id,
    }, req ? h('span', { class: 'run-req', text: req }) : h('span', { class: 'run-req na-text', text: '(no request recorded)' }));
    const p50 = isObj(r.stage_p50_ms) ? r.stage_p50_ms : {};
    const when = isNum(r.started_ts)
      ? h('time', { datetime: isoOf(r.started_ts), title: fmtFull(r.started_ts), text: fmtWhen(r.started_ts) }) : na();
    let task;
    if (has(r.task_id)) task = h('span', { class: 'task-id', title: `Sandbox task ${r.task_id}` }, sv('task '), str(r.task_id));
    else if (r.console_run_id) task = h('span', { class: 'task-id is-free', title: 'Free text: no sandbox check', text: 'free text' });
    else task = h('span', { class: 'task-id is-free' }, h('span', { 'aria-hidden': 'true', text: 'task —' }), sv('no task recorded'));
    const tr = h('tr', { class: 'run-row', 'data-run-id': r.run_id, 'data-status': str(r.status) || 'unknown' },
      cell(when, 'c-time'),
      cell(h('div', { class: 'req-cell' }, link, task), 'c-request'),
      cell(agentBadge(r.agent), 'c-agent'),
      cell(statusChip(r.status), 'c-status'),
      cell(tick(r.success_claimed, 'claimed success', 'did not claim success', 'no claim recorded'), 'num c-check'),
      cell(tick(r.verified, 'verified by the sandbox', 'the sandbox check failed', 'no sandbox check'), 'num c-check'),
      cell(val(r.steps, (v) => nf0.format(v)), 'num'),
      cell(isNum(r.total_ms) ? h('span', { title: fmtMs(r.total_ms), text: fmtMsShort(r.total_ms) }) : na(), 'num'),
      ...STAGES.map((k) => cell(isNum(p50[k]) ? fmtMsShort(p50[k]) : na(), 'num p50')),
      cell(modelsCell(r.models), 'c-models'));
    return tr;
  }

  /** Re-render the table without losing keyboard focus (rows are rebuilt only when their data changed). */
  function renderRuns() {
    const focusedId = ui.runsBody.contains(document.activeElement) ? document.activeElement.dataset.runId : null;
    const old = new Map([...ui.runsBody.children].map((tr) => [tr.dataset.runId, tr]));
    const rows = S.runs.map((r) => {
      const sig = JSON.stringify(r);
      const prev = old.get(r.run_id);
      if (prev && prev.dataset.sig === sig) return prev;
      const tr = runRow(r);
      tr.dataset.sig = sig;
      return tr;
    });
    const same = rows.length === ui.runsBody.children.length && rows.every((tr, i) => ui.runsBody.children[i] === tr);
    if (!same) ui.runsBody.replaceChildren(...rows);
    ui.runsEmpty.hidden = rows.length > 0;
    if (!rows.length) {
      setText(ui.runsEmpty, S.runsError || (S.runsLoaded ? 'No runs yet. Start one from the console.' : 'Loading runs…'));
    }
    ui.runsEmpty.classList.toggle('is-error', Boolean(S.runsError));
    const nMock = S.runs.filter((r) => r.agent === 'mock').length;
    setText(ui.runsMeta, S.runsError && rows.length ? S.runsError
      : rows.length ? `${S.runsTotal > rows.length ? `newest ${rows.length} of ${S.runsTotal}` : rows.length} `
        + `run${rows.length === 1 ? '' : 's'}${nMock ? ` · ${nMock} mock` : ''} · newest first` : '');
    markSelectedRow();
    if (focusedId) {
      const link = runLink(focusedId);
      if (link && document.activeElement !== link) link.focus({ preventScroll: true });
    }
  }

  const runLinks = () => [...ui.runsBody.querySelectorAll('.run-link')];
  const runLink = (id) => runLinks().find((a) => a.dataset.runId === id) || null;

  function markSelectedRow() {
    const links = runLinks();
    let stop = null;
    for (const a of links) {
      const on = a.dataset.runId === S.selectedId;
      a.closest('tr').classList.toggle('is-selected', on);
      if (on) { a.setAttribute('aria-current', 'true'); stop = a; } else a.removeAttribute('aria-current');
    }
    const focused = links.find((a) => a === document.activeElement);
    stop = focused || stop || links[0] || null;
    for (const a of links) a.tabIndex = a === stop ? 0 : -1;
  }

  function onRunsKey(e) {
    const link = e.target.closest && e.target.closest('.run-link');
    if (!link) return;
    const links = runLinks();
    const i = links.indexOf(link);
    let j = null;
    if (e.key === 'ArrowDown') j = Math.min(i + 1, links.length - 1);
    else if (e.key === 'ArrowUp') j = Math.max(i - 1, 0);
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = links.length - 1;
    else if (e.key === 'PageDown') j = Math.min(i + 10, links.length - 1);
    else if (e.key === 'PageUp') j = Math.max(i - 10, 0);
    else if (e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault();
      selectRun(link.dataset.runId);
      return;
    } else if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && !e.altKey && !e.ctrlKey && !e.metaKey) {
      // a narrow window: the table is wider than its box (logs.css min-width); scroll it sideways
      const box = ui.runsScroll;
      if (box.scrollWidth > box.clientWidth + 1) {
        e.preventDefault();
        box.scrollLeft += (e.key === 'ArrowRight' ? 1 : -1) * Math.max(120, Math.round(box.clientWidth / 3));
      }
      return;
    }
    if (j === null || e.altKey || e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    for (const a of links) a.tabIndex = a === links[j] ? 0 : -1;
    links[j].focus();
    revealRow(links[j].closest('tr'));
  }

  /** Keep the focused row visible inside the scrolling table (under its sticky header). */
  function revealRow(tr) {
    const box = ui.runsScroll.getBoundingClientRect();
    const head = ui.runsTable.tHead ? ui.runsTable.tHead.getBoundingClientRect().height : 0;
    const r = tr.getBoundingClientRect();
    if (r.top < box.top + head) ui.runsScroll.scrollTop -= box.top + head - r.top + 2;
    else if (r.bottom > box.bottom) ui.runsScroll.scrollTop += r.bottom - box.bottom + 2;
  }

  function onRunsClick(e) {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const tr = e.target.closest && e.target.closest('tr.run-row');
    if (!tr) return;
    if (e.target.closest('a')) e.preventDefault();
    else if (String(window.getSelection ? window.getSelection() : '')) return;  // selecting text in a cell
    const link = runLink(tr.dataset.runId);
    selectRun(tr.dataset.runId);
    if (link && document.activeElement !== link) link.focus({ preventScroll: true });
  }

  function selectRun(id) {
    if (!id) return;
    if (id === S.selectedId && S.detailFor === id) {
      markSelectedRow();
      return;
    }
    S.selectedId = id;
    writeUrl(true);
    markSelectedRow();
    loadDetail(id, { announceIt: true });
  }

  // --- the selected run -------------------------------------------------------------------------------------------
  function showPlaceholder() {
    window.clearTimeout(S.refreshTimer);
    S.refreshTimer = 0;
    S.detail = null;
    S.detailFor = null;
    ui.runSection.hidden = true;
    ui.runPlaceholder.hidden = false;
    setText(ui.runPlaceholder, S.runs.length ? 'Select a run above to see its summary and every event.'
      : 'Runs appear here after the console runs a request.');
  }

  async function loadDetail(id, { refresh: isRefresh = false, announceIt = false } = {}) {
    const seq = ++S.detailSeq;
    if (!isRefresh) {
      ui.runSection.setAttribute('aria-busy', 'true');
      if (S.detailFor !== id) setText(ui.eventsMeta, 'Loading…');
    }
    let data;
    let text;
    try {
      if (!RUN_ID_RE.test(id)) {
        const err = new Error('not a valid run id');
        err.status = 404;
        throw err;
      }
      ({ data, text } = await fetchJson(`/api/logs/runs/${encodeURIComponent(id)}`));
    } catch (err) {
      if (seq !== S.detailSeq) return;
      ui.runSection.removeAttribute('aria-busy');
      if (isRefresh && S.detailFor === id) {  // keep what we have; say the refresh failed
        setText(ui.refreshNote, `Refresh failed: ${err.message}`);
        scheduleRefresh();
        return;
      }
      renderMissing(id, err);
      return;
    }
    if (seq !== S.detailSeq) return;
    ui.runSection.removeAttribute('aria-busy');
    const fresh = S.detailFor !== id;
    const wasRunning = !fresh && S.detail && isObj(S.detail.summary) && S.detail.summary.status === 'running';
    S.detail = normalizeDetail(data, id, text);
    S.detailFor = id;
    if (fresh) {
      S.expanded.clear();
      S.evEls = [];
      S.activeIdx = null;
      ui.eventList.replaceChildren();
      ui.copyFallbackWrap.hidden = true;
      setText(ui.copyStatus, '');
    }
    renderDetail();
    scheduleRefresh();
    const s = S.detail.summary;
    if (announceIt) {
      announce(`Showing run ${id}: ${oneLine(s.request) || 'no request'}, ${(STATUS[s.status] || [s.status])[0]}, `
        + `${S.detail.events.length} events.`);
    } else if (wasRunning && s.status !== 'running') {  // the auto-refresh saw the run end: say so once
      const verdict = s.verified === true ? ', verified' : s.verified === false ? ', not verified' : '';
      announce(`Run ${id} finished: ${(STATUS[s.status] || [s.status || 'unknown'])[0]}${verdict}, `
        + `${S.detail.events.length} events.`);
    }
  }

  function normalizeDetail(data, id, text) {
    const d = isObj(data) ? data : {};
    const summary = isObj(d.summary) ? d.summary : { run_id: id };
    const events = Array.isArray(d.events) ? d.events.map((e) => (isObj(e) ? e : { kind: 'event', payload: e })) : [];
    return {
      raw: data, text: typeof text === 'string' ? text : null, summary, console: isObj(d.console) ? d.console : null,
      events,
    };
  }

  function renderMissing(id, err) {
    window.clearTimeout(S.refreshTimer);
    S.refreshTimer = 0;
    S.detail = null;
    S.detailFor = null;
    ui.runPlaceholder.hidden = true;
    ui.runSection.hidden = false;
    setText(ui.runHeading, `Run ${id}`);
    ui.runBadges.replaceChildren();
    ui.runSummary.replaceChildren();
    ui.kindChips.replaceChildren();
    ui.eventList.replaceChildren();
    S.evEls = [];
    ui.copyBtn.disabled = true;
    ui.downloadLink.hidden = true;
    ui.eventsEmpty.hidden = true;
    setText(ui.eventsMeta, '');
    ui.runError.textContent = err.status === 404
      ? `There is no run “${id}” in the trace folders.`
      : `Couldn’t load run ${id}: ${err.message}`;
  }

  function renderDetail() {
    const d = S.detail;
    const s = d.summary;
    const id = S.detailFor;
    ui.runPlaceholder.hidden = true;
    ui.runSection.hidden = false;
    ui.runError.textContent = '';
    ui.copyBtn.disabled = false;
    ui.downloadLink.hidden = false;
    ui.downloadLink.href = `/api/logs/runs/${encodeURIComponent(id)}/download`;
    ui.downloadLink.setAttribute('download', `run-${id}.json`);
    setText(ui.runHeading, `Run ${id}`);
    ui.runBadges.replaceChildren(
      agentBadge(s.agent),
      statusChip(s.status),
      s.verified === true ? h('span', { class: 'status-chip tone-ok', text: '✓ Verified' })
        : s.verified === false ? h('span', { class: 'status-chip tone-bad', text: '✗ Not verified' })
          : h('span', { class: 'status-chip tone-muted',
            text: d.console && !d.console.task_id ? 'Free text: no sandbox check' : 'No verdict recorded' }),
    );
    renderSummary();
    renderFilters();
    renderEvents();
  }

  function tile(label, value, note, tone) {
    return h('div', { class: `tile${tone ? ` tone-${tone}` : ''}` },
      h('dt', { class: 'tile-label', text: label }),
      h('dd', { class: 'tile-value' }, value),
      note ? h('dd', { class: 'tile-note' }, note) : null);
  }

  function kvRow(label, ...value) {
    const vals = value.flat().filter((v) => v !== null && v !== undefined && v !== '');
    return h('div', { class: 'kv-row' }, h('dt', { text: label }), h('dd', {}, vals.length ? vals : [na()]));
  }

  function joinNodes(parts, sep = ' · ') {
    const out = [];
    parts.filter(Boolean).forEach((p, i) => {
      if (i) out.push(sep);
      out.push(p);
    });
    return out;
  }

  function renderSummary() {
    const d = S.detail;
    const s = d.summary;
    const c = d.console;
    const p50 = isObj(s.stage_p50_ms) ? s.stage_p50_ms : {};
    const models = isObj(s.models) ? s.models : {};
    const startEv = d.events.find((e) => e.kind === 'start');
    const cfg = startEv && isObj(startEv.payload) && isObj(startEv.payload.config) ? startEv.payload.config : {};

    const status = STATUS[s.status] || [str(s.status) || 'Unknown', 'muted'];
    const verifiedValue = s.verified === true ? '✓ Passed' : s.verified === false ? '✗ Failed' : na('no sandbox check');
    const verifiedNote = s.verified === null || s.verified === undefined
      ? (c && !c.task_id ? 'free text: no sandbox check' : 'no verdict in the trace')
      : 'sandbox state, checked by the console';
    const claimedValue = s.success_claimed === true ? '✓ Yes' : s.success_claimed === false ? '✗ No' : na();
    const errs = isNum(s.n_errors) ? s.n_errors : null;
    const inv = isNum(s.n_invalid_tool_calls) ? s.n_invalid_tool_calls : null;
    const bad = isNum(s.n_bad_lines) && s.n_bad_lines > 0 ? s.n_bad_lines : 0;
    const eventsNote = [errs !== null ? `${errs} error${errs === 1 ? '' : 's'}` : null,
      inv !== null ? `${inv} invalid call${inv === 1 ? '' : 's'}` : null,
      bad ? `${bad} unreadable line${bad === 1 ? '' : 's'} skipped` : null].filter(Boolean).join(' · ');

    const tiles = h('dl', { class: 'tiles' },
      tile('Status', status[0], s.status === 'running' ? 'still going' : (REASON_WORDS[s.status] || ''), status[1]),
      tile('Verified', verifiedValue, verifiedNote, s.verified === true ? 'ok' : s.verified === false ? 'bad' : ''),
      tile('Claimed success', claimedValue, 'what the agent said', ''),
      tile('Steps', val(s.steps, (v) => nf0.format(v)), cfg.max_steps ? `of at most ${cfg.max_steps}` : ''),
      tile('Total time', isNum(s.total_ms) ? fmtMs(s.total_ms) : na(), 'first to last agent event'),
      tile('Agent events', val(s.n_events, (v) => nf0.format(v)), eventsNote,
        (errs || inv || bad) ? 'warn' : ''),
    );

    const stageList = h('ul', { class: 'stages' }, STAGES.map((k) => h('li', { class: 'stage' },
      h('span', { class: 'stage-name', title: STAGE_WORDS[k], text: k }),
      h('span', { class: 'stage-val' }, isNum(p50[k]) ? fmtMs(p50[k]) : na()))));

    const modelBits = [
      ['Lightning', models.lightning], ['Grounder', models.grounder], ['Screen summary', models.summary],
      ['Speech', models.asr],
    ].map(([k, v]) => h('span', { class: 'kv-pair' }, h('span', { class: 'kv-k', text: `${k} ` }),
      has(v) ? h('span', { class: 'kv-v', text: str(v) }) : na()));
    const extraModel = [];
    if (isObj(cfg.lightning) && cfg.lightning.root) {
      extraModel.push(h('span', { class: 'kv-sub', text: `Lightning endpoint serves ${cfg.lightning.root}` }));
    }
    if (cfg.model_family) extraModel.push(h('span', { class: 'kv-sub', text: `VLM family ${cfg.model_family}` }));

    const shas = isObj(s.prompt_sha) ? s.prompt_sha : {};
    const promptBits = [
      ['version', s.prompt_version], ['Lightning SHA', shas.lightning], ['Grounder SHA', shas.grounder],
      ['Screen summary SHA', shas.summary],
    ].map(([k, v]) => h('span', { class: 'kv-pair' }, h('span', { class: 'kv-k', text: `${k} ` }),
      has(v) ? h('span', { class: 'kv-v', text: str(v) }) : na()));

    const opts = c && isObj(c.options) ? c.options : null;
    const optBits = opts ? Object.entries(opts).map(([k, v]) => h('span', { class: 'kv-pair' },
      h('span', { class: 'kv-k', text: `${k} ` }), h('span', { class: 'kv-v', text: str(v) }))) : [];

    const voice = isObj(s.voice) ? s.voice : (c && isObj(c.voice) ? c.voice : null);
    const voiceBits = voice ? [
      h('q', { class: 'voice-text', text: oneLine(voice.text) }),
      h('span', { class: 'kv-sub' }, joinNodes([
        has(voice.model) ? `model ${voice.model}` : null,
        isNum(voice.latency_ms) ? `transcribed in ${fmtMs(voice.latency_ms)}` : null,
        isNum(voice.duration_s) ? `${voice.duration_s.toFixed(1)} s of audio` : null,
      ])),
    ] : [c ? h('span', { class: 'muted-text', text: 'no voice input recorded' }) : na()];

    let verdictBits;
    const verdict = c && isObj(c.verdict) ? c.verdict : null;
    if (has(s.verdict_explanation) || verdict) {
      verdictBits = [h('span', { text: oneLine(s.verdict_explanation ?? (verdict && verdict.explanation)) || '—' }),
        h('span', { class: 'kv-sub', text: oneLine(verdict && verdict.label) || VERDICT_LABEL })];
    } else {
      verdictBits = [na('no sandbox verdict')];
    }

    let task;
    if (has(s.task_id)) task = h('span', { class: 'task-id', text: str(s.task_id) });
    else if (c) task = h('span', { class: 'muted-text', text: 'free text (no sandbox check)' });
    else task = na('no console record for this run');

    const ids = [h('span', { class: 'kv-pair' }, h('span', { class: 'kv-k', text: 'agent ' }),
      h('code', { text: str(s.run_id || S.detailFor) }))];
    ids.push(h('span', { class: 'kv-pair' }, h('span', { class: 'kv-k', text: 'console ' }),
      has(s.console_run_id) ? h('code', { text: str(s.console_run_id) }) : na()));

    const when = [];
    if (isNum(s.started_ts)) when.push(h('time', { datetime: isoOf(s.started_ts), text: fmtFull(s.started_ts) }));
    if (isNum(s.ended_ts)) when.push(h('span', { class: 'kv-sub', text: `to ${fmtClock(s.ended_ts)}` }));

    const startUrl = (c && c.start_url) || (startEv && isObj(startEv.payload) && startEv.payload.start_url);
    const consoleNote = c ? null : h('span', { class: 'kv-sub', text: 'no console.json: run outside the console, or before it recorded runs' });

    const grid = h('dl', { class: 'kv' },
      kvRow('Request', has(s.request) ? h('q', { class: 'req-text', text: oneLine(s.request) }) : null),
      kvRow('Task', task),
      kvRow('Verdict', verdictBits),
      kvRow('Voice', voiceBits),
      kvRow('Models', h('span', { class: 'kv-line' }, joinNodes(modelBits)), extraModel),
      kvRow('Prompts', h('span', { class: 'kv-line' }, joinNodes(promptBits)),
        h('span', { class: 'kv-sub', text: s.agent === 'mock' ? 'the mock sends no prompts'
          : 'SHA = first 10 hex digits of the SHA-256 of the exact prompt (from the events)' })),
      kvRow('p50 per stage', stageList),
      kvRow('Options', optBits.length ? h('span', { class: 'kv-line' }, joinNodes(optBits)) : null,
        consoleNote),
      kvRow('Agent', agentBadge(s.agent), h('span', { class: 'kv-sub', text: s.agent === 'mock'
        ? 'scripted replay: no Lightning or VLM calls, so no model latencies' : s.agent === 'real'
          ? 'Lightning plans, the VLM summarizes the screen and grounds each click' : '' })),
      kvRow('When', when),
      kvRow('Start page', startUrl ? h('code', { text: displayUrl(startUrl) }) : null),
      kvRow('Run ids', h('span', { class: 'kv-line' }, joinNodes(ids))),
    );
    ui.runSummary.replaceChildren(tiles, grid);
  }

  // --- filters ------------------------------------------------------------------------------------------------------
  function presentKinds() {
    const counts = new Map();
    for (const ev of S.detail ? S.detail.events : []) counts.set(str(ev.kind), (counts.get(str(ev.kind)) || 0) + 1);
    return [...counts.entries()].sort((a, b) => kindRank(a[0]) - kindRank(b[0]) || a[0].localeCompare(b[0]));
  }

  function presentModels() {
    const counts = new Map();
    for (const ev of S.detail ? S.detail.events : []) {
      const m = eventModel(ev);
      if (has(m)) counts.set(str(m), (counts.get(str(m)) || 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }

  function renderFilters() {
    if (!ui.kindChips) return;
    const kinds = presentKinds();
    const focusedKind = ui.kindChips.contains(document.activeElement) ? document.activeElement.value : null;
    const sig = JSON.stringify(kinds);
    if (ui.kindChips.dataset.sig !== sig) {
      ui.kindChips.dataset.sig = sig;
      ui.kindChips.replaceChildren(...kinds.map(([k, n]) => {
        const id = `kind-${k.replace(/[^A-Za-z0-9_-]/g, '_')}`;
        return h('span', { class: 'kind-chip', 'data-kind': KIND_ORDER.includes(k) ? k : 'other' },
          h('input', { type: 'checkbox', id, value: k }),
          h('label', { for: id }, h('span', { class: 'chip-mark', 'aria-hidden': 'true' }),
            h('span', { class: 'chip-name', text: kindLabel(k) }), ' ', h('span', { class: 'chip-count', text: String(n) }),
            sv(` event${n === 1 ? '' : 's'}`)));
      }));
    }
    for (const input of ui.kindChips.querySelectorAll('input')) {
      input.checked = !S.filters.kinds || S.filters.kinds.has(input.value);
    }
    if (focusedKind) {
      const back = [...ui.kindChips.querySelectorAll('input')].find((i) => i.value === focusedKind);
      if (back && document.activeElement !== back) back.focus();
    }

    const models = presentModels();
    const want = S.filters.model;
    const opts = [['', `All models`]];
    for (const [m, n] of models) opts.push([m, `${m} (${n})`]);
    if (want && !models.some(([m]) => m === want)) opts.push([want, `${want} (not in this run)`]);
    const msig = JSON.stringify(opts);
    if (ui.modelFilter.dataset.sig !== msig) {
      ui.modelFilter.dataset.sig = msig;
      ui.modelFilter.replaceChildren(...opts.map(([v, t]) => new Option(t, v)));
    }
    ui.modelFilter.value = want;
    if (ui.eventSearch.value !== S.filters.q) ui.eventSearch.value = S.filters.q;
  }

  function onKindChange(e) {
    const input = e.target;
    if (!input || input.type !== 'checkbox') return;
    const all = presentKinds().map(([k]) => k);
    const set = S.filters.kinds ? new Set(S.filters.kinds) : new Set(all);
    if (input.checked) set.add(input.value);
    else set.delete(input.value);
    S.filters.kinds = all.every((k) => set.has(k)) && [...set].every((k) => all.includes(k)) ? null : set;
    applyFilters();
    writeUrl(false);
  }

  function filtersActive() {
    return Boolean(S.filters.kinds || S.filters.model || S.filters.q.trim());
  }

  // --- event log ----------------------------------------------------------------------------------------------------
  function eventModel(ev) {
    if (ev.model !== undefined) return ev.model;
    return null;
  }

  function eventSig(ev) {
    let n = 0;
    try { n = JSON.stringify(ev).length; } catch { n = -1; }
    return `${str(ev.kind)}|${str(ev.ts)}|${str(ev.step)}|${n}`;
  }

  function renderEvents() {
    const events = S.detail.events;
    const focusIdx = ui.eventList.contains(document.activeElement)
      ? Number((document.activeElement.closest('.ev') || {}).dataset?.idx) : null;
    let rebuildFrom = events.length;
    for (let i = 0; i < Math.max(events.length, S.evEls.length); i += 1) {
      if (i >= events.length || i >= S.evEls.length || S.evEls[i].sig !== eventSig(events[i])) {
        rebuildFrom = i;
        break;
      }
    }
    if (rebuildFrom < S.evEls.length) {  // something before the end changed: drop from there on
      for (const rec of S.evEls.slice(rebuildFrom)) rec.li.remove();
      S.evEls.length = rebuildFrom;
    }
    for (let i = S.evEls.length; i < events.length; i += 1) {
      const rec = buildEvent(events[i], i);
      S.evEls.push(rec);
      ui.eventList.append(rec.li);
    }
    applyFilters({ quiet: true });
    if (Number.isInteger(focusIdx) && S.evEls[focusIdx] && document.activeElement === document.body) {
      S.evEls[focusIdx].row.focus({ preventScroll: true });
    }
  }

  function buildEvent(ev, idx) {
    const p = isObj(ev.payload) ? ev.payload : {};
    const desc = describe(ev);
    const lat = eventLatency(ev);
    const model = eventModel(ev);
    const src = str(ev.source) || 'agent';
    const detailsId = `ev-${idx}-details`;
    const tone = eventTone(ev);
    const row = h('button', {
      type: 'button', class: 'ev-row', 'aria-expanded': 'false', 'aria-controls': detailsId, tabindex: '-1',
      'data-idx': String(idx),
    },
    h('span', { class: 'ev-t' }, has(ev.t_ms) && isNum(ev.t_ms) ? [sv('time '), fmtT(ev.t_ms)] : na('no time')),
    h('span', { class: 'ev-step' }, Number.isInteger(ev.step) ? [sv('step '), String(ev.step)] : na('no step')),
    h('span', { class: 'ev-kind' }, kindBadge(ev.kind)),
    h('span', { class: `ev-src src-${src === 'console' ? 'console' : 'agent'}` }, sv('from '), src),
    h('span', { class: 'ev-model', title: has(model) ? str(model) + (has(ev.prompt_sha) ? ` · prompt SHA ${ev.prompt_sha}` : '') : null },
      has(model) ? [sv('model '), str(model)] : na('no model')),
    h('span', { class: 'ev-lat' }, isNum(lat) ? [sv('latency '), fmtMs(lat)] : na('no latency')),
    h('span', { class: 'ev-desc' }, sv(': '), desc),
    h('span', { class: 'ev-caret', 'aria-hidden': 'true' }));
    const details = h('div', { class: 'ev-details', id: detailsId, hidden: true });
    const li = h('li', {
      class: 'ev', 'data-idx': String(idx), 'data-kind': str(ev.kind), 'data-source': src,
      'data-tone': tone || null,
    }, row, details);
    let hay = `${desc} ${str(ev.kind)} ${kindLabel(ev.kind)} ${str(model)} ${src}`;
    try { hay += ` ${JSON.stringify(p)}`; } catch { /* unserializable payload */ }
    const rec = { li, row, details, sig: eventSig(ev), hay: hay.toLowerCase(), built: false, visible: true };
    if (S.expanded.has(idx)) setExpanded(rec, idx, true);
    return rec;
  }

  function setExpanded(rec, idx, open) {
    if (open && !rec.built) {
      rec.details.replaceChildren(...eventDetails(S.detail.events[idx], idx));
      rec.built = true;
    }
    rec.details.hidden = !open;
    rec.row.setAttribute('aria-expanded', String(open));
    rec.li.classList.toggle('is-open', open);
    if (open) S.expanded.add(idx);
    else S.expanded.delete(idx);
  }

  function applyFilters({ quiet = false } = {}) {
    if (!S.detail) return;
    const { kinds, model } = S.filters;
    const q = S.filters.q.trim().toLowerCase();
    let shown = 0;
    S.evEls.forEach((rec, i) => {
      const ev = S.detail.events[i];
      const ok = (!kinds || kinds.has(str(ev.kind))) && (!model || str(eventModel(ev)) === model)
        && (!q || rec.hay.includes(q));
      rec.visible = ok;
      if (rec.li.hidden === ok) rec.li.hidden = !ok;
      if (ok) shown += 1;
    });
    const total = S.evEls.length;
    const hadFocus = ui.eventList.contains(document.activeElement);
    if (S.activeIdx === null || !S.evEls[S.activeIdx] || !S.evEls[S.activeIdx].visible) {
      const first = S.evEls.findIndex((r) => r.visible);
      setActiveEvent(first >= 0 ? first : null, hadFocus);
    } else {
      setActiveEvent(S.activeIdx, false);
    }
    ui.eventsEmpty.hidden = shown > 0 || total === 0;
    if (shown === 0 && total > 0) {
      setText(ui.eventsEmpty, 'No events match these filters. Use “Show all events” to see them all.');
    }
    if (total === 0) {
      ui.eventsEmpty.hidden = false;
      setText(ui.eventsEmpty, 'This run has no events yet.');
    }
    const text = total === 0 ? '0 events' : shown === total ? `${total} event${total === 1 ? '' : 's'}`
      : `Showing ${shown} of ${total} events`;
    setText(ui.eventsMeta, text);
    ui.clearFilters.disabled = !filtersActive();
    if (!quiet) announce(text);  // the one live region for filter results (#events-meta itself is not live)
  }

  function visibleRows() {
    return S.evEls.filter((r) => r.visible).map((r) => r.row);
  }

  function setActiveEvent(idx, focus) {
    S.activeIdx = Number.isInteger(idx) ? idx : null;
    for (const [i, rec] of S.evEls.entries()) {
      const t = i === S.activeIdx ? 0 : -1;
      if (rec.row.tabIndex !== t) rec.row.tabIndex = t;
    }
    if (focus && S.activeIdx !== null) S.evEls[S.activeIdx].row.focus();
  }

  function onEventKey(e) {
    const row = e.target.closest && e.target.closest('.ev-row');
    if (!row || e.target !== row || e.altKey || e.ctrlKey || e.metaKey) return;
    const rows = visibleRows();
    const i = rows.indexOf(row);
    let j = null;
    if (e.key === 'ArrowDown') j = Math.min(i + 1, rows.length - 1);
    else if (e.key === 'ArrowUp') j = Math.max(i - 1, 0);
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = rows.length - 1;
    else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      const idx = Number(row.dataset.idx);
      const open = e.key === 'ArrowRight';
      if (S.expanded.has(idx) !== open) {
        e.preventDefault();
        setExpanded(S.evEls[idx], idx, open);
      }
      return;
    }
    if (j === null || i < 0) return;
    e.preventDefault();
    setActiveEvent(Number(rows[j].dataset.idx), true);
    rows[j].scrollIntoView({ block: 'nearest' });
  }

  function onEventClick(e) {
    const row = e.target.closest && e.target.closest('.ev-row');
    if (row) {
      const idx = Number(row.dataset.idx);
      setActiveEvent(idx, false);
      setExpanded(S.evEls[idx], idx, !S.expanded.has(idx));
      return;
    }
    const thumb = e.target.closest && e.target.closest('.thumb-btn');
    if (thumb) openImage(thumb);
  }

  // --- event details ----------------------------------------------------------------------------------------------
  function dl(...rows) {
    return h('dl', { class: 'd-kv' }, rows.filter(Boolean));
  }

  function dRow(label, value, { mono = false } = {}) {
    const v = value === null || value === undefined || value === '' ? na()
      : value instanceof Node || Array.isArray(value) ? value : h('span', { class: mono ? 'mono' : null, text: str(value) });
    return h('div', { class: 'd-row' }, h('dt', { text: label }), h('dd', {}, v));
  }

  function yesNo(v, yes = 'yes', no = 'no') {
    if (v === true) return h('span', { class: 'yn yn-yes', text: `✓ ${yes}` });
    if (v === false) return h('span', { class: 'yn yn-no', text: `✗ ${no}` });
    return null;
  }

  function changedValue(c) {
    if (c === true) return h('span', { class: 'yn yn-yes', text: '✓ changed' });
    if (c === false) return h('span', { class: 'yn yn-warn', text: '⚠ no change' });
    return null;
  }

  function codeBlock(text, { label, open = false, cls = '' } = {}) {
    const pre = h('pre', { class: `d-pre ${cls}`.trim(), tabindex: '0', text: str(text) });
    if (!label) return pre;
    const det = h('details', { class: 'd-more' }, h('summary', { text: label }), pre);
    if (open) det.open = true;
    return det;
  }

  function timingsValue(t) {
    if (!isObj(t) || !Object.keys(t).length) return null;
    return Object.entries(t).map(([k, v]) => `${k} ${isNum(v) ? fmtMs(v) : str(v)}`).join(' · ');
  }

  /** A model name plus where it came from (the event itself, or the run's config). */
  function modelValue(ev, payloadModel) {
    const m = eventModel(ev);
    if (!has(m)) return null;
    const from = has(payloadModel) ? 'from this event' : 'from the run config';
    return [h('span', { class: 'mono', text: str(m) }), ' ', h('span', { class: 'd-sub', text: `(${from})` })];
  }

  function runConfig() {
    const startEv = S.detail.events.find((e) => e.kind === 'start');
    return startEv && isObj(startEv.payload) && isObj(startEv.payload.config) ? startEv.payload.config : {};
  }

  function promptVersion(p, { config = true } = {}) {
    if (has(p.prompt_version)) return [h('span', { class: 'mono', text: str(p.prompt_version) })];
    const cfg = runConfig();
    if (config && has(cfg.prompt_version)) {
      return [h('span', { class: 'mono', text: str(cfg.prompt_version) }), ' ',
        h('span', { class: 'd-sub', text: '(agent prompt, from the run config)' })];
    }
    return null;
  }

  /** A prompt_sha (agent/trace.py: the first 10 hex of the SHA-256 of the exact prompt). The mock writes "mock". */
  function shaValue(sha, mock) {
    if (!has(sha)) return null;
    if (mock) return [h('span', { class: 'mono', text: str(sha) }), ' ', h('span', { class: 'd-sub', text: '(no prompt was sent)' })];
    return [h('span', { class: 'mono', title: 'first 10 hex digits of the SHA-256 of the exact prompt', text: str(sha) })];
  }

  function thumb(url, title, alt) {
    if (!url) return null;
    return h('button', { type: 'button', class: 'thumb-btn', 'data-src': url, 'data-title': title, 'aria-label': `Enlarge: ${title}` },
      h('img', { src: url, alt, loading: 'lazy', decoding: 'async', width: '320', height: '200' }),
      h('span', { class: 'thumb-cap', 'aria-hidden': 'true', text: 'Click to enlarge' }));
  }

  function usageValue(u) {
    if (!isObj(u)) return null;
    const parts = [];
    if (isNum(u.prompt_tokens)) parts.push(`${nf0.format(u.prompt_tokens)} prompt`);
    if (isNum(u.completion_tokens)) parts.push(`${nf0.format(u.completion_tokens)} completion`);
    const r = isObj(u.completion_tokens_details) ? u.completion_tokens_details.reasoning_tokens : undefined;
    if (isNum(r)) parts.push(`${nf0.format(r)} reasoning`);
    if (isNum(u.total_tokens)) parts.push(`${nf0.format(u.total_tokens)} total`);
    return parts.length ? parts.join(' · ') : compactJson(u);
  }

  function statsValue(stats) {
    if (!isObj(stats)) return null;
    const scal = [];
    const lat = [];
    for (const [k, v] of Object.entries(stats)) {
      if (k === 'latency_ms' && isObj(v)) {
        for (const [stage, xs] of Object.entries(v)) {
          const arr = Array.isArray(xs) ? xs.filter(isNum) : [];
          if (!arr.length) continue;
          const sorted = [...arr].sort((a, b) => a - b);
          const mid = sorted.length % 2 ? sorted[(sorted.length - 1) / 2]
            : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
          lat.push(h('tr', {}, h('th', { scope: 'row', text: stage }), h('td', { class: 'num', text: String(arr.length) }),
            h('td', { class: 'num', text: fmtMs(mid) }), h('td', { class: 'num', text: fmtMs(sorted[sorted.length - 1]) })));
        }
      } else if (!isObj(v) && !Array.isArray(v)) {
        scal.push(`${k.replace(/^n_/, '')} ${str(v)}`);
      } else {
        scal.push(`${k} ${compactJson(v, 80)}`);
      }
    }
    const out = [];
    if (scal.length) out.push(h('p', { class: 'd-stats', text: scal.join(' · ') }));
    if (lat.length) {
      out.push(h('table', { class: 'd-lat' },
        h('caption', { text: 'Latency recorded by the agent, per stage' }),
        h('thead', {}, h('tr', {}, h('th', { scope: 'col', text: 'stage' }), h('th', { scope: 'col', class: 'num', text: 'calls' }),
          h('th', { scope: 'col', class: 'num', text: 'p50' }), h('th', { scope: 'col', class: 'num', text: 'max' }))),
        h('tbody', {}, lat)));
    }
    return out.length ? out : null;
  }

  function eventDetails(ev, idx) {
    const p = isObj(ev.payload) ? ev.payload : {};
    const parts = [];
    const stepWord = Number.isInteger(ev.step) ? `step ${ev.step}` : 'this event';
    switch (ev.kind) {
      case 'voice':
        parts.push(dl(
          dRow('Speech model', has(p.model) ? str(p.model) : null, { mono: true }),
          dRow('Transcribed in', isNum(p.latency_ms) ? fmtMs(p.latency_ms) : null),
          dRow('Audio length', isNum(p.duration_s) ? `${p.duration_s.toFixed(2)} s` : null),
          dRow('Heard', has(p.text) ? h('q', { class: 'voice-text', text: str(p.text) }) : null),
          dRow('Note', 'Recorded by the console only. The agent gets the request text, never the audio.'),
        ));
        break;
      case 'start': {
        const cfg = isObj(p.config) ? p.config : {};
        parts.push(dl(
          dRow('Request', has(p.request) ? h('q', { text: str(p.request) }) : null),
          dRow('Start URL', has(p.start_url) ? displayUrl(p.start_url) : null, { mono: true }),
          dRow('Lightning', isObj(cfg.lightning) ? [cfg.lightning.model ? h('span', { class: 'mono', text: str(cfg.lightning.model) }) : na(),
            cfg.lightning.root ? h('span', { class: 'd-sub', text: ` serving ${cfg.lightning.root}` }) : null]
            : has(cfg.lightning) ? `${str(cfg.lightning)} (console option)` : null),
          dRow('Grounder', has(cfg.grounder_model) ? str(cfg.grounder_model) : null, { mono: true }),
          dRow('Screen summary', has(cfg.summary_model) ? str(cfg.summary_model) : null, { mono: true }),
          dRow('Thinking', has(cfg.thinking) ? str(cfg.thinking) : null),
          dRow('Prompt version', promptVersion(cfg, { config: false })),
          dRow('Step limit', has(cfg.max_steps) ? str(cfg.max_steps) : null),
        ));
        parts.push(codeBlock(prettyJson(p.config), { label: 'Full run config (JSON)' }));
        break;
      }
      case 'observation': {
        const o = isObj(p.observation) ? p.observation : {};
        const sum = isObj(p.summary) ? p.summary : {};
        const t = isObj(ev.timings_ms) ? ev.timings_ms : {};
        const reused = sum.reused === true;
        const tokens = usageValue(sum.usage);
        parts.push(dl(
          dRow('Summary model', modelValue(ev, sum.model)),
          reused ? dRow('Summary call', h('span', { class: 'yn yn-warn', text: 'reused: no new summarizer call' })) : null,
          dRow('Prompt version', promptVersion(sum, { config: false })),
          dRow('Prompt SHA', shaValue(sum.prompt_sha, p.mock === true)),
          dRow('Tokens', tokens && reused ? [tokens, ' ', h('span', { class: 'd-sub', text: '(the earlier call that wrote it)' })]
            : tokens),
          dRow('Summary latency', isNum(t.summary) ? fmtMs(t.summary) : null),
          dRow('Page', [has(o.title) ? h('span', { text: str(o.title) }) : na(), ' ',
            has(o.url) ? h('span', { class: 'mono d-sub', text: displayUrl(o.url) }) : null]),
          dRow('Screen changed', changedValue(o.screen_changed)),
          dRow('Last action', has(o.last_action_result) ? str(o.last_action_result) : null),
          dRow('Timings', timingsValue(t)),
          sum.error ? dRow('Summary error', str(sum.error)) : null,
        ));
        const text = str(o.screen_summary || (typeof p.summary === 'string' ? p.summary : ''));
        if (text) parts.push(codeBlock(text, { label: 'The screen summary Lightning read (raw text)', open: true }));
        if (has(p.text) && p.text !== text) parts.push(codeBlock(p.text, { label: 'The full observation text sent to Lightning' }));
        if (ev.screenshot_url) {
          parts.push(h('div', { class: 'd-thumbs' },
            thumb(ev.screenshot_url, `Screenshot after ${stepWord}`, `Screenshot the agent observed at ${stepWord}`)));
        }
        break;
      }
      case 'plan': {
        const calls = Array.isArray(p.tool_calls) ? p.tool_calls : [];
        parts.push(dl(
          dRow('Model', modelValue(ev, p.model)),
          dRow('Prompt version', promptVersion(p)),
          dRow('Prompt SHA', shaValue(p.prompt_sha, p.mock === true)),
          dRow('Thinking', p.thinking === true ? 'on for this call' : p.thinking === false ? 'off for this call' : null),
          dRow('Finish reason', has(p.finish_reason) ? str(p.finish_reason) : null, { mono: true }),
          dRow('Tokens', usageValue(p.usage)),
          dRow('Latency', isObj(ev.timings_ms) && isNum(ev.timings_ms.plan) ? fmtMs(ev.timings_ms.plan) : null),
          has(p.wait_ms) ? dRow('Waited', isNum(p.wait_ms) ? fmtMs(p.wait_ms) : str(p.wait_ms)) : null,
          dRow(calls.length > 1 ? `Tool calls (${calls.length})` : 'Tool call', calls.length
            ? calls.map((c, i) => [i ? h('br') : null, h('span', { class: 'd-call', text: callWords(c) })])
            : h('span', { class: 'warn-text', text: 'none (Lightning replied without a tool call)' })),
        ));
        calls.forEach((c, i) => parts.push(codeBlock(prettyJson(parseArgs(c && c.arguments)), {
          label: `${calls.length > 1 ? `Call ${i + 1}: ` : ''}${str(c && c.name) || '?'} arguments (JSON)`, open: true,
        })));
        const reasoning = str(p.reasoning);
        if (reasoning) parts.push(codeBlock(reasoning, { label: `Reasoning (${nf0.format(reasoning.length)} characters)`, cls: 'prose' }));
        else parts.push(h('p', { class: 'd-note', text: p.thinking === true ? 'Reasoning: none recorded.' : 'Reasoning: none (thinking was off).' }));
        if (has(p.content) && oneLine(p.content)) parts.push(codeBlock(str(p.content), { label: 'Message content', cls: 'prose' }));
        break;
      }
      case 'tool_call':
        parts.push(dl(
          dRow('Tool', has(p.name) ? str(p.name) : null, { mono: true }),
          dRow('Valid', p.valid === true ? yesNo(true, 'valid') : p.valid === false ? yesNo(false, 'rejected: not executed') : null),
          p.error ? dRow('Error', str(p.error)) : null,
          dRow('In plain words', has(p.description) ? str(p.description) : null),
          p.from_content ? dRow('Parsed from', 'the message text (not a structured tool call)') : null,
        ));
        parts.push(codeBlock(prettyJson(parseArgs(p.arguments)), { label: 'Arguments (JSON)', open: true }));
        break;
      case 'grounding': {
        const pt = Array.isArray(p.point_1000) ? p.point_1000 : null;
        parts.push(h('div', { class: 'd-split' },
          dl(
            dRow('Served name', modelValue(ev, p.model)),
            dRow('Prompt version', promptVersion(p, { config: false })),
            dRow('Prompt SHA', shaValue(p.prompt_sha, p.mock === true)),
            dRow('Tokens', usageValue(p.usage)),
            dRow('VLM family', has(runConfig().model_family) ? [h('span', { class: 'mono', text: str(runConfig().model_family) }), ' ',
              h('span', { class: 'd-sub', text: '(from the run config)' })] : null),
            dRow('Target', has(p.target) ? h('q', { text: str(p.target) }) : null),
            dRow('Point (0–1000 grid)', pt ? pt.map((n) => (isNum(n) ? String(n) : str(n))).join(', ') : null, { mono: true }),
            dRow('Pixel x, y', isNum(p.x) && isNum(p.y) ? `${+p.x.toFixed(1)}, ${+p.y.toFixed(1)}` : null, { mono: true }),
            dRow('Latency', isObj(ev.timings_ms) && isNum(ev.timings_ms.ground) ? fmtMs(ev.timings_ms.ground) : null),
            p.error ? dRow('Error', str(p.error)) : null,
          ),
          ev.overlay_url ? h('div', { class: 'd-thumbs' },
            thumb(ev.overlay_url, `What the grounder saw at ${stepWord}, with its click point`,
              `Screenshot the grounder saw at ${stepWord}, with a crosshair on its click point for ${oneLine(p.target)}`)) : null,
        ));
        parts.push(codeBlock(str(p.raw), {
          label: p.mock === true ? 'Raw output (written by the mock: no model ran)' : 'Raw model output', open: true,
        }));
        break;
      }
      case 'action': {
        const a = parseArgs(p.arguments);
        let what = null;
        if (isObj(p.click) && isNum(p.click.x)) what = `click at ${Math.round(p.click.x)}, ${Math.round(p.click.y)}`;
        else if (p.name === 'type_text') what = [`type `, h('q', { text: str(a.text) })];
        else if (p.name === 'press_key') what = `press ${str(a.key)}`;
        else if (p.name === 'scroll') what = `scroll ${str(a.direction)}${a.amount ? ` (${str(a.amount)})` : ''}`;
        else if (p.name === 'done') what = 'done (no browser action)';
        parts.push(dl(
          dRow('Action', has(p.name) ? [h('span', { class: 'mono', text: str(p.name) }), what ? [' · ', what] : null] : null),
          dRow('Result', p.ok === true ? yesNo(true, 'ok') : p.ok === false ? yesNo(false, 'failed') : null),
          p.error ? dRow('Error', str(p.error)) : null,
          dRow('Screen changed', changedValue(p.screen_changed)),
          dRow('Latency', isObj(ev.timings_ms) && isNum(ev.timings_ms.act) ? fmtMs(ev.timings_ms.act) : null),
          dRow('Timings', timingsValue(ev.timings_ms)),
        ));
        parts.push(codeBlock(prettyJson(a), { label: 'Arguments (JSON)' }));
        break;
      }
      case 'error':
        parts.push(dl(
          dRow('Message', has(p.message) ? str(p.message) : null),
          dRow('Where', has(p.where) ? str(p.where) : null, { mono: true }),
          dRow('Recoverable', p.recoverable === true ? 'yes: the agent kept going' : p.recoverable === false ? 'no: the run ended' : null),
          has(p.finish_reason) ? dRow('Finish reason', str(p.finish_reason), { mono: true }) : null,
        ));
        break;
      case 'done':
        parts.push(dl(
          dRow('Reason', has(p.reason) ? [h('span', { class: 'mono', text: str(p.reason) }), REASON_WORDS[p.reason] ? h('span', { class: 'd-sub', text: ` (${REASON_WORDS[p.reason]})` }) : null] : null),
          dRow('Claimed success', p.success_claimed === true ? yesNo(true) : p.success_claimed === false ? yesNo(false) : null),
          dRow('Steps', has(p.steps) ? str(p.steps) : null),
          dRow('Summary', has(p.summary) ? h('q', { text: str(p.summary) }) : null),
          dRow('Stats', statsValue(p.stats)),
        ));
        break;
      case 'verdict':
        parts.push(dl(
          dRow('Passed', p.passed === true ? yesNo(true, 'passed') : p.passed === false ? yesNo(false, 'failed') : null),
          dRow('Explanation', has(p.explanation) ? str(p.explanation) : null),
          dRow('Label', has(p.label) ? str(p.label) : null),
          dRow('Task', has(p.task_id) ? str(p.task_id) : null, { mono: true }),
          dRow('Note', 'The console read the sandbox state after the run. The agent never sees it.'),
        ));
        break;
      default:
        parts.push(codeBlock(prettyJson(ev.payload === undefined ? ev : ev.payload), { label: null }));
    }
    if (ev.screenshot_url && ev.kind !== 'observation') {
      parts.push(h('div', { class: 'd-thumbs' }, thumb(ev.screenshot_url, `Screenshot at ${stepWord}`, `Screenshot at ${stepWord}`)));
    }
    if (ev.overlay_url && ev.kind !== 'grounding') {
      parts.push(h('div', { class: 'd-thumbs' }, thumb(ev.overlay_url, `Click overlay at ${stepWord}`,
        `Screenshot the grounder saw at ${stepWord}, with its click point marked`)));
    }
    parts.push(codeBlock(prettyJson(ev), { label: `Raw event #${idx + 1} (JSON)` }));
    return parts;
  }

  // --- image dialog -------------------------------------------------------------------------------------------------
  function openImage(btn) {
    S.dialogReturn = btn;
    ui.imgDialogImg.src = btn.dataset.src;
    ui.imgDialogImg.alt = (btn.querySelector('img') || {}).alt || '';
    setText(ui.imgDialogTitle, btn.dataset.title || 'Screenshot');
    if (typeof ui.imgDialog.showModal === 'function') ui.imgDialog.showModal();
    else ui.imgDialog.setAttribute('open', '');
    ui.imgDialogClose.focus();
  }

  // --- copy / download ----------------------------------------------------------------------------------------------
  async function copyJson() {
    if (!S.detail) return;
    // the detail response's own text: the same bytes as Download JSON (console/logs.py detail_json)
    const text = S.detail.text || `${JSON.stringify(S.detail.raw, null, 2)}\n`;
    const id = S.detailFor;
    try {
      if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') throw new Error('no clipboard');
      await navigator.clipboard.writeText(text);
      ui.copyFallbackWrap.hidden = true;
      copied(`Copied: run ${id} as JSON (${nf0.format(text.length)} characters).`);
    } catch {
      ui.copyFallback.value = text;
      ui.copyFallbackWrap.hidden = false;
      ui.copyFallback.focus();
      ui.copyFallback.select();
      let ok = false;
      try { ok = document.execCommand && document.execCommand('copy'); } catch { ok = false; }
      copied(ok ? `Copied: run ${id} as JSON.` : 'The JSON is selected below. Press Ctrl+C (⌘C on a Mac) to copy it.');
    }
  }

  let copiedFlip = false;
  function copied(text) {
    copiedFlip = !copiedFlip;
    ui.copyStatus.textContent = text + (copiedFlip ? '' : ' ');
  }
})();
