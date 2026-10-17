/* GUI assist console: vanilla JS, no build step, no external resources.
 *
 * Talks to the console backend (console/app.py): GET /api/config, GET /api/tasks,
 * POST /api/run, GET /api/runs/{id}, SSE /api/runs/{id}/events, POST /api/runs/{id}/stop.
 * StepEvent payloads are documented in agent/trace.py. Everything the agent says is
 * rendered with textContent (never innerHTML), and unknown shapes fall back to JSON.
 */
'use strict';

(() => {
  const HC_KEY = 'gui-assist.high-contrast';
  const RUN_KEY = 'gui-assist.run';
  const REASON_MAX = 200;
  const OPTION_NAMES = ['grounder', 'thinking', 'lightning'];
  const DIFFICULTIES = ['easy', 'medium', 'hard'];
  const TAG_WORDS = { icon_only: 'unlabeled icon buttons', canvas: 'canvas widget', scroll: 'needs scrolling' };
  const TIMING_ORDER = ['plan', 'ground', 'act', 'summary'];
  const TIMING_HIDDEN = new Set(['settle', 'screenshot']);
  const FINAL = new Set(['done', 'stopped', 'error']);
  const STATUS_TEXT = {
    idle: 'Idle', starting: 'Starting', running: 'Running', reconnecting: 'Reconnecting',
    done: 'Done', stopped: 'Stopped', error: 'Error',
  };
  const REASON_WORDS = {
    max_steps: 'Stopped: reached the step limit before finishing.',
    stopped: 'Stopped by the user.',
    error: 'Stopped after an error.',
  };

  // --- storage (may be unavailable in private modes) -----------------------------
  const storage = (area) => ({
    get(k) { try { return window[area].getItem(k); } catch { return null; } },
    set(k, v) { try { window[area].setItem(k, v); } catch { /* ignore */ } },
    del(k) { try { window[area].removeItem(k); } catch { /* ignore */ } },
  });
  const prefs = storage('localStorage');
  const tabStore = storage('sessionStorage');

  // Theme first: this file loads in <head>, so the saved theme applies before first paint.
  (() => {
    const saved = prefs.get(HC_KEY);
    const on = saved !== null ? saved === '1'
      : Boolean(window.matchMedia && window.matchMedia('(prefers-contrast: more)').matches);
    document.documentElement.classList.toggle('hc', on);
  })();

  // --- state ------------------------------------------------------------------------
  const S = {
    config: null,
    tasksById: new Map(),
    taskId: null,
    starting: false,
    run: null,          // the run being shown (see attach())
    es: null,
    lastSeq: 0,
    steps: new Map(),   // step number -> step record (see getStep())
    latest: null,       // {step, kind: 'overlay'|'shot'}: the newest image
    selected: null,     // step being replayed; null = follow live
    viewPref: 'auto',   // auto | before | after
    autoScroll: true,
    pending: null,      // {text, since}
    timer: null,
    runSeq: 0,          // bumps per attached run (unique element ids)
  };
  const ui = {};

  // --- small helpers ------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  function h(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else node.setAttribute(k, v === true ? '' : String(v));
    }
    for (const c of children) if (c !== null && c !== undefined) node.append(c);
    return node;
  }

  function setText(node, text) {
    if (node.textContent !== text) node.textContent = text;
  }

  const str = (v) => (v === null || v === undefined ? '' : String(v));
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
  const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
  const oneLine = (s) => str(s).replace(/\s+/g, ' ').trim();

  function compactJson(v, max = 240) {
    let out;
    try { out = JSON.stringify(v); } catch { out = String(v); }
    out = str(out);
    return out.length > max ? `${out.slice(0, max)}…` : out;
  }

  function fmtSeconds(ms) {
    const s = ms / 1000;
    return s < 10 ? `${s.toFixed(1)}\u00a0s` : `${Math.round(s)}\u00a0s`;
  }

  function fmtDuration(ms) {
    if (!isNum(ms) || ms < 0) return '';
    const s = ms / 1000;
    if (s < 60) return `${s.toFixed(1)}\u00a0s`;
    return `${Math.floor(s / 60)}\u00a0min ${Math.round(s % 60)}\u00a0s`;
  }

  function fmtClock(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
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

  async function getJson(url) {
    const res = await fetch(url, { headers: { Accept: 'application/json' }, cache: 'no-store' });
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  }

  function announce(text) {
    ui.srStatus.textContent = '';
    window.setTimeout(() => { ui.srStatus.textContent = text; }, 50);
  }

  // --- plain words for tool calls --------------------------------------------------------
  function parseArgs(args) {
    if (isObj(args)) return args;
    if (typeof args === 'string') {
      try { const v = JSON.parse(args); return isObj(v) ? v : {}; } catch { return {}; }
    }
    return {};
  }

  function describeCall(name, rawArgs, fallback) {
    const a = parseArgs(rawArgs);
    if (typeof rawArgs === 'string' && !Object.keys(a).length && rawArgs.trim()) {
      return `${name || 'Unknown action'}(${oneLine(rawArgs).slice(0, 120)})`;  // arguments that aren't valid JSON
    }
    switch (name) {
      case 'click':
        return `Clicking: ${oneLine(a.target_description) || '(no target given)'}`;
      case 'type_text': {
        const target = oneLine(a.target_description);
        return `Typing “${str(a.text)}”${target ? ` into ${target}` : ''}`;
      }
      case 'press_key':
        return `Pressing ${oneLine(a.key) || '(no key given)'}`;
      case 'scroll':
        return `Scrolling ${oneLine(a.direction) || '?'}${a.amount ? ` (${oneLine(a.amount)})` : ''}`;
      case 'done':
        return `Done: ${oneLine(a.summary) || '(no summary)'}`;
      default:
        if (fallback) return oneLine(fallback);
        return name ? `${name}(${compactJson(rawArgs ?? {}, 120)})` : 'Unknown action';
    }
  }

  function firstPlanCall(plan) {
    const calls = plan && Array.isArray(plan.tool_calls) ? plan.tool_calls : [];
    return isObj(calls[0]) ? calls[0] : null;
  }

  function actionWords(s) {
    const call = s.toolCall || firstPlanCall(s.plan);
    if (call && call.name) return describeCall(str(call.name), call.arguments, call.description);
    if (s.n === 0) return s.start && s.start.start_url ? `Started at ${displayUrl(s.start.start_url)}` : 'Start screen';
    if (s.plan) {
      const failed = s.errors.some((e) => isObj(e) && !/no tool call/i.test(str(e.message)));
      return failed ? 'Planning failed: the Lightning call did not succeed' : 'No action: Lightning replied without a tool call';
    }
    if (s.done) return s.done.reason === 'done' ? 'Done' : 'Run ended';
    if (s.errors.length) return 'Error';
    if (s.observation) return 'Screen';
    return `Step ${s.n}`;
  }

  function reasonText(plan) {
    if (!isObj(plan)) return '';
    return oneLine(plan.reasoning) || oneLine(typeof plan.content === 'string' ? plan.content : '');
  }

  function timingsText(t) {
    const keys = [
      ...TIMING_ORDER.filter((k) => k in t),
      ...Object.keys(t).filter((k) => !TIMING_ORDER.includes(k) && !TIMING_HIDDEN.has(k)),
    ];
    return keys.filter((k) => isNum(Number(t[k])) && t[k] !== null)
      .map((k) => `${k}\u00a0${fmtSeconds(Number(t[k]))}`).join(' · ');
  }

  function configText(cfg, options) {
    const c = isObj(cfg) ? cfg : {};
    const o = isObj(options) ? options : {};
    const parts = [];
    const grounder = c.grounder_model || o.grounder;
    if (grounder) parts.push(`grounder: ${grounder}`);
    if (c.summary_model) parts.push(`screen summary: ${c.summary_model}`);
    const thinking = c.thinking || o.thinking;
    if (thinking) parts.push(`thinking: ${thinking}`);
    const lightning = (isObj(c.lightning) && c.lightning.model) || o.lightning;
    if (lightning) parts.push(`Lightning: ${lightning}${o.lightning && o.lightning !== lightning ? ` (${o.lightning})` : ''}`);
    if (c.use_oracle_grounder) parts.push('oracle grounder (test only)');
    return parts.join(' · ');
  }

  function groundingLine(g) {
    const who = g.model ? `Grounder (${oneLine(g.model)})` : 'Grounder';
    if (isNum(g.x) && isNum(g.y)) {
      return { cls: '', text: `${who} chose (${Math.round(g.x)}, ${Math.round(g.y)})` };
    }
    return { cls: 'warn', text: `⚠ ${who} couldn’t place the click${g.error ? `: ${oneLine(g.error)}` : ''}` };
  }

  function errorLine(e) {
    const p = isObj(e) ? e : { message: compactJson(e) };
    const where = p.where ? ` (${oneLine(p.where)})` : '';
    let msg = oneLine(p.message) || 'unknown error';
    if (p.recoverable === true) msg += `${/[.!?]$/.test(msg) ? '' : '.'} Retrying.`;
    return `Error${where}: ${msg}`;
  }

  // --- init -----------------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    for (const id of [
      'request', 'request-note', 'task-select', 'task-meta', 'run-form', 'run-btn', 'stop-btn', 'run-status',
      'form-error', 'follow-btn', 'log-scroll', 'step-log', 'log-empty', 'pending', 'banner', 'banner-icon',
      'banner-title', 'banner-label', 'banner-expl', 'banner-summary', 'banner-claim', 'banner-extra',
      'options', 'options-summary', 'live-btn', 'view-toggle', 'view-before', 'view-after', 'screen-img', 'screen-empty',
      'screen-caption', 'thumbs', 'sr-status', 'hc-toggle', 'agent-badge',
    ]) ui[id.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = $(id);
    ui.optionSets = OPTION_NAMES.map((n) => $(`opt-${n}`));

    ui.hcToggle.setAttribute('aria-pressed', String(document.documentElement.classList.contains('hc')));
    ui.hcToggle.addEventListener('click', () => {
      const on = !document.documentElement.classList.contains('hc');
      document.documentElement.classList.toggle('hc', on);
      ui.hcToggle.setAttribute('aria-pressed', String(on));
      prefs.set(HC_KEY, on ? '1' : '0');
    });

    ui.runForm.addEventListener('submit', (e) => { e.preventDefault(); startRun(); });
    ui.request.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        startRun();
      }
    });
    ui.request.addEventListener('input', onRequestInput);
    ui.taskSelect.addEventListener('change', onTaskChange);
    ui.stopBtn.addEventListener('click', stopRun);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && isRunning() && !e.defaultPrevented) {
        e.preventDefault();
        stopRun();
      }
    });

    ui.logScroll.setAttribute('tabindex', '0');
    ui.logScroll.setAttribute('role', 'region');
    ui.logScroll.setAttribute('aria-label', 'Step log, scrollable');
    ui.logScroll.addEventListener('scroll', () => {
      const el = ui.logScroll;
      S.autoScroll = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      ui.followBtn.hidden = S.autoScroll || !isRunning();
    });
    ui.followBtn.addEventListener('click', () => {
      S.autoScroll = true;
      ui.logScroll.focus();
      scrollLog();
    });

    ui.liveBtn.addEventListener('click', () => {
      S.selected = null;
      S.viewPref = 'auto';
      renderViewer();
    });
    for (const input of [ui.viewBefore, ui.viewAfter]) {
      input.addEventListener('change', () => {
        if (!input.checked) return;
        S.viewPref = input.value;
        renderViewer();
      });
    }
    ui.thumbs.addEventListener('keydown', onThumbKey);
    ui.options.addEventListener('change', renderOptionsSummary);
    ui.screenImg.addEventListener('error', () => {
      if (!ui.screenImg.getAttribute('src')) return;
      ui.screenImg.hidden = true;
      ui.screenEmpty.hidden = false;
      setText(ui.screenEmpty, 'Couldn’t load this screenshot.');
    });
    ui.screenImg.addEventListener('load', () => {
      ui.screenImg.hidden = false;
      ui.screenEmpty.hidden = true;
    });

    renderViewer();
    renderOptionsSummary();
    await Promise.all([loadConfig(), loadTasks()]);
    const last = tabStore.get(RUN_KEY);
    if (last) watchRun(last, { quietIfDone: true });
  }

  async function loadConfig() {
    try {
      const cfg = await getJson('/api/config');
      S.config = cfg;
      const agent = str(cfg.agent);
      ui.agentBadge.dataset.agent = agent === 'mock' || agent === 'real' ? agent : 'unknown';
      ui.agentBadge.textContent = agent === 'mock' ? 'Agent: mock (scripted, no models)' : `Agent: ${agent || '?'}`;
      const d = isObj(cfg.defaults) ? cfg.defaults : {};
      for (const name of OPTION_NAMES) setRadio(name, d[name]);
      renderOptionsSummary();
      if (cfg.headed) {
        ui.logEmpty.textContent = 'Steps appear here while the agent works. Its browser window opens beside this one.';
      }
    } catch {
      ui.agentBadge.dataset.agent = 'offline';
      ui.agentBadge.textContent = 'Agent: offline';
      showFormError('Can’t reach the console backend. Is it running? (scripts/run_console.sh)');
    }
  }

  async function loadTasks() {
    try {
      const tasks = await getJson('/api/tasks');
      if (!Array.isArray(tasks)) throw new Error('tasks is not a list');
      for (const t of tasks) {
        if (!isObj(t) || !t.id || !t.request) continue;
        S.tasksById.set(String(t.id), t);
        ui.taskSelect.add(new Option(`${t.request} (${difficulty(t)})`, String(t.id)));
      }
    } catch {
      ui.taskMeta.textContent = 'Couldn’t load the sandbox tasks. Free text still works.';
    }
  }

  function difficulty(t) {
    const tags = Array.isArray(t.tags) ? t.tags : [];
    return DIFFICULTIES.find((d) => tags.includes(d)) || 'task';
  }

  function taskMetaText(t) {
    const tags = Array.isArray(t.tags) ? t.tags.map(String) : [];
    const areas = tags.filter((x) => !DIFFICULTIES.includes(x) && !TAG_WORDS[x]).map(cap);
    const special = tags.filter((x) => TAG_WORDS[x]).map((x) => TAG_WORDS[x]);
    const n = Number(t.steps_min);
    const steps = n > 0 ? `${n}+ step${n === 1 ? '' : 's'}` : '';
    return [cap(difficulty(t)), ...areas, ...special, steps].filter(Boolean).join(' · ');
  }

  function setRadio(name, value) {
    if (!value) return;
    const input = document.querySelector(`input[name="${name}"][value="${CSS.escape(String(value))}"]`);
    if (input) input.checked = true;
  }

  function renderOptionsSummary() {
    const o = readOptions();
    setText(ui.optionsSummary, OPTION_NAMES.map((n) => o[n] || '?').join(' · '));
  }

  function readOptions() {
    const out = {};
    for (const name of OPTION_NAMES) {
      const input = document.querySelector(`input[name="${name}"]:checked`);
      if (input) out[name] = input.value;
    }
    return out;
  }

  // --- request form ---------------------------------------------------------------------
  function onTaskChange() {
    const id = ui.taskSelect.value || null;
    const prev = S.taskId ? S.tasksById.get(S.taskId) : null;
    clearFormError();
    ui.requestNote.textContent = '';
    if (id && S.tasksById.has(id)) {
      const t = S.tasksById.get(id);
      S.taskId = id;
      ui.request.value = t.request;
      ui.taskMeta.textContent = taskMetaText(t);
    } else {
      if (prev && ui.request.value === prev.request) ui.request.value = '';
      S.taskId = null;
      ui.taskMeta.textContent = '';
    }
  }

  function onRequestInput() {
    clearFormError();
    if (!S.taskId) return;
    const t = S.tasksById.get(S.taskId);
    if (t && ui.request.value !== t.request) {
      S.taskId = null;
      ui.taskSelect.value = '';
      ui.taskMeta.textContent = '';
      ui.requestNote.textContent = 'You edited the task text, so this runs as free text (no sandbox check).';
    }
  }

  function showFormError(message, activeRunId) {
    ui.formError.replaceChildren(document.createTextNode(message));
    if (activeRunId) {
      const btn = h('button', { type: 'button', class: 'btn btn-small', text: 'Watch that run' });
      btn.addEventListener('click', () => { clearFormError(); watchRun(activeRunId, {}); });
      ui.formError.append(btn);
    }
  }

  function clearFormError() {
    if (ui.formError.firstChild) ui.formError.replaceChildren();
  }

  function setLocked(locked) {
    const hadRunFocus = document.activeElement === ui.runBtn;
    ui.runBtn.disabled = locked;
    ui.request.readOnly = locked;
    ui.taskSelect.disabled = locked;
    for (const fs of ui.optionSets) fs.disabled = locked;
    ui.stopBtn.disabled = !(locked && S.run && !S.run.finished && !S.run.stopRequested);
    ui.stopBtn.textContent = S.run && S.run.stopRequested && !S.run.finished ? 'Stopping…' : 'Stop';
    if (locked && hadRunFocus) ui.request.focus();
  }

  const isRunning = () => Boolean(S.run && !S.run.finished);

  async function startRun() {
    if (S.starting || isRunning()) return;
    clearFormError();
    const request = ui.request.value.trim();
    if (!request) {
      showFormError('Type what you want to do first.');
      ui.request.focus();
      return;
    }
    if (request.length > 500) {
      showFormError('That request is too long (500 characters at most).');
      ui.request.focus();
      return;
    }
    const options = readOptions();
    const taskId = S.taskId;
    S.starting = true;
    if (document.activeElement === ui.runBtn) ui.request.focus();  // Run is disabled until the run ends
    ui.runBtn.disabled = true;
    let res;
    let data = {};
    try {
      res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ request, task_id: taskId, options }),
      });
      data = await res.json().catch(() => ({}));
    } catch {
      S.starting = false;
      ui.runBtn.disabled = false;
      showFormError('Can’t reach the console backend. Is it still running?');
      return;
    }
    S.starting = false;
    ui.runBtn.disabled = false;
    if (!res.ok || !data.run_id) {
      const msg = oneLine(data.error) || `HTTP ${res.status}`;
      if (res.status === 409) {
        showFormError(`Another run is already in progress${data.active_run_id ? ` (${data.active_run_id})` : ''}. `
          + 'Wait for it to finish or stop it.', data.active_run_id);
      } else if (res.status === 503) {
        showFormError(`Can’t start: ${msg}. Check that the sandbox (and for the real agent, the models) are running.`);
      } else {
        showFormError(`Can’t start: ${msg}`);
      }
      return;
    }
    attach(String(data.run_id), {
      request, taskId, options, startUrl: data.start_url, focusOnEnd: true,
    });
  }

  async function watchRun(runId, { quietIfDone = false } = {}) {
    let info;
    try {
      info = await getJson(`/api/runs/${encodeURIComponent(runId)}`);
    } catch {
      tabStore.del(RUN_KEY);
      if (!quietIfDone) showFormError('That run is no longer available.');
      return;
    }
    if (isRunning()) return;
    const active = !FINAL.has(str(info.status));
    ui.request.value = str(info.request);
    const taskId = info.task_id && S.tasksById.has(String(info.task_id)) ? String(info.task_id) : null;
    S.taskId = taskId;
    ui.taskSelect.value = taskId || '';
    ui.taskMeta.textContent = taskId ? taskMetaText(S.tasksById.get(taskId)) : '';
    ui.requestNote.textContent = '';
    if (isObj(info.options)) for (const name of OPTION_NAMES) setRadio(name, info.options[name]);
    renderOptionsSummary();
    attach(String(info.run_id || runId), {
      request: str(info.request), taskId: info.task_id || null, options: info.options,
      focusOnEnd: active || !quietIfDone, quiet: !active,
    });
  }

  async function stopRun() {
    if (!isRunning() || S.run.stopRequested) return;
    const run = S.run;
    run.stopRequested = true;
    setLocked(true);
    setPending('Stopping…');
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(run.id)}/stop`, { method: 'POST' });
      if (res.status === 404) {
        run.status = 'error';
        run.statusMessage = 'The console no longer knows this run.';
        finishRun();
      } else if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch {
      if (S.run !== run || run.finished) return;
      run.stopRequested = false;
      setLocked(true);
      showFormError('Couldn’t stop the run. Try again.');
    }
  }

  // --- run lifecycle ---------------------------------------------------------------------
  function attach(runId, info) {
    closeStream();
    resetRunView();
    S.run = {
      id: runId,
      request: str(info.request),
      taskId: info.taskId || null,
      options: isObj(info.options) ? info.options : null,
      startUrl: info.startUrl || null,
      focusOnEnd: Boolean(info.focusOnEnd),
      status: 'starting',
      statusMessage: '',
      finished: false,
      stopRequested: false,
      reconnecting: false,
      startedAt: Date.now(),
      finishedAt: null,
      startTs: null,
      endTs: null,
      lastTs: null,
      done: null,
      result: null,
      verdict: null,
    };
    S.lastSeq = 0;
    S.runSeq += 1;
    tabStore.set(RUN_KEY, runId);
    ui.options.open = false;  // give the step log the room; the summary line still shows the choices
    setLocked(true);
    setStatus('starting');
    setPending('Starting a browser session…');
    if (!info.quiet) announce(`Run started: ${S.run.request}`);
    S.timer = window.setInterval(tick, 1000);
    openStream(runId);
  }

  function openStream(runId) {
    const es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
    S.es = es;
    const on = (type, handler) => es.addEventListener(type, (e) => {
      if (S.es !== es || !S.run) return;
      const seq = Number.parseInt(e.lastEventId, 10);
      if (Number.isFinite(seq)) {
        if (seq <= S.lastSeq) return;  // replayed after a reconnect
        S.lastSeq = seq;
      }
      let data;
      try { data = e.data ? JSON.parse(e.data) : {}; } catch { data = { unparsed: str(e.data) }; }
      try {
        handler(isObj(data) ? data : { value: data });
      } catch (err) {
        logUnknown('console', { error: str(err && err.message), event: type });
      }
    });
    on('status', onStatus);
    on('step', onStep);
    on('result', (d) => { S.run.result = d; });
    on('verdict', (d) => { S.run.verdict = d; });
    on('end', () => finishRun());
    es.addEventListener('open', () => {
      if (S.es === es && S.run && S.run.reconnecting) {
        S.run.reconnecting = false;
        setStatus(S.run.status);
      }
    });
    es.addEventListener('error', () => {
      if (S.es !== es || !S.run || S.run.finished) return;
      if (es.readyState === EventSource.CLOSED) {
        recoverStream(runId);
      } else {
        S.run.reconnecting = true;
        setStatus('reconnecting');
      }
    });
  }

  function closeStream() {
    if (S.es) S.es.close();
    S.es = null;
  }

  /** The stream closed for good (e.g. an HTTP error). Ask the backend what happened. */
  async function recoverStream(runId) {
    closeStream();
    const run = S.run;
    try {
      const info = await getJson(`/api/runs/${encodeURIComponent(runId)}`);
      if (S.run !== run || run.finished) return;
      if (FINAL.has(str(info.status))) {
        run.status = str(info.status);
        if (isObj(info.result)) run.result = info.result;
        if (isObj(info.verdict)) run.verdict = info.verdict;
        finishRun();
      } else {
        run.reconnecting = true;
        setStatus('reconnecting');
        window.setTimeout(() => { if (S.run === run && !run.finished) openStream(runId); }, 2000);
      }
    } catch {
      if (S.run !== run || run.finished) return;
      run.status = 'error';
      run.statusMessage = 'Lost the connection to this run (was the console backend restarted?).';
      finishRun();
    }
  }

  function onStatus(d) {
    const st = str(d.status);
    if (!st) return;
    S.run.status = st;
    if (d.message) S.run.statusMessage = oneLine(d.message);
    if (!S.run.reconnecting) setStatus(st);
    if (st === 'running' && S.pending && S.pending.text.startsWith('Starting')) setPending('Opening the app…');
  }

  function finishRun() {
    const run = S.run;
    if (!run || run.finished) return;
    run.finished = true;
    run.finishedAt = Date.now();
    closeStream();
    window.clearInterval(S.timer);
    S.timer = null;
    setPending(null);
    setLocked(false);
    ui.followBtn.hidden = true;
    setStatus(finalStatus(run));
    for (const s of S.steps.values()) renderStep(s);
    renderBanner();
    renderViewer();
    if (run.focusOnEnd) ui.banner.focus();
  }

  function finalStatus(run) {
    const reason = run.done && run.done.reason;
    if (reason === 'stopped' || reason === 'error') return reason;
    if (FINAL.has(run.status)) return run.status;
    return run.statusMessage ? 'error' : 'done';
  }

  function elapsedMs() {
    const run = S.run;
    if (!run) return 0;
    if (run.finished) {
      if (isNum(run.startTs) && isNum(run.endTs)) return (run.endTs - run.startTs) * 1000;
      return run.finishedAt - run.startedAt;
    }
    const start = isNum(run.startTs) ? Math.min(run.startTs * 1000, Date.now()) : run.startedAt;
    return Date.now() - start;
  }

  function setStatus(st) {
    ui.runStatus.dataset.state = st;
    renderStatusText();
  }

  function renderStatusText() {
    const st = ui.runStatus.dataset.state;
    let text = STATUS_TEXT[st] || cap(st);
    if (S.run && st !== 'idle') text += ` · ${fmtClock(elapsedMs())}`;
    setText(ui.runStatus, text);
  }

  function setPending(text) {
    S.pending = text ? { text, since: Date.now() } : null;
    renderPending();
  }

  function renderPending() {
    const p = S.pending;
    ui.pending.hidden = !p;
    if (!p) return;
    const secs = Math.floor((Date.now() - p.since) / 1000);
    setText(ui.pending, secs >= 1 ? `${p.text} ${secs} s` : p.text);
  }

  function tick() {
    renderStatusText();
    renderPending();
  }

  function resetRunView() {
    S.steps.clear();
    S.latest = null;
    S.selected = null;
    S.viewPref = 'auto';
    S.autoScroll = true;
    ui.stepLog.replaceChildren();
    ui.thumbs.replaceChildren();
    ui.logEmpty.hidden = false;
    ui.followBtn.hidden = true;
    ui.banner.hidden = true;
    window.clearInterval(S.timer);
    S.timer = null;
    renderViewer();
  }

  // --- step events -------------------------------------------------------------------------
  function onStep(ev) {
    const run = S.run;
    const n = Number.isInteger(ev.step) && ev.step >= 0 ? ev.step : lastStepNumber();
    const s = getStep(n);
    const p = isObj(ev.payload) ? ev.payload : {};
    if (isNum(ev.ts)) {
      if (run.startTs === null) run.startTs = ev.ts;
      run.lastTs = ev.ts;
      renderStatusText();
    }
    switch (ev.kind) {
      case 'start':
        s.start = p;
        if (isNum(ev.ts)) run.startTs = ev.ts;
        if (!run.startUrl && p.start_url) run.startUrl = str(p.start_url);
        setPending('Reading the screen…');
        break;
      case 'observation':
        s.observation = p;
        setPending(`Lightning is planning step ${n + 1}…`);
        break;
      case 'plan': {
        s.plan = p;
        const call = firstPlanCall(p);
        setPending(call ? pendingFor(call) : 'Lightning replied without an action; asking again…');
        break;
      }
      case 'tool_call':
        s.toolCall = p;
        break;
      case 'grounding':
        s.grounding = p;
        setPending('Acting on the screen…');
        break;
      case 'action':
        s.action = p;
        setPending(p.done || p.name === 'done' ? 'Finishing…' : 'Reading the new screen…');
        break;
      case 'error':
        s.errors.push(p);
        setPending(p.recoverable === false ? 'Finishing…' : `Lightning is planning step ${n + 1}…`);
        break;
      case 'done':
        s.done = p;
        run.done = p;
        if (isNum(ev.ts)) run.endTs = ev.ts;
        setPending(null);
        break;
      default:
        s.unknown.push({ kind: str(ev.kind) || 'event', payload: ev.payload === undefined ? ev : ev.payload });
    }
    if (isObj(ev.timings_ms)) Object.assign(s.timings, ev.timings_ms);
    if (ev.overlay_url) {
      s.overlayUrl = str(ev.overlay_url);
      S.latest = { step: n, kind: 'overlay' };
    }
    if (ev.screenshot_url) {
      s.shotUrl = str(ev.screenshot_url);
      S.latest = { step: n, kind: 'shot' };
    }
    if (s.overlayUrl || s.shotUrl) ensureThumb(s);
    renderStep(s);
    renderViewer();
    scrollLog();
  }

  function pendingFor(call) {
    const a = parseArgs(call.arguments);
    const target = oneLine(a.target_description);
    switch (call.name) {
      case 'click': return `Finding “${target}” on the screen…`;
      case 'type_text': return target ? `Finding “${target}” on the screen…` : 'Typing…';
      case 'press_key': return `Pressing ${oneLine(a.key)}…`;
      case 'scroll': return 'Scrolling…';
      case 'done': return 'Finishing…';
      default: return 'Acting…';
    }
  }

  function logUnknown(kind, payload) {
    const s = getStep(lastStepNumber());
    s.unknown.push({ kind, payload });
    renderStep(s);
  }

  function lastStepNumber() {
    let max = 0;
    for (const n of S.steps.keys()) max = Math.max(max, n);
    return max;
  }

  function getStep(n) {
    let s = S.steps.get(n);
    if (s) return s;
    const prevLast = S.steps.size ? S.steps.get(lastStepNumber()) : null;
    s = {
      n, start: null, observation: null, plan: null, toolCall: null, grounding: null, action: null, done: null,
      errors: [], unknown: [], timings: {}, overlayUrl: null, shotUrl: null, reasonOpen: false, screenOpen: false,
      lineEls: new Map(), thumb: null, thumbImg: null,
    };
    buildStepEl(s);
    S.steps.set(n, s);
    const after = [...ui.stepLog.children].find((li) => Number(li.dataset.step) > n);
    ui.stepLog.insertBefore(s.el, after || null);
    ui.logEmpty.hidden = true;
    if (prevLast && prevLast.n < n) renderStep(prevLast);  // it is no longer the active step
    return s;
  }

  function buildStepEl(s) {
    const num = h('span', { class: 'step-num', 'aria-hidden': 'true', text: String(s.n) });
    const action = h('span', { class: 'step-action' });
    const chips = h('span', { class: 'step-chips' });
    const head = h('p', { class: 'step-head' },
      h('span', { class: 'visually-hidden', text: s.n === 0 ? 'Start: ' : `Step ${s.n}: ` }), action, chips);
    const reasonTxt = h('span');
    const reasonBtn = h('button', { type: 'button', class: 'btn-link', 'aria-expanded': 'false', hidden: true });
    reasonBtn.addEventListener('click', () => {
      s.reasonOpen = !s.reasonOpen;
      renderStep(s);
    });
    const reason = h('p', { class: 'step-line step-reason', hidden: true },
      h('span', { class: 'visually-hidden', text: 'Reasoning: ' }), reasonTxt, reasonBtn);
    const lines = h('div', { class: 'step-lines' });
    const timing = h('span');
    const screenId = `screen-text-${S.runSeq}-${s.n}`;
    const screenBtn = h('button', {
      type: 'button', class: 'btn-link', 'aria-expanded': 'false', 'aria-controls': screenId, hidden: true,
      text: 'Screen text',
    });
    screenBtn.addEventListener('click', () => {
      s.screenOpen = !s.screenOpen;
      renderStep(s);
    });
    const meta = h('p', { class: 'step-line step-meta', hidden: true }, timing, screenBtn);
    const pre = h('pre', { class: 'step-screen', id: screenId, hidden: true });
    const body = h('div', { class: 'step-body' }, head, reason, lines, meta, pre);
    s.el = h('li', { class: 'step', 'data-step': String(s.n) }, num, body);
    s.slots = { action, chips, reason, reasonTxt, reasonBtn, lines, meta, timing, screenBtn, pre };
  }

  function renderStep(s) {
    const { slots } = s;
    const words = actionWords(s);
    setText(slots.action, words);

    const chips = [];
    if (s.plan && s.plan.thinking === true) chips.push(['info', 'thinking']);
    if (s.toolCall && s.toolCall.valid === false) chips.push(['warn', 'invalid call']);
    const a = s.action;
    if (a && !a.done && a.name !== 'done') {
      if (a.ok === false) chips.push(['bad', 'failed']);
      else if (a.screen_changed === true) chips.push(['ok', 'screen changed']);
      else if (a.screen_changed === false) chips.push(['warn', 'no change']);
    }
    const chipKey = JSON.stringify(chips);
    if (slots.chips.dataset.key !== chipKey) {
      slots.chips.dataset.key = chipKey;
      slots.chips.replaceChildren(...chips.map(([c, t]) => h('span', { class: `chip chip-${c}`, text: t })));
    }

    const full = reasonText(s.plan);
    slots.reason.hidden = !full;
    if (full) {
      const long = full.length > REASON_MAX;
      setText(slots.reasonTxt, `“${long && !s.reasonOpen ? `${full.slice(0, REASON_MAX).trimEnd()}…` : full}”`);
      slots.reasonBtn.hidden = !long;
      slots.reasonBtn.setAttribute('aria-expanded', String(s.reasonOpen));
      setText(slots.reasonBtn, s.reasonOpen ? 'Show less' : 'Show more');
    }

    const lines = [];
    if (s.n === 0 && S.run && S.run.request) lines.push(['request', '', `You asked: “${S.run.request}”`]);
    if (s.start) {
      const c = configText(s.start.config, S.run && S.run.options);
      if (c) lines.push(['config', '', c]);
    }
    if (s.toolCall && s.toolCall.valid === false) {
      lines.push(['invalid', 'warn', `⚠ Invalid tool call${s.toolCall.error ? `: ${oneLine(s.toolCall.error)}` : ''}`]);
    }
    if (s.grounding) {
      const g = groundingLine(s.grounding);
      lines.push(['ground', g.cls, g.text]);
    }
    if (a && a.ok === false && a.error && !(s.toolCall && oneLine(s.toolCall.error) === oneLine(a.error))) {
      lines.push(['actionerr', 'error', `Action failed: ${oneLine(a.error)}`]);
    }
    s.errors.forEach((e, i) => lines.push([`err${i}`, 'error', errorLine(e)]));
    if (s.done && s.done.reason && s.done.reason !== 'done') {
      lines.push(['done', s.done.reason === 'error' ? 'error' : 'warn',
        oneLine(s.done.summary) || REASON_WORDS[s.done.reason] || `Run ended (${oneLine(s.done.reason)}).`]);
    }
    s.unknown.forEach((u, i) => lines.push([`unknown${i}`, 'json', `${u.kind}: ${compactJson(u.payload)}`]));
    for (const [key, cls, text] of lines) {
      let node = s.lineEls.get(key);
      if (!node) {
        node = h('p');
        s.lineEls.set(key, node);
        slots.lines.append(node);
      }
      const className = `step-line${cls ? ` step-${cls}` : ''}`;
      if (node.className !== className) node.className = className;
      setText(node, text);
    }

    const timing = timingsText(s.timings);
    const obs = s.observation;
    const obsText = obs ? str(obs.text || (isObj(obs.observation) && obs.observation.screen_summary)) : '';
    slots.meta.hidden = !timing && !obsText;
    setText(slots.timing, timing && obsText ? `${timing} ·` : timing);
    slots.screenBtn.hidden = !obsText;
    slots.screenBtn.setAttribute('aria-expanded', String(s.screenOpen));
    setText(slots.screenBtn, s.screenOpen ? 'Hide screen text' : 'Screen text');
    slots.screenBtn.setAttribute('aria-label', s.screenOpen ? 'Hide the screen description'
      : `Show the screen description Lightning read${s.n === 0 ? ' at the start' : ` after step ${s.n}`}`);
    slots.pre.hidden = !(obsText && s.screenOpen);
    setText(slots.pre, obsText);

    const state = stepState(s);
    if (s.el.dataset.state !== state) s.el.dataset.state = state;

    if (s.thumb) {
      const src = s.overlayUrl || s.shotUrl;
      if (s.thumbImg.getAttribute('src') !== src) s.thumbImg.src = src;
      s.thumbImg.alt = thumbAlt(s);
    }
  }

  function stepState(s) {
    const a = s.action;
    if (s.errors.some((e) => isObj(e) && e.recoverable === false) || (a && a.ok === false)
        || (s.done && s.done.reason === 'error')) return 'bad';
    if (s.errors.length || (s.toolCall && s.toolCall.valid === false) || (a && a.screen_changed === false)
        || (s.grounding && s.grounding.error) || (s.done && s.done.reason && s.done.reason !== 'done')) return 'warn';
    if ((a && a.ok) || (s.done && s.done.reason === 'done') || (s.n === 0 && s.observation)) return 'ok';
    if (isRunning() && s.n === lastStepNumber()) return 'active';
    return '';
  }

  function scrollLog() {
    if (S.autoScroll) {
      ui.logScroll.scrollTop = ui.logScroll.scrollHeight;
    } else {
      ui.followBtn.hidden = !isRunning();
    }
  }

  // --- screen viewer -----------------------------------------------------------------------
  function pickImage(n, pref) {
    const s = S.steps.get(n);
    if (!s) return null;
    const order = pref === 'after' ? ['shot', 'overlay'] : ['overlay', 'shot'];
    for (const kind of order) if (kind === 'overlay' ? s.overlayUrl : s.shotUrl) return { step: n, kind };
    return null;
  }

  function shownImage() {
    if (S.selected === null) {
      if (!S.latest) return null;
      return S.viewPref === 'auto' ? S.latest : pickImage(S.latest.step, S.viewPref);
    }
    return pickImage(S.selected, S.viewPref === 'after' ? 'after' : 'before');
  }

  function stepLabel(s) {
    return s.n === 0 ? 'Start' : `Step ${s.n}`;
  }

  function thumbAlt(s) {
    return `${stepLabel(s)}: ${s.n === 0 ? displayUrl(s.start && s.start.start_url) || 'start screen' : actionWords(s)}`;
  }

  function imageAlt(s, kind) {
    if (kind === 'overlay') {
      const target = oneLine(s.grounding && s.grounding.target) || actionWords(s);
      return `${stepLabel(s)}, before the click: the screen the grounder saw, with its click point marked on ${target}.`;
    }
    if (s.n === 0) return 'Start: the app screen when the run began.';
    return `${stepLabel(s)}, after the action: the app screen after “${actionWords(s)}”.`;
  }

  function renderViewer() {
    const cur = shownImage();
    const live = S.selected === null;
    ui.liveBtn.setAttribute('aria-pressed', String(live));
    ui.liveBtn.dataset.running = String(isRunning());
    if (!cur) {
      ui.screenImg.hidden = true;
      ui.screenImg.removeAttribute('src');
      ui.screenEmpty.hidden = false;
      setText(ui.screenEmpty, S.run ? 'Waiting for the first screenshot…'
        : 'No screenshot yet. Run a request to watch the agent work.');
      ui.screenCaption.replaceChildren(document.createTextNode(S.run ? 'Waiting for the first screenshot.' : 'Waiting for a run.'));
      ui.viewToggle.disabled = true;
      ui.viewBefore.checked = false;
      ui.viewAfter.checked = false;
    } else {
      const s = S.steps.get(cur.step);
      const url = cur.kind === 'overlay' ? s.overlayUrl : s.shotUrl;
      if (ui.screenImg.getAttribute('src') !== url) {
        ui.screenImg.src = url;
        ui.screenImg.hidden = false;
        ui.screenEmpty.hidden = true;
      }
      ui.screenImg.alt = imageAlt(s, cur.kind);
      const startPath = displayUrl(s.start && s.start.start_url);
      const what = s.n === 0 ? (startPath ? `opened ${startPath}` : 'start screen') : actionWords(s);
      const kindText = cur.kind === 'overlay' ? 'before click' : (s.n === 0 ? 'start screen' : 'after action');
      ui.screenCaption.replaceChildren(
        document.createTextNode(`${stepLabel(s)} · ${what}`),
        h('span', { class: `chip ${cur.kind === 'overlay' ? 'chip-info' : 'chip-ok'}`, text: kindText }),
      );
      if (!live) ui.screenCaption.append(h('span', { class: 'chip chip-warn', text: 'replay' }));
      ui.viewToggle.disabled = false;
      ui.viewBefore.disabled = !s.overlayUrl;
      ui.viewAfter.disabled = !s.shotUrl;
      ui.viewBefore.checked = cur.kind === 'overlay';
      ui.viewAfter.checked = cur.kind === 'shot';
    }

    const shownStep = cur ? cur.step : null;
    for (const s of S.steps.values()) s.el.classList.toggle('is-shown', s.n === shownStep && !live);
    const thumbs = [...ui.thumbs.children];
    const focusThumb = thumbs.find((t) => Number(t.dataset.step) === shownStep) || thumbs[thumbs.length - 1];
    for (const t of thumbs) {
      t.tabIndex = t === focusThumb ? 0 : -1;
      if (Number(t.dataset.step) === shownStep) t.setAttribute('aria-current', 'true');
      else t.removeAttribute('aria-current');
    }
    if (live && focusThumb && document.activeElement !== focusThumb && !ui.thumbs.contains(document.activeElement)) {
      ui.thumbs.scrollLeft = ui.thumbs.scrollWidth;
    }
  }

  function ensureThumb(s) {
    if (s.thumb) return;
    const img = h('img', { alt: thumbAlt(s), src: s.overlayUrl || s.shotUrl, decoding: 'async' });
    const btn = h('button', { type: 'button', class: 'thumb', tabindex: '-1', 'data-step': String(s.n) },
      img, h('span', { class: 'thumb-num', 'aria-hidden': 'true', text: s.n === 0 ? 'Start' : String(s.n) }));
    btn.addEventListener('click', () => selectStep(s.n));
    const after = [...ui.thumbs.children].find((t) => Number(t.dataset.step) > s.n);
    ui.thumbs.insertBefore(btn, after || null);
    s.thumb = btn;
    s.thumbImg = img;
  }

  function selectStep(n) {
    S.selected = n;
    renderViewer();
  }

  function onThumbKey(e) {
    const thumbs = [...ui.thumbs.children];
    const i = thumbs.indexOf(document.activeElement);
    if (i < 0) return;
    let j = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(i + 1, thumbs.length - 1);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(i - 1, 0);
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = thumbs.length - 1;
    if (j === null) return;
    e.preventDefault();
    selectStep(Number(thumbs[j].dataset.step));
    thumbs[j].focus();
  }

  // --- final banner ------------------------------------------------------------------------
  function renderBanner() {
    const run = S.run;
    const st = finalStatus(run);
    const done = isObj(run.done) ? run.done : {};
    const result = isObj(run.result) ? run.result : {};
    const summary = oneLine(result.summary ?? done.summary);
    const claimed = typeof result.success_claimed === 'boolean' ? result.success_claimed
      : (typeof done.success_claimed === 'boolean' ? done.success_claimed : null);
    const steps = isNum(result.steps) ? result.steps : (isNum(done.steps) ? done.steps : lastStepNumber());
    const v = isObj(run.verdict) && typeof run.verdict.passed === 'boolean' ? run.verdict : null;

    let tone;
    let icon;
    let title;
    let label = '';
    let expl = '';
    if (v) {
      tone = v.passed ? 'pass' : 'fail';
      icon = v.passed ? '✓' : '✗';
      title = v.passed ? 'Verified: the task is done' : 'Not verified: the task is not done';
      label = oneLine(v.label) || 'verified by sandbox state, not seen by the agent';
      expl = oneLine(v.explanation);
    } else if (st === 'error') {
      tone = 'error';
      icon = '!';
      title = 'Run failed';
      expl = run.statusMessage || summary;
      label = run.taskId ? '' : 'Free-text request: no sandbox check';
    } else if (st === 'stopped') {
      tone = 'stopped';
      icon = '■';
      title = 'Run stopped';
      label = run.taskId ? 'No sandbox verdict for this run' : 'Free-text request: no sandbox check';
    } else {
      tone = 'neutral';
      icon = 'ⓘ';
      title = 'Run finished';
      label = run.taskId ? 'No sandbox verdict received' : 'Free-text request: no sandbox check';
    }
    ui.banner.dataset.tone = tone;
    setText(ui.bannerIcon, icon);
    setText(ui.bannerTitle, title);
    setText(ui.bannerLabel, label);
    setText(ui.bannerExpl, expl);

    setText(ui.bannerSummary, summary ? `Agent’s summary: “${summary}”` : 'The agent gave no summary.');
    const claim = [];
    if (claimed === true) claim.push('The agent claimed success.');
    else if (claimed === false) claim.push('The agent did not claim success.');
    if (v && claimed !== null && v.passed !== claimed) {
      claim.push(claimed ? 'The sandbox state says otherwise.' : 'The sandbox state shows it is done anyway.');
    }
    if (st === 'stopped') claim.push(run.stopRequested ? 'You stopped the run.' : 'The run was stopped.');
    if (st === 'error' && v && run.statusMessage) claim.push(`Error: ${run.statusMessage}`);
    const dur = fmtDuration(elapsedMs());
    claim.push(`${steps} step${steps === 1 ? '' : 's'}${dur ? ` · ${dur}` : ''}.`);
    setText(ui.bannerClaim, claim.join(' '));

    const extra = [];
    const stats = isObj(done.stats) ? done.stats : {};
    const plural = (k, one, many) => `${stats[k]} ${stats[k] === 1 ? one : many}`;
    if (stats.n_clicks > 0) extra.push(plural('n_clicks', 'click', 'clicks'));
    if (stats.n_no_change) extra.push(plural('n_no_change', 'no-change', 'no-changes'));
    if (stats.n_recoveries) extra.push(plural('n_recoveries', 'recovery', 'recoveries'));
    if (stats.n_thinking_calls) extra.push(plural('n_thinking_calls', 'thinking call', 'thinking calls'));
    ui.bannerExtra.replaceChildren();
    if (extra.length) ui.bannerExtra.append(document.createTextNode(extra.join(' · ')));
    const href = safeHttpUrl(run.startUrl);
    if (href) {
      if (extra.length) ui.bannerExtra.append(document.createTextNode(' · '));
      ui.bannerExtra.append(h('a', { href, target: '_blank', rel: 'noopener', text: 'Open this session in the sandbox' }));
    }
    ui.banner.hidden = false;
  }

  function safeHttpUrl(u) {
    if (!u) return null;
    try {
      const url = new URL(u, window.location.href);
      return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null;
    } catch {
      return null;
    }
  }
})();
