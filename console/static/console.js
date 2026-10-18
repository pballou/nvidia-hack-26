/* GUI assist console: vanilla JS, no build step, no external resources.
 *
 * Talks to the console backend (console/app.py): GET /api/config, GET /api/tasks,
 * POST /api/run, GET /api/runs/{id}, SSE /api/runs/{id}/events, POST /api/runs/{id}/stop.
 * StepEvent payloads are documented in agent/trace.py. Everything the agent says is
 * rendered with textContent (never innerHTML), and unknown shapes fall back to JSON.
 *
 * /api/config `endpoints` (real agent only; null for the mock) drive the header ("VLM: <label> · <host:port> ·
 * serves <ids>", "Lightning: <label>", a chip when the VLM is unreachable) and disable grounder radios the VLM
 * doesn't serve. They are re-read while the page is idle (every 15 s while the VLM is down, else every 60 s, on
 * tab focus, after a run, after a "not served" 400). A run that ends with done.reason "stuck" (agent/progress.py)
 * gets a banner that leads with the agent's question; an action's `stuck` verdict adds a step-log line.
 *
 * Voice input lives in voice.js and drives the page through window.AcmeConsole (below):
 * ready (resolves after /api/config and /api/tasks were tried), config(), isRunning(),
 * isStarting(), setRequest(text, voice?) -> {taskId, request}, run(), stop(). The page fires
 * `acme:run-start` / `acme:run-end` on document (detail: {runId, request, taskId}).
 * `voice` ({text, model, latency_ms, duration_s} from /api/transcribe) goes into POST /api/run
 * for the logs page, but only while the request box still holds exactly what that transcript put
 * there, and only for the first run that uses it. The backend never passes it to the agent.
 */
'use strict';

(() => {
  const HC_KEY = 'gui-assist.high-contrast';
  const RUN_KEY = 'gui-assist.run';
  const RUN_ID_RE = /^[A-Za-z0-9._-]{1,80}$/;  // an agent run id the logs page can open
  const VOICE_TEXT_MAX = 500;
  const VOICE_MODEL_MAX = 120;
  const REASON_MAX = 200;
  const SUMMARY_MAX = 220;  // the banner's agent summary; longer ones get Show more
  const CHECK_HEAD_MAX = 110;  // longer sandbox-check explanations are cut, with the full text one click away
  const OPTION_NAMES = ['grounder', 'thinking', 'lightning'];
  const DIFFICULTIES = ['easy', 'medium', 'hard'];
  const TAG_WORDS = { icon_only: 'unlabeled icon buttons', canvas: 'canvas widget', scroll: 'needs scrolling' };
  const TIMING_ORDER = ['plan', 'ground', 'act', 'summary'];
  const TIMING_HIDDEN = new Set(['settle', 'screenshot']);
  const FINAL = new Set(['done', 'stopped', 'error', 'stuck']);
  const STATUS_TEXT = {
    idle: 'Idle', starting: 'Starting', running: 'Running', reconnecting: 'Reconnecting',
    done: 'Done', stopped: 'Stopped', error: 'Error', stuck: 'Stuck',
  };
  // The header's Lightning line when the toggle isn't the server default (/api/config endpoints describe the
  // default only). The console's per-run choice always picks one of agent.yaml's two presets.
  const LIGHTNING_PRESET_LABELS = { local: 'local BF16 (:8000)', hosted: 'hosted API' };
  const SERVED_SHOWN_MAX = 3;
  const CONFIG_POLL_MS = 15000;      // re-check /api/config this often while the VLM is unreachable...
  const CONFIG_REFRESH_MS = 60000;   // ...and this often otherwise (idle, page visible, real agent only)
  // The collapsed Options line, e.g. "base · thinking retry · local Lightning".
  const OPTION_SHORT = {
    grounder: { base: 'base', grounder: 'LoRA grounder' },
    thinking: { off: 'thinking off', retry: 'thinking retry', on: 'thinking on' },
    lightning: { local: 'local Lightning', hosted: 'hosted Lightning' },
  };
  const FREE_TEXT_LABEL = 'Free-text request: no sandbox check';
  const REASON_WORDS = {
    max_steps: 'Stopped: reached the step limit before finishing.',
    stuck: 'Stopped: stuck',
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
    voice: null,        // {request, voice}: the last transcript put in the box (see setRequestText)
    configAt: 0,        // when /api/config was last read (the header's endpoints; see refreshConfig)
    refreshing: false,
  };
  const ui = {};

  // --- API for voice.js ----------------------------------------------------------------
  let resolveReady;
  const ready = new Promise((resolve) => { resolveReady = resolve; });
  window.AcmeConsole = Object.freeze({
    ready,
    config: () => S.config,
    isRunning: () => isRunning(),
    isStarting: () => S.starting,
    setRequest: (text, voice) => setRequestText(text, voice),
    run: () => startRun(),
    stop: () => stopRun(),
    matchTask: (text) => matchTask(text),
  });

  function emit(name, detail) {
    try { document.dispatchEvent(new CustomEvent(name, { detail })); } catch { /* a listener failed; not ours */ }
  }

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
    if (Math.round(ms) < 1000) return `${Math.round(ms)}\u00a0ms`;  // a real click acts in ~20 ms: not "0.0 s"
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

  let announceTimer = 0;
  function announce(text) {  // a burst (e.g. replayed events) speaks only its last message
    window.clearTimeout(announceTimer);
    ui.srStatus.textContent = '';
    announceTimer = window.setTimeout(() => { ui.srStatus.textContent = text; }, 50);
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
      const errs = s.errors.filter(isObj);
      if (s.plan.finish_reason === 'length' || errs.some((e) => e.finish_reason === 'length')) {
        return 'No action: Lightning’s reply was cut off before it chose one';
      }
      // T2's no-tool-call errors carry the reply's finish_reason; a failed Lightning call has none
      const failed = errs.some((e) => e.where === 'planner' && !('finish_reason' in e));
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

  /** An action's agent/progress.py verdict: 'warn' (Lightning was told it is stuck), 'stuck' (the run stops), or ''. */
  function stuckVerdict(a) {
    const v = isObj(a) && isObj(a.stuck) ? str(a.stuck.verdict) : '';
    return v === 'warn' || v === 'stuck' ? v : '';
  }

  function stuckLine(a) {
    const v = stuckVerdict(a);
    if (!v) return '';
    const reason = oneLine(a.stuck.reason);
    return v === 'warn' ? `Stuck warning: ${reason || 'no progress'}` : `Stopped: stuck${reason ? ` (${reason})` : ''}`;
  }

  /** A stuck stop's summary (agent/progress.py: "I'm stuck: <why>. I tried: <…>. None of it changed the screen.
   *  What would you like me to do next? <…>") as {question, context}: the question runs from the first sentence
   *  that asks one (question marks inside quoted targets don't count); context is what comes before it. */
  function stuckParts(summary) {
    const text = oneLine(summary);
    const masked = text.replace(/"[^"]*"|“[^”]*”/g, (q) => '_'.repeat(q.length));
    const q = masked.indexOf('?');
    if (q < 0) return { question: text, context: '' };
    let start = 0;
    const end = /[.!…]["”’)]*\s+/g;
    for (let m = end.exec(masked); m && m.index < q; m = end.exec(masked)) start = m.index + m[0].length;
    return { question: text.slice(start).trim(), context: text.slice(0, start).trim() };
  }

  // --- init -----------------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    for (const id of [
      'request', 'request-note', 'task-select', 'task-meta', 'run-form', 'run-btn', 'stop-btn', 'run-status',
      'form-error', 'follow-btn', 'log-scroll', 'step-log', 'log-empty', 'pending', 'banner', 'banner-icon',
      'banner-title', 'banner-label', 'banner-expl', 'banner-details', 'banner-summary', 'banner-claim', 'banner-extra',
      'options', 'options-summary', 'live-btn', 'view-toggle', 'view-before', 'view-after', 'screen-img', 'screen-empty',
      'screen', 'screen-caption', 'thumbs', 'sr-status', 'hc-toggle', 'agent-badge', 'endpoints', 'vlm-endpoint',
      'vlm-endpoint-text', 'vlm-warning', 'lightning-endpoint', 'grounder-note', 'banner-question',
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
      S.autoScroll = el.scrollTop >= followTop() - 24;
      ui.followBtn.hidden = S.autoScroll || !isRunning();
      el.classList.toggle('is-scrolled', el.scrollTop > 2);
    }, { passive: true });
    ui.followBtn.addEventListener('click', () => {
      S.autoScroll = true;
      ui.logScroll.focus();
      scrollLog();
    });

    ui.liveBtn.addEventListener('click', () => {
      S.selected = null;
      S.viewPref = 'auto';
      S.autoScroll = true;
      renderViewer();
      scrollLog();
    });
    for (const input of [ui.viewBefore, ui.viewAfter]) {
      input.addEventListener('change', () => {
        if (!input.checked) return;
        S.viewPref = input.value;
        renderViewer();
      });
    }
    ui.thumbs.addEventListener('keydown', onThumbKey);
    document.addEventListener('keydown', onPageArrowKey);
    ui.options.addEventListener('change', () => {
      renderOptionsSummary();
      renderEndpoints();  // the Lightning line follows the toggle
    });
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
    resolveReady({ config: S.config });
    const last = tabStore.get(RUN_KEY);
    if (last) watchRun(last, { quietIfDone: true });
    // The VLM can come up (or go down) while the page is open: re-read the endpoints now and then.
    window.setInterval(() => {
      if (!hasEndpoints() || document.visibilityState !== 'visible') return;
      const { vlm } = endpoints();
      const due = vlm && (vlm.reachable === false || !servedList(vlm)) ? CONFIG_POLL_MS : CONFIG_REFRESH_MS;
      if (Date.now() - S.configAt >= due - 500) refreshConfig();
    }, CONFIG_POLL_MS);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') refreshConfig();
    });
  }

  async function loadConfig() {
    try {
      const cfg = await getJson('/api/config');
      S.config = cfg;
      S.configAt = Date.now();
      const agent = str(cfg.agent);
      ui.agentBadge.dataset.agent = agent === 'mock' || agent === 'real' ? agent : 'unknown';
      ui.agentBadge.textContent = agent === 'mock' ? 'Agent: mock (scripted, no models)' : `Agent: ${agent || '?'}`;
      for (const name of OPTION_NAMES) setRadio(name, configDefaults()[name]);
      applyEndpoints();
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

  /** Re-read /api/config for the endpoints (where the VLM and Lightning run, what the VLM serves). The option
   *  choices on the page are the user's now: defaults are applied once, at load. Idle real-agent pages only. */
  async function refreshConfig() {
    if (S.refreshing || !hasEndpoints() || isRunning() || S.starting) return;
    S.refreshing = true;
    const before = vlmProblem();
    try {
      const cfg = await getJson('/api/config');
      if (isRunning() || !isObj(cfg)) return;
      S.config = { ...S.config, endpoints: cfg.endpoints };
      applyEndpoints();
      renderOptionsSummary();
      const after = vlmProblem();
      if ((after && after.sentence) !== (before && before.sentence)) {
        announce(after ? `Warning: ${after.sentence}.`
          : `The VLM ${before.chip === 'not reachable' ? 'is reachable' : 'serves the screen-summary model'} again.`);
      }
    } catch {
      /* the backend is down: the next run reports it */
    } finally {
      S.configAt = Date.now();
      S.refreshing = false;
    }
  }

  function configDefaults() {
    return S.config && isObj(S.config.defaults) ? S.config.defaults : {};
  }

  /** /api/config endpoints ({vlm, lightning}, each an object or null). Both null for the mock: it runs no models. */
  function endpoints() {
    const ep = S.config && S.config.agent !== 'mock' && isObj(S.config.endpoints) ? S.config.endpoints : {};
    return { vlm: isObj(ep.vlm) ? ep.vlm : null, lightning: isObj(ep.lightning) ? ep.lightning : null };
  }

  function hasEndpoints() {
    const { vlm, lightning } = endpoints();
    return Boolean(vlm || lightning);
  }

  /** The model ids the VLM serves (GET <base_url>/models), or null when unknown (unreachable). */
  function servedList(vlm) {
    return vlm && Array.isArray(vlm.served) ? vlm.served.map(String) : null;
  }

  /** '127.0.0.1:8011' from 'http://127.0.0.1:8011/v1'. */
  function hostPort(u) {
    const s = str(u);
    try { return new URL(s).host || s; } catch { return oneLine(s); }
  }

  function shortList(items) {
    const shown = items.slice(0, SERVED_SHOWN_MAX).join(', ');
    return items.length > SERVED_SHOWN_MAX ? `${shown} +${items.length - SERVED_SHOWN_MAX} more` : shown;
  }

  /** What is wrong with the VLM endpoint: {chip (a few words for the header), sentence (for announcements)}, or
   *  null if nothing is known to be. */
  function vlmProblem() {
    const { vlm } = endpoints();
    if (!vlm) return null;
    const served = servedList(vlm);
    if (vlm.reachable === false || (vlm.reachable !== true && !served)) {
      return { chip: 'not reachable', sentence: `the VLM is not reachable at ${hostPort(vlm.base_url)}` };
    }
    const summary = oneLine(vlm.summary_model);
    if (served && summary && !served.includes(summary)) {
      return { chip: `summary model ${summary} not served`,
        sentence: `the VLM doesn’t serve ${summary}, the model that writes the screen summaries` };
    }
    return null;
  }

  function applyEndpoints() {
    renderEndpoints();
    applyServed();
  }

  /** The header line(s): 'VLM: Mac (vllm-metal) · 127.0.0.1:8011 · serves grounder' and 'Lightning: local BF16
   *  (:8000)'. Real agent only; the mock's badge already says it runs no models. */
  function renderEndpoints() {
    const { vlm, lightning } = endpoints();
    ui.endpoints.hidden = !(vlm || lightning);
    if (ui.endpoints.hidden) return;

    ui.vlmEndpoint.hidden = !vlm;
    if (vlm) {
      const where = hostPort(vlm.base_url);
      const label = oneLine(vlm.label);
      const served = servedList(vlm);
      const parts = [label || where || '?'];
      if (label && where && label !== where) parts.push(where);
      if (served) parts.push(served.length ? `serves ${shortList(served)}` : 'serves no models');
      setText(ui.vlmEndpointText, `VLM: ${parts.join(' · ')}`);
      const summary = oneLine(vlm.summary_model);
      ui.vlmEndpoint.title = [str(vlm.base_url), summary && `screen summaries: ${summary}`].filter(Boolean).join(' · ');
      const problem = vlmProblem();
      ui.vlmWarning.hidden = !problem;
      setText(ui.vlmWarning, problem ? `⚠ ${problem.chip}` : '');
    }

    const lt = lightningEndpoint(lightning);
    ui.lightningEndpoint.hidden = !lt;
    if (lt) {
      setText(ui.lightningEndpoint, `Lightning: ${lt.label}`);
      ui.lightningEndpoint.title = lt.title;
    }
  }

  /** The Lightning the next run uses: /api/config's endpoint when the toggle is on the server default (that is
   *  what it describes), else the preset the toggle picks. */
  function lightningEndpoint(lightning) {
    const choice = readOptions().lightning;
    const dflt = configDefaults().lightning;
    if (lightning && (!choice || !dflt || choice === dflt)) {
      return {
        label: oneLine(lightning.label) || hostPort(lightning.base_url) || '?',
        title: [str(lightning.base_url), lightning.model ? `model ${oneLine(lightning.model)}` : ''].filter(Boolean)
          .join(' · '),
      };
    }
    return choice ? { label: LIGHTNING_PRESET_LABELS[choice] || choice, title: '' } : null;
  }

  /** Grounder radios: a model the VLM doesn't serve is disabled, with '(not served at <host:port>)' in its label
   *  (arrow keys skip it; Tab lands on the checked one). The choice moves off a disabled model. */
  function applyServed() {
    const { vlm } = endpoints();
    const served = servedList(vlm);
    const where = vlm ? hostPort(vlm.base_url) : '';
    const inputs = [...document.querySelectorAll('input[name="grounder"]')];
    for (const input of inputs) {
      const off = Boolean(served) && !served.includes(input.value);
      input.disabled = off;
      const label = input.labels && input.labels[0];
      if (!label) continue;
      let note = label.querySelector('.opt-note');
      if (off) {
        if (!note) {
          note = h('span', { class: 'opt-note' });
          label.append(note);
        }
        setText(note, ` (not served at ${where || 'the VLM'})`);
      } else if (note) {
        note.remove();
      }
      label.classList.toggle('is-unserved', off);
    }
    keepServedChoice();
    const none = Boolean(served) && inputs.length > 0 && inputs.every((i) => i.disabled);
    ui.grounderNote.hidden = !none;
    setText(ui.grounderNote, none ? `The VLM at ${where} serves neither model, so runs can’t start until it does.` : '');
  }

  /** If the checked grounder isn't served, check the default (or the first served one) instead. */
  function keepServedChoice() {
    const inputs = [...document.querySelectorAll('input[name="grounder"]')];
    const checked = inputs.find((i) => i.checked);
    if (!checked || !checked.disabled) return;
    const dflt = configDefaults().grounder;
    const pick = inputs.find((i) => !i.disabled && i.value === dflt) || inputs.find((i) => !i.disabled);
    if (pick) pick.checked = true;
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
    setText(ui.optionsSummary, OPTION_NAMES.map((n) => (OPTION_SHORT[n] && OPTION_SHORT[n][o[n]]) || o[n] || '?')
      .join(' · '));
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
    S.voice = null;  // the box now holds the picked task, not what was heard
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
    if (S.voice && ui.request.value !== S.voice.request) S.voice = null;  // edited: no longer the transcript
    if (!S.taskId) return;
    const t = S.tasksById.get(S.taskId);
    if (t && requestKey(ui.request.value) !== requestKey(t.request)) {  // same words: still that task
      S.taskId = null;
      ui.taskSelect.value = '';
      ui.taskMeta.textContent = '';
      ui.requestNote.textContent = 'You edited the task text, so this runs as free text (no sandbox check).';
    }
  }

  const NUMBER_WORDS = new Map(Object.entries({
    zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
    eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18,
    nineteen: 19, twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90,
  }));
  const NUMBER_WORD = `(?:${[...NUMBER_WORDS.keys(), 'hundred'].join('|')})`;
  const NUMBER_RUN_RE = new RegExp(`(?<![\\p{L}\\p{N}])${NUMBER_WORD}(?:[\\s-]+${NUMBER_WORD})*(?![\\p{L}\\p{N}])`, 'gu');

  /** 'fifty' -> '50', 'fifty-five' -> '55', 'two hundred five' -> '205'; words that don't combine stay apart. */
  function numberWordsToDigits(run) {
    const out = [];
    let cur = null;
    for (const w of run.split(/[\s-]+/)) {
      if (w === 'hundred') {
        if (cur !== null && cur % 100 === 0 && cur > 0) { out.push(cur); cur = null; }  // 'hundred hundred'
        cur = (cur === null ? 1 : cur) * 100;
        continue;
      }
      const v = NUMBER_WORDS.get(w);
      if (cur !== null && cur >= 100 && cur % 100 === 0 && v < 100) cur += v;        // two hundred | five
      else if (cur !== null && cur % 100 >= 20 && cur % 10 === 0 && v < 10) cur += v;  // fifty | five
      else {
        if (cur !== null) out.push(cur);
        cur = v;
      }
    }
    if (cur !== null) out.push(cur);
    return out.join(' ');
  }

  /** Request text as a comparison key: case, accents, punctuation and whitespace don't matter, and neither does
   *  how an amount is written ('$50' = '50 dollars' = 'fifty dollars': speech-to-text spells numbers out). */
  function requestKey(text) {
    return str(text).normalize('NFKD').replace(/\p{M}+/gu, '').toLowerCase()  // NFKD splits off accents; dropped
      .replace(NUMBER_RUN_RE, numberWordsToDigits)
      .replace(/\$\s*(\d+)(?:\.00)?(?!\d)/g, '$1')
      .replace(/(\d)\s*dollars?(?![\p{L}\p{N}])/gu, '$1')
      .replace(/[^\p{L}\p{N}]+/gu, '');
  }

  function matchTask(text) {
    const key = requestKey(text);
    if (!key) return null;
    for (const t of S.tasksById.values()) if (requestKey(t.request) === key) return t;
    return null;
  }

  /** Select sandbox task `t` and put its exact text in the request box. */
  function selectTask(t) {
    S.taskId = String(t.id);
    ui.taskSelect.value = S.taskId;
    ui.request.value = t.request;
    ui.taskMeta.textContent = taskMetaText(t);
  }

  /** Put text in the request box (voice input). A sandbox task's text selects that task, so its verdict shows.
   *  `voice` (the /api/transcribe reply) is kept with the text it produced, for the next run's POST body. */
  function setRequestText(text, voice) {
    const clean = oneLine(text).slice(0, 500);
    const t = matchTask(clean);
    clearFormError();
    ui.requestNote.textContent = '';
    if (t) {
      selectTask(t);
    } else {
      S.taskId = null;
      ui.taskSelect.value = '';
      ui.request.value = clean;
      ui.taskMeta.textContent = '';
    }
    const v = cleanVoice(voice);
    S.voice = v ? { request: ui.request.value, voice: v } : null;
    return { taskId: S.taskId, request: ui.request.value };
  }

  /** {text, model, latency_ms, duration_s} as POST /api/run accepts it, or null (then no voice is sent). */
  function cleanVoice(v) {
    if (!isObj(v)) return null;
    const text = oneLine(v.text);
    const model = oneLine(v.model);
    const latency = Number(v.latency_ms);
    const duration = Number(v.duration_s);
    if (!text || text.length > VOICE_TEXT_MAX || !model || model.length > VOICE_MODEL_MAX) return null;
    if (!isNum(v.latency_ms) || !isNum(v.duration_s) || latency < 0 || duration < 0) return null;
    return { text, model, latency_ms: latency, duration_s: duration };
  }

  /** The voice record for a run of `request`: only if the box still holds exactly what the transcript put there. */
  function voiceFor(request) {
    return S.voice && S.voice.request.trim() === request ? S.voice.voice : null;
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
    const prev = document.activeElement;
    ui.runBtn.disabled = locked;
    ui.request.readOnly = locked;
    ui.taskSelect.disabled = locked;
    for (const fs of ui.optionSets) fs.disabled = locked;
    ui.stopBtn.disabled = !(locked && S.run && !S.run.finished && !S.run.stopRequested);
    ui.stopBtn.textContent = S.run && S.run.stopRequested && !S.run.finished ? 'Stopping…' : 'Stop';
    // A control that just got disabled (Run, Stop, the task picker, an option radio) drops focus to <body>,
    // where it is invisible and Space starts the mic. Keep it in the request box instead.
    if (locked && prev && prev !== document.body && (document.activeElement !== prev || prev.matches(':disabled'))) {
      ui.request.focus();
    }
  }

  const isRunning = () => Boolean(S.run && !S.run.finished);

  async function startRun() {
    if (S.starting || isRunning()) return;
    clearFormError();
    let request = ui.request.value.trim();
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
    const voice = voiceFor(request);
    const match = matchTask(request);
    if (match && (!S.taskId || S.taskId === String(match.id))) {
      // typed text that is a sandbox task's text runs as that task (its start page and verdict), as voice does
      selectTask(match);
      ui.requestNote.textContent = '';
      request = ui.request.value.trim();
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
        body: JSON.stringify({ request, task_id: taskId, options, ...(voice ? { voice } : {}) }),
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
        if (/not served/.test(msg)) refreshConfig();  // the VLM changed since the page last looked: update the choices
      }
      return;
    }
    if (voice && S.voice && S.voice.voice === voice) S.voice = null;  // one transcript, one run
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
    S.voice = null;
    ui.request.value = str(info.request);
    const taskId = info.task_id && S.tasksById.has(String(info.task_id)) ? String(info.task_id) : null;
    S.taskId = taskId;
    ui.taskSelect.value = taskId || '';
    ui.taskMeta.textContent = taskId ? taskMetaText(S.tasksById.get(taskId)) : '';
    ui.requestNote.textContent = '';
    if (isObj(info.options)) for (const name of OPTION_NAMES) setRadio(name, info.options[name]);
    keepServedChoice();  // that run's grounder may not be served any more: the radios describe the next run
    renderOptionsSummary();
    renderEndpoints();
    attach(String(info.run_id || runId), {
      request: str(info.request), taskId: info.task_id || null, options: info.options,
      focusOnEnd: active || !quietIfDone, quiet: !active, agentRunId: info.agent_run_id || null,
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
    const prev = document.activeElement;
    closeStream();
    resetRunView();
    S.run = {
      id: runId,
      request: str(info.request),
      taskId: info.taskId || null,
      options: isObj(info.options) ? info.options : null,
      startUrl: info.startUrl || null,
      focusOnEnd: Boolean(info.focusOnEnd),
      quiet: Boolean(info.quiet),  // a finished run being shown again: its steps aren't announced
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
      agentRunId: info.agentRunId ? str(info.agentRunId) : null,  // the trace's id: /logs?run=<it>
    };
    S.lastSeq = 0;
    S.runSeq += 1;
    tabStore.set(RUN_KEY, runId);
    ui.options.open = false;  // give the step log the room; the summary line still shows the choices
    setLocked(true);
    // the banner (focused after the last run) is hidden now and Options is closed: focus inside either is lost
    if (prev && prev !== ui.request && (ui.banner.contains(prev) || ui.options.contains(prev))) ui.request.focus();
    setStatus('starting');
    setPending('Starting a browser session…');
    if (!info.quiet) announce(`Run started: ${S.run.request}`);
    S.timer = window.setInterval(tick, 1000);
    openStream(runId);
    emit('acme:run-start', { runId, request: S.run.request, taskId: S.run.taskId });
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
    on('result', (d) => {
      S.run.result = d;
      if (!S.run.agentRunId && d.run_id) S.run.agentRunId = str(d.run_id);
    });
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
    scrollLog();
    if (run.focusOnEnd) ui.banner.focus();
    emit('acme:run-end', { runId: run.id, request: run.request, taskId: run.taskId });
    if (!run.quiet) refreshConfig();  // e.g. a grounding error: is the VLM still up?
  }

  function finalStatus(run) {
    const reason = run.done && run.done.reason;
    if (reason === 'stopped' || reason === 'error' || reason === 'stuck') return reason;
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
    ui.logScroll.classList.remove('is-scrolled');
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
    if (!run.agentRunId && ev.run_id) run.agentRunId = str(ev.run_id);
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
        setPending(stuckVerdict(s.action) === 'stuck' ? 'Finishing…' : `Lightning is planning step ${n + 1}…`);
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
        setPending(p.done || p.name === 'done' || stuckVerdict(p) === 'stuck' ? 'Finishing…' : 'Reading the new screen…');
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
    announceStep(s, str(ev.kind), p);
  }

  /** One sentence per step for screen readers, once its outcome is known. The step log is not a live region:
   *  its timing line and grounder coordinates change several times per step and would be read out each time. */
  function announceStep(s, kind, p) {
    if (!S.run || S.run.quiet || s.n === 0) return;
    const sentence = (t) => (/[.!?…]$/.test(t) ? t : `${t}.`);
    let tail = null;
    if (kind === 'action') {
      if (p.done || p.name === 'done') tail = '';
      else if (p.ok === false) tail = sentence(`The action failed${p.error ? `: ${oneLine(p.error)}` : ''}`);
      else if (p.screen_changed === true) tail = 'Screen changed.';
      else if (p.screen_changed === false) tail = 'No change on screen.';
      else tail = '';
      const stuck = stuckLine(p);  // once: each step has one action event
      if (stuck) tail = `${tail ? `${tail} ` : ''}${sentence(stuck)}`;
    } else if (kind === 'tool_call' && p.valid === false) {
      tail = 'Invalid tool call, asking again.';
    } else if (kind === 'error' && p.recoverable === false) {
      announce(`${stepLabel(s)}: ${sentence(errorLine(p))}`);
      return;
    } else if (kind === 'error' && !s.action) {
      tail = 'Asking again.';
    }
    if (tail === null) return;
    announce(`${stepLabel(s)}: ${sentence(actionWords(s))}${tail ? ` ${tail}` : ''}`);
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
    s.el.addEventListener('click', (e) => {  // a compact step opens it in the viewer; the open one is left alone
      if (!s.el.classList.contains('is-compact') || e.target.closest('button, a, pre')) return;
      if (String(window.getSelection ? window.getSelection() : '')) return;  // the user is selecting text
      if (s.overlayUrl || s.shotUrl) selectStep(s.n);
      else if (s.n === lastStepNumber()) ui.liveBtn.click();
    });
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
    if (stuckVerdict(a)) lines.push(['stuck', 'warn', stuckLine(a)]);
    s.errors.forEach((e, i) => lines.push([`err${i}`, 'error', errorLine(e)]));
    if (s.done && s.done.reason === 'stuck') {
      // the question itself is the banner's headline; here it closes the transcript, after "Stopped: stuck"
      if (stuckVerdict(a) !== 'stuck') lines.push(['stuck', 'warn', REASON_WORDS.stuck]);
      const ask = stuckParts(s.done.summary).question;
      if (ask) lines.push(['done', 'ask', `The agent asks: “${ask}”`]);
    } else if (s.done && s.done.reason && s.done.reason !== 'done') {
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
    const screenLabel = s.screenOpen ? 'Hide screen text' : 'Screen text';
    setText(slots.screenBtn, screenLabel);
    // the accessible name starts with the visible words, so "click Screen text" works in voice control
    slots.screenBtn.setAttribute('aria-label',
      `${screenLabel}: what Lightning read${s.n === 0 ? ' at the start' : ` after step ${s.n}`}`);
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
    if (s.errors.length || (s.toolCall && s.toolCall.valid === false) || (a && a.screen_changed === false) || stuckVerdict(a)
        || (s.grounding && s.grounding.error) || (s.done && s.done.reason && s.done.reason !== 'done')) return 'warn';
    if ((a && a.ok) || (s.done && s.done.reason === 'done') || (s.n === 0 && s.observation)) return 'ok';
    if (isRunning() && s.n === lastStepNumber()) return 'active';
    return '';
  }

  /** Where "following" puts the log: the bottom, but never past the newest step's first line. */
  function followTop() {
    const el = ui.logScroll;
    const bottom = el.scrollHeight - el.clientHeight;
    const newest = S.steps.get(lastStepNumber());
    if (!newest) return bottom;
    const top = newest.el.getBoundingClientRect().top - el.getBoundingClientRect().top + el.scrollTop - 6;
    return Math.max(0, Math.min(bottom, top));
  }

  function scrollLog() {
    if (S.autoScroll) {
      ui.logScroll.scrollTop = followTop();
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
    const open = live ? lastStepNumber() : S.selected;
    for (const s of S.steps.values()) {
      s.el.classList.toggle('is-shown', s.n === shownStep && !live);
      s.el.classList.toggle('is-compact', s.n !== open);
    }
    const thumbs = [...ui.thumbs.children];
    const focusThumb = thumbs.find((t) => Number(t.dataset.step) === shownStep) || thumbs[thumbs.length - 1];
    for (const t of thumbs) {
      t.tabIndex = t === focusThumb ? 0 : -1;
      if (Number(t.dataset.step) === shownStep) t.setAttribute('aria-current', 'true');
      else t.removeAttribute('aria-current');
    }
    if (live && focusThumb && document.activeElement !== focusThumb && !ui.thumbs.contains(document.activeElement)) {
      ui.thumbs.scrollLeft = ui.thumbs.scrollWidth;
    } else if (!live && focusThumb) {
      revealThumb(focusThumb);  // replaying from the log or by arrow key: the strip shows which step it is
    }
  }

  /** Scroll the thumbnail strip (only it, not the page) just enough to show `btn` with its focus ring. */
  function revealThumb(btn) {
    const box = ui.thumbs.getBoundingClientRect();
    const r = btn.getBoundingClientRect();
    const pad = 8;
    if (r.left < box.left + pad) ui.thumbs.scrollLeft -= box.left + pad - r.left;
    else if (r.right > box.right - pad) ui.thumbs.scrollLeft += r.right - (box.right - pad);
  }

  /** Scroll the log just enough to show `el` (not scrollIntoView: that would scroll the page too). */
  function revealStep(el) {
    const box = ui.logScroll.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    if (r.top < box.top) ui.logScroll.scrollTop -= box.top - r.top + 4;
    else if (r.bottom > box.bottom) ui.logScroll.scrollTop += Math.min(r.bottom - box.bottom + 4, r.top - box.top - 4);
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
    const s = S.steps.get(n);
    if (s) revealStep(s.el);
  }

  function onThumbKey(e) {
    moveThumb(e, [...ui.thumbs.children].indexOf(document.activeElement));
  }

  /** ←/→/Home/End replay steps from where focus lands after a run (the banner) or where it has no other use (the
   *  page itself, the screenshot), not only from the thumbnails. Fields, radios and the log keep their arrows. */
  function onPageArrowKey(e) {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    const a = document.activeElement;
    if (a && a !== document.body && a !== ui.banner && !ui.screen.contains(a)) return;
    moveThumb(e, [...ui.thumbs.children].findIndex((t) => t.tabIndex === 0));  // the roving tab stop: the step on screen
  }

  function moveThumb(e, i) {
    const thumbs = [...ui.thumbs.children];
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
    let question = '';
    let agentSays = summary;
    const check = v ? checkParts(oneLine(v.explanation)) : { head: '', details: [] };
    if (st === 'stuck') {  // the agent's question is the headline: not a failure, but not done either
      const ask = stuckParts(summary);
      tone = 'stuck';
      icon = '?';
      title = 'The agent is stuck and asks:';
      question = ask.question || 'What would you like me to do next?';
      agentSays = ask.context;
      if (v) {
        label = oneLine(v.label) || 'verified by sandbox state, not seen by the agent';
        expl = v.passed ? `✓ ${check.head}` : `✗ Not done. ${check.head}`;
      } else {
        label = run.taskId ? 'No sandbox verdict received' : FREE_TEXT_LABEL;
        expl = 'Not done: the agent stopped to ask you.';
      }
    } else if (v && st === 'stopped') {  // the stop is the headline; the sandbox check still says where things stand
      tone = 'stopped';
      icon = '■';
      title = v.passed ? 'Stopped, but the task is done' : 'Stopped: the task is not done';
      label = oneLine(v.label) || 'verified by sandbox state, not seen by the agent';
      expl = `${v.passed ? '✓' : '✗'} ${check.head}`;
    } else if (v) {
      tone = v.passed ? 'pass' : 'fail';
      icon = v.passed ? '✓' : '✗';
      title = v.passed ? 'Verified: the task is done' : 'Not verified: the task is not done';
      label = oneLine(v.label) || 'verified by sandbox state, not seen by the agent';
      expl = check.head;
    } else if (st === 'error') {
      tone = 'error';
      icon = '!';
      title = 'Run failed';
      expl = run.statusMessage || summary;
      label = run.taskId ? '' : FREE_TEXT_LABEL;
    } else if (st === 'stopped') {
      tone = 'stopped';
      icon = '■';
      title = 'Run stopped';
      label = run.taskId ? 'No sandbox verdict for this run' : FREE_TEXT_LABEL;
    } else {
      tone = 'neutral';
      icon = 'ⓘ';
      title = 'Run finished';
      label = run.taskId ? 'No sandbox verdict received' : FREE_TEXT_LABEL;
    }
    ui.banner.dataset.tone = tone;
    setText(ui.bannerIcon, icon);
    setText(ui.bannerTitle, title);
    ui.bannerQuestion.hidden = !question;
    setText(ui.bannerQuestion, question ? `“${question}”` : '');
    // focus lands here after a run: a screen reader then reads the question as the banner's name
    ui.banner.setAttribute('aria-labelledby', question ? 'banner-title banner-question' : 'banner-title');
    setText(ui.bannerLabel, label);
    renderCheck(expl, check.details, check.more);

    renderSummary(agentSays, st === 'stuck' ? '' : 'The agent gave no summary.');
    const claim = [];
    if (claimed === true) claim.push('The agent claimed success.');
    else if (claimed === false) claim.push('The agent did not claim success.');
    if (v && claimed !== null && v.passed !== claimed) {
      claim.push(claimed ? 'The sandbox state says otherwise.' : 'The sandbox state shows it is done anyway.');
    }
    if (st === 'stuck') claim.push('To answer, add what it needs to your request and run it again.');
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
    if (stats.n_stuck_warnings) extra.push(plural('n_stuck_warnings', 'stuck warning', 'stuck warnings'));
    ui.bannerExtra.replaceChildren();
    if (extra.length) ui.bannerExtra.append(document.createTextNode(extra.join(' · ')));
    const links = [];
    const href = safeHttpUrl(run.startUrl);
    if (href) links.push(h('a', { href, target: '_blank', rel: 'noopener', text: 'Open this session in the sandbox' }));
    const traceId = run.agentRunId || (isObj(run.result) && run.result.run_id ? str(run.result.run_id) : '');
    if (RUN_ID_RE.test(traceId)) {
      links.push(h('a', { id: 'banner-logs-link', href: `/logs?run=${encodeURIComponent(traceId)}`,
        text: 'View this run in Logs' }));
    }
    for (const a of links) {
      if (ui.bannerExtra.firstChild) ui.bannerExtra.append(document.createTextNode(' · '));
      ui.bannerExtra.append(a);
    }
    ui.banner.hidden = false;
  }

  /** "Agent’s summary: “…”", cut to SUMMARY_MAX characters with a Show more button, so a long one (a real agent's,
   *  or a stuck stop's list of what it tried) doesn't push the screenshot and the thumbnails out of view. */
  function renderSummary(text, empty) {
    const run = S.run;
    ui.bannerSummary.replaceChildren();
    if (!text) {
      if (empty) ui.bannerSummary.append(empty);
      return;
    }
    const long = text.length > SUMMARY_MAX;
    const shown = long && !run.summaryOpen ? `${text.slice(0, SUMMARY_MAX).trimEnd()}…` : text;
    ui.bannerSummary.append(`Agent’s summary: “${shown}”`);
    if (!long) return;
    const btn = h('button', {
      type: 'button', class: 'btn-link', 'aria-expanded': String(Boolean(run.summaryOpen)),
      text: run.summaryOpen ? 'Show less' : 'Show more',
    });
    btn.addEventListener('click', () => {
      run.summaryOpen = !run.summaryOpen;
      renderSummary(text, empty);
      ui.bannerSummary.querySelector('button').focus();
    });
    ui.bannerSummary.append(' ', btn);
  }

  /** A sandbox/checks.py explanation as a short headline plus details shown on request (keeps the banner small). */
  function checkParts(text) {
    let m = /^all (\d+) conditions? passed$/.exec(text);
    if (m) return { head: `Sandbox check: ${m[1]}/${m[1]} condition${m[1] === '1' ? '' : 's'} passed.`, details: [] };
    m = /^(\d+) of (\d+) conditions? failed: (.+)$/.exec(text);
    if (m) return { head: `Sandbox check: ${m[1]} of ${m[2]} conditions failed.`, details: m[3].split('; ') };
    m = /^any: none passed: (.+)$/.exec(text);
    if (m) return { head: 'Sandbox check: none of the alternatives passed.', details: m[1].split('; ') };
    if (text.length > CHECK_HEAD_MAX) {
      return { head: `Sandbox check: ${text.slice(0, CHECK_HEAD_MAX).trimEnd()}…`, details: [text], more: 'details' };
    }
    return { head: text ? `Sandbox check: ${text}` : '', details: [] };
  }

  function renderCheck(expl, details, more = 'which') {
    const run = S.run;
    ui.bannerExpl.replaceChildren(document.createTextNode(expl));
    ui.bannerDetails.replaceChildren(...details.map((d) => h('li', { text: d })));
    ui.bannerDetails.hidden = !(details.length && run.checkOpen);
    if (!details.length) return;
    const btn = h('button', {
      type: 'button', class: 'btn-link', 'aria-expanded': String(Boolean(run.checkOpen)), 'aria-controls': 'banner-details',
      text: run.checkOpen ? 'Hide' : (more === 'which' ? `Show which${details.length > 1 ? '' : ' one'}` : 'Show details'),
    });
    btn.addEventListener('click', () => {
      run.checkOpen = !run.checkOpen;
      renderCheck(expl, details, more);
      ui.bannerExpl.querySelector('button').focus();
    });
    ui.bannerExpl.append(' ', btn);
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
