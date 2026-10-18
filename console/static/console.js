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
 * isStarting(), setRequest(text, voice?) -> {taskId, request}, run(), stop(), matchTask(text) -> task | null,
 * matchTaskDetailed(text) -> {task, method: 'exact'|'fuzzy', score, runner_up} | null (see createTaskMatcher) and
 * explainMatch(text) (every task's score and what ruled it out). The page fires
 * `acme:run-start` / `acme:run-end` on document (detail: {runId, request, taskId}).
 * `voice` ({text, model, latency_ms, duration_s} from /api/transcribe) goes into POST /api/run
 * for the logs page, but only while the request box still holds exactly what that transcript put
 * there, and only for the first run that uses it. The backend never passes it to the agent.
 *
 * A run is checked as a sandbox task (task_id: its start page, state and verdict) when the task was picked from the
 * list or the request matches one, exactly or fuzzily (createTaskMatcher). The picker shows the match before the
 * run (voice input at once, typed text when the box loses focus) so it can be undone (Free text). The agent always
 * gets the user's request: a fuzzy match never swaps in the task's text. POST /api/run carries task_match {method:
 * picked|fuzzy, score, matched_from}; an exact match is left out (the backend records it). The banner says "Checked as
 * task: <id> (matched from your request)" or "(picked from the list)" above the verdict.
 *
 * "Ask when unclear" (options.clarify on|off, agent/clarify.py): with it on, the agent's step 0 is a `clarify` event
 * (after start, before any observation). Without that event (off, or an agent without the step) nothing changes.
 * - proceed: the step log says the request was clear, and "What the agent understood: <normalized_request>" shows
 *   above the log (display only: the agent works from the user's words). A run with no task whose restatement
 *   matches one (matchTaskDetailed) asks the backend to check it as that task: POST /api/runs/{id}/task_match
 *   {task_id, method: "restated", score, matched_from}. The backend refuses (409/400, said in the log) unless the
 *   run has no task or verdict yet and the task starts where the run did. The banner then says "Checked as task: <id>
 *   (matched from the agent's restatement: "<text>")".
 * - ask: the run ends with done reason "needs_input" (no browser opened). The banner asks the question with an
 *   answer box; the answer starts a follow-up run: POST /api/run {request: previous + " (clarified: <answer>)",
 *   clarifies: {parent_run_id: <the asking run's agent run id>, answer}, task_id/task_match from matching that text}.
 *   The backend builds the same text (agent.clarify.follow_up_request) and refuses a different one. Voice input
 *   answers too: while a question waits, the next transcript fills the answer box (voice.js, pendingQuestion()).
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
  const OPTION_NAMES = ['grounder', 'thinking', 'clarify', 'lightning'];
  const DIFFICULTIES = ['easy', 'medium', 'hard'];
  const TAG_WORDS = { icon_only: 'unlabeled icon buttons', canvas: 'canvas widget', scroll: 'needs scrolling' };
  const TIMING_ORDER = ['clarify', 'plan', 'ground', 'act', 'summary'];
  const TIMING_HIDDEN = new Set(['settle', 'screenshot']);
  const FINAL = new Set(['done', 'stopped', 'error', 'stuck', 'needs_input']);
  const STATUS_TEXT = {
    idle: 'Idle', starting: 'Starting', running: 'Running', reconnecting: 'Reconnecting',
    done: 'Done', stopped: 'Stopped', error: 'Error', stuck: 'Stuck', needs_input: 'Needs your answer',
  };
  const ANSWER_MAX = 300;               // POST /api/run clarifies.answer: 1..300 characters
  const CLARIFIED = ' (clarified: ';    // agent/clarify.py follow_up_request: previous + " (clarified: <answer>)"
  const TASK_MATCH_METHODS = ['picked', 'exact', 'fuzzy', 'restated'];
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
    clarify: { on: 'asks if unclear', off: 'never asks first' },
    lightning: { local: 'local Lightning', hosted: 'hosted Lightning' },
  };
  const FREE_TEXT_LABEL = 'Free-text request: no sandbox check';
  const REASON_WORDS = {
    max_steps: 'Stopped: reached the step limit before finishing.',
    stuck: 'Stopped: stuck',
    stopped: 'Stopped by the user.',
    error: 'Stopped after an error.',
    needs_input: 'Stopped to ask you first.',
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
    taskMatch: null,    // how taskId was chosen: {method: picked|exact|fuzzy, score, matched_from} (see setTask)
    declined: null,     // a task id the user chose Free text over: not matched again until the box is emptied
    matcher: null,      // createTaskMatcher over /api/tasks
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
    answerVoice: null,  // {answer, voice}: the last transcript put in the answer box (see setAnswerText)
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
    matchTaskDetailed: (text) => matchTaskDetailed(text),
    explainMatch: (text) => (S.matcher ? S.matcher.explain(oneLine(text)) : null),
    matchSupport: (taskId, text) => (S.matcher ? S.matcher.support(taskId, oneLine(text)) : null),
    // a needs_input question waiting for an answer: {question, request, parentRunId} | null (voice.js routes the
    // next transcript to the answer box while there is one)
    pendingQuestion: () => {
      const q = pendingQuestion();
      return q ? { question: q.question, request: q.request, parentRunId: q.parentRunId } : null;
    },
    setAnswer: (text, voice) => setAnswerText(text, voice),
    answer: () => answerQuestion(),
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
    if (s.n === 0 && ((s.clarify && s.clarify.action === 'ask') || (s.done && s.done.reason === 'needs_input'))) {
      return 'Asked you first, before opening the app';
    }
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
    const clarify = 'clarify' in c ? c.clarify : o.clarify;  // AgentConfig: a bool; the options: 'on' | 'off'
    if (clarify === true || clarify === 'on') parts.push('ask when unclear: on');
    else if (clarify === false || clarify === 'off') parts.push('ask when unclear: off');
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
      'vlm-endpoint-text', 'vlm-warning', 'lightning-endpoint', 'grounder-note', 'banner-question', 'banner-task',
      'banner-answer', 'answer-input', 'answer-btn', 'answer-error', 'understood',
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
    ui.bannerAnswer.addEventListener('submit', (e) => { e.preventDefault(); answerQuestion(); });
    ui.answerInput.addEventListener('input', () => {
      clearAnswerError();
      if (S.answerVoice && ui.answerInput.value !== S.answerVoice.answer) S.answerVoice = null;  // edited
    });
    ui.request.addEventListener('change', () => syncMatch(true));  // leaving the box: the picker shows its match
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
      S.matcher = createTaskMatcher([...S.tasksById.values()]);
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

  // --- matching a request to a sandbox task ---------------------------------------------------
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

  // Exact: the request's requestKey is a task's (same words; case, punctuation and how an amount is written aside).
  // Fuzzy: both texts become content tokens (fuzzyTokens: filler dropped, synonyms folded, light stemming), the
  // task's anchor words (TASK_ANCHORS: its number, month, district, on/off and key nouns) must all be in the request,
  // nothing in the request may contradict the task (another number, month, district or channel, the opposite on/off,
  // a negated anchor, a schedule word, another task's key word, a second request), and the weighted overlap must be
  // >= MATCH_MIN_SCORE and beat the next eligible task by MATCH_MIN_MARGIN. Anything else is free text.
  const MATCH_MIN_SCORE = 0.5;
  const MATCH_MIN_MARGIN = 0.15;
  const ANCHOR_WEIGHT = 2.5;   // an anchor counts this much in the overlap; other words 1 (0.5 if 3+ tasks have them)
  const MISSING_WEIGHT = 0.5;  // a task word the request leaves out costs half; a word the request adds costs fully

  const W = (src) => new RegExp(`(?<![\\p{L}\\p{N}])(?:${src})(?![\\p{L}\\p{N}])`, 'gu');
  // Spoken-language rewrites, in order, on lowercase text without punctuation: filler goes, synonyms become the
  // word the task texts use, multi-word names become one token. Task texts and requests both go through them.
  const PHRASES = [
    [W('(?:can|could|would|will) (?:you|u)'), ' '],
    [W('i (?:would|d) (?:like|love) (?:you )?(?:to )?|id (?:like|love) (?:you )?(?:to )?'), ' '],
    [W('i (?:want|need) (?:you )?to|i wanna|help me|go ahead and|for me|thank you'), ' '],
    [W(['please', 'pls', 'plz', 'um+', 'uh+', 'uhm', 'erm?', 'hmm+', 'okay', 'ok', 'hey', 'hi', 'hello', 'so', 'just',
      'kindly', 'thanks', 'now', 'actually', 'quickly', 'right away', 'alright', 'all right'].join('|')), ' '],
    [W('(?:sms|text) messages?|sms texts?|texts|texting|s m s'), ' sms '],  // before 'text' (the font)
    [W('text (alerts?|notifications?)'), ' sms $1 '],
    [W('auto pay(?:ments?)?|autopayments?|automatic (?:bill )?pay(?:ments?)?|auto payments?'), ' autopay '],
    [W('(?:power )?outages?|blackouts?|power cuts?|power failures?'), ' outage '],
    [W('(?:is |are )?(?:not|isnt|arent) working|(?:does not|doesnt|do not|dont) work|stopped working|broke|busted|'
      + 'faulty|malfunctioning|not functioning'), ' broken '],
    [W('(?:every|each) (month|week|day)'), (_, unit) => ` ${unit === 'day' ? 'daily' : `${unit}ly`} `],
    [W('make a payment(?: of)?|payment of'), ' pay '],
    [W('switch|flip'), ' turn '],
    [W('enable|activate|set up|sign up for|start using'), ' turn on '],
    [W('disable|deactivate|unsubscribe(?: me)?(?: from)?|opt out(?: of)?|cancel|no more'), ' turn off '],
    [W('stop'), ' off '],
    [W('every|everything|each'), ' all '],
    [W('favou?rites?'), ' star '],
    [W('fonts?|letters|lettering|writing'), ' text '],
    [W('larger|increase|enlarge|big'), ' bigger '],
    [W('highest|biggest|largest|peak|max|maximum'), ' most '],
    [W('electricity|energy|usage|consumption|kwh|kilowatt hours?'), ' power '],
    [W('used|using|uses|consumed'), ' use '],
    [W('invoice'), ' statement '],
    [W('can not|cannot|unable to'), ' cant '],
    [W('customer (?:support|service)|support team|help desk|helpdesk'), ' support '],
    [W('(?:electric|electricity|power|smart) meter'), ' meter '],
    [W('landlady|property manager'), ' landlord '],
    [W('promos?|promotional|marketing|offers|deals|ads|advertising|advertisements'), ' promotion '],
    [W('e mails?'), ' email '],
    [W('mail|inbox'), ' message '],
    [W('neighbou?rhood|area'), ' district '],
    [W('old town'), ' oldtown '], [W('maple grove'), ' maplegrove '], [W('harbou?r point'), ' harborpoint '],
    [W('north gate'), ' northgate '], [W('hill crest'), ' hillcrest '], [W('river side'), ' riverside '],
    [W('sept?'), ' september '], [W('jan'), ' january '], [W('feb'), ' february '], [W('apr'), ' april '],
    [W('jun'), ' june '], [W('jul'), ' july '], [W('aug'), ' august '], [W('oct'), ' october '],
    [W('nov'), ' november '], [W('dec'), ' december '],
    [W('twice'), ' 2 '], [W('thrice'), ' 3 '],
  ];
  const STOP = new Set(('a an the my me mine i im ive it its this that these those to too for of in at from with as '
    + 'is are was were be been being am do does did done doing and or you your yours we our us they them their he she '
    + 'his her him can could would will should shall might must have has had having there here what which who whom '
    + 'whose when where why how some any one about into onto over up down out then than also very really much more '
    + 'again if whether say said saying says tell telling told let lets want need get got go going gonna wanna like by '
    + 'turn').split(' '));  // 'turn' too: "turn on" / "turn off" carry their meaning in on / off
  const NEGATORS = new Set(['not', 'dont', 'never', 'wont', 'doesnt', 'didnt', 'isnt', 'arent', 'without', 'except',
    'neither', 'nor']);
  const NEGATION_SPAN = 3;  // a negator covers the next 3 content words: "don't turn on high contrast"
  const MONTHS = new Set(['january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october',
    'november', 'december']);  // not "may": far more often the verb
  const DISTRICTS = new Set(['riverside', 'hillcrest', 'oldtown', 'maplegrove', 'harborpoint', 'northgate']);
  const POLARITY = new Set(['on', 'off']);
  const CHANNELS = new Set(['email', 'sms', 'push']);
  const SCHEDULE = new Set(['monthly', 'weekly', 'daily', 'recurring', 'schedule', 'scheduled', 'tomorrow', 'later',
    'automatically', 'automatic']);
  const CONJUNCTIONS = new Set(['and', 'also', 'then', 'plus', 'after', 'afterwards']);
  // Opposite intent: "cancel my outage report", "remove the star". Read from the raw words, because synonyms
  // fold "cancel"/"stop" into "turn off" before tokenizing. Only "off"-type tasks may contain them.
  const UNDO_WORDS = new Set(['cancel', 'cancelled', 'canceled', 'remove', 'undo', 'unstar', 'delete', 'withdraw',
    'revoke', 'refund', 'retract', 'unmark', 'reverse']);
  // A question is not a request ("Is high contrast on?", "Should I pay $50?"), unless the task itself asks one
  // ("Which month did I use the most power? Open it."). Polite "can/could/would you ...?" stays a request.
  const QUESTION_START = new Set(['is', 'are', 'was', 'were', 'did', 'does', 'should', 'how', 'why', 'what', 'when',
    'where', 'which', 'who', 'have', 'has']);
  const CLAUSE_SKIP = new Set(['i', 'you', 'we', 'me', 'to', 'can', 'could', 'would', 'will', 'please']);
  // Verbs that start a second request after "and" / "then". Light verbs ('turn', 'make', 'set') only count when the
  // words after them are foreign to the task ("... and turn on high contrast"), not when they repeat it.
  const ACTION_VERBS = ['turn', 'make', 'set', 'change', 'reduce', 'decrease', 'lower', 'raise', 'send', 'pay',
    'download', 'open', 'close', 'mark', 'star', 'unstar', 'report', 'delete', 'remove', 'archive', 'add', 'update',
    'edit', 'show', 'find', 'check', 'click', 'select', 'email', 'call', 'reply', 'forward', 'print', 'schedule',
    'start', 'save', 'share', 'upload', 'move', 'toggle', 'write', 'type', 'submit', 'buy', 'order', 'book', 'search',
    'view', 'sort', 'filter', 'hide', 'mute', 'block', 'dismiss', 'zoom', 'navigate', 'log', 'sign', 'read', 'pin',
    'flag'];
  const LIGHT_VERBS = new Set(['turn', 'mak', 'set']);
  const NO_STEM = new Set([...MONTHS, ...DISTRICTS, ...POLARITY, ...CHANNELS, 'autopay', 'bus', 'gas', 'yes', 'this',
    'its', 'is', 'us', 'as', 'has', 'was', 'does', 'news', 'status', 'access', 'press', 'less', 'process']);

  /** Light stemming (both sides get it, so it only has to be consistent): -s, -es, -ies, -ed, -ing, a final -e and a
   *  doubled final consonant ('starred' -> 'star', 'forgetting' -> 'forget', 'messages' / 'message' -> 'messag'). */
  function stem(w) {
    if (w.length <= 3 || NO_STEM.has(w) || /^\d/.test(w)) return w;
    let s = w;
    if (s.endsWith('ies') && s.length > 4) s = `${s.slice(0, -3)}y`;
    else if (s.endsWith('ing') && s.length >= 6) s = s.slice(0, -3);
    else if (s.endsWith('ed') && s.length >= 5) s = s.slice(0, -2);
    else if (s.endsWith('es') && s.length >= 5) s = s.slice(0, -2);
    else if (s.endsWith('s') && !/(?:ss|us|is)$/.test(s)) s = s.slice(0, -1);
    if (s.length >= 4 && s.endsWith('e')) s = s.slice(0, -1);
    if (/([b-df-hj-km-np-rtv-z])\1$/.test(s)) s = s.slice(0, -1);  // not ll or ss: 'bill', 'press'
    return s;
  }
  const ACTION_STEMS = new Set(ACTION_VERBS.map(stem));

  const plainNumber = (n) => {  // '50.00' -> '50', '50.50' -> '50point5'
    const [whole, frac = ''] = n.split('.');
    const f = frac.replace(/0+$/, '');
    return f ? `${Number(whole)}point${f}` : String(Number(whole));
  };

  /** Request text -> {tokens: content stems in order, negated: stems a negator covers, clauses: [{verb, after}] for
   *  "... and <verb> ..."}. Amounts: '$50', '50 dollars', 'fifty dollars', 'fifty bucks' -> '50'. */
  function fuzzyTokens(text) {
    let s = str(text).normalize('NFKD').replace(/\p{M}+/gu, '').toLowerCase().replace(/[’‘`´]/g, "'");
    s = s.replace(/(\p{L})'s(?![\p{L}\p{N}])/gu, '$1')  // possessive: "landlord's" -> "landlord"
      .replace(/(\p{L})'(\p{L})/gu, '$1$2')               // "can't" -> "cant", "i'd" -> "id"
      .replace(/(\d),(?=\d{3}(?!\d))/g, '$1')             // 1,000 -> 1000
      .replace(NUMBER_RUN_RE, (run, at, all) => (run === 'one' && !/^\s*(?:dollars?|bucks?)\b/.test(all.slice(at + 3))
        ? run : ` ${numberWordsToDigits(run)} `))       // a lone 'one' is a pronoun ("the one from my landlord")
      .replace(/\$\s*(\d+(?:\.\d+)?)/g, (_, n) => ` ${plainNumber(n)} `)
      .replace(/(\d+\.\d+)/g, (_, n) => plainNumber(n))
      .replace(/(\d+(?:point\d+)?)\s*(?:dollars?|bucks?|usd)(?![\p{L}\p{N}])/gu, '$1')
      .replace(/(\d+)\s*cents?(?![\p{L}\p{N}])/gu, '$1c');
    s = ` ${s.replace(/[^\p{L}\p{N}]+/gu, ' ').trim()} `;
    const raw = s.trim().split(/\s+/).filter(Boolean);
    const undo = raw.filter((w) => UNDO_WORDS.has(w));
    const first = (raw[0] === 'do' && /\?\s*$/.test(str(text))) || QUESTION_START.has(raw[0]) ? raw[0] : null;
    for (const [re, to] of PHRASES) s = s.replace(re, to);
    const all = s.split(/\s+/).filter(Boolean);
    const clauses = [];
    all.forEach((w, i) => {
      if (!CONJUNCTIONS.has(w) || CONJUNCTIONS.has(all[i - 1])) return;
      let j = i + 1;
      while (j < all.length && (CONJUNCTIONS.has(all[j]) || CLAUSE_SKIP.has(all[j]))) j += 1;
      if (j < all.length && ACTION_STEMS.has(stem(all[j]))) {
        clauses.push({ verb: stem(all[j]),
          after: all.slice(j + 1).filter((x) => !STOP.has(x) && !NEGATORS.has(x)).slice(0, 3).map(stem) });
      }
    });
    const tokens = [];
    const negated = new Set();
    let span = 0;
    for (const w of all) {
      if (STOP.has(w)) continue;
      if (NEGATORS.has(w)) { span = NEGATION_SPAN; continue; }
      const t = stem(w);
      tokens.push(t);
      if (span > 0) { negated.add(t); span -= 1; }
    }
    return { tokens, negated, clauses, undo, question: first };
  }

  const tokenSet = (text) => new Set(fuzzyTokens(text).tokens);

  // Per task: the words a request must contain (each inner list: any one of them), and words that are fine in a
  // request though the task text lacks them (the portal's statements are 2026's). A task id not listed here gets
  // its numbers, months, districts, on / off and the words no other task has.
  const TASK_ANCHORS = {
    'text-size-large': { need: [['text'], ['bigger']] },
    'download-september-statement': { need: [['download'], ['september'], ['statement', 'bill']], also: ['2026'] },
    'mark-all-read': { need: [['mark'], ['all'], ['read']] },
    'turn-on-autopay': { need: [['autopay'], ['on']] },
    'high-contrast': { need: [['high'], ['contrast'], ['on']] },
    'sms-billing-off': { need: [['sms'], ['bill'], ['off']] },
    'max-usage-month': { need: [['month'], ['most']] },
    'report-outage-riverside': { need: [['report'], ['outage'], ['riverside']] },
    'star-landlord-message': { need: [['star'], ['landlord']] },
    'pay-50': { need: [['pay'], ['50']] },
    'message-support-meter': { need: [['support'], ['meter'], ['broken']] },
    'stop-promotions': { need: [['promotion'], ['off']] },
  };
  // Anchors too common to mean "that other task" when they turn up in a request for this one.
  const GENERIC_ANCHORS = new Set(['on', 'off', 'all', 'most', 'month', 'read', 'high', 'mark', 'report', 'broken',
    'bill']);
  const classOf = (t) => (/^\d/.test(t) ? 'number' : MONTHS.has(t) ? 'month' : DISTRICTS.has(t) ? 'district'
    : POLARITY.has(t) ? 'polarity' : CHANNELS.has(t) ? 'channel' : '');
  const byClass = (tokens) => {
    const out = {};
    for (const t of tokens) {
      const c = classOf(t);
      if (c) (out[c] = out[c] || new Set()).add(t);
    }
    return out;
  };

  /** The matcher for a task list: {detailed(text), explain(text)}. */
  function createTaskMatcher(tasks) {
    const models = tasks.map((t) => ({ task: t, id: String(t.id), toks: tokenSet(t.request), key: requestKey(t.request),
      question: fuzzyTokens(t.request).question,
      spec: Object.prototype.hasOwnProperty.call(TASK_ANCHORS, String(t.id)) ? TASK_ANCHORS[String(t.id)] : null }));
    const df = new Map();
    for (const m of models) for (const tok of m.toks) df.set(tok, (df.get(tok) || 0) + 1);
    for (const m of models) {
      m.need = m.spec ? m.spec.need.map((alts) => alts.flatMap((a) => [...tokenSet(a)]))
        : [...m.toks].filter((tok) => classOf(tok) || df.get(tok) === 1).map((tok) => [tok]);
      m.anchors = new Set(m.need.flat());
      m.also = new Set(((m.spec && m.spec.also) || []).flatMap((a) => [...tokenSet(a)]));
      m.accept = new Set([...m.toks, ...m.anchors, ...m.also]);
      m.cls = byClass(m.toks);
    }
    const foreign = new Set();
    for (const m of models) for (const a of m.anchors) if (!GENERIC_ANCHORS.has(a) && !classOf(a)) foreign.add(a);
    const weight = (m, tok) => (m.anchors.has(tok) ? ANCHOR_WEIGHT : (df.get(tok) || 0) >= 3 ? 0.5 : 1);

    /** One task against a request: {id, task, score (weighted overlap, 0..1), eligible, why: [what rules it out]}. */
    function judge(m, req) {
      const R = new Set(req.tokens);
      const why = [];
      const tside = new Set(m.toks);
      for (const alts of m.need) {
        const hit = alts.find((a) => R.has(a));
        if (!hit) {
          why.push(`missing ${alts.join('|')}`);
          continue;
        }
        if (req.negated.has(hit)) why.push(`negated ${hit}`);
        for (const a of alts) tside.delete(a);
        tside.add(hit);
      }
      const rcls = byClass(R);
      const mine = (c) => m.cls[c] || new Set();
      for (const c of ['number', 'month', 'district']) {
        for (const tok of rcls[c] || []) if (!mine(c).has(tok) && !m.also.has(tok)) why.push(`other ${c} ${tok}`);
      }
      if (mine('polarity').size) {
        for (const tok of rcls.polarity || []) if (!mine('polarity').has(tok)) why.push(`opposite ${tok}`);
      }
      if (mine('channel').size && rcls.channel) {  // channels: none named, or exactly the task's
        const said = [...rcls.channel].sort().join('+');
        const want = [...mine('channel')].sort().join('+');
        if (said !== want) why.push(`channels ${said}, not ${want}`);
      }
      if (req.undo.length && !mine('polarity').has('off')) why.push(`opposite intent: ${req.undo[0]}`);
      if (req.question && m.question !== req.question) why.push(`a question (${req.question} …), not a request`);
      for (const tok of R) {
        if (SCHEDULE.has(tok) && !m.toks.has(tok)) why.push(`schedule word ${tok}`);
        if (foreign.has(tok) && !m.accept.has(tok)) why.push(`another task's word ${tok}`);
      }
      for (const c of req.clauses) {
        if (!m.toks.has(c.verb) && (!LIGHT_VERBS.has(c.verb) || c.after.some((t) => !m.accept.has(t)))) {
          why.push(`a second request: and ${[c.verb, ...c.after].join(' ')}`);
        }
      }
      for (const o of models) {  // all of another task's anchors, some of them foreign to this one: two requests
        if (o !== m && o.need.every((alts) => alts.some((a) => R.has(a)))
            && [...o.anchors].some((a) => R.has(a) && !m.accept.has(a))) why.push(`also asks for ${o.id}`);
      }
      const rside = new Set([...R].filter((tok) => !m.also.has(tok) || tside.has(tok)));
      let inter = 0;
      let union = 0;
      for (const tok of new Set([...rside, ...tside])) {
        const w = weight(m, tok);
        const both = rside.has(tok) && tside.has(tok);
        if (both) inter += w;
        union += both || rside.has(tok) ? w : w * MISSING_WEIGHT;
      }
      const score = union ? Math.round((inter / union) * 1000) / 1000 : 0;
      return { id: m.id, task: m.task, score, eligible: why.length === 0, why };
    }

    /** The best other task: {id, score (0 when a rule rules it out), similarity, ruled_out (the first reason)}. */
    function runnerUp(judged, bestId) {
      const o = judged.filter((j) => j.id !== bestId)
        .sort((a, b) => Number(b.eligible) - Number(a.eligible) || b.score - a.score)[0];
      return o ? { id: o.id, score: o.eligible ? o.score : 0, similarity: o.score, ruled_out: o.why[0] || null } : null;
    }

    /** {task, method: 'exact'|'fuzzy', score, runner_up} or null (free text: no sandbox check). */
    function detailed(text) {
      const key = requestKey(text);
      if (!key) return null;
      const req = fuzzyTokens(text);
      const judged = models.map((m) => judge(m, req));
      const exact = models.find((m) => m.key === key);
      if (exact) return { task: exact.task, method: 'exact', score: 1, runner_up: runnerUp(judged, exact.id) };
      const ok = judged.filter((j) => j.eligible).sort((a, b) => b.score - a.score);
      if (!ok.length) return null;
      const [best, second] = ok;
      if (best.score < MATCH_MIN_SCORE || best.score - (second ? second.score : 0) < MATCH_MIN_MARGIN) return null;
      return { task: best.task, method: 'fuzzy', score: best.score, runner_up: runnerUp(judged, best.id) };
    }

    /** Why a request did or didn't match: its tokens and every task's score and reasons (for tests and debugging). */
    function explain(text) {
      const req = fuzzyTokens(text);
      return {
        tokens: req.tokens, negated: [...req.negated], clauses: req.clauses,
        tasks: models.map((m) => judge(m, req)).map(({ task, ...j }) => j)
          .sort((a, b) => Number(b.eligible) - Number(a.eligible) || b.score - a.score),
      };
    }

    /** Guard against a self-fulfilling restated match (orchestrator, 2026-10-18): every distinguishing word of the
     *  task (its number/month/district/channel tokens, and required words no other task has) must already be in the
     *  user's own words (original request + clarification answer). -> {ok, missing: [words]} */
    function support(taskId, text) {
      const m = models.find((x) => x.id === String(taskId));
      if (!m) return { ok: false, missing: [] };
      const R = new Set(fuzzyTokens(text).tokens);
      const KEY_CLASSES = ['number', 'month', 'district', 'channel'];  // not on/off: polarity is the request's form
      const distinguishing = (a) => KEY_CLASSES.includes(classOf(a)) || (df.get(a) || 0) === 1;
      const missing = [];
      for (const alts of m.need) {
        if (alts.some(distinguishing) && !alts.some((a) => R.has(a))) missing.push(alts.join('|'));
      }
      for (const c of ['number', 'month', 'district', 'channel']) {
        for (const tok of m.cls[c] || []) {
          if (!R.has(tok) && !missing.some((x) => x.split('|').includes(tok))) missing.push(tok);
        }
      }
      return { ok: missing.length === 0, missing };
    }

    return { detailed, explain, support };
  }

  function matchTaskDetailed(text) {
    return S.matcher ? S.matcher.detailed(oneLine(text)) : null;
  }

  function matchTask(text) {
    const m = matchTaskDetailed(text);
    return m ? m.task : null;
  }

  // --- request form ---------------------------------------------------------------------
  // S.taskId is the sandbox task the next run is checked as; S.taskMatch says how it was chosen: {method: 'picked'
  // (from the list) | 'exact' | 'fuzzy' (matched from the request), score, matched_from}. The agent always gets the
  // request box's text: a fuzzy match keeps the user's own words; an exact one shows the task's own spelling of the
  // same words. S.declined is a matched task the user chose Free text over: it isn't matched again (while they edit
  // the same words) until the box is emptied, a task is picked or voice input replaces the text.
  function setTask(t, match) {
    S.taskId = String(t.id);
    S.taskMatch = match;
    ui.taskSelect.value = S.taskId;
    ui.taskMeta.textContent = taskMetaText(t);
  }

  function clearTask() {
    S.taskId = null;
    S.taskMatch = null;
    ui.taskSelect.value = '';
    ui.taskMeta.textContent = '';
  }

  const MATCH_HINT = 'This matches sandbox task ';

  function matchNote(id, method) {
    return `Will be checked as task ${id} (matched from your request${method === 'fuzzy'
      ? '; the agent gets your words as they are' : ''}). Choose Free text to skip the check.`;
  }

  function onTaskChange() {
    const id = ui.taskSelect.value || null;
    const prev = S.taskId ? S.tasksById.get(S.taskId) : null;
    const how = S.taskMatch && S.taskMatch.method;
    clearFormError();
    if (id && S.tasksById.has(id)) {
      const t = S.tasksById.get(id);
      S.voice = null;  // the box now holds the picked task, not what was heard
      S.declined = null;
      setTask(t, { method: 'picked', score: null, matched_from: t.request });
      ui.request.value = t.request;
      ui.requestNote.textContent = '';
      return;
    }
    if (prev && how === 'picked' && ui.request.value === prev.request) {
      ui.request.value = '';  // that text came from the picker
      S.voice = null;
      ui.requestNote.textContent = '';
    } else if (prev && oneLine(ui.request.value)) {  // the user's own words were matched: keep them, unmatched
      S.declined = S.taskId;
      ui.requestNote.textContent = 'Runs as free text: no sandbox check.';
    } else {
      ui.requestNote.textContent = '';
    }
    clearTask();
  }

  function onRequestInput() {
    clearFormError();
    if (S.voice && ui.request.value !== S.voice.request) S.voice = null;  // edited: no longer the transcript
    syncMatch(false);
  }

  /** Keep the task picker in step with the request box. A task stays selected while the box still says it (a picked
   *  task: the same words; a matched one: still that match). select=false (typing): a new match is only announced
   *  in the note; the picker follows when the box loses focus, on Run, and for voice input. */
  function syncMatch(select) {
    const text = oneLine(ui.request.value);
    if (!text) S.declined = null;  // an empty box starts over
    const cur = S.taskId ? S.tasksById.get(S.taskId) : null;
    const how = S.taskMatch && S.taskMatch.method;
    if (cur && how === 'picked' && requestKey(text) === requestKey(cur.request)) return;  // same words: still that task
    let m = text ? matchTaskDetailed(text) : null;
    if (m && S.declined === String(m.task.id)) m = null;  // the user chose Free text over this task
    if (m && (select || (cur && String(m.task.id) === S.taskId))) {
      const changed = S.taskId !== String(m.task.id) || how !== m.method;
      setTask(m.task, { method: m.method, score: m.score, matched_from: text });
      if (changed) ui.requestNote.textContent = matchNote(m.task.id, m.method);
      return;
    }
    let note = ui.requestNote.textContent;
    if (cur) {
      clearTask();
      note = how === 'picked' ? 'You edited the task text, so this runs as free text (no sandbox check).'
        : 'No longer matches that sandbox task, so this runs as free text (no sandbox check).';
    }
    if (m) note = `${MATCH_HINT}${m.task.id}: running it checks the portal’s state afterwards.`;  // picked on blur/Run
    else if (!text || note.startsWith(MATCH_HINT)) note = '';
    ui.requestNote.textContent = note;
  }

  /** Put text in the request box (voice input). Text that matches a sandbox task (matchTaskDetailed) selects it in
   *  the picker, so its check runs and the user can see (and undo) that before the run. An exact match shows the
   *  task's own text (the same words); a fuzzy one keeps what was said. `voice` (the /api/transcribe reply) is kept
   *  with the text it produced, for the next run's POST body. */
  function setRequestText(text, voice) {
    const clean = oneLine(text).slice(0, 500);
    const m = matchTaskDetailed(clean);
    clearFormError();
    S.declined = null;
    if (m) {
      setTask(m.task, { method: m.method, score: m.score, matched_from: clean });
      ui.request.value = m.method === 'exact' ? m.task.request : clean;
      ui.requestNote.textContent = matchNote(m.task.id, m.method);
    } else {
      clearTask();
      ui.request.value = clean;
      ui.requestNote.textContent = '';
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

  function showFormError(message, activeRunId, where = ui.formError) {
    where.replaceChildren(document.createTextNode(message));
    if (activeRunId) {
      const btn = h('button', { type: 'button', class: 'btn btn-small', text: 'Watch that run' });
      btn.addEventListener('click', () => { where.replaceChildren(); watchRun(activeRunId, {}); });
      where.append(btn);
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
    // typed text that matches a sandbox task runs as that task (its start page, state and verdict), as voice does
    syncMatch(true);
    const task = S.taskId ? S.tasksById.get(S.taskId) : null;
    if (task && S.taskMatch && S.taskMatch.method === 'exact' && request !== task.request) {
      ui.request.value = task.request;  // the same words, as the task writes them
      request = task.request;
    }
    const options = readOptions();
    const taskId = S.taskId;
    const taskMatch = taskId && S.taskMatch ? { ...S.taskMatch } : null;
    const declined = taskId ? null : S.declined;  // the user chose Free text over this task: no late match to it
    // An exact match isn't sent: the backend records one itself when the request is the task's text.
    const sentMatch = taskMatch && taskMatch.method !== 'exact'
      ? { method: taskMatch.method, score: isNum(taskMatch.score) ? taskMatch.score : null,
        matched_from: str(taskMatch.method === 'picked' ? request : taskMatch.matched_from).slice(0, 500) }
      : null;
    if (document.activeElement === ui.runBtn) ui.request.focus();  // Run is disabled until the run ends
    const data = await postRun({
      request, task_id: taskId, options, ...(voice ? { voice } : {}), ...(sentMatch ? { task_match: sentMatch } : {}),
    }, showFormError);
    if (!data) return;
    if (voice && S.voice && S.voice.voice === voice) S.voice = null;  // one transcript, one run
    attach(String(data.run_id), {
      request, taskId, taskMatch, options, startUrl: data.start_url, focusOnEnd: true, declined,
    });
  }

  /** POST /api/run: the reply ({run_id, sid, start_url, request?}), or null once `showError(message, activeRunId?)`
   *  (the form's error line, or the answer box's) has said why it didn't start. */
  async function postRun(body, showError) {
    S.starting = true;
    ui.runBtn.disabled = true;
    ui.answerBtn.disabled = true;
    let res;
    let data = {};
    try {
      res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(body),
      });
      data = await res.json().catch(() => ({}));
    } catch {
      showError('Can’t reach the console backend. Is it still running?');
      return null;
    } finally {
      S.starting = false;
      ui.runBtn.disabled = false;
      ui.answerBtn.disabled = false;
    }
    if (!isObj(data)) data = {};
    if (res.ok && data.run_id) return data;
    const msg = oneLine(data.error) || `HTTP ${res.status}`;
    if (res.status === 409) {
      showError(`Another run is already in progress${data.active_run_id ? ` (${data.active_run_id})` : ''}. `
        + 'Wait for it to finish or stop it.', data.active_run_id);
    } else if (res.status === 503) {
      showError(`Can’t start: ${msg}. Check that the sandbox (and for the real agent, the models) are running.`);
    } else {
      showError(`Can’t start: ${msg}`);
      if (/not served/.test(msg)) refreshConfig();  // the VLM changed since the page last looked: update the choices
    }
    return null;
  }

  // --- answering the agent's question (done reason "needs_input") -----------------------------
  /** agent/clarify.py follow_up_request: the previous request plus " (clarified: <answer>)". */
  function followUpRequest(previous, answer) {
    const a = oneLine(answer);
    const p = str(previous).trim();
    return a ? `${p}${CLARIFIED}${a})` : p;
  }

  /** "<previous> (clarified: <answer>)" -> {previous, answer}, or null (a follow-up shown again, e.g. after a reload). */
  function splitFollowUp(text) {
    const t = str(text).trim();
    const i = t.lastIndexOf(CLARIFIED);
    if (i <= 0 || !t.endsWith(')')) return null;
    return { previous: t.slice(0, i).trim(), answer: t.slice(i + CLARIFIED.length, -1).trim() };
  }

  /** The question a needs_input run asks: the done event's, else the result's, else the clarify event's. */
  function questionOf(run) {
    const done = isObj(run.done) ? run.done : {};
    const result = isObj(run.result) ? run.result : {};
    const c = isObj(run.clarify) ? run.clarify : {};
    return oneLine(done.question || result.question || c.question || done.summary || result.summary)
      || 'What would you like me to do?';
  }

  /** The request the question was about (the backend builds the follow-up from the done event's `request`). */
  function previousRequest(run) {
    const done = isObj(run.done) ? run.done : {};
    const start = S.run === run && S.steps.get(0) && isObj(S.steps.get(0).start) ? S.steps.get(0).start : {};
    return str(done.request || start.request || run.request).trim();
  }

  /** The question the shown run is waiting to have answered, or null: the run ended with done reason needs_input, the
   *  agent's run id is known (the follow-up names it as its parent) and it wasn't answered yet. */
  function pendingQuestion() {
    const run = S.run;
    if (!run || !run.finished || run.answered || S.starting || finalStatus(run) !== 'needs_input') return null;
    const parentRunId = run.agentRunId || (isObj(run.result) ? str(run.result.run_id) : '');
    if (!RUN_ID_RE.test(parentRunId)) return null;
    return { question: questionOf(run), request: previousRequest(run), parentRunId, run };
  }

  /** Put text in the answer box (voice input). `voice` (the /api/transcribe reply) goes with the follow-up's POST
   *  while the box still holds exactly this text. */
  function setAnswerText(text, voice) {
    const clean = oneLine(text).slice(0, ANSWER_MAX);
    clearAnswerError();
    ui.answerInput.value = clean;
    const v = cleanVoice(voice);
    S.answerVoice = v ? { answer: clean, voice: v } : null;
    return { answer: clean };
  }

  function showAnswerError(message, activeRunId) {
    showFormError(message, activeRunId, ui.answerError);
  }

  function clearAnswerError() {
    if (ui.answerError.firstChild) ui.answerError.replaceChildren();
  }

  /** Answer: start the follow-up run. Its request is the asking run's request plus the answer; the backend builds
   *  the same text from the parent run (clarifies) and refuses another one. The text is matched to a sandbox task
   *  like any request (its task_id and task_match go with it). */
  async function answerQuestion() {
    const q = pendingQuestion();
    if (!q || S.starting || isRunning()) return;
    clearAnswerError();
    const answer = oneLine(ui.answerInput.value);
    if (!answer || answer.length > ANSWER_MAX) {
      showAnswerError(answer ? `That answer is too long (${ANSWER_MAX} characters at most).` : 'Type your answer first.');
      ui.answerInput.focus();
      return;
    }
    const request = followUpRequest(q.request, answer);
    const m = matchTaskDetailed(request);
    const taskId = m ? String(m.task.id) : null;
    const taskMatch = m ? { method: m.method, score: isNum(m.score) ? m.score : null, matched_from: request } : null;
    const sentMatch = taskMatch && taskMatch.method !== 'exact'
      ? { method: taskMatch.method, score: taskMatch.score, matched_from: request.slice(0, 500) } : null;
    const voice = S.answerVoice && S.answerVoice.answer === answer ? S.answerVoice.voice : null;
    const options = readOptions();
    if (document.activeElement === ui.answerBtn) ui.answerInput.focus();
    const data = await postRun({
      // the page's text too (when it fits a request): the backend answers 400 if its own differs
      ...(request.length <= 500 ? { request } : {}), task_id: taskId, options,
      clarifies: { parent_run_id: q.parentRunId, answer },
      ...(voice ? { voice } : {}), ...(sentMatch ? { task_match: sentMatch } : {}),
    }, showAnswerError);
    if (!data) return;
    q.run.answered = true;
    S.answerVoice = null;
    const final = str(data.request).trim() || request;  // the backend's text: what the agent gets
    // the request box and the picker describe the follow-up, as they do for any run shown
    S.voice = null;
    S.declined = null;
    ui.request.value = final;
    if (taskId && S.tasksById.has(taskId)) setTask(S.tasksById.get(taskId), taskMatch);
    else clearTask();
    ui.requestNote.textContent = '';
    attach(String(data.run_id), {
      request: final, taskId, taskMatch, options, startUrl: data.start_url, focusOnEnd: true,
      followUp: { parentRunId: q.parentRunId, previous: q.request, answer, question: q.question },
    });
  }

  /** GET /api/runs/{id} task_match ({method, score, matched_from}) as the page uses it, or null. */
  function cleanTaskMatch(v) {
    if (!isObj(v) || !TASK_MATCH_METHODS.includes(v.method)) return null;
    return { method: v.method, score: isNum(v.score) ? v.score : null, matched_from: str(v.matched_from) };
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
    const taskMatch = taskId ? cleanTaskMatch(info.task_match) : null;
    S.declined = null;
    // a task matched from the agent's restatement isn't the request box's: the next run of that text is free text
    if (taskId && !(taskMatch && taskMatch.method === 'restated')) {
      setTask(S.tasksById.get(taskId), taskMatch || { method: 'picked', score: null, matched_from: str(info.request) });
    } else {
      clearTask();
    }
    ui.requestNote.textContent = '';
    if (isObj(info.options)) for (const name of OPTION_NAMES) setRadio(name, info.options[name]);
    keepServedChoice();  // that run's grounder may not be served any more: the radios describe the next run
    renderOptionsSummary();
    renderEndpoints();
    attach(String(info.run_id || runId), {
      request: str(info.request), taskId: info.task_id || null, taskMatch, options: info.options,
      focusOnEnd: active || !quietIfDone, quiet: !active, agentRunId: info.agent_run_id || null,
      followUp: followUpInfo(info),
    });
  }

  /** A follow-up run's {parentRunId, previous, answer} from GET /api/runs/{id} (clarifies, or parent_run_id plus
   *  the request's "(clarified: …)"), or null. The start event's parent_run_id fills it in later otherwise. */
  function followUpInfo(info) {
    const c = isObj(info.clarifies) ? info.clarifies : {};
    const parentRunId = str(c.parent_run_id || info.parent_run_id);
    if (!parentRunId) return null;
    const split = splitFollowUp(info.request) || { previous: '', answer: '' };
    return { parentRunId, previous: split.previous, answer: oneLine(c.answer || info.answer) || split.answer };
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
      taskMatch: info.taskId ? cleanTaskMatch(info.taskMatch) : null,  // how the task was chosen (the banner says)
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
      followUp: isObj(info.followUp) ? info.followUp : null,  // {parentRunId, previous, answer}: it answers a question
      clarify: null,     // the clarify event's payload (step 0)
      lateMatch: null,   // {status: pending|ok|refused, taskId, match, reason}: the task its restatement matched
      answered: false,   // needs_input: the follow-up was started
      declined: info.declined ? str(info.declined) : null,  // a task the user chose Free text over (see lateMatch)
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
    renderViewer();  // "Waiting for the first screenshot…" (resetRunView rendered it for the previous run)
    if (!info.quiet) {
      const f = S.run.followUp;
      announce(f && f.answer ? `Follow-up run started with your answer: ${f.answer}` : `Run started: ${S.run.request}`);
    }
    S.timer = window.setInterval(tick, 1000);
    openStream(runId);
    emit('acme:run-start', {
      runId, request: S.run.request, taskId: S.run.taskId, answer: S.run.followUp ? S.run.followUp.answer : null,
    });
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
    on('verdict', (d) => {
      const run = S.run;
      run.verdict = d;
      // checked as the task the agent's restatement matched, before that POST's reply got here
      const late = run.lateMatch;
      if (!run.taskId && d.task_id && late && late.taskId === str(d.task_id)) {
        run.taskId = late.taskId;
        run.taskMatch = late.match;
        late.status = 'ok';
      }
    });
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
    // a question to answer: focus goes straight to the answer box (its description reads the question)
    if (run.focusOnEnd) (ui.bannerAnswer.hidden ? ui.banner : ui.answerInput).focus();
    const q = pendingQuestion();
    emit('acme:run-end', { runId: run.id, request: run.request, taskId: run.taskId, question: q ? q.question : null });
    if (!run.quiet) refreshConfig();  // e.g. a grounding error: is the VLM still up?
  }

  function finalStatus(run) {
    const reason = run.done && run.done.reason;
    if (reason === 'stopped' || reason === 'error' || reason === 'stuck' || reason === 'needs_input') return reason;
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
    ui.bannerAnswer.hidden = true;
    ui.answerInput.value = '';
    S.answerVoice = null;
    clearAnswerError();
    ui.understood.hidden = true;
    ui.understood.replaceChildren();
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
        if (!run.followUp && p.parent_run_id) {  // a follow-up shown again: its start event names the asking run
          const split = splitFollowUp(p.request || run.request) || { previous: '', answer: '' };
          run.followUp = { parentRunId: str(p.parent_run_id), ...split };
        }
        setPending(clarifyOn(run, p) ? 'Checking your request…' : 'Reading the screen…');
        break;
      case 'clarify':
        s.clarify = p;
        run.clarify = p;
        if (!(isObj(ev.timings_ms) && 'clarify' in ev.timings_ms) && isNum(p.latency_ms)) s.timings.clarify = p.latency_ms;
        setPending(p.action === 'ask' ? 'Finishing…' : 'Reading the screen…');
        renderUnderstood();
        if (p.action !== 'ask') lateMatch(run, p);
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
    if (!S.run || S.run.quiet) return;
    if (kind === 'clarify' && p.action !== 'ask' && oneLine(p.normalized_request)) {  // an ask is the banner's
      announce(`Request check: clear. What the agent understood: ${oneLine(p.normalized_request)}`);
      return;
    }
    if (s.n === 0) return;
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

  // --- the clarify step (agent/clarify.py) ------------------------------------------------------
  /** Whether this run checks the request first: its options say so, or the agent's start config does. */
  function clarifyOn(run, start) {
    const c = isObj(start) && isObj(start.config) ? start.config.clarify : undefined;
    if (c === true || c === 'on') return true;
    if (c === false || c === 'off') return false;
    return Boolean(run.options && run.options.clarify === 'on');
  }

  /** Step 0's lines for the clarify event: what the check decided, the question or the restatement. */
  function clarifyLines(c) {
    const out = [];
    const why = (t) => oneLine(t).replace(/[.!]+$/, '');
    // who checked it, with which prompt (the Logs page has the rest: usage, the tool call)
    const by = [oneLine(c.model), oneLine(c.prompt_version), c.prompt_sha ? `sha ${oneLine(c.prompt_sha)}` : '']
      .filter(Boolean).join(' · ');
    if (c.action === 'ask') {
      out.push(['clarify', '', 'Request check: unclear, so the agent asks before doing anything.', by]);
      if (oneLine(c.question)) out.push(['ask', 'ask', `The agent asks: “${oneLine(c.question)}”`]);
      return out;
    }
    if (c.error) {
      out.push(['clarify', 'warn', `Request check failed (${why(c.error)}), so the agent went ahead anyway.`, by]);
    } else if (c.skipped) {
      out.push(['clarify', '', `Request check skipped (${why(c.skipped)}), so the agent goes ahead.`, by]);
    } else {
      out.push(['clarify', '', 'Request check: clear, so the agent goes ahead.', by]);
    }
    if (oneLine(c.normalized_request)) {
      out.push(['understood', 'understood', `What the agent understood: “${oneLine(c.normalized_request)}”`]);
    }
    return out;
  }

  /** The step-0 line about the task the restatement matched: checked as it, or why it can't be. */
  function lateMatchLine(run) {
    const late = run.lateMatch;
    if (!late || late.status === 'pending') return null;
    if (late.status === 'ok') {
      return ['latematch', '', `Checked as task ${late.taskId}: it matches the agent’s restatement, not your words.`];
    }
    const why = oneLine(late.reason).replace(/[.!]+$/, '') || 'the console refused';
    return ['latematch', 'warn', `The agent’s restatement matches task ${late.taskId}, but this run can’t be checked `
      + `as it: ${why}.`];
  }

  /** The line above the step log: what the agent understood (display only), and the task that restatement matched. */
  function renderUnderstood() {
    const run = S.run;
    const c = run && isObj(run.clarify) ? run.clarify : null;
    const text = c && c.action !== 'ask' ? oneLine(c.normalized_request) : '';
    ui.understood.hidden = !text;
    if (!text) return;
    const parts = [h('span', { class: 'understood-label', text: 'What the agent understood: ' }),
      document.createTextNode(`“${text}”`)];
    const late = run.lateMatch;
    if (late && late.status === 'ok' && run.taskId) {
      parts.push(h('span', { class: 'understood-task', text: ` · checked as task ${run.taskId}` }));
    }
    ui.understood.replaceChildren(...parts);
    ui.understood.title = 'The agent’s restatement of your request, shown only: the agent works from your words';
  }

  /** Contract C: a run with no sandbox task whose clarify restatement matches one asks the backend to check it as
   *  that task (POST /api/runs/{id}/task_match). The backend refuses unless the run has no task or verdict yet and the
   *  task starts where the run did; the step log says why. Live runs only, once. */
  async function lateMatch(run, c) {
    if (run.quiet || run.finished || run.taskId || run.lateMatch || run.verdict) return;
    const text = oneLine(c.normalized_request).slice(0, 500);
    const m = text ? matchTaskDetailed(text) : null;
    if (!m || String(m.task.id) === run.declined) return;  // (the user said this isn't that task)
    const taskId = String(m.task.id);
    const sup = S.matcher.support(taskId, run.request);
    if (!sup.ok) {  // the restatement brought in the task's distinguishing intent itself: no self-fulfilling ✓
      run.lateMatch = { status: 'refused', taskId, match: null,
        reason: `your own words don’t mention ${sup.missing.join(', ')}; only the agent’s restatement does` };
      if (S.run !== run) return;
      const first = S.steps.get(0);
      if (first) renderStep(first);
      renderUnderstood();
      if (run.finished) renderBanner();
      return;
    }
    const match = { method: 'restated', score: isNum(m.score) ? m.score : null, matched_from: text };
    const late = { status: 'pending', taskId, match, reason: '' };
    run.lateMatch = late;
    let res;
    let data = {};
    try {
      res = await fetch(`/api/runs/${encodeURIComponent(run.id)}/task_match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ task_id: taskId, ...match }),
      });
      data = await res.json().catch(() => ({}));
    } catch {
      res = null;
    }
    if (!isObj(data)) data = {};
    if (res && res.ok) {
      late.status = 'ok';
      if (!run.taskId) {
        run.taskId = taskId;
        run.taskMatch = cleanTaskMatch(data.task_match) || match;
      }
    } else if (late.status === 'pending') {  // (the verdict may have shown it was accepted meanwhile)
      late.status = 'refused';
      late.reason = res ? oneLine(data.error) || `HTTP ${res.status}` : 'the console backend didn’t answer';
    }
    if (S.run !== run) return;
    const step0 = S.steps.get(0);
    if (step0) renderStep(step0);
    renderUnderstood();
    // one sentence with the restatement: announce() speaks only the last of a burst
    if (late.status === 'ok' && !run.quiet) {
      announce(`What the agent understood: ${text}. It will be checked as sandbox task ${taskId}.`);
    }
    if (run.finished) renderBanner();
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
      n, start: null, clarify: null, observation: null, plan: null, toolCall: null, grounding: null, action: null, done: null,
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
    const f = s.n === 0 && S.run ? S.run.followUp : null;
    if (f && f.previous) {  // a follow-up: what it follows, and the answer (the request box holds the combined text)
      lines.push(['followup', 'followup', `Follow-up to: “${f.previous}”`]);
      lines.push(['request', '', f.answer ? `Your answer: “${f.answer}”` : `You asked: “${S.run.request}”`]);
    } else if (s.n === 0 && S.run && S.run.request) {
      lines.push(['request', '', `You asked: “${S.run.request}”`]);
    }
    if (s.start) {
      const c = configText(s.start.config, S.run && S.run.options);
      if (c) lines.push(['config', '', c]);
    }
    if (s.clarify) lines.push(...clarifyLines(s.clarify));
    const late = s.n === 0 && S.run ? lateMatchLine(S.run) : null;
    if (late) lines.push(late);
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
    } else if (s.done && s.done.reason === 'needs_input') {
      // the question is the clarify line's (same key), unless that event is missing
      const ask = oneLine(s.done.question || s.done.summary);
      if (ask && !(s.clarify && oneLine(s.clarify.question))) lines.push(['ask', 'ask', `The agent asks: “${ask}”`]);
    } else if (s.done && s.done.reason && s.done.reason !== 'done') {
      lines.push(['done', s.done.reason === 'error' ? 'error' : 'warn',
        oneLine(s.done.summary) || REASON_WORDS[s.done.reason] || `Run ended (${oneLine(s.done.reason)}).`]);
    }
    s.unknown.forEach((u, i) => lines.push([`unknown${i}`, 'json', `${u.kind}: ${compactJson(u.payload)}`]));
    for (const [key, cls, text, title] of lines) {
      let node = s.lineEls.get(key);
      if (!node) {
        node = h('p');
        s.lineEls.set(key, node);
        slots.lines.append(node);
      }
      const className = `step-line${cls ? ` step-${cls}` : ''}`;
      if (node.className !== className) node.className = className;
      setText(node, text);
      if (title) node.title = title;
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
      const over = Boolean(S.run && S.run.finished);
      const asked = over && finalStatus(S.run) === 'needs_input';
      setText(ui.screenEmpty, asked ? 'No screenshots: the agent asked you first, before opening the app.'
        : over ? 'This run has no screenshots.'
          : S.run ? 'Waiting for the first screenshot…' : 'No screenshot yet. Run a request to watch the agent work.');
      ui.screenCaption.replaceChildren(document.createTextNode(over ? 'No screenshots for this run.'
        : S.run ? 'Waiting for the first screenshot.' : 'Waiting for a run.'));
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
    } else if (st === 'needs_input') {  // the clarify step asked before the browser opened: nothing was done yet
      tone = 'ask';
      icon = '?';
      title = 'The agent asks:';
      question = questionOf(run);
      agentSays = '';  // the summary is the question
      if (v) {
        label = oneLine(v.label) || 'verified by sandbox state, not seen by the agent';
        expl = v.passed ? `✓ ${check.head}` : `✗ Not done yet. ${check.head}`;
      } else if (run.taskId) {
        label = 'No sandbox verdict received';
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
    // needs_input: answer here, and the follow-up run starts (pendingQuestion: the asking run's id is known)
    const answerable = st === 'needs_input' && Boolean(pendingQuestion());
    ui.bannerAnswer.hidden = !answerable;
    const task = taskLine(run, Boolean(v));
    ui.bannerTask.hidden = !task;
    setText(ui.bannerTask, task);
    const how = run.taskMatch && isNum(run.taskMatch.score) ? run.taskMatch.method : '';
    ui.bannerTask.title = how === 'fuzzy'
      ? `Close match (score ${run.taskMatch.score.toFixed(2)}): the agent got your words, the check is this task's`
      : how === 'restated' ? `Matched (score ${run.taskMatch.score.toFixed(2)}) from how the agent restated your `
        + 'request; the agent got your words, the check is this task’s' : '';
    setText(ui.bannerLabel, label);
    renderCheck(expl, check.details, check.more);

    renderSummary(agentSays, st === 'stuck' || st === 'needs_input' ? '' : 'The agent gave no summary.');
    const claim = [];
    if (st === 'needs_input') claim.push('Nothing was done yet: it asked before opening the app.');
    else if (claimed === true) claim.push('The agent claimed success.');
    else if (claimed === false) claim.push('The agent did not claim success.');
    if (v && claimed !== null && v.passed !== claimed && st !== 'needs_input') {
      claim.push(claimed ? 'The sandbox state says otherwise.' : 'The sandbox state shows it is done anyway.');
    }
    if (st === 'stuck' || (st === 'needs_input' && !answerable)) {
      claim.push('To answer, add what it needs to your request and run it again.');
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
    if (stats.n_stuck_warnings) extra.push(plural('n_stuck_warnings', 'stuck warning', 'stuck warnings'));
    ui.bannerExtra.replaceChildren();
    if (extra.length) ui.bannerExtra.append(document.createTextNode(extra.join(' · ')));
    const links = [];
    const href = st === 'needs_input' ? null : safeHttpUrl(run.startUrl);  // needs_input: no browser was opened
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

  /** "Checked as task: pay-50 (matched from your request)" (or "(picked from the list)", or "(matched from the
   *  agent's restatement: "<normalized_request>")": never hide that the agent's words chose the check); "Sandbox task:
   *  …" when no verdict came; '' for free text. The logs page shows the same line (console/logs.py task_line). */
  function taskLine(run, checked) {
    if (!run.taskId) return '';
    const m = run.taskMatch;
    const how = m && m.method;
    const why = how === 'picked' ? ' (picked from the list)'
      : how === 'exact' || how === 'fuzzy' ? ' (matched from your request)'
        : how === 'restated' ? ` (matched from the agent's restatement: "${oneLine(m.matched_from)}")` : '';
    return `${checked ? 'Checked as task' : 'Sandbox task'}: ${run.taskId}${why}`;
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
