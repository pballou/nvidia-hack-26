/* Voice input for the GUI assist console: mic button, push-to-talk, local capture, POST /api/transcribe.
 *
 * No cloud speech. The Web Speech API is never used. The microphone is read as raw PCM with Web Audio
 * (pcm-worklet.js, or a ScriptProcessor where AudioWorklet is missing), encoded to a 16 kHz mono PCM16 WAV in the
 * page (wav.js), and sent only to this console's backend (console/asr.py runs a local engine).
 *
 * Flow: press Speak (or hold the push-to-talk key) -> "Listening" with a live level meter -> release / Stop
 * listening -> "Transcribing" (Esc or the mic button cancels it) -> the text goes into the request box (a sandbox
 * task's text selects that task) -> a 1.5 s countdown, then the run starts (Esc or Cancel stops the countdown).
 * Options sets what happens after voice input: run in 1.5 s, run in 5 s, or don't auto-run (localStorage).
 * Nothing runs by itself when the transcript is over 500 characters (the box gets the first part), when the user
 * edited the request while it was transcribing (their text is kept), or when the engine is the `--asr stub`
 * (fixed text, labelled as such). While a run is going, a stop phrase ("stop", "stop it", "cancel that",
 * "please stop", ...) stops it; anything else only fills the box. With nothing running, a stop phrase starts
 * nothing. While voice input is off because the recognizer isn't up, /api/config is checked again every 4 s.
 *
 * Push-to-talk: hold Right Option (event.code AltRight; configurable in Options, stored in localStorage), or hold
 * Space while focus is on the page itself (not a field, button, link, scrollable region or other control, where
 * Space keeps its usual meaning). A modifier key held in a text field opens the mic only after ~200 ms alone, so
 * Option+E (é) and other chords never touch the microphone. Every state change is announced in #voice-status
 * (role=status), which is also the visible line under the mic. Nothing here throws into the page: every failure
 * becomes a message.
 */
'use strict';

(() => {
  const PTT_STORE = 'gui-assist.ptt-key';
  const DEFAULT_PTT = 'AltRight';
  const AUTORUN_STORE = 'gui-assist.voice-autorun';
  const AUTORUN_CHOICES = [['1.5', 'run in 1.5 s'], ['5', 'run in 5 s'], ['off', 'don’t auto-run']];
  const DEFAULT_AUTORUN = '1.5';
  const TARGET_RATE = 16000;
  const MAX_SECONDS = 30;
  const MIN_SECONDS = 0.25;
  const REQUEST_MAX = 500;  // the request box's limit (POST /api/run)
  const TRANSCRIBE_TIMEOUT_MS = 60000;
  const CANCEL_GRACE_MS = 600;  // a mic click this soon after "Transcribing" is a double-click, not a cancel
  const ARM_MS = 200;  // a modifier push-to-talk key in a text field must be held alone this long
  const RECHECK_MS = 4000;  // voice input off because the recognizer isn't up: ask /api/config again
  const FLUSH_TIMEOUT_MS = 300;
  const DONE_MS = 2500;
  const SILENT_RMS = 0.002;  // about -54 dBFS: a muted or dead microphone
  // A whole transcript that means "stop", after lowercasing and turning punctuation into spaces. Anchored at both
  // ends with a closed list of words, so a request such as "Stop sending me promotions." is not a stop.
  const STOP_RE = /^(?:(?:please|ok|okay|hey) )?(?:stop|cancel)(?: (?:it|that|this|now|the run|everything|please|stop|cancel))*$/;
  const MIC_CONSTRAINTS = { audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } };

  const platform = String((navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '');
  const IS_MAC = /mac|iphone|ipad/i.test(platform);
  const IS_WIN = /win/i.test(platform);

  const MODIFIER_RE = /^(Alt|Control|Meta|Shift|OS)(Left|Right)$/;
  const FUNCTION_RE = /^F([1-9]|1\d|2[0-4])$/;
  const INTERACTIVE_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'SUMMARY', 'OPTION', 'AUDIO', 'VIDEO',
    'IFRAME', 'OBJECT', 'EMBED']);
  const INTERACTIVE_ROLES = new Set(['button', 'link', 'checkbox', 'radio', 'switch', 'tab', 'menuitem',
    'menuitemcheckbox', 'menuitemradio', 'option', 'slider', 'spinbutton', 'textbox', 'searchbox', 'combobox',
    'listbox', 'gridcell', 'treeitem', 'scrollbar']);

  const V = {
    asr: null,           // /api/config "asr": {available, model, label, error?}
    reasons: new Map(),  // why voice input is off: source -> message (empty = available)
    wasOff: false,       // it was off after loading, so coming back gets announced
    state: 'off',        // off | idle | starting | listening | transcribing | done | error
    rec: null,           // the recording in progress (see startRecording)
    held: null,          // event.code of the key holding push-to-talk right now
    armed: 0,            // timer: a modifier held in a text field, not recording yet (see onKeyDown)
    pttCode: DEFAULT_PTT,
    capturing: false,    // Options: waiting for the new push-to-talk key
    autorun: DEFAULT_AUTORUN,  // Options: what happens after voice input ('1.5' | '5' | 'off')
    countdown: null,     // {endsAt, ms, timer}
    tx: null,            // the transcription in flight: {ctrl, since, request, task, edited}
    recheck: 0,          // timer: the next /api/config check while the recognizer is down
    heard: null,         // {request, used}: the last transcript put in the box
    doneTimer: null,
    rafId: 0,
    level: 0,
    sayId: 0,            // bumped by every status write, so a delayed re-announce never overwrites newer text
  };
  const el = {};

  // --- small helpers -----------------------------------------------------------------
  const str = (v) => (v === null || v === undefined ? '' : String(v));
  const oneLine = (s) => str(s).replace(/\s+/g, ' ').trim();

  const prefs = {
    get(k) { try { return window.localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch { /* private mode */ } },
    del(k) { try { window.localStorage.removeItem(k); } catch { /* private mode */ } },
  };

  function setText(node, text) {
    if (node && node.textContent !== text) node.textContent = text;
  }

  function fmtClock(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }

  function asrLabel() {
    return oneLine(V.asr && (V.asr.label || V.asr.model)) || 'local speech recognizer';
  }

  /** `--asr stub`: the backend returns fixed text, not what was said (/api/config asr.fixed_text). */
  const isFixedText = () => Boolean(V.asr && V.asr.fixed_text === true);

  /** True if the whole transcript asks to stop ("Stop.", "stop it", "Cancel that.", "please stop", "stop stop"). */
  function isStopIntent(text) {
    return STOP_RE.test(str(text).toLowerCase().replace(/[\p{P}\p{S}]+/gu, ' ').replace(/\s+/g, ' ').trim());
  }

  /** `s` cut to at most `max` characters, at a word boundary when there is one in the second half. */
  function cutToWord(s, max) {
    if (s.length <= max) return s;
    const head = s.slice(0, max);
    if (s[max] === ' ') return head.trimEnd();
    const sp = head.lastIndexOf(' ');
    return (sp > max / 2 ? head.slice(0, sp) : head).trimEnd();
  }

  /** Show `text` under the mic and announce it (role=status). tone: '' | 'ok' | 'warn' | 'error'. srExtra is read
   *  to screen readers only (e.g. the countdown, which is visible in its own row). */
  function say(text, { tone = '', srExtra = '' } = {}) {
    const node = el.status;
    if (!node) return;
    V.sayId += 1;
    const id = V.sayId;
    node.dataset.tone = tone;
    const write = () => {
      if (V.sayId !== id) return;  // something newer was said (or the line was cleared) meanwhile
      node.replaceChildren(document.createTextNode(text));
      if (srExtra) {
        const extra = document.createElement('span');
        extra.className = 'visually-hidden';
        extra.textContent = ` ${srExtra}`;
        node.append(extra);
      }
    };
    if (node.textContent === text + (srExtra ? ` ${srExtra}` : '')) {  // same words again: re-announce
      node.replaceChildren();
      window.setTimeout(write, 60);
    } else {
      write();
    }
  }

  // --- key names and push-to-talk rules -------------------------------------------------
  function keyName(code) {
    let m = MODIFIER_RE.exec(code);
    if (m) {
      const base = {
        Alt: IS_MAC ? 'Option' : 'Alt',
        Control: 'Control',
        Shift: 'Shift',
        Meta: IS_MAC ? 'Command' : (IS_WIN ? 'Windows' : 'Meta'),
        OS: IS_WIN ? 'Windows' : 'Super',
      }[m[1]];
      return `${m[2]} ${base}`;
    }
    if ((m = /^Key([A-Z])$/.exec(code))) return m[1];
    if ((m = /^Digit(\d)$/.exec(code))) return m[1];
    if ((m = /^Numpad(.+)$/.exec(code))) return `Numpad ${m[1]}`;
    if ((m = /^Arrow(Up|Down|Left|Right)$/.exec(code))) return `${m[1]} arrow`;
    const named = {
      Space: 'Space', Backquote: '`', Minus: '-', Equal: '=', BracketLeft: '[', BracketRight: ']', Backslash: '\\',
      Semicolon: ';', Quote: "'", Comma: ',', Period: '.', Slash: '/', IntlBackslash: '§', ContextMenu: 'Menu',
      PageUp: 'Page Up', PageDown: 'Page Down',
    };
    return named[code] || code;
  }

  /** Modifier and function keys don't type, so they work as push-to-talk anywhere, even in the request box. */
  const worksEverywhere = (code) => MODIFIER_RE.test(code) || FUNCTION_RE.test(code);

  /** A control where Space (or a typing key) already means something: fields, buttons, links, custom widgets,
   *  scrollable regions. */
  function isInteractive(node) {
    if (!node || node === document.body || node === document.documentElement) return false;
    if (node.isContentEditable) return true;
    if (INTERACTIVE_TAGS.has(node.tagName)) return true;
    if (node.tagName === 'A' && node.hasAttribute('href')) return true;
    if (INTERACTIVE_ROLES.has(str(node.getAttribute('role')).toLowerCase())) return true;
    if (node.hasAttribute('tabindex') && node.tabIndex >= 0) return true;  // e.g. the scrollable step log
    return isScroller(node);
  }

  /** A box that scrolls and has more to show. Chromium lets Tab focus such a box (the "Screen text" of a step,
   *  for example) even without a tabindex, and then Space scrolls it. */
  function isScroller(node) {
    if (!(node instanceof Element)) return false;
    const cs = window.getComputedStyle(node);
    const scrolls = (v) => v === 'auto' || v === 'scroll';
    return (scrolls(cs.overflowY) && node.scrollHeight > node.clientHeight)
      || (scrolls(cs.overflowX) && node.scrollWidth > node.clientWidth);
  }

  const NON_TEXT_INPUTS = new Set(['button', 'checkbox', 'color', 'file', 'hidden', 'image', 'radio', 'range',
    'reset', 'submit']);

  /** Where a modifier types characters (macOS Option+E is é, AltGr on Windows): text inputs and the request box. */
  function isTextField(node) {
    if (!node) return false;
    if (node.isContentEditable || node.tagName === 'TEXTAREA') return true;
    return node.tagName === 'INPUT' && !NON_TEXT_INPUTS.has(str(node.type).toLowerCase());
  }

  function pttKeyHere(code) {
    return worksEverywhere(code) || !isInteractive(document.activeElement);
  }

  function rejectKey(code) {
    if (!code || code === 'Unidentified') {
      return 'The browser can’t tell which key that was (the Fn key, for example). Press another key.';
    }
    if (code === 'Enter' || code === 'NumpadEnter') return 'Enter runs the request. Press another key.';
    if (code === 'Space') return 'Space already works when focus isn’t on a control. Press another key.';
    if (code === 'CapsLock') return 'Caps Lock can’t be held down reliably. Press another key.';
    return '';
  }

  // --- availability ---------------------------------------------------------------------
  function setReason(source, message) {
    if (message) V.reasons.set(source, message);
    else V.reasons.delete(source);
    renderAvailability();
  }

  const isAvailable = () => V.reasons.size === 0 && V.asr !== null;

  function renderAvailability() {
    if (!el.mic) return;
    const on = isAvailable();
    if (!on && V.rec) stopRecording('cancel', { quiet: true });
    const checking = V.asr === null && !V.reasons.size;
    if (!on) {
      cancelCountdown('');
      if (V.state !== 'transcribing') setState('off');
      if (!checking) V.wasOff = true;
      const why = [...V.reasons.values()][0];
      el.hint.dataset.tone = checking ? '' : 'off';
      el.hint.replaceChildren(document.createTextNode(checking ? 'Checking voice input…' : `Voice input is off: ${why}`));
    } else {
      if (V.state === 'off') setState('idle');
      if (V.wasOff) {  // e.g. the microphone permission was granted in site settings, or the recognizer came up
        V.wasOff = false;
        say('Voice input is available again.', { tone: 'ok' });
      }
      const k = keyName(V.pttCode);
      if (isFixedText()) {
        el.hint.dataset.tone = 'off';
        el.hint.replaceChildren(document.createTextNode('Voice input is a stub: it fills in fixed text, whatever you '
          + 'say, and never runs it by itself. Hold '), kbd(k), document.createTextNode(' to try it.'));
      } else {
        el.hint.dataset.tone = '';
        el.hint.replaceChildren(document.createTextNode('Or hold '), kbd(k),
          document.createTextNode(' to talk, release to send.'));
      }
    }
    if (el.kbdWrap) el.kbdWrap.hidden = !on;
    setText(el.kbdPtt, keyName(V.pttCode));
    if (el.asrNote) {
      el.asrNote.hidden = !on;
      setText(el.asrNote, isFixedText()
        ? `Voice input is the stub (${asrLabel()}): it recognizes no speech. For real voice input, start `
          + 'scripts/run_asr.sh and the console with --asr parakeet.'
        : `Speech is recognized on this computer (${asrLabel()}); audio goes only to this console.`);
    }
  }

  function kbd(text) {
    const k = document.createElement('kbd');
    k.textContent = text;
    return k;
  }

  async function watchMicPermission() {
    if (!navigator.permissions || !navigator.permissions.query) return;
    try {
      const status = await navigator.permissions.query({ name: 'microphone' });
      const apply = () => setReason('permission', status.state === 'denied'
        ? 'microphone access is blocked for this page. Allow the microphone in the browser’s site settings.' : '');
      apply();
      status.onchange = apply;
    } catch { /* this browser can't query the microphone permission */ }
  }

  async function micPermissionState() {
    try {
      return (await navigator.permissions.query({ name: 'microphone' })).state;
    } catch {
      return 'unknown';
    }
  }

  // --- UI state ---------------------------------------------------------------------------
  function setState(next) {
    // While voice input is off, every resting state is 'off' (a transcript can still arrive and fill the box).
    const state = next !== 'transcribing' && !isAvailable() ? 'off' : next;
    V.state = state;
    const mic = el.mic;
    if (!mic) return;
    window.clearTimeout(V.doneTimer);
    mic.dataset.state = state;
    const listening = state === 'starting' || state === 'listening';
    // A plain action button: its name says what pressing it does ("Speak", "Stop listening"), so no aria-pressed.
    mic.removeAttribute('aria-pressed');
    mic.disabled = state === 'off';
    if (state === 'transcribing') mic.setAttribute('aria-label', 'Transcribing… press to cancel');
    else mic.removeAttribute('aria-label');
    setText(el.micLabel, listening ? 'Stop listening' : (state === 'transcribing' ? 'Transcribing…' : 'Speak'));
    el.timer.hidden = !listening;
    if (!listening) mic.style.setProperty('--level', '0');
    if (state === 'done') {
      V.doneTimer = window.setTimeout(() => { if (V.state === 'done') setState('idle'); }, DONE_MS);
    }
  }

  function startMeter(rec) {
    const frame = () => {
      if (V.rec !== rec) return;
      const db = 20 * Math.log10(rec.lastRms + 1e-9);
      const target = Math.min(1, Math.max(0, (db + 60) / 48));  // -60 dBFS .. -12 dBFS
      V.level = target > V.level ? target : V.level * 0.85 + target * 0.15;
      el.mic.style.setProperty('--level', V.level.toFixed(3));
      const secs = rec.frames / rec.rate;
      setText(el.timer, `${fmtClock(secs)} / ${fmtClock(MAX_SECONDS)}`);
      V.rafId = window.requestAnimationFrame(frame);
    };
    V.level = 0;
    window.cancelAnimationFrame(V.rafId);
    V.rafId = window.requestAnimationFrame(frame);
  }

  // --- capture ----------------------------------------------------------------------------
  async function startRecording(mode) {
    if (V.rec || !isAvailable() || V.state === 'transcribing') return;
    cancelCountdown('');  // a new utterance replaces a pending auto-run
    const rec = {
      mode, live: false, stopping: false, stopWanted: null, chunks: [], frames: 0, rate: TARGET_RATE,
      lastRms: 0, peakRms: 0, stream: null, ctx: null, src: null, node: null, kind: '', flushed: null, maxTimer: 0,
    };
    V.rec = rec;
    setState('starting');
    say('Starting the microphone…');
    try {
      rec.stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
      if (V.rec !== rec) { teardown(rec); return; }
      await openGraph(rec);
      if (V.rec !== rec) { teardown(rec); return; }
    } catch (err) {
      teardown(rec);
      if (V.rec !== rec) return;  // cancelled meanwhile; its message is already shown
      V.rec = null;
      V.held = null;
      await micFailed(err);
      return;
    }
    rec.live = true;
    rec.maxTimer = window.setTimeout(() => {
      if (V.rec === rec) stopRecording('send', { limit: true });
    }, MAX_SECONDS * 1000);
    setState('listening');
    say(mode === 'ptt' ? 'Listening… release to send' : 'Listening… press again to send', { tone: 'live' });
    startMeter(rec);
    if (rec.stopWanted) stopRecording(rec.stopWanted);
  }

  async function openGraph(rec) {
    const AC = window.AudioContext || window.webkitAudioContext;
    let ctx = null;
    let src = null;
    try {
      ctx = new AC({ sampleRate: TARGET_RATE });
      rec.ctx = ctx;
      src = ctx.createMediaStreamSource(rec.stream);
    } catch {
      // Firefox can't connect a microphone to a context at another rate: record at the default and resample.
      if (ctx) ctx.close().catch(() => {});
      ctx = new AC();
      rec.ctx = ctx;  // so teardown closes it even if the next line throws
      src = ctx.createMediaStreamSource(rec.stream);
    }
    rec.src = src;
    rec.rate = ctx.sampleRate;
    if (ctx.state !== 'running') {
      await Promise.race([ctx.resume().catch(() => {}), new Promise((r) => { window.setTimeout(r, 1000); })]);
      if (ctx.state !== 'running') {
        const e = new Error('the browser kept audio paused. Click anywhere on the page once, then try again.');
        e.name = 'AudioPaused';
        throw e;
      }
    }
    const onSamples = (f32) => {
      if (V.rec !== rec || !f32 || !f32.length) return;
      rec.chunks.push(f32);
      rec.frames += f32.length;
      rec.lastRms = window.AcmeWav.rms(f32);
      if (rec.lastRms > rec.peakRms) rec.peakRms = rec.lastRms;
    };
    if (ctx.audioWorklet && window.AudioWorkletNode) {
      try {
        await ctx.audioWorklet.addModule('/static/pcm-worklet.js');
        const node = new window.AudioWorkletNode(ctx, 'pcm-capture', {
          numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
          channelCount: 1, channelCountMode: 'explicit', channelInterpretation: 'speakers',
          processorOptions: { chunkFrames: Math.max(128, Math.round(ctx.sampleRate * 0.064)) },
        });
        node.port.onmessage = (e) => {
          const d = e.data || {};
          if (d.type === 'chunk') onSamples(d.samples);
          else if (d.type === 'flushed' && rec.flushed) rec.flushed();
        };
        src.connect(node);
        node.connect(ctx.destination);  // keeps the node pulled; it outputs silence
        rec.node = node;
        rec.kind = 'worklet';
        return;
      } catch { /* fall back to ScriptProcessor */ }
    }
    const size = ctx.sampleRate <= 24000 ? 1024 : 2048;
    const sp = ctx.createScriptProcessor(size, 1, 1);
    sp.onaudioprocess = (e) => onSamples(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(sp);
    sp.connect(ctx.destination);
    rec.node = sp;
    rec.kind = 'script';
  }

  function flushGraph(rec) {
    if (rec.kind !== 'worklet' || !rec.node) return Promise.resolve();
    return new Promise((resolve) => {
      const t = window.setTimeout(resolve, FLUSH_TIMEOUT_MS);
      rec.flushed = () => { window.clearTimeout(t); resolve(); };
      try { rec.node.port.postMessage({ type: 'flush' }); } catch { resolve(); }
    });
  }

  /** Mic off: stop every track and close the context (the browser's recording indicator goes away). */
  function teardown(rec) {
    window.clearTimeout(rec.maxTimer);
    try { if (rec.src) rec.src.disconnect(); } catch { /* already gone */ }
    try {
      if (rec.node) {
        if (rec.kind === 'worklet') { rec.node.port.postMessage({ type: 'stop' }); rec.node.port.onmessage = null; }
        else rec.node.onaudioprocess = null;
        rec.node.disconnect();
      }
    } catch { /* already gone */ }
    try { if (rec.stream) rec.stream.getTracks().forEach((t) => t.stop()); } catch { /* already stopped */ }
    try { if (rec.ctx && rec.ctx.state !== 'closed') rec.ctx.close().catch(() => {}); } catch { /* closed */ }
    rec.src = null;
    rec.node = null;
    rec.stream = null;
    rec.ctx = null;
  }

  /** how: 'send' | 'cancel'. */
  async function stopRecording(how, { limit = false, quiet = false, why = '' } = {}) {
    const rec = V.rec;
    if (!rec || rec.stopping) return;
    V.held = null;
    if (!rec.live) {
      if (how === 'send') {  // released before the mic was ready: finish once it is (it will be too short)
        rec.stopWanted = 'send';
        return;
      }
      V.rec = null;  // startRecording sees this and tears down
      setState(isAvailable() ? 'idle' : 'off');
      if (!quiet) say(why || 'Cancelled. Nothing was sent.', { tone: 'warn' });
      return;
    }
    rec.stopping = true;
    window.cancelAnimationFrame(V.rafId);
    if (how === 'send') await flushGraph(rec);
    teardown(rec);
    V.rec = null;
    if (how === 'cancel') {
      setState(isAvailable() ? 'idle' : 'off');
      if (!quiet) say(why || 'Cancelled. Nothing was sent.', { tone: 'warn' });
      return;
    }
    const seconds = rec.frames / rec.rate;
    if (seconds < MIN_SECONDS) {
      setState('idle');
      say(rec.mode === 'ptt' ? 'Too short. Hold the key while you speak.' : 'Too short. Nothing was sent.', { tone: 'warn' });
      return;
    }
    let wav;
    try {
      const samples = new Float32Array(rec.frames);
      let off = 0;
      for (const c of rec.chunks) { samples.set(c, off); off += c.length; }
      const pcm = rec.rate === TARGET_RATE ? samples : window.AcmeWav.resample(samples, rec.rate, TARGET_RATE);
      wav = window.AcmeWav.encodeWav(pcm, TARGET_RATE);
    } catch (err) {
      showError(`Couldn’t encode the recording: ${oneLine(err && err.message) || 'unknown error'}`);
      return;
    }
    await transcribe(wav, { seconds, peakRms: rec.peakRms, limit });
  }

  async function micFailed(err) {
    const name = str(err && err.name) || 'Error';
    const msg = oneLine(err && err.message);
    const detail = msg ? `${name}: ${msg}` : name;
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      if ((await micPermissionState()) !== 'denied') {  // dismissed, or we can't tell: allow a retry
        showError(`The microphone wasn’t allowed (${detail}). Press Speak to be asked again, or allow the `
          + 'microphone in the browser’s site settings.');
        return;
      }
      // Off until the permission changes (watchMicPermission clears this reason when it does).
      setReason('permission', `microphone access was denied (${detail}). Allow the microphone for this page in `
        + 'the browser’s site settings.');
      say('Microphone access was denied, so voice input is off.', { tone: 'error' });
      return;
    }
    if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      setReason('device', `no microphone was found (${detail}). Connect one; this re-checks automatically.`);
      say('No microphone was found, so voice input is off.', { tone: 'error' });
      return;
    }
    showError(`Couldn’t start the microphone (${detail}).`);
  }

  // --- transcription ------------------------------------------------------------------------
  async function transcribe(wav, info) {
    // What the request box held when the recording was sent: if the user changes it meanwhile, the late
    // transcript must not replace their text (see onTranscript).
    const tx = {
      ctrl: typeof AbortController === 'function' ? new AbortController() : null,
      since: performance.now(),
      request: el.request ? el.request.value : '',
      task: el.taskSelect ? el.taskSelect.value : '',
      edited: false,
    };
    V.tx = tx;
    setState('transcribing');
    say(info.limit ? `Reached the ${MAX_SECONDS}-second limit. Transcribing…` : 'Transcribing…');
    const timer = window.setTimeout(() => { if (tx.ctrl) tx.ctrl.abort(); }, TRANSCRIBE_TIMEOUT_MS);
    let res;
    let data = {};
    try {
      res = await fetch('/api/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav', Accept: 'application/json' },
        body: wav,
        cache: 'no-store',
        signal: tx.ctrl ? tx.ctrl.signal : undefined,
      });
      data = await res.json().catch(() => ({}));
    } catch (err) {
      window.clearTimeout(timer);
      if (V.tx !== tx) return;  // cancelled with Esc or the mic button; that already said so
      V.tx = null;
      showError(err && err.name === 'AbortError'
        ? `Transcribing took longer than ${TRANSCRIBE_TIMEOUT_MS / 1000} s, so it was abandoned. Try again.`
        : 'Couldn’t reach the console backend to transcribe. Is it still running?');
      return;
    }
    window.clearTimeout(timer);
    if (V.tx !== tx) return;  // cancelled meanwhile: a late reply changes nothing
    V.tx = null;
    if (!data || typeof data !== 'object' || Array.isArray(data)) data = {};
    if (!res.ok) {
      const msg = oneLine(data.error);
      if (res.status === 400) showError(`Couldn’t use that recording${msg ? `: ${msg}` : ''}.`);
      else if (res.status === 413) showError(`That recording is too long${msg ? `: ${msg}` : ''}.`);
      else if (res.status === 503) showError(`Speech recognition is unavailable${msg ? `: ${msg}` : ''}.`);
      else showError(`Transcribing failed (HTTP ${res.status})${msg ? `: ${msg}` : ''}.`);
      return;
    }
    try {
      onTranscript(data, info, tx);
    } catch (err) {
      showError(`Something went wrong with the transcript: ${oneLine(err && err.message) || 'unknown error'}`);
    }
  }

  /** Esc or the mic button while transcribing: drop the recording; its reply, if it still comes, is ignored. */
  function cancelTranscribe() {
    const tx = V.tx;
    if (!tx || V.state !== 'transcribing') return false;
    V.tx = null;
    if (tx.ctrl) tx.ctrl.abort();
    setState(isAvailable() ? 'idle' : 'off');
    say('Transcription cancelled. Nothing will run.', { tone: 'warn' });
    return true;
  }

  function showError(message) {
    setState(isAvailable() ? 'error' : 'off');
    say(message.replace(/\.\.$/, '.'), { tone: 'error' });
  }

  function onTranscript(data, info, tx) {
    const C = window.AcmeConsole;
    const text = oneLine(data.text);
    if (!text) {
      setState('idle');
      say(info.peakRms < SILENT_RMS ? 'Didn’t catch that: the microphone sounded silent. Is it muted?'
        : 'Didn’t catch that. Try again.', { tone: 'warn' });
      return;
    }
    const latency = Number(data.latency_ms);
    const ms = Number.isFinite(latency) ? `, ${Math.round(latency)} ms` : '';
    const fixed = isFixedText();
    const label = oneLine(V.asr && V.asr.label) || oneLine(data.model) || 'ASR';
    showHeard(fixed ? `Stub (fixed text${ms}): ${text}` : `Heard (${label}${ms}): ${text}`);
    const heard = fixed ? `The stub’s fixed text (not what you said), “${text}”` : `Heard: “${text}”`;

    // The user typed or picked a task while this was transcribing: keep their request, act on nothing.
    const edited = tx && (tx.edited || (el.request && el.request.value !== tx.request)
      || (el.taskSelect && el.taskSelect.value !== tx.task));
    if (edited) {
      V.heard = { request: null, used: false };
      setState('done');
      say(`${heard}. You changed the request meanwhile, so yours was kept and nothing runs by itself.`,
        { tone: 'warn' });
      return;
    }

    // For the logs page: console.js sends this with the run only while the box still holds what it produced.
    const voice = { text, model: data.model, latency_ms: data.latency_ms, duration_s: data.duration_s };
    const stop = isStopIntent(text);
    if (C.isRunning()) {
      if (stop) {
        V.heard = { request: null, used: true };
        setState('done');
        say('Stopping', { tone: 'ok' });
        C.stop();
        return;
      }
      const fill = fillBox(text, voice);
      V.heard = { request: fill.request, used: false };
      setState('done');
      say(`A run is in progress — say "stop" to stop it. “${fill.request}” is in the request box for later.`
        + `${fill.cut ? ` What you said was too long (${text.length} characters), so the box has the first part.` : ''}`,
      { tone: 'warn' });
      return;
    }
    if (stop) {  // a stop phrase never starts a run
      V.heard = { request: null, used: true };
      setState('done');
      say(C.isStarting() ? `${heard}. The run is still starting: press Escape once it’s running to stop it.`
        : `${heard}. Nothing is running, so there’s nothing to stop.`, { tone: C.isStarting() ? 'warn' : '' });
      return;
    }
    const fill = fillBox(text, voice);
    V.heard = { request: fill.request, used: false };
    setState('done');
    const task = fill.taskId ? ' (a sandbox task)' : '';
    if (C.isStarting()) {
      say(`A run is starting, so “${fill.request}” won’t run by itself. It’s in the request box.`, { tone: 'warn' });
    } else if (fill.cut) {
      say(`Too long: you said ${text.length} characters and a request holds ${REQUEST_MAX} at most. The request `
        + `box has the first ${fill.request.length}; check it, then press Enter to run it.`, { tone: 'warn' });
      if (el.request) el.request.focus();
    } else if (fixed) {
      say(`${heard}${task} is in the request box. The stub never runs by itself: press Enter in the request box `
        + 'to run it.', { tone: 'warn' });
    } else if (!autoRunMs()) {
      say(`Heard: “${fill.request}”${task}. Press Enter in the request box to run it.`, { tone: 'ok' });
    } else {
      startCountdown(fill.request, fill.taskId, autoRunMs());
    }
  }

  /** Put the transcript in the request box, cut to the box's 500 characters at a word boundary. */
  function fillBox(text, voice) {
    const cut = cutToWord(text, REQUEST_MAX);
    const shortened = cut.length < text.length;
    const { taskId, request } = window.AcmeConsole.setRequest(cut, shortened ? { ...voice, text: cut } : voice);
    if (shortened && el.heard) {
      showHeard(`${el.heard.textContent} [too long: the request box kept the first ${request.length} characters]`);
    }
    return { taskId, request, cut: shortened };
  }

  function showHeard(line) {
    if (!el.heard) return;
    el.heard.textContent = line;
    el.heard.hidden = false;
  }

  function hideHeard() {
    if (el.heard) el.heard.hidden = true;
    V.heard = null;
  }

  // --- auto-run countdown -------------------------------------------------------------------------
  function startCountdown(request, taskId, ms) {
    cancelCountdown('');
    const cd = { endsAt: performance.now() + ms, ms, timer: 0 };
    V.countdown = cd;
    el.countdown.hidden = false;
    renderCountdown(cd);
    cd.timer = window.setInterval(() => {
      if (V.countdown !== cd) return;
      if (performance.now() >= cd.endsAt) finishCountdown(cd);
      else renderCountdown(cd);
    }, 100);
    // Short, and still true when it's read late: Escape cancels the countdown, and stops the run once it started.
    say(`Heard: “${request}”${taskId ? ' (a sandbox task)' : ''}`, {
      tone: 'ok', srExtra: `Runs in ${ms / 1000} seconds. Escape stops it.`,
    });
  }

  function renderCountdown(cd) {
    const left = Math.max(0, cd.endsAt - performance.now());
    setText(el.countdownText, `Running in ${(Math.ceil(left / 100) / 10).toFixed(1)} s — press Esc to cancel`);
    el.countdown.style.setProperty('--left', (left / cd.ms).toFixed(3));
  }

  function finishCountdown(cd) {
    if (V.countdown !== cd) return;
    clearCountdown();
    const sr = el.status.querySelector('.visually-hidden');  // "Runs in 1.5 seconds…" is spent (removal isn't read)
    if (sr) sr.remove();
    const C = window.AcmeConsole;
    if (C.isRunning() || C.isStarting()) return;
    C.run();
  }

  function clearCountdown() {
    const cd = V.countdown;
    if (!cd) return false;
    window.clearInterval(cd.timer);
    V.countdown = null;
    const hadFocus = el.countdown.contains(document.activeElement);
    el.countdown.hidden = true;
    if (hadFocus) document.getElementById('request').focus();
    return true;
  }

  /** message '' = silent (e.g. the run started anyway). */
  function cancelCountdown(message) {
    if (!clearCountdown()) return;
    if (message) say(message, { tone: 'warn' });
  }

  // --- push-to-talk key setting ------------------------------------------------------------------
  function loadPttKey() {
    const saved = oneLine(prefs.get(PTT_STORE));
    V.pttCode = saved && !rejectKey(saved) ? saved : DEFAULT_PTT;
  }

  function renderPtt() {
    const name = keyName(V.pttCode);
    setText(el.pttKey, V.capturing ? '…' : name);
    setText(el.pttChange, V.capturing ? 'Press a key…' : 'Change');
    el.pttChange.setAttribute('aria-label', V.capturing ? 'Press a key… (Escape cancels)'
      : `Change push-to-talk key (now ${name})`);
    el.pttChange.dataset.capturing = String(V.capturing);
    el.pttReset.hidden = V.pttCode === DEFAULT_PTT || V.capturing;
    renderAvailability();
  }

  function pttMsg(text) {
    if (!el.pttMsg) return;
    el.pttMsg.textContent = '';
    window.setTimeout(() => { el.pttMsg.textContent = text; }, 40);
  }

  function beginCapture() {
    if (V.capturing) return;
    if (V.armed) disarm();
    if (V.rec) stopRecording('cancel', { quiet: true });
    V.capturing = true;
    renderPtt();
    pttMsg('Press the key you want to hold for talking. Escape cancels.');
  }

  function endCapture(code, message) {
    V.capturing = false;
    if (code) {
      V.pttCode = code;
      if (code === DEFAULT_PTT) prefs.del(PTT_STORE);
      else prefs.set(PTT_STORE, code);
    }
    renderPtt();
    if (message) pttMsg(message);
  }

  function onCaptureKey(e) {
    if (e.repeat) { e.preventDefault(); return; }
    if (e.code === 'Escape' || e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      endCapture(null, `Kept ${keyName(V.pttCode)}.`);
      return;
    }
    if (e.code === 'Tab' || e.key === 'Tab') {  // let Tab move focus as usual; nothing changes
      endCapture(null, `Kept ${keyName(V.pttCode)}.`);
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    const problem = rejectKey(e.code);
    if (problem) { pttMsg(problem); return; }
    const where = worksEverywhere(e.code) ? '' : ' It works when focus isn’t in a field or on a button.';
    endCapture(e.code, `Push-to-talk key: ${keyName(e.code)}.${where}`);
  }

  // --- "after voice input" setting ---------------------------------------------------------------
  function loadAutorun() {
    const saved = oneLine(prefs.get(AUTORUN_STORE));
    V.autorun = AUTORUN_CHOICES.some(([v]) => v === saved) ? saved : DEFAULT_AUTORUN;
  }

  /** ms before a transcript runs by itself; 0 = it never does. */
  const autoRunMs = () => (V.autorun === 'off' ? 0 : Math.round(Number(V.autorun) * 1000));

  /** Options: "After voice input: run in 1.5 s / run in 5 s / don't auto-run", below the push-to-talk key. */
  function buildAutorunSetting() {
    const anchor = document.getElementById('opt-ptt');
    if (!anchor || document.getElementById('voice-autorun')) return;
    const select = document.createElement('select');
    select.id = 'voice-autorun';
    select.style.width = 'auto';  // next to its label, not the full-width task picker
    for (const [value, text] of AUTORUN_CHOICES) select.add(new Option(text, value));
    select.value = V.autorun;
    const label = document.createElement('label');
    label.className = 'ptt-title';
    label.htmlFor = select.id;
    label.textContent = 'After voice input:';
    const line = document.createElement('p');
    line.className = 'ptt-line';
    line.append(label, select);
    const wrap = document.createElement('div');
    wrap.className = 'opt opt-ptt';
    wrap.id = 'opt-autorun';
    wrap.append(line);
    anchor.after(wrap);
    select.addEventListener('change', () => {
      V.autorun = AUTORUN_CHOICES.some(([v]) => v === select.value) ? select.value : DEFAULT_AUTORUN;
      if (V.autorun === DEFAULT_AUTORUN) prefs.del(AUTORUN_STORE);
      else prefs.set(AUTORUN_STORE, V.autorun);
      if (!autoRunMs()) cancelCountdown('Auto-run cancelled.');
    });
  }

  // --- keyboard and pointer ---------------------------------------------------------------------
  function disarm() {
    window.clearTimeout(V.armed);
    V.armed = 0;
    V.held = null;
  }

  function onKeyDown(e) {
    if (V.capturing) { onCaptureKey(e); return; }
    if (e.key === 'Escape') {
      if (V.countdown) {  // Esc cancels the countdown; console.js's "Esc stops the run" sees defaultPrevented
        e.preventDefault();
        cancelCountdown('Auto-run cancelled. Press Enter in the request box to run it.');
      } else if (V.rec) {
        e.preventDefault();
        stopRecording('cancel');
      } else if (V.armed) {
        disarm();
      } else if (V.state === 'transcribing') {
        // During a run, Esc also stops the run (console.js): no preventDefault then.
        if (!(window.AcmeConsole && window.AcmeConsole.isRunning())) e.preventDefault();
        cancelTranscribe();
      }
      return;
    }
    if (V.held) {
      if (e.code === V.held) { e.preventDefault(); return; }  // auto-repeat while holding
      if (V.armed) {  // a chord such as Option+E (é) or a shortcut: it never was push-to-talk, and says nothing
        disarm();
        return;
      }
      if (MODIFIER_RE.test(V.held) && !MODIFIER_RE.test(e.code)) {  // the modifier was part of a shortcut
        stopRecording('cancel', { why: `Cancelled: ${keyName(V.held)} was used with another key. Nothing was sent.` });
      }
      return;
    }
    if (e.isComposing || !isAvailable()) return;
    const isPtt = e.code === V.pttCode && pttKeyHere(e.code);
    const isSpace = e.code === 'Space' && !e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey
      && !isInteractive(document.activeElement);
    if (!isPtt && !isSpace) return;
    e.preventDefault();  // Option doesn't type, Space doesn't scroll
    if (e.repeat) return;
    if (V.rec) return;  // already listening (the Speak button)
    if (V.state === 'transcribing') {
      say('Still transcribing the last recording… Esc cancels it.');
      return;
    }
    const code = e.code;
    V.held = code;
    if (MODIFIER_RE.test(code) && isTextField(document.activeElement)) {
      // In a text field the modifier may be the start of a character (Option+E, AltGr+Q): wait until it has been
      // held alone for a moment. A chord or a quick release clears this silently, before any getUserMedia.
      V.armed = window.setTimeout(() => {
        V.armed = 0;
        if (V.held === code && !V.rec) startRecording('ptt');
      }, ARM_MS);
      return;
    }
    startRecording('ptt');
  }

  function onKeyUp(e) {
    if (!V.held || e.code !== V.held) return;
    e.preventDefault();
    if (V.armed) { disarm(); return; }  // released before the mic was asked for: nothing happened
    V.held = null;
    stopRecording('send');
  }

  function onMicClick() {
    if (V.state === 'transcribing') {  // the button cancels it (not the second click of a double-click)
      if (V.tx && performance.now() - V.tx.since >= CANCEL_GRACE_MS) cancelTranscribe();
      return;
    }
    if (!isAvailable()) return;
    if (V.rec) stopRecording('send');
    else startRecording('toggle');
  }

  // --- init -------------------------------------------------------------------------------------
  function grab() {
    const ids = {
      mic: 'mic-btn', micLabel: 'mic-label', status: 'voice-status', timer: 'voice-timer', hint: 'voice-hint',
      countdown: 'voice-countdown', countdownText: 'voice-countdown-text', cancel: 'voice-cancel', heard: 'heard',
      pttKey: 'ptt-key', pttChange: 'ptt-change', pttReset: 'ptt-reset', pttMsg: 'ptt-msg',
      kbdWrap: 'kbd-ptt-wrap', kbdPtt: 'kbd-ptt', request: 'request', taskSelect: 'task-select', asrNote: 'asr-note',
    };
    for (const [k, id] of Object.entries(ids)) el[k] = document.getElementById(id);
    return Boolean(el.mic && el.status && el.hint && el.countdown && el.pttChange);
  }

  async function init() {
    if (!grab()) return;
    loadPttKey();
    loadAutorun();
    buildAutorunSetting();

    el.mic.addEventListener('click', onMicClick);
    el.mic.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.repeat) e.preventDefault();  // holding Enter must not toggle over and over
    });
    el.cancel.addEventListener('click', () => {
      cancelCountdown('Auto-run cancelled. Press Enter in the request box to run it.');
    });
    el.pttChange.addEventListener('click', () => { if (!V.capturing) beginCapture(); });
    el.pttChange.addEventListener('blur', () => { if (V.capturing) endCapture(null, `Kept ${keyName(V.pttCode)}.`); });
    el.pttReset.addEventListener('click', () => {
      endCapture(DEFAULT_PTT, `Push-to-talk key reset to ${keyName(DEFAULT_PTT)}.`);
      el.pttChange.focus();
    });
    window.addEventListener('keydown', onKeyDown, true);  // capture: runs before console.js's Esc handler
    window.addEventListener('keyup', onKeyUp, true);
    window.addEventListener('blur', () => {
      if (V.capturing) endCapture(null, '');
      if (V.armed) disarm();
      else if (V.held) { V.held = null; stopRecording('send'); }  // the key-up will never arrive
    });
    if (el.request) {
      el.request.addEventListener('input', () => {
        if (V.tx) V.tx.edited = true;  // a transcript still on its way must not replace this
        cancelCountdown('Auto-run cancelled because you edited the request.');
      });
    }
    if (el.taskSelect) {
      el.taskSelect.addEventListener('change', () => {
        if (V.tx) V.tx.edited = true;
        cancelCountdown('Auto-run cancelled.');
      });
    }
    document.addEventListener('acme:run-start', (e) => {
      cancelCountdown('');
      const req = e.detail && e.detail.request;
      if (V.heard && !V.heard.used && V.heard.request === req) V.heard.used = true;  // this run came from voice
      else hideHeard();
      // Old "Heard…" / "Cancelled…" lines give the step log their room; errors and "Listening…" stay.
      if (!V.rec && el.status.dataset.tone !== 'error') {
        V.sayId += 1;  // drops a pending re-announce too
        el.status.replaceChildren();
        el.status.dataset.tone = '';
      }
    });

    renderPtt();

    const missing = [];
    if (!window.isSecureContext) missing.push('the page isn’t a secure context (open it as http://127.0.0.1 or https)');
    else if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
      missing.push('this browser gives the page no microphone access (getUserMedia is unavailable)');
    }
    if (!(window.AudioContext || window.webkitAudioContext)) missing.push('this browser has no Web Audio');
    if (!window.AcmeWav) missing.push('wav.js didn’t load');
    if (!window.AcmeConsole) missing.push('console.js didn’t load');
    if (missing.length) {
      setReason('browser', `${missing[0]}.`);
      V.asr = V.asr || {};
      renderAvailability();
      return;
    }
    if (navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener('devicechange', () => setReason('device', ''));
    }
    watchMicPermission();

    let cfg = null;
    try {
      cfg = (await window.AcmeConsole.ready).config;
    } catch { /* treated as offline below */ }
    if (applyAsr(cfg)) scheduleRecheck();
  }

  /** /api/config's "asr" -> the 'asr' reason. Returns true when checking again can help (the recognizer is down,
   *  e.g. scripts/run_asr.sh is still loading the model). */
  function applyAsr(cfg) {
    const asr = cfg && typeof cfg.asr === 'object' && cfg.asr ? cfg.asr : null;
    V.asr = asr || {};
    if (!cfg) {
      setReason('asr', 'the console backend isn’t reachable.');
      return false;
    }
    if (!asr) {
      setReason('asr', 'this console backend has no speech recognizer (/api/config has no "asr").');
      return false;
    }
    if (asr.available === true) {
      setReason('asr', '');
      return false;
    }
    const err = oneLine(asr.error).replace(/\.+$/, '');
    setReason('asr', `the speech recognizer (${oneLine(asr.label || asr.model) || 'ASR'}) isn’t available`
      + `${err ? `: ${err}` : ''}. This checks again every few seconds and turns voice input on by itself.`);
    return true;
  }

  function scheduleRecheck() {
    window.clearTimeout(V.recheck);
    V.recheck = window.setTimeout(recheckAsr, RECHECK_MS);
  }

  async function recheckAsr() {
    V.recheck = 0;
    if (!V.reasons.has('asr')) return;
    let cfg = null;
    if (!document.hidden) {
      try {
        const res = await fetch('/api/config', { cache: 'no-store', headers: { Accept: 'application/json' } });
        if (res.ok) cfg = await res.json();
      } catch { /* the backend is busy or restarting: try again later */ }
    }
    if (!V.reasons.has('asr')) return;
    if (!cfg || typeof cfg !== 'object' || applyAsr(cfg)) scheduleRecheck();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
