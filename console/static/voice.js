/* Voice input for the GUI Assist console: mic button, push-to-talk, local capture, POST /api/transcribe.
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
 *
 * A warm microphone, so the first words aren't lost. The first press opens the microphone (getUserMedia, an
 * AudioContext, the worklet: about half a second), says "Starting the microphone…", and records from the very first
 * frame; "Listening… start speaking now" (and the red button) appear only once frames arrive. After that take the
 * stream, context and worklet stay open ("warm"): the worklet keeps the last ~400 ms in a ring (pcm-worklet.js), and
 * the next press starts with that pre-roll, so audio from just before the key went down is included (never any
 * older than ~0.5 s), with no new getUserMedia. While warm, a line in the voice area says "Microphone ready
 * (releases after 5 min idle)" with a Turn off button. Everything is released (tracks stopped, context closed) after
 * 5 minutes without a recording, when the page is hidden or unloads, when voice input goes off, when the track
 * ends, and by Turn off; the next press opens it again. A press resumes a suspended context (Safari) inside the key
 * or click handler.
 *
 * Answering the agent's question (done reason "needs_input", console.js): while one waits (AcmeConsole.pendingQuestion()),
 * the next transcript is the answer, not a new request: it fills the answer box and the same countdown sends it
 * ("Answer: “…”", "Answering in 1.5 s", Esc or Cancel stops it). The rules above hold for it too: no auto-send over
 * 300 characters, after an edit meanwhile, with the fixed-text stub or with auto-run off; a stop phrase is never an
 * answer. The banner says the next voice input answers.
 *
 * window.AcmeVoice (for tests): setIdleReleaseMs(ms) -> ms in use, mic() -> {warm, recording, kind, rate,
 * idleReleaseMs}, lastTake() -> {cold, preRollMs, seconds, heldSeconds, rate, sent} | null, release().
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
  const ANSWER_MAX = 300;   // the answer box's limit (POST /api/run clarifies.answer)
  const ANSWER_CANCELLED = 'Cancelled: the answer wasn’t sent. It’s in the answer box: press Enter there to send it.';
  const TRANSCRIBE_TIMEOUT_MS = 60000;
  const CANCEL_GRACE_MS = 600;  // a mic click this soon after "Transcribing" is a double-click, not a cancel
  const ARM_MS = 200;  // a modifier push-to-talk key in a text field must be held alone this long
  const RECHECK_MS = 4000;  // voice input off because the recognizer isn't up: ask /api/config again
  const FLUSH_TIMEOUT_MS = 300;
  const DONE_MS = 2500;
  const PRE_ROLL_MS = 400;           // a warm take starts with this much audio from just before the press...
  const PRE_ROLL_MAX_AGE_MS = 500;   // ...and never with audio older than this
  const IDLE_RELEASE_MS = 5 * 60 * 1000;  // a warm microphone is released after this long without a recording
  const WARM_SLOW_MS = 250;          // a warm take with no frames yet after this says "Starting the microphone…"
  const WARM_WATCHDOG_MS = 1500;     // ...and after this the warm microphone is opened again from scratch
  const COLD_WATCHDOG_MS = 4000;     // a just-opened microphone that sends no frames for this long is an error
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
    // opening = a take has started but no frames have arrived yet (not red); listening = frames are arriving
    state: 'off',        // off | idle | opening | listening | transcribing | done | error
    rec: null,           // the take in progress (see startRecording)
    mic: null,           // the open ("warm") microphone, kept between takes (see openMic)
    takeSeq: 0,
    lastTake: null,      // what the last take held (window.AcmeVoice.lastTake)
    idleMs: IDLE_RELEASE_MS,
    idleTimer: 0,
    held: null,          // event.code of the key holding push-to-talk right now
    armed: 0,            // timer: a modifier held in a text field, not recording yet (see onKeyDown)
    pttCode: DEFAULT_PTT,
    capturing: false,    // Options: waiting for the new push-to-talk key
    autorun: DEFAULT_AUTORUN,  // Options: what happens after voice input ('1.5' | '5' | 'off')
    countdown: null,     // {endsAt, ms, timer, kind: 'run' | 'answer'}
    tx: null,            // the transcription in flight: {ctrl, since, request, task, answer, edited}
    recheck: 0,          // timer: the next /api/config check while the recognizer is down
    heard: null,         // {request, answer?, used}: the last transcript put in the request (or answer) box
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
      releaseMic('');  // voice input is off: the microphone is not kept open for it
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
    if (el.answerVoice) {  // the needs_input banner: say that voice answers too
      el.answerVoice.hidden = !on;
      el.answerVoice.replaceChildren(...(on ? [document.createTextNode(' Or hold '), kbd(keyName(V.pttCode)),
        document.createTextNode(' and say it: your next voice input answers.')] : []));
    }
    if (el.asrNote) {
      el.asrNote.hidden = !on;
      setText(el.asrNote, isFixedText()
        ? `Voice input is the ${asrLabel()} engine: it recognizes no speech. For real voice input, start `
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
    const listening = state === 'opening' || state === 'listening';
    // A plain action button: its name says what pressing it does ("Speak", "Stop listening"), so no aria-pressed.
    mic.removeAttribute('aria-pressed');
    mic.disabled = state === 'off';
    if (state === 'transcribing') mic.setAttribute('aria-label', 'Transcribing… press to cancel');
    else mic.removeAttribute('aria-label');
    setText(el.micLabel, listening ? 'Stop listening' : (state === 'transcribing' ? 'Transcribing…' : 'Speak'));
    el.timer.hidden = state !== 'listening';
    if (state !== 'listening') mic.style.setProperty('--level', '0');
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
      const secs = (rec.frames - rec.preFrames) / rec.rate;  // since the press (the pre-roll is extra)
      setText(el.timer, `${fmtClock(secs)} / ${fmtClock(MAX_SECONDS)}`);
      V.rafId = window.requestAnimationFrame(frame);
    };
    V.level = 0;
    window.cancelAnimationFrame(V.rafId);
    V.rafId = window.requestAnimationFrame(frame);
  }

  // --- the microphone ---------------------------------------------------------------------
  // V.mic, once a take has had frames: {stream, ctx, src, node, kind: 'worklet'|'script', rate, ring (script only),
  // take (the take it feeds, or null), onEnded, announced}. It stays open between takes (see the header).

  const micIsWarm = () => Boolean(V.mic && V.mic.ctx && V.mic.ctx.state !== 'closed');

  /** Safari (and a context made outside a gesture) can start or fall back to 'suspended' / 'interrupted'. Called
   *  in the key or click handler, where resuming is allowed. */
  function wake(ctx) {
    if (!ctx || ctx.state === 'running' || ctx.state === 'closed') return;
    try { ctx.resume().catch(() => {}); } catch { /* an old webkitAudioContext: no promise */ }
  }

  function newContext(rate) {
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = rate ? new AC({ sampleRate: rate }) : new AC();
    wake(ctx);
    return ctx;
  }

  /** The first press (or one after a release): open the microphone for take `rec`. The context is made here,
   *  still inside the press handler, before getUserMedia. */
  async function openMic(rec) {
    const mic = {
      stream: null, ctx: null, src: null, node: null, kind: '', rate: TARGET_RATE, ring: null, take: null,
      onEnded: null, announced: false,
    };
    try {
      try { mic.ctx = newContext(TARGET_RATE); } catch { mic.ctx = null; }  // openGraph tries the default rate
      mic.stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
      if (V.rec !== rec) { closeMic(mic); return; }  // cancelled while asking: nothing was recorded, nothing kept
      await openGraph(mic);
      if (V.rec !== rec) { closeMic(mic); return; }
    } catch (err) {
      closeMic(mic);
      if (V.rec !== rec) return;  // cancelled meanwhile; its message is already shown
      clearTake(rec);
      V.rec = null;
      V.held = null;
      await micFailed(err);
      return;
    }
    V.mic = mic;
    for (const t of mic.stream.getAudioTracks()) t.onended = () => { if (V.mic === mic) micLost(mic); };
    attach(rec);
    rec.watchdog = window.setTimeout(() => noFrames(rec), COLD_WATCHDOG_MS);
  }

  async function openGraph(mic) {
    let ctx = mic.ctx;
    let src = null;
    try {
      if (!ctx) throw new Error('no context yet');
      src = ctx.createMediaStreamSource(mic.stream);
    } catch {
      // Firefox can't connect a microphone to a context at another rate: record at the default and resample.
      if (ctx) ctx.close().catch(() => {});
      ctx = newContext(0);
      mic.ctx = ctx;  // so closeMic closes it even if the next line throws
      src = ctx.createMediaStreamSource(mic.stream);
    }
    mic.src = src;
    mic.rate = ctx.sampleRate;
    if (ctx.state !== 'running') {
      await Promise.race([ctx.resume().catch(() => {}), new Promise((r) => { window.setTimeout(r, 1000); })]);
      if (ctx.state !== 'running') {
        const e = new Error('the browser kept audio paused. Click anywhere on the page once, then try again.');
        e.name = 'AudioPaused';
        throw e;
      }
    }
    const ringFrames = Math.round((mic.rate * PRE_ROLL_MS) / 1000);
    if (ctx.audioWorklet && window.AudioWorkletNode) {
      try {
        await ctx.audioWorklet.addModule('/static/pcm-worklet.js');
        const node = new window.AudioWorkletNode(ctx, 'pcm-capture', {
          numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
          channelCount: 1, channelCountMode: 'explicit', channelInterpretation: 'speakers',
          processorOptions: { chunkFrames: Math.max(128, Math.round(mic.rate * 0.064)), ringFrames },
        });
        node.port.onmessage = (e) => onWorkletMessage(mic, e.data || {});
        src.connect(node);
        node.connect(ctx.destination);  // keeps the node pulled; it outputs silence
        mic.node = node;
        mic.kind = 'worklet';
        return;
      } catch { /* fall back to ScriptProcessor */ }
    }
    const size = mic.rate <= 24000 ? 1024 : 2048;
    const sp = ctx.createScriptProcessor(size, 1, 1);
    mic.ring = { chunks: [], frames: 0, cap: ringFrames };
    sp.onaudioprocess = (e) => onScriptSamples(mic, new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(sp);
    sp.connect(ctx.destination);
    mic.node = sp;
    mic.kind = 'script';
  }

  function post(mic, msg) {
    try { mic.node.port.postMessage(msg); return true; } catch { return false; }
  }

  function onWorkletMessage(mic, d) {
    if (d.type === 'ended') {
      if (mic.onEnded && mic.onEnded.id === d.id) mic.onEnded();
      return;
    }
    const rec = mic.take;
    if (!rec || d.id !== rec.id || V.mic !== mic) return;  // a take that already ended
    if (d.type === 'preroll') addPreRoll(rec, d.samples);
    else if (d.type === 'chunk') addSamples(rec, d.samples);
  }

  /** ScriptProcessor fallback: between takes the newest ~400 ms are kept here, as the worklet does. */
  function onScriptSamples(mic, f32) {
    if (V.mic !== mic || !f32.length) return;
    if (mic.take) { addSamples(mic.take, f32); return; }
    const ring = mic.ring;
    ring.chunks.push({ f32, at: performance.now() });
    ring.frames += f32.length;
    while (ring.chunks.length > 1 && ring.frames - ring.chunks[0].f32.length >= ring.cap) {
      ring.frames -= ring.chunks.shift().f32.length;
    }
  }

  /** The script ring's newest samples: at most PRE_ROLL_MS, none older than PRE_ROLL_MAX_AGE_MS. Empties it. */
  function takeScriptRing(mic) {
    const ring = mic.ring;
    const now = performance.now();
    const cap = Math.floor((mic.rate * PRE_ROLL_MS) / 1000);
    const parts = [];
    let n = 0;
    for (let i = ring.chunks.length - 1; i >= 0 && n < cap; i -= 1) {
      const { f32, at } = ring.chunks[i];
      // its last sample arrived at `at`; the j-th from the end is j / rate older
      const young = Math.floor(((PRE_ROLL_MAX_AGE_MS - (now - at)) * mic.rate) / 1000);
      const k = Math.min(f32.length, young, cap - n);
      if (k <= 0) break;
      parts.unshift(f32.subarray(f32.length - k));
      n += k;
      if (k < f32.length) break;
    }
    ring.chunks = [];
    ring.frames = 0;
    const out = new Float32Array(n);
    let off = 0;
    for (const p of parts) { out.set(p, off); off += p.length; }
    return out;
  }

  /** Connect take `rec` to the open microphone: it starts with the pre-roll, then live audio. */
  function attach(rec) {
    const mic = V.mic;
    rec.rate = mic.rate;
    mic.take = rec;
    if (mic.kind === 'worklet') {
      post(mic, { type: 'start', id: rec.id, preRollMs: PRE_ROLL_MS, maxAgeMs: PRE_ROLL_MAX_AGE_MS });
    } else {
      addPreRoll(rec, takeScriptRing(mic));
    }
  }

  function addPreRoll(rec, f32) {
    if (!f32 || !f32.length) return;
    rec.chunks.push(f32);
    rec.frames += f32.length;
    rec.preFrames += f32.length;
    const r = window.AcmeWav.rms(f32);
    if (r > rec.peakRms) rec.peakRms = r;
  }

  function addSamples(rec, f32) {
    if (!f32 || !f32.length) return;
    rec.chunks.push(f32);
    rec.frames += f32.length;
    rec.lastRms = window.AcmeWav.rms(f32);
    if (rec.lastRms > rec.peakRms) rec.peakRms = rec.lastRms;
    if (!rec.live && V.rec === rec && !rec.stopping) goLive(rec);
  }

  /** Frames are arriving for this take: only now is it "Listening…" (and red). */
  function goLive(rec) {
    rec.live = true;
    window.clearTimeout(rec.slowTimer);
    window.clearTimeout(rec.watchdog);
    rec.maxTimer = window.setTimeout(() => {
      if (V.rec === rec) stopRecording('send', { limit: true });
    }, MAX_SECONDS * 1000);
    setState('listening');
    const send = rec.mode === 'ptt' ? 'release to send' : 'press again to send';
    say(rec.cold ? `Listening… start speaking now, ${send}` : `Listening… ${send}`, { tone: 'live' });
    startMeter(rec);
    if (rec.stopWanted) stopRecording(rec.stopWanted);
  }

  /** No frames for this take. A warm microphone may have died quietly (a context that won't resume, a stalled
   *  device): open it again once. A just-opened one that sends nothing is an error. */
  function noFrames(rec) {
    if (V.rec !== rec || rec.live || rec.stopping) return;
    if (!rec.cold) {
      rec.cold = true;
      if (V.mic) V.mic.take = null;
      releaseMic('');
      say('Starting the microphone…');
      openMic(rec);
      return;
    }
    clearTake(rec);
    V.rec = null;
    V.held = null;
    releaseMic('');
    showError('The microphone sent no audio. Check that it isn’t muted or in use by another app, then try again.');
  }

  /** The track ended without us: the device was unplugged, or the browser or the system took the microphone. */
  function micLost(mic) {
    const rec = V.rec;
    if (rec && mic.take === rec) {
      clearTake(rec);
      mic.take = null;
      V.rec = null;
      V.held = null;
      releaseMic('');
      showError('The microphone stopped (unplugged, or taken by the system or another app). Nothing was sent. '
        + 'Press Speak to start it again.');
      return;
    }
    releaseMic('');
    say('The microphone was turned off by the browser or the system. The next recording opens it again.',
      { tone: 'warn' });
  }

  /** The worklet posts what it still buffers, then 'ended'; resolves then (or after FLUSH_TIMEOUT_MS).
   *  discard: drop the take's unsent audio, don't wait. */
  function endTake(rec, discard) {
    const mic = V.mic;
    if (!mic || mic.take !== rec) return Promise.resolve();
    if (mic.kind !== 'worklet' || discard) {
      if (mic.kind === 'worklet') post(mic, { type: 'end', id: rec.id, discard: true });
      mic.take = null;
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      let t = 0;
      const done = () => {
        window.clearTimeout(t);
        if (mic.onEnded === done) mic.onEnded = null;
        if (mic.take === rec) mic.take = null;
        resolve();
      };
      done.id = rec.id;
      mic.onEnded = done;
      t = window.setTimeout(done, FLUSH_TIMEOUT_MS);
      if (!post(mic, { type: 'end', id: rec.id })) done();
    });
  }

  /** Close everything: stop every track and close the context (the browser's recording indicator goes away). */
  function closeMic(mic) {
    try { if (mic.src) mic.src.disconnect(); } catch { /* already gone */ }
    try {
      if (mic.node) {
        if (mic.kind === 'worklet') { mic.node.port.postMessage({ type: 'stop' }); mic.node.port.onmessage = null; }
        else mic.node.onaudioprocess = null;
        mic.node.disconnect();
      }
    } catch { /* already gone */ }
    try {
      if (mic.stream) mic.stream.getTracks().forEach((t) => { t.onended = null; t.stop(); });
    } catch { /* already stopped */ }
    try { if (mic.ctx && mic.ctx.state !== 'closed') mic.ctx.close().catch(() => {}); } catch { /* closed */ }
    mic.take = null;
    if (mic.onEnded) mic.onEnded();
    mic.src = null;
    mic.node = null;
    mic.stream = null;
    mic.ctx = null;
  }

  /** why: 'off' (Turn off) | 'idle' (the idle timeout) | '' (quiet: page hidden, voice off, a lost track). */
  function releaseMic(why) {
    window.clearTimeout(V.idleTimer);
    V.idleTimer = 0;
    const mic = V.mic;
    if (!mic) return false;
    V.mic = null;
    closeMic(mic);
    renderWarm();
    if (why === 'off') announceMic('Microphone off.');
    else if (why === 'idle') announceMic(`Microphone off: released after ${fmtSpan(V.idleMs)} without a recording.`);
    return true;
  }

  function armIdle() {
    window.clearTimeout(V.idleTimer);
    V.idleTimer = window.setTimeout(() => {
      V.idleTimer = 0;
      if (V.mic && !V.rec) releaseMic('idle');
    }, V.idleMs);
  }

  /** A take is over (sent, cancelled or too short): the microphone stays warm, or is released if the page is hidden
   *  or voice input went off meanwhile. */
  function afterTake() {
    if (!V.mic) return;
    if (document.hidden || !isAvailable()) { releaseMic(''); return; }
    armIdle();
    renderWarm();
    if (!V.mic.announced) {
      V.mic.announced = true;
      announceMic(`${warmText()}. Turn off releases it now.`);
    }
  }

  /** "5 min", "90 s", "1.5 s". */
  function fmtSpan(ms) {
    if (ms >= 60000 && ms % 60000 === 0) return `${ms / 60000} min`;
    return `${Math.round(ms / 100) / 10} s`;
  }

  const warmText = () => `Microphone ready (releases after ${fmtSpan(V.idleMs)} idle)`;

  /** The warm-microphone line in the voice area: shown while the microphone is open, between takes and during one. */
  function renderWarm() {
    if (!el.warm) return;
    const open = Boolean(V.mic);
    if (!open && el.warm.contains(document.activeElement)) (el.mic.disabled ? el.request : el.mic).focus();
    el.warm.hidden = !open;
    if (open) setText(el.warmText, V.rec ? 'Microphone on (recording)' : warmText());
  }

  function announceMic(text) {
    if (!el.warmSr) return;
    el.warmSr.textContent = '';
    window.setTimeout(() => { el.warmSr.textContent = text; }, 40);
  }

  function buildWarmLine() {
    const anchor = el.hint;
    if (!anchor || document.getElementById('mic-warm')) return;
    const line = document.createElement('p');
    line.id = 'mic-warm';
    line.className = 'note voice-warm';
    line.hidden = true;
    const text = document.createElement('span');
    text.id = 'mic-warm-text';
    const off = document.createElement('button');
    off.id = 'mic-off';
    off.type = 'button';
    off.className = 'btn btn-small';
    off.textContent = 'Turn off';
    off.setAttribute('aria-label', 'Turn off the microphone');
    line.append(text, document.createTextNode(' '), off);  // a space, not a margin: no indent when it wraps
    const sr = document.createElement('p');
    sr.id = 'mic-warm-status';
    sr.className = 'visually-hidden';
    sr.setAttribute('role', 'status');
    sr.setAttribute('aria-live', 'polite');
    anchor.after(line, sr);
    el.warm = line;
    el.warmText = text;
    el.warmOff = off;
    el.warmSr = sr;
    off.addEventListener('click', turnOff);
  }

  /** Turn off: release the microphone now; a take in progress is dropped. */
  function turnOff() {
    if (V.rec) stopRecording('cancel', { why: 'Cancelled: the microphone was turned off. Nothing was sent.' });
    return releaseMic('off');
  }

  function clearTake(rec) {
    window.clearTimeout(rec.maxTimer);
    window.clearTimeout(rec.slowTimer);
    window.clearTimeout(rec.watchdog);
    if (V.rec === rec) window.cancelAnimationFrame(V.rafId);
  }

  // --- takes ------------------------------------------------------------------------------
  /** Called inside the key or click handler (or the push-to-talk arm timer). */
  function startRecording(mode) {
    if (V.rec || !isAvailable() || V.state === 'transcribing') return;
    cancelCountdown('');  // a new utterance replaces a pending auto-run
    window.clearTimeout(V.idleTimer);
    V.takeSeq += 1;
    const rec = {
      id: V.takeSeq, mode, cold: !micIsWarm(), live: false, stopping: false, stopWanted: null, chunks: [], frames: 0,
      preFrames: 0, rate: TARGET_RATE, lastRms: 0, peakRms: 0, maxTimer: 0, slowTimer: 0, watchdog: 0,
    };
    V.rec = rec;
    setState('opening');
    if (!rec.cold) {
      wake(V.mic.ctx);  // still in the handler: Safari resumes a suspended context only here
      V.sayId += 1;  // the old line goes; "Listening…" follows within a few ms (no "Starting…" in between)
      el.status.replaceChildren();
      el.status.dataset.tone = '';
      attach(rec);
      renderWarm();
      rec.slowTimer = window.setTimeout(() => {
        if (V.rec === rec && !rec.live) say('Starting the microphone…');
      }, WARM_SLOW_MS);
      rec.watchdog = window.setTimeout(() => noFrames(rec), WARM_WATCHDOG_MS);
      return;
    }
    if (V.mic) releaseMic('');  // its context was closed under us: start over
    say('Starting the microphone…');
    openMic(rec);
  }

  /** how: 'send' | 'cancel'. */
  async function stopRecording(how, { limit = false, quiet = false, why = '' } = {}) {
    const rec = V.rec;
    if (!rec || rec.stopping) return;
    V.held = null;
    if (!rec.live && how === 'send') {  // released before frames arrived: finish once they do (it will be too short)
      rec.stopWanted = 'send';
      return;
    }
    rec.stopping = true;
    clearTake(rec);
    if (how === 'send') await endTake(rec, false);
    else endTake(rec, true);
    V.rec = null;  // a cold take still opening sees this and closes the microphone
    const held = (rec.frames - rec.preFrames) / rec.rate;
    V.lastTake = {
      cold: rec.cold, preRollMs: Math.round((rec.preFrames / rec.rate) * 1000), seconds: rec.frames / rec.rate,
      heldSeconds: held, rate: rec.rate, sent: false,
    };
    afterTake();
    if (how === 'cancel') {
      setState(isAvailable() ? 'idle' : 'off');
      if (!quiet) say(why || 'Cancelled. Nothing was sent.', { tone: 'warn' });
      return;
    }
    if (held < MIN_SECONDS) {
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
    V.lastTake.sent = true;
    await transcribe(wav, { seconds: rec.frames / rec.rate, peakRms: rec.peakRms, limit });
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
      answer: el.answer ? el.answer.value : '',
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
      || (el.taskSelect && el.taskSelect.value !== tx.task) || (el.answer && el.answer.value !== tx.answer));
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
    if (stop) {  // a stop phrase never starts a run (and is never an answer)
      V.heard = { request: null, used: true };
      setState('done');
      say(C.isStarting() ? `${heard}. The run is still starting: press Escape once it’s running to stop it.`
        : `${heard}. Nothing is running, so there’s nothing to stop.`, { tone: C.isStarting() ? 'warn' : '' });
      return;
    }
    if (!C.isStarting() && C.pendingQuestion && C.pendingQuestion()) {  // the agent asked: this is the answer
      onAnswer(text, voice, heard, fixed);
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

  /** A transcript while the agent's question waits: it fills the answer box, and the countdown sends it. */
  function onAnswer(text, voice, heard, fixed) {
    const C = window.AcmeConsole;
    const cut = cutToWord(text, ANSWER_MAX);
    const shortened = cut.length < text.length;
    const { answer } = C.setAnswer(cut, shortened ? { ...voice, text: cut } : voice);
    V.heard = { request: null, answer, used: false };
    setState('done');
    if (shortened) {
      if (el.heard) showHeard(`${el.heard.textContent} [too long: the answer box kept the first ${answer.length} characters]`);
      say(`Too long for an answer: you said ${text.length} characters and an answer holds ${ANSWER_MAX} at most. The `
        + `answer box has the first ${answer.length}; check it, then press Enter to send it.`, { tone: 'warn' });
      if (el.answer) el.answer.focus();
    } else if (fixed) {
      say(`${heard} is in the answer box. The stub never sends by itself: press Enter in the answer box to send it.`,
        { tone: 'warn' });
    } else if (!autoRunMs()) {
      say(`Answer: “${answer}”. Press Enter in the answer box to send it.`, { tone: 'ok' });
    } else {
      startCountdown(answer, null, autoRunMs(), 'answer');
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
  /** kind 'run': the request runs when it ends; 'answer': the answer box's text is sent (a follow-up run). */
  function startCountdown(request, taskId, ms, kind = 'run') {
    cancelCountdown('');
    const cd = { endsAt: performance.now() + ms, ms, timer: 0, kind };
    V.countdown = cd;
    el.countdown.hidden = false;
    if (el.cancel) {
      el.cancel.setAttribute('aria-label', kind === 'answer' ? 'Cancel sending the answer' : 'Cancel the automatic run');
    }
    renderCountdown(cd);
    cd.timer = window.setInterval(() => {
      if (V.countdown !== cd) return;
      if (performance.now() >= cd.endsAt) finishCountdown(cd);
      else renderCountdown(cd);
    }, 100);
    // Short, and still true when it's read late: Escape cancels the countdown, and stops the run once it started.
    if (kind === 'answer') {
      say(`Answer: “${request}”`, { tone: 'ok', srExtra: `Sends in ${ms / 1000} seconds. Escape stops it.` });
      return;
    }
    say(`Heard: “${request}”${taskId ? ' (a sandbox task)' : ''}`, {
      tone: 'ok', srExtra: `Runs in ${ms / 1000} seconds. Escape stops it.`,
    });
  }

  function renderCountdown(cd) {
    const left = Math.max(0, cd.endsAt - performance.now());
    const verb = cd.kind === 'answer' ? 'Answering' : 'Running';
    setText(el.countdownText, `${verb} in ${(Math.ceil(left / 100) / 10).toFixed(1)} s — press Esc to cancel`);
    el.countdown.style.setProperty('--left', (left / cd.ms).toFixed(3));
  }

  function finishCountdown(cd) {
    if (V.countdown !== cd) return;
    clearCountdown();
    const sr = el.status.querySelector('.visually-hidden');  // "Runs in 1.5 seconds…" is spent (removal isn't read)
    if (sr) sr.remove();
    const C = window.AcmeConsole;
    if (C.isRunning() || C.isStarting()) return;
    if (cd.kind === 'answer') C.answer();
    else C.run();
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

  /** message '' = silent (e.g. the run started anyway). An answer's countdown says so in its own words
   *  (`answerMessage`, else ANSWER_CANCELLED). */
  function cancelCountdown(message, answerMessage = ANSWER_CANCELLED) {
    const kind = V.countdown ? V.countdown.kind : 'run';
    if (!clearCountdown() || !message) return;
    say(kind === 'answer' ? answerMessage : message, { tone: 'warn' });
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
    if (V.mic) wake(V.mic.ctx);  // a suspended warm context (Safari) resumes only in a key or click handler
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
      answer: 'answer-input', answerVoice: 'answer-voice',
    };
    for (const [k, id] of Object.entries(ids)) el[k] = document.getElementById(id);
    return Boolean(el.mic && el.status && el.hint && el.countdown && el.pttChange);
  }

  async function init() {
    if (!grab()) return;
    loadPttKey();
    loadAutorun();
    buildAutorunSetting();
    buildWarmLine();

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
    // The page is hidden: a take with audio is sent (as on blur), and the microphone is released (afterTake).
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) return;
      if (V.armed) disarm();
      if (V.rec && V.rec.live) {
        V.held = null;
        stopRecording('send');
      } else if (V.rec) {
        stopRecording('cancel', { quiet: true });
      }
      if (!V.rec) releaseMic('');
    });
    window.addEventListener('pagehide', () => {
      if (V.rec) stopRecording('cancel', { quiet: true });
      releaseMic('');
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
    if (el.answer) {
      el.answer.addEventListener('input', () => {
        if (V.tx) V.tx.edited = true;  // a transcript still on its way must not replace this
        cancelCountdown('Cancelled because you edited the answer.',
          'Cancelled because you edited the answer: press Enter in the answer box to send it.');
      });
    }
    document.addEventListener('acme:run-start', (e) => {
      cancelCountdown('');
      const req = e.detail && e.detail.request;
      const answer = e.detail && e.detail.answer;
      if (V.heard && !V.heard.used && (V.heard.request === req || (answer && V.heard.answer === answer))) {
        V.heard.used = true;  // this run came from voice (a request, or the answer that started a follow-up)
      } else {
        hideHeard();
      }
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

  // For tests and the console's own debugging; nothing here records or sends audio.
  window.AcmeVoice = Object.freeze({
    /** The idle release (default 5 min); restarts a running idle timer. Returns the value in use. */
    setIdleReleaseMs(ms) {
      const n = Math.round(Number(ms));
      if (Number.isFinite(n) && n >= 50) {
        V.idleMs = n;
        if (V.mic && !V.rec) armIdle();
        renderWarm();
      }
      return V.idleMs;
    },
    mic: () => ({
      warm: Boolean(V.mic), recording: Boolean(V.rec), kind: V.mic ? V.mic.kind : '', rate: V.mic ? V.mic.rate : 0,
      idleReleaseMs: V.idleMs,
    }),
    lastTake: () => (V.lastTake ? { ...V.lastTake } : null),
    release: () => turnOff(),
  });

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
