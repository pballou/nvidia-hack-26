"""Test-only stand-in for T1's sandbox (docs/tasks/contracts.md §1), until sandbox/ lands.

A tiny FastAPI app with the same HTTP contract: `?sid=` sets a cookie, per-sid
in-memory state, `POST /api/reset {sid, task_id}`, `GET /api/state?sid=`. Two pages at
1280x800: a dashboard with an icon-only gear button (no text, no aria-label) and a
settings page with a High contrast toggle, a Text size S/M/L control and a Display
name field. Every state change is visible on screen.

`FAKE_TASKS` follows the sandbox/tasks.json schema and `evaluate` is the real
sandbox/checks.py one, so agent and Gym tests can run without T1's pages. Only tests and verifiers read
/api/state; the agent never does.
"""

from __future__ import annotations

import copy
import socket
import threading
import time
from typing import Any

from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from sandbox.checks import evaluate  # noqa: F401  (re-exported: the real contract's checker)

DEFAULT_STATE: dict[str, Any] = {
    "settings": {"accessibility": {"high_contrast": False, "text_size": "M"}},
    "profile": {"name": "Sam Rivera"},
    "notifications": {"opened": False},
}

FAKE_TASKS: list[dict] = [
    {"id": "fake_high_contrast", "request": "Turn on high contrast.", "start_path": "/settings",
     "initial_state_overrides": {}, "steps_min": 1, "tags": ["easy", "toggle"],
     "check": {"all": [{"path": "settings.accessibility.high_contrast", "op": "eq", "value": True}]}},
    {"id": "fake_text_large", "request": "Make the text bigger, I can't read this.", "start_path": "/settings",
     "initial_state_overrides": {}, "steps_min": 1, "tags": ["easy", "segmented"],
     "check": {"all": [{"path": "settings.accessibility.text_size", "op": "eq", "value": "L"}]}},
    {"id": "fake_notifications", "request": "Show me my notifications.", "start_path": "/",
     "initial_state_overrides": {}, "steps_min": 1, "tags": ["easy", "icon_only"],
     "check": {"all": [{"path": "notifications.opened", "op": "eq", "value": True}]}},
    {"id": "fake_gear_high_contrast", "request": "I need high contrast turned on.", "start_path": "/",
     "initial_state_overrides": {}, "steps_min": 2, "tags": ["medium", "icon_only"],
     "check": {"all": [{"path": "settings.accessibility.high_contrast", "op": "eq", "value": True}]}},
    {"id": "fake_display_name", "request": "Change my display name to Alex Kim.", "start_path": "/settings",
     "initial_state_overrides": {}, "steps_min": 2, "tags": ["medium", "text_input"],
     "check": {"all": [{"path": "profile.name", "op": "eq", "value": "Alex Kim"}]}},
    {"id": "fake_already_contrast_off", "request": "Turn off high contrast.", "start_path": "/settings",
     "initial_state_overrides": {"settings.accessibility.high_contrast": True}, "steps_min": 1, "tags": ["easy"],
     "check": {"all": [{"path": "settings.accessibility.high_contrast", "op": "eq", "value": False}]}},
]
TASKS_BY_ID = {t["id"]: t for t in FAKE_TASKS}


# --- app ---------------------------------------------------------------------------------

_GEAR_SVG = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
    '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 '
    '1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 '
    '1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 '
    '2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 '
    '2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>'
)
_BELL_SVG = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
    '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>'
)

_CSS = """
* { box-sizing: border-box; transition: none !important; animation: none !important; }
body { margin: 0; font-family: Helvetica, Arial, sans-serif; background: #f4f6f8; color: #1d2733; }
body.hc { background: #000; color: #fff; }
body.hc .card, body.hc header { background: #000; border-color: #fff; color: #fff; }
body.size-S { font-size: 13px; } body.size-M { font-size: 16px; } body.size-L { font-size: 22px; }
header { height: 64px; background: #0b4f8a; color: #fff; display: flex; align-items: center; padding: 0 24px; }
header h1 { font-size: 22px; margin: 0; flex: 1; }
.icon-btn { background: none; border: none; color: inherit; cursor: pointer; padding: 8px; margin-left: 8px; }
main { padding: 32px 48px; }
.card { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px; width: 640px; }
.row { display: flex; align-items: center; justify-content: space-between; margin: 14px 0; }
.switch { width: 56px; height: 30px; border-radius: 15px; border: 2px solid #555; background: #ccc; position: relative; cursor: pointer; }
.switch::after { content: ""; position: absolute; top: 3px; left: 3px; width: 20px; height: 20px; border-radius: 10px; background: #fff; }
.switch.on { background: #1a7f37; } .switch.on::after { left: 29px; }
.seg button { padding: 6px 16px; border: 1px solid #555; background: #fff; cursor: pointer; font-size: inherit; }
.seg button.sel { background: #0b4f8a; color: #fff; }
input[type=text] { font-size: inherit; padding: 6px 8px; width: 260px; }
.primary { background: #0b4f8a; color: #fff; border: none; padding: 8px 18px; border-radius: 6px; cursor: pointer; font-size: inherit; }
#toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); background: #1d2733; color: #fff; padding: 10px 20px; border-radius: 6px; display: none; }
#notif { position: absolute; top: 64px; right: 60px; width: 300px; display: none; }
"""

_JS = """
async function act(action, value) {
  const r = await fetch('/api/action', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                       body: JSON.stringify({action: action, value: value})});
  const s = await r.json(); render(s); return s;
}
function toast(msg) { const t = document.getElementById('toast'); t.textContent = msg; t.style.display = 'block'; }
function render(s) {
  const a = s.settings.accessibility;
  document.body.className = (a.high_contrast ? 'hc ' : '') + 'size-' + a.text_size;
  const sw = document.getElementById('hc'); if (sw) { sw.classList.toggle('on', a.high_contrast); }
  document.querySelectorAll('.seg button').forEach(b => b.classList.toggle('sel', b.dataset.size === a.text_size));
  const nm = document.getElementById('who'); if (nm) nm.textContent = s.profile.name;
}
"""


def _page(title: str, body: str, state: dict) -> str:
    a = state["settings"]["accessibility"]
    cls = ("hc " if a["high_contrast"] else "") + "size-" + a["text_size"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{_CSS}</style></head>
<body class="{cls}"><header><h1>Acme Utilities</h1>
<button class="icon-btn" id="bell" onclick="document.getElementById('notif').style.display='block'; act('open_notifications')">{_BELL_SVG}</button>
<button class="icon-btn" id="gear" onclick="location.href='/settings'">{_GEAR_SVG}</button></header>
<div class="card" id="notif"><b>Notifications</b><p>Your September bill is ready.</p></div>
<main>{body}</main><div id="toast"></div><script>{_JS}</script></body></html>"""


def create_app() -> FastAPI:
    app = FastAPI()
    sessions: dict[str, dict] = {}

    def state_for(sid: str) -> dict:
        return sessions.setdefault(sid, copy.deepcopy(DEFAULT_STATE))

    def sid_of(request: Request) -> str:
        return request.query_params.get("sid") or request.cookies.get("sid") or "default"

    def html(request: Request, title: str, body: str) -> HTMLResponse:
        sid = sid_of(request)
        resp = HTMLResponse(_page(title, body, state_for(sid)))
        resp.set_cookie("sid", sid)
        return resp

    @app.get("/")
    def dashboard(request: Request):
        s = state_for(sid_of(request))
        return html(request, "Dashboard - Acme Utilities",
                    f"<h2>Welcome back, <span id='who'>{s['profile']['name']}</span></h2>"
                    "<div class='card'><h3>Current balance</h3><p style='font-size:2em'>$84.20</p>"
                    "<p>Due October 28</p></div>")

    @app.get("/settings")
    def settings(request: Request):
        s = state_for(sid_of(request))
        a = s["settings"]["accessibility"]
        seg = "".join(f"<button data-size='{z}' class='{'sel' if a['text_size'] == z else ''}' "
                      f"onclick=\"act('text_size','{z}')\">{z}</button>" for z in "SML")
        return html(request, "Settings - Acme Utilities", f"""<h2>Settings</h2>
<div class="card"><h3>Accessibility</h3>
<div class="row"><span>High contrast</span><div id="hc" class="switch {'on' if a['high_contrast'] else ''}"
  onclick="act('high_contrast', !this.classList.contains('on'))"></div></div>
<div class="row"><span>Text size</span><div class="seg">{seg}</div></div></div>
<div class="card"><h3>Profile</h3>
<div class="row"><span>Display name</span><input type="text" id="name" value="{s['profile']['name']}"></div>
<div class="row"><span></span><button class="primary" onclick="act('name', document.getElementById('name').value).then(() => toast('Profile saved'))">Save</button></div></div>""")

    @app.post("/api/action")
    def action(request: Request, body: dict = Body(...)):
        s = state_for(sid_of(request))
        kind, value = body.get("action"), body.get("value")
        if kind == "high_contrast":
            s["settings"]["accessibility"]["high_contrast"] = bool(value)
        elif kind == "text_size" and value in ("S", "M", "L"):
            s["settings"]["accessibility"]["text_size"] = value
        elif kind == "name" and isinstance(value, str):
            s["profile"]["name"] = value.strip()
        elif kind == "open_notifications":
            s["notifications"]["opened"] = True
        return JSONResponse(s)

    @app.post("/api/reset")
    def reset(body: dict = Body(...)):
        sid = body.get("sid") or "default"
        st = copy.deepcopy(DEFAULT_STATE)
        task = TASKS_BY_ID.get(body.get("task_id") or "")
        for path, value in (task or {}).get("initial_state_overrides", {}).items():
            cur = st
            *parents, leaf = path.split(".")
            for p in parents:
                cur = cur.setdefault(p, {})
            cur[leaf] = value
        sessions[sid] = st
        return {"ok": True}

    @app.get("/api/state")
    def state(sid: str):
        return state_for(sid)

    app.state.sessions = sessions
    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSandboxServer:
    """Runs the fake sandbox with uvicorn in a background thread: `with FakeSandboxServer() as url: ...`."""

    def __init__(self, port: int | None = None):
        import uvicorn

        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.app = create_app()
        self._server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> str:
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("fake sandbox did not start")
            time.sleep(0.02)
        return self.url

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


if __name__ == "__main__":  # manual look: python -m agent.testing.fake_sandbox
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8090)
