"""Acme Utilities: the sandbox target app (docs/tasks/T1-app.md, contracts.md §1).

Run: uvicorn sandbox.app:app --host 127.0.0.1 --port 8090

Sessions: the first request carries ?sid=<sid>; the server sets a `sid` cookie and
later navigation uses it. Each sid has isolated in-memory state (sandbox/state.py).
Pages are server-rendered from that state. Every mutation goes through
POST /api/action, then the page reloads and shows the toast ("flash").

/api/state is for verifiers only; the agent never calls it.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sandbox import state as st

HERE = Path(__file__).resolve().parent
TASKS_PATH = HERE / "tasks.json"
DEFAULT_SID = "default"
SID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
NAV = [
    ("dashboard", "/", "Dashboard"),
    ("billing", "/billing", "Billing"),
    ("usage", "/usage", "Usage"),
    ("outages", "/outages", "Outage map"),
    ("messages", "/messages", "Messages"),
    ("settings", "/settings", "Settings"),
    ("help", "/help", "Help"),
]


def load_tasks(path: Path = TASKS_PATH) -> list[dict]:
    return json.loads(path.read_text())


TASKS = load_tasks()
TASKS_BY_ID = {t["id"]: t for t in TASKS}

store = st.SessionStore()
app = FastAPI(title="Acme Utilities sandbox", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


def fmt_date(iso: str, style: str = "short") -> str:
    """'2026-10-01' -> 'Oct 1, 2026' (short) or 'October 1, 2026' (long)."""
    d = date.fromisoformat(iso)
    month = d.strftime("%B" if style == "long" else "%b")
    return f"{month} {d.day}, {d.year}"


templates.env.globals.update(
    money=st.money,
    fmt_date=fmt_date,
    NAV=NAV,
    TODAY=st.TODAY,
    TODAY_LABEL=st.TODAY_LABEL,
    ACCOUNT=st.ACCOUNT,
    CONTACTS=st.CONTACTS,
    DISTRICTS=st.DISTRICTS,
    DISTRICT_NAMES=st.DISTRICT_NAMES,
    OUTAGE_TYPES=st.OUTAGE_TYPES,
    HELP_TOPICS=st.HELP_TOPICS,
    NOTIFICATION_CATEGORIES=st.NOTIFICATION_CATEGORIES,
    CHANNELS=st.CHANNELS,
    TEXT_SIZES=st.TEXT_SIZES,
)
templates.env.filters["tojson_attr"] = lambda v: json.dumps(v, separators=(",", ":"))


@app.middleware("http")
async def session_cookie(request: Request, call_next):
    q = request.query_params.get("sid")
    c = request.cookies.get("sid")
    sid = q or c or DEFAULT_SID
    if not SID_RE.match(sid):
        return JSONResponse({"ok": False, "error": "invalid sid"}, status_code=400)
    request.state.sid = sid
    response = await call_next(request)
    if q and q != c:
        response.set_cookie("sid", q, httponly=True, samesite="lax", path="/")
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-store"
    return response


def render(request: Request, template: str, nav: str, status_code: int = 200, **ctx) -> HTMLResponse:
    """Render a page from a snapshot of the session state; consumes the flash toast."""
    with store.lock():
        live = store.get(request.state.sid)
        if not live["session"]["signed_in"]:
            template, nav = "signed_out.html", ""
        flash = live["ui"]["flash"]
        live["ui"]["flash"] = None
        state = copy.deepcopy(live)
    context = {"state": state, "nav": nav, "flash": flash, "query": dict(request.query_params), **ctx}
    return templates.TemplateResponse(request, template, context, status_code=status_code)


# --- pages -------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return render(request, "dashboard.html", "dashboard")


@app.get("/billing", response_class=HTMLResponse)
def billing(request: Request):
    return render(request, "billing.html", "billing")


@app.get("/usage", response_class=HTMLResponse)
def usage(request: Request):
    return render(request, "usage.html", "usage")


@app.get("/outages", response_class=HTMLResponse)
def outages(request: Request):
    return render(request, "outages.html", "outages")


@app.get("/messages", response_class=HTMLResponse)
def messages(request: Request):
    tab = request.query_params.get("tab", "inbox")
    return render(request, "messages.html", "messages", tab=tab if tab in ("inbox", "sent") else "inbox")


@app.get("/messages/compose", response_class=HTMLResponse)
def compose(request: Request):
    return render(request, "compose.html", "messages")


@app.get("/messages/{mid}", response_class=HTMLResponse)
def message(request: Request, mid: str):
    with store.lock():
        state = store.get(request.state.sid)
        found = state["session"]["signed_in"] and st.mark_message_read(state, mid)
    if not found:
        return render(request, "not_found.html", "messages", status_code=404, what="That message")
    return render(request, "message.html", "messages", mid=mid)


@app.get("/settings", response_class=HTMLResponse)
def settings(request: Request):
    tab = request.query_params.get("tab", "profile")
    tabs = ("profile", "notifications", "accessibility")
    return render(request, "settings.html", "settings", tab=tab if tab in tabs else "profile")


@app.get("/help", response_class=HTMLResponse)
def help_page(request: Request):
    return render(request, "help.html", "help")


# --- API -----------------------------------------------------------------------


class ActionBody(BaseModel):
    type: str
    payload: dict = {}


class ResetBody(BaseModel):
    sid: str
    task_id: str | None = None


@app.post("/api/action")
def action(request: Request, body: ActionBody):
    """The UI's single mutation endpoint (session from the cookie)."""
    with store.lock():
        state = store.get(request.state.sid)
        try:
            toast = st.apply_action(state, body.type, body.payload)
        except st.ActionError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True, "toast": toast}


@app.post("/api/reset")
def reset(body: ResetBody):
    if not SID_RE.match(body.sid):
        return JSONResponse({"ok": False, "error": "invalid sid"}, status_code=400)
    task = None
    if body.task_id is not None:
        task = TASKS_BY_ID.get(body.task_id)
        if task is None:
            return JSONResponse({"ok": False, "error": f"unknown task_id {body.task_id!r}"}, status_code=404)
    store.reset(body.sid, task["initial_state_overrides"] if task else {}, body.task_id)
    return {"ok": True}


@app.get("/api/state")
def get_state(request: Request, sid: str | None = None):
    """Full session state, for verifiers only. Never called by the agent."""
    return store.snapshot(sid or request.state.sid)


@app.get("/api/health")
def health():
    return {"ok": True, "app": "acme-utilities-sandbox", "tasks": len(TASKS)}
