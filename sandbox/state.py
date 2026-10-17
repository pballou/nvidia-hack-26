"""Per-session state for the Acme Utilities sandbox.

Everything a verifier checks lives in the state dict returned by GET /api/state.
`default_state()` is fixed seed data (no clock, no randomness). `apply_action` is
the only way the UI mutates state; it returns the toast text the page shows.

Money is integer cents. Dates are ISO strings; "today" is fixed at TODAY.
"""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from typing import Any

TODAY = "2026-10-17"
TODAY_LABEL = "Saturday, October 17, 2026"

ACCOUNT = {
    "holder": "Jordan Lee",
    "initials": "JL",
    "number": "4417-2093-88",
    "service_address": "212 Maple Court, Apt 3B, Riverside",
    "payment_method": "Checking ••4821",
}

# Statements, newest first. id = billing month.
_STATEMENTS = [
    ("2026-09", "September 2026", "2026-10-01", 14237, "due"),
    ("2026-08", "August 2026", "2026-09-01", 15120, "paid"),
    ("2026-07", "July 2026", "2026-08-01", 16388, "paid"),
    ("2026-06", "June 2026", "2026-07-01", 13905, "paid"),
    ("2026-05", "May 2026", "2026-06-01", 10412, "paid"),
    ("2026-04", "April 2026", "2026-05-01", 9640, "paid"),
    ("2026-03", "March 2026", "2026-04-01", 10877, "paid"),
    ("2026-02", "February 2026", "2026-03-01", 12990, "paid"),
    ("2026-01", "January 2026", "2026-02-01", 14211, "paid"),
    ("2025-12", "December 2025", "2026-01-01", 13162, "paid"),
    ("2025-11", "November 2025", "2025-12-01", 11245, "paid"),
    ("2025-10", "October 2025", "2025-11-01", 10530, "paid"),
]

# Monthly usage, oldest first (chart order). July is the unique maximum; August is
# the runner-up so "the most power" is a real visual comparison.
_USAGE = [
    ("2025-10", "Oct", "October 2025", 612),
    ("2025-11", "Nov", "November 2025", 655),
    ("2025-12", "Dec", "December 2025", 801),
    ("2026-01", "Jan", "January 2026", 876),
    ("2026-02", "Feb", "February 2026", 790),
    ("2026-03", "Mar", "March 2026", 640),
    ("2026-04", "Apr", "April 2026", 548),
    ("2026-05", "May", "May 2026", 602),
    ("2026-06", "Jun", "June 2026", 834),
    ("2026-07", "Jul", "July 2026", 968),
    ("2026-08", "Aug", "August 2026", 902),
    ("2026-09", "Sep", "September 2026", 705),
]

DISTRICTS = [
    ("riverside", "Riverside"),
    ("hillcrest", "Hillcrest"),
    ("old_town", "Old Town"),
    ("maple_grove", "Maple Grove"),
    ("harbor_point", "Harbor Point"),
    ("northgate", "Northgate"),
]
DISTRICT_NAMES = dict(DISTRICTS)

OUTAGE_TYPES = {
    "no_power": "No power",
    "flickering": "Flickering lights",
    "downed_line": "Downed power line",
}

CONTACTS = {
    "support": "Customer Support",
    "billing": "Billing Department",
    "outages": "Outage Team",
    "field_services": "Field Services",
    "landlord": "Dana Whitfield (Landlord)",
}

HELP_TOPICS = {
    "billing": "Billing question",
    "outage": "Outage or safety",
    "account": "Account access",
    "other": "Something else",
}

# Inbox, newest first; the order never changes. Keys are message ids.
_INBOX = [
    ("msg-outage-hillcrest", "Acme Outage Team", "Power restored in Hillcrest", "2026-10-16", False, False,
     "Crews restored power to all customers in Hillcrest at 6:42 PM. Thank you for your patience."),
    ("msg-statement-sep", "Acme Billing", "Your September statement is ready", "2026-10-01", False, False,
     "Your September 2026 statement is ready. Amount due: $142.37 by October 28, 2026."),
    ("msg-autopay-promo", "Acme Billing", "Never miss a due date: try Autopay", "2026-09-28", False, False,
     "Turn on Autopay in Billing and we will pay your bill on each due date."),
    ("msg-usage-alert", "Acme Energy Insights", "Your usage is 22% lower than last month", "2026-09-25", True, False,
     "You used 705 kWh in September, down from 902 kWh in August."),
    ("msg-support-reply", "Customer Support", "Re: Question about my meter reading", "2026-09-22", True, True,
     "Thanks for reaching out. Your August reading was an actual reading, not an estimate."),
    ("msg-tree-trimming", "Acme Outage Team", "Scheduled tree trimming near your street", "2026-09-19", False, False,
     "Crews will trim trees near Maple Court on October 21. Short outages are possible."),
    ("msg-payment-received", "Acme Billing", "Payment received. Thank you!", "2026-09-18", True, False,
     "We received your payment of $151.20 on September 18, 2026."),
    ("msg-rebate", "Acme Energy Insights", "You qualify for a smart thermostat rebate", "2026-09-15", True, False,
     "Get $75 back when you install a qualifying smart thermostat."),
    ("msg-survey", "Acme Utilities", "How did we do? 2-minute survey", "2026-09-12", True, False,
     "Tell us about your recent support call."),
    ("msg-heat-advisory", "Acme Outage Team", "Heat advisory: tips to save energy", "2026-09-08", True, False,
     "Set your thermostat to 78°F when you are home and close blinds in the afternoon."),
    ("msg-landlord", "Dana Whitfield (Landlord)", "Water heater replacement on September 20", "2026-09-05", True, False,
     "Hi Jordan, a plumber will replace the water heater in 3B on September 20 between 9 and 11 AM. - Dana"),
    ("msg-paperless", "Acme Billing", "You're now on paperless billing", "2026-08-30", True, False,
     "You will get statements by email instead of paper."),
    ("msg-welcome-portal", "Acme Utilities", "Welcome to the new customer portal", "2026-08-21", True, False,
     "Pay bills, track usage and report outages in one place."),
    ("msg-meter-upgrade", "Acme Field Services", "Smart meter upgrade complete", "2026-08-14", True, False,
     "Your new smart meter is installed. Readings now update daily."),
]

NOTIFICATION_CATEGORIES = [
    ("billing", "Billing alerts", "New statements and upcoming due dates"),
    ("payments", "Payment confirmations", "Receipts when a payment goes through"),
    ("outages", "Outage alerts", "Outages and restorations at your address"),
    ("usage", "Usage insights", "Monthly usage summaries and spikes"),
    ("messages", "New messages", "When someone sends you a message"),
    ("service", "Service updates", "Planned maintenance and meter work"),
    ("promotions", "Offers and promotions", "Rebates, programs and special offers"),
    ("surveys", "Surveys and feedback", "Short surveys about your experience"),
]
CHANNELS = [("email", "Email"), ("sms", "SMS"), ("push", "Push")]
_NOTIFICATION_DEFAULTS = {
    "billing": (True, True, False),
    "payments": (True, False, True),
    "outages": (True, True, True),
    "usage": (True, False, False),
    "messages": (True, False, True),
    "service": (True, False, False),
    "promotions": (True, True, True),
    "surveys": (True, False, False),
}

TEXT_SIZES = ("S", "M", "L")


def default_state() -> dict:
    """Fresh seed state. Deterministic: same output on every call."""
    return {
        "meta": {"sid": None, "task_id": None, "today": TODAY, "action_count": 0},
        "session": {"signed_in": True},
        "ui": {"nav_collapsed": False, "flash": None},
        "account": dict(ACCOUNT),
        "billing": {
            "balance_cents": 14237,
            "due_date": "2026-10-28",
            "autopay": False,
            "statements": [
                {"id": sid, "label": label, "issued": issued, "amount_cents": cents, "status": status}
                for sid, label, issued, cents, status in _STATEMENTS
            ],
            "downloaded": [],
            "payments": [],
        },
        "usage": {
            "months": [
                {"id": mid, "short": short, "label": label, "kwh": kwh} for mid, short, label, kwh in _USAGE
            ],
            "selected_month": None,
        },
        "outages": {
            "districts": [{"id": did, "name": name} for did, name in DISTRICTS],
            "active": [
                {"district": "harbor_point", "status": "Crews on site", "since": "2026-10-17", "customers": 214},
            ],
            "reports": [],
        },
        "messages": {
            "inbox": {
                mid: {"id": mid, "from": sender, "subject": subject, "date": date, "read": read,
                      "starred": starred, "body": body}
                for mid, sender, subject, date, read, starred, body in _INBOX
            },
            "sent": [],
        },
        "settings": {
            "profile": {
                "name": ACCOUNT["holder"],
                "email": "jordan.lee@example.com",
                "phone": "(919) 555-0142",
            },
            "notifications": {
                cat: {"email": e, "sms": s, "push": p} for cat, (e, s, p) in _NOTIFICATION_DEFAULTS.items()
            },
            "accessibility": {"high_contrast": False, "text_size": "M", "reduce_motion": False},
        },
        "notifications": {
            "items": [
                {"id": "n-statement", "text": "Your September statement is ready", "href": "/billing", "read": False},
                {"id": "n-outage", "text": "Outage in Harbor Point: crews on site", "href": "/outages", "read": False},
                {"id": "n-message", "text": "New message from Acme Outage Team", "href": "/messages", "read": False},
            ],
        },
        "help": {"contact_requests": []},
        "activity": [
            {"date": "2026-10-16", "text": "Power restored in Hillcrest"},
            {"date": "2026-10-01", "text": "September statement issued: $142.37"},
            {"date": "2026-09-18", "text": "Payment received: $151.20"},
            {"date": "2026-09-02", "text": "August statement issued: $151.20"},
            {"date": "2026-08-14", "text": "Smart meter upgrade complete"},
        ],
        "history": [],
    }


class ActionError(ValueError):
    """Invalid action from the UI. The message is shown to the user."""


def money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


# --- overrides -----------------------------------------------------------------


def set_path(state: dict, path: str, value: Any) -> None:
    """Set a dotted path that must already exist (catches typos in tasks.json)."""
    segs = path.split(".")
    cur: Any = state
    for seg in segs[:-1]:
        cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
    last = segs[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        if last not in cur:
            raise KeyError(f"override path not in state: {path}")
        cur[last] = value


def apply_overrides(state: dict, overrides: dict) -> dict:
    for path, value in (overrides or {}).items():
        set_path(state, path, copy.deepcopy(value))
    return state


# --- actions -------------------------------------------------------------------


def _need(payload: dict, key: str, kind: type) -> Any:
    val = payload.get(key)
    # bool is a subclass of int; don't let true pass as an amount.
    if not isinstance(val, kind) or (kind is not bool and isinstance(val, bool)):
        raise ActionError(f"'{key}' is required")
    return val


def _activity(state: dict, text: str) -> None:
    state["activity"].insert(0, {"date": TODAY, "text": text})


def _nav_toggle(state, p):
    state["ui"]["nav_collapsed"] = not state["ui"]["nav_collapsed"]
    return None  # visible change is the nav itself


def _billing_download(state, p):
    sid = _need(p, "statement_id", str)
    stmt = next((s for s in state["billing"]["statements"] if s["id"] == sid), None)
    if stmt is None:
        raise ActionError("Statement not found")
    if sid not in state["billing"]["downloaded"]:
        state["billing"]["downloaded"].append(sid)
    return f"Downloaded {stmt['label']} statement (PDF)"


def _billing_autopay(state, p):
    on = _need(p, "enabled", bool)
    state["billing"]["autopay"] = on
    if on:
        _activity(state, "Autopay turned on")
        return f"Autopay is on. We'll pay from {ACCOUNT['payment_method']} on each due date."
    _activity(state, "Autopay turned off")
    return "Autopay is off"


def _billing_pay(state, p):
    amount = _need(p, "amount_cents", int)
    bal = state["billing"]["balance_cents"]
    if amount <= 0:
        raise ActionError("Enter an amount greater than $0.00")
    if amount > bal:
        raise ActionError(f"You can pay at most {money(bal)}")
    state["billing"]["balance_cents"] = bal - amount
    n = len(state["billing"]["payments"]) + 1
    conf = f"PAY-{1000 + n}"
    state["billing"]["payments"].append(
        {"amount_cents": amount, "method": ACCOUNT["payment_method"], "date": TODAY, "confirmation": conf}
    )
    _activity(state, f"Payment received: {money(amount)}")
    return f"Payment of {money(amount)} received. Confirmation {conf}."


def _usage_select(state, p):
    mid = _need(p, "month", str)
    month = next((m for m in state["usage"]["months"] if m["id"] == mid), None)
    if month is None:
        raise ActionError("Month not found")
    state["usage"]["selected_month"] = mid
    return None  # the details panel is the visible change


def _outage_report(state, p):
    district = _need(p, "district", str)
    if district not in DISTRICT_NAMES:
        raise ActionError("Choose a district")
    otype = p.get("type") or "no_power"
    if otype not in OUTAGE_TYPES:
        raise ActionError("Choose what's happening")
    details = p.get("details") or ""
    if not isinstance(details, str):
        raise ActionError("'details' must be text")
    reports = state["outages"]["reports"]
    ref = f"OUT-{1001 + len(reports)}"
    reports.append({"district": district, "type": otype, "details": details.strip(), "date": TODAY, "reference": ref})
    _activity(state, f"Outage reported in {DISTRICT_NAMES[district]}")
    return f"Outage reported in {DISTRICT_NAMES[district]}. Reference {ref}."


def _msg(state, p):
    mid = _need(p, "id", str)
    msg = state["messages"]["inbox"].get(mid)
    if msg is None:
        raise ActionError("Message not found")
    return msg


def _messages_star(state, p):
    msg = _msg(state, p)
    on = _need(p, "starred", bool)
    msg["starred"] = on
    return f"{'Starred' if on else 'Unstarred'}: {msg['subject']}"


def _messages_mark_all_read(state, p):
    n = 0
    for msg in state["messages"]["inbox"].values():
        if not msg["read"]:
            msg["read"] = True
            n += 1
    return f"Marked {n} message{'s' if n != 1 else ''} as read" if n else "All messages are already read"


def mark_message_read(state: dict, mid: str) -> bool:
    """Opening a message marks it read. Returns False if the id is unknown."""
    msg = state["messages"]["inbox"].get(mid)
    if msg is None:
        return False
    msg["read"] = True
    return True


def _messages_send(state, p):
    to = _need(p, "to", str)
    if to not in CONTACTS:
        raise ActionError("Choose who to send this to")
    subject = p.get("subject") or ""
    body = p.get("body") or ""
    if not isinstance(subject, str) or not isinstance(body, str):
        raise ActionError("Subject and message must be text")
    if not body.strip():
        raise ActionError("Write a message before sending")
    state["messages"]["sent"].append(
        {"to": to, "subject": subject.strip(), "body": body.strip(), "date": TODAY}
    )
    _activity(state, f"Message sent to {CONTACTS[to]}")
    return f"Message sent to {CONTACTS[to]}"


def _settings_notification(state, p):
    cat = _need(p, "category", str)
    channel = _need(p, "channel", str)
    on = _need(p, "enabled", bool)
    prefs = state["settings"]["notifications"]
    if cat not in prefs or channel not in prefs[cat]:
        raise ActionError("Unknown notification setting")
    prefs[cat][channel] = on
    label = dict((c, l) for c, l, _ in NOTIFICATION_CATEGORIES)[cat]
    chan = dict(CHANNELS)[channel]
    return f"{chan} for {label} turned {'on' if on else 'off'}"


def _settings_accessibility(state, p):
    key = _need(p, "key", str)
    acc = state["settings"]["accessibility"]
    if key == "text_size":
        val = _need(p, "value", str)
        if val not in TEXT_SIZES:
            raise ActionError("Text size must be S, M or L")
        acc["text_size"] = val
        return f"Text size set to {dict(S='Small', M='Medium', L='Large')[val]}"
    if key in ("high_contrast", "reduce_motion"):
        on = _need(p, "value", bool)
        acc[key] = on
        name = "High contrast" if key == "high_contrast" else "Reduce motion"
        return f"{name} turned {'on' if on else 'off'}"
    raise ActionError("Unknown accessibility setting")


def _settings_profile(state, p):
    prof = state["settings"]["profile"]
    for key in ("name", "email", "phone"):
        val = p.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ActionError(f"{key.capitalize()} can't be empty")
        prof[key] = val.strip()
    return "Profile saved"


def _help_contact(state, p):
    topic = _need(p, "topic", str)
    if topic not in HELP_TOPICS:
        raise ActionError("Choose a topic")
    message = p.get("message") or ""
    if not isinstance(message, str) or not message.strip():
        raise ActionError("Tell us how we can help")
    reqs = state["help"]["contact_requests"]
    ref = f"HELP-{2001 + len(reqs)}"
    reqs.append({"topic": topic, "message": message.strip(), "date": TODAY, "reference": ref})
    return f"Thanks! We'll reply within 1 business day. Reference {ref}."


def _notifications_mark_read(state, p):
    for item in state["notifications"]["items"]:
        item["read"] = True
    return "Notifications marked as read"


def _session_sign_out(state, p):
    state["session"]["signed_in"] = False
    return None


def _session_sign_in(state, p):
    state["session"]["signed_in"] = True
    return f"Welcome back, {ACCOUNT['holder'].split()[0]}"


ACTIONS = {
    "nav.toggle": _nav_toggle,
    "billing.download": _billing_download,
    "billing.autopay": _billing_autopay,
    "billing.pay": _billing_pay,
    "usage.select_month": _usage_select,
    "outages.report": _outage_report,
    "messages.star": _messages_star,
    "messages.mark_all_read": _messages_mark_all_read,
    "messages.send": _messages_send,
    "settings.notification": _settings_notification,
    "settings.accessibility": _settings_accessibility,
    "settings.profile": _settings_profile,
    "help.contact": _help_contact,
    "notifications.mark_read": _notifications_mark_read,
    "session.sign_out": _session_sign_out,
    "session.sign_in": _session_sign_in,
}


def apply_action(state: dict, action_type: str, payload: dict | None = None) -> str | None:
    """Apply one UI action in place. Returns toast text (or None). Raises ActionError."""
    payload = payload or {}
    fn = ACTIONS.get(action_type)
    if fn is None:
        raise ActionError(f"Unknown action {action_type!r}")
    if action_type != "session.sign_in" and not state["session"]["signed_in"]:
        raise ActionError("You're signed out")
    toast = fn(state, payload)
    state["meta"]["action_count"] += 1
    state["history"].append({"type": action_type, "payload": copy.deepcopy(payload)})
    state["ui"]["flash"] = toast
    return toast


# --- session store -------------------------------------------------------------


class SessionStore:
    """In-memory state per sid. Thread-safe; evicts the oldest sid past `cap`."""

    def __init__(self, cap: int = 2048):
        self._lock = threading.RLock()
        self._states: OrderedDict[str, dict] = OrderedDict()
        self.cap = cap

    def reset(self, sid: str, overrides: dict | None = None, task_id: str | None = None) -> dict:
        state = apply_overrides(default_state(), overrides or {})
        state["meta"]["sid"] = sid
        state["meta"]["task_id"] = task_id
        with self._lock:
            self._states[sid] = state
            self._states.move_to_end(sid)
            while len(self._states) > self.cap:
                self._states.popitem(last=False)
        return state

    def get(self, sid: str) -> dict:
        """The live state for sid, created with defaults on first use."""
        with self._lock:
            state = self._states.get(sid)
            if state is None:
                return self.reset(sid)
            self._states.move_to_end(sid)
            return state

    def lock(self) -> threading.RLock:
        return self._lock

    def snapshot(self, sid: str) -> dict:
        with self._lock:
            return copy.deepcopy(self.get(sid))
