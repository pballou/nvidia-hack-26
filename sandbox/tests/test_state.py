"""State transitions, reset/overrides, and session isolation (HTTP)."""

import pytest
from fastapi.testclient import TestClient

from sandbox import state as st
from sandbox.app import app


@pytest.fixture
def s():
    return st.default_state()


def test_default_state_is_deterministic():
    assert st.default_state() == st.default_state()
    a = st.default_state()
    a["billing"]["downloaded"].append("x")
    assert st.default_state()["billing"]["downloaded"] == [], "no shared mutable seed data"


def test_usage_max_is_unique_july(s):
    kwh = [m["kwh"] for m in s["usage"]["months"]]
    assert kwh.count(max(kwh)) == 1
    assert max(s["usage"]["months"], key=lambda m: m["kwh"])["id"] == "2026-07"


def test_download_dedupes_and_validates(s):
    assert "September 2026" in st.apply_action(s, "billing.download", {"statement_id": "2026-09"})
    st.apply_action(s, "billing.download", {"statement_id": "2026-09"})
    st.apply_action(s, "billing.download", {"statement_id": "2026-08"})
    assert s["billing"]["downloaded"] == ["2026-09", "2026-08"]
    with pytest.raises(st.ActionError):
        st.apply_action(s, "billing.download", {"statement_id": "1999-01"})


def test_pay_updates_balance_and_validates(s):
    st.apply_action(s, "billing.pay", {"amount_cents": 5000})
    assert s["billing"]["balance_cents"] == 9237
    assert s["billing"]["payments"][0]["amount_cents"] == 5000
    assert s["activity"][0]["text"] == "Payment received: $50.00"
    for bad in (0, -5, 999999, True, "50", None):
        with pytest.raises(st.ActionError):
            st.apply_action(s, "billing.pay", {"amount_cents": bad})
    assert len(s["billing"]["payments"]) == 1


def test_autopay_requires_bool(s):
    st.apply_action(s, "billing.autopay", {"enabled": True})
    assert s["billing"]["autopay"] is True
    with pytest.raises(st.ActionError):
        st.apply_action(s, "billing.autopay", {"enabled": 1})


def test_usage_select(s):
    st.apply_action(s, "usage.select_month", {"month": "2026-07"})
    assert s["usage"]["selected_month"] == "2026-07"
    with pytest.raises(st.ActionError):
        st.apply_action(s, "usage.select_month", {"month": "2027-01"})


def test_outage_report(s):
    toast = st.apply_action(s, "outages.report", {"district": "riverside"})
    assert "Riverside" in toast
    r = s["outages"]["reports"][0]
    assert (r["district"], r["type"], r["details"], r["reference"]) == ("riverside", "no_power", "", "OUT-1001")
    with pytest.raises(st.ActionError):
        st.apply_action(s, "outages.report", {"district": "atlantis"})
    with pytest.raises(st.ActionError):
        st.apply_action(s, "outages.report", {"district": "riverside", "type": "aliens"})
    with pytest.raises(st.ActionError):  # a non-string type is a 400, not a TypeError/500
        st.apply_action(s, "outages.report", {"district": "riverside", "type": ["no_power"]})


def test_messages(s):
    st.apply_action(s, "messages.star", {"id": "msg-landlord", "starred": True})
    assert s["messages"]["inbox"]["msg-landlord"]["starred"] is True
    unread = [m for m in s["messages"]["inbox"].values() if not m["read"]]
    assert len(unread) == 4
    assert "4 messages" in st.apply_action(s, "messages.mark_all_read")
    assert all(m["read"] for m in s["messages"]["inbox"].values())
    assert st.mark_message_read(s, "msg-landlord") and not st.mark_message_read(s, "nope")


def test_send_message(s):
    st.apply_action(s, "messages.send", {"to": "support", "subject": " Meter ", "body": " My meter is broken "})
    assert s["messages"]["sent"] == [{"to": "support", "subject": "Meter", "body": "My meter is broken", "date": st.TODAY}]
    with pytest.raises(st.ActionError):
        st.apply_action(s, "messages.send", {"to": "nobody", "body": "x"})
    with pytest.raises(st.ActionError):
        st.apply_action(s, "messages.send", {"to": "support", "body": "   "})


def test_settings(s):
    st.apply_action(s, "settings.notification", {"category": "billing", "channel": "sms", "enabled": False})
    assert s["settings"]["notifications"]["billing"] == {"email": True, "sms": False, "push": False}
    st.apply_action(s, "settings.accessibility", {"key": "text_size", "value": "L"})
    st.apply_action(s, "settings.accessibility", {"key": "high_contrast", "value": True})
    assert s["settings"]["accessibility"] == {"high_contrast": True, "text_size": "L", "reduce_motion": False}
    for bad in ({"key": "text_size", "value": "XL"}, {"key": "high_contrast", "value": "yes"}, {"key": "zoom", "value": 2}):
        with pytest.raises(st.ActionError):
            st.apply_action(s, "settings.accessibility", bad)
    with pytest.raises(st.ActionError):
        st.apply_action(s, "settings.notification", {"category": "billing", "channel": "fax", "enabled": False})


def test_sign_in_greets_the_saved_profile_name(s):
    st.apply_action(s, "settings.profile", {"name": "Alex Kim", "email": "a@example.com", "phone": "555"})
    st.apply_action(s, "session.sign_out")
    assert st.apply_action(s, "session.sign_in") == "Welcome back, Alex"


def test_sign_out_blocks_actions(s):
    st.apply_action(s, "session.sign_out")
    with pytest.raises(st.ActionError):
        st.apply_action(s, "billing.autopay", {"enabled": True})
    st.apply_action(s, "session.sign_in")
    st.apply_action(s, "billing.autopay", {"enabled": True})


def test_history_and_flash(s):
    toast = st.apply_action(s, "billing.autopay", {"enabled": True})
    assert s["ui"]["flash"] == toast
    assert s["history"][-1] == {"type": "billing.autopay", "payload": {"enabled": True}}
    assert s["meta"]["action_count"] == 1
    with pytest.raises(st.ActionError):
        st.apply_action(s, "no.such.action")


def test_overrides_and_bad_override_paths():
    s = st.apply_overrides(st.default_state(), {"ui.nav_collapsed": True, "settings.accessibility.text_size": "S"})
    assert s["ui"]["nav_collapsed"] is True and s["settings"]["accessibility"]["text_size"] == "S"
    with pytest.raises(KeyError):
        st.apply_overrides(st.default_state(), {"ui.nav_colapsed": True})


# --- HTTP: reset, state, isolation --------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


def act(client, sid, type_, payload=None):
    r = client.post("/api/action", params={"sid": sid}, json={"type": type_, "payload": payload or {}})
    return r


def test_reset_applies_task_overrides(client):
    assert client.post("/api/reset", json={"sid": "t1", "task_id": "max-usage-month"}).json() == {"ok": True}
    state = client.get("/api/state", params={"sid": "t1"}).json()
    assert state["ui"]["nav_collapsed"] is True
    assert state["meta"] == {"sid": "t1", "task_id": "max-usage-month", "today": st.TODAY, "action_count": 0}
    assert client.post("/api/reset", json={"sid": "t1", "task_id": None}).json() == {"ok": True}
    assert client.get("/api/state", params={"sid": "t1"}).json()["ui"]["nav_collapsed"] is False


def test_reset_errors(client):
    assert client.post("/api/reset", json={"sid": "t1", "task_id": "nope"}).status_code == 404
    assert client.post("/api/reset", json={"sid": "bad sid!", "task_id": None}).status_code == 400
    assert client.post("/api/reset", json={"task_id": None}).status_code == 422


def test_reset_discards_changes(client):
    client.post("/api/reset", json={"sid": "t2", "task_id": None})
    assert act(client, "t2", "billing.autopay", {"enabled": True}).json()["ok"]
    assert client.get("/api/state", params={"sid": "t2"}).json()["billing"]["autopay"] is True
    client.post("/api/reset", json={"sid": "t2", "task_id": "turn-on-autopay"})
    assert client.get("/api/state", params={"sid": "t2"}).json()["billing"]["autopay"] is False


def test_sessions_are_isolated(client):
    for sid in ("iso-a", "iso-b"):
        client.post("/api/reset", json={"sid": sid, "task_id": None})
    act(client, "iso-a", "settings.accessibility", {"key": "high_contrast", "value": True})
    act(client, "iso-a", "billing.pay", {"amount_cents": 100})
    a = client.get("/api/state", params={"sid": "iso-a"}).json()
    b = client.get("/api/state", params={"sid": "iso-b"}).json()
    assert a["settings"]["accessibility"]["high_contrast"] is True and a["billing"]["balance_cents"] == 14137
    assert b["settings"]["accessibility"]["high_contrast"] is False and b["billing"]["balance_cents"] == 14237
    assert b["history"] == []


def test_action_errors_are_400_and_change_nothing(client):
    client.post("/api/reset", json={"sid": "err", "task_id": None})
    r = act(client, "err", "billing.pay", {"amount_cents": 10**9})
    assert r.status_code == 400 and r.json()["ok"] is False and "at most" in r.json()["error"]
    assert client.get("/api/state", params={"sid": "err"}).json()["billing"]["payments"] == []


def test_sid_query_sets_cookie(client):
    r = client.get("/api/health", params={"sid": "cookie-test"})
    assert r.cookies.get("sid") == "cookie-test"
    assert client.get("/api/health", params={"sid": "bad sid!"}).status_code == 400
