"""Scripted oracle: every task in tasks.json is solvable through the UI and verifiable.

For each task, from a fresh reset at its start_path:
- solve: follow the minimal path a real user would take, with real clicks, typing and
  mouse-wheel scrolls on visible elements. After EVERY step the 1280x800 viewport
  screenshot must change (decoded pixels, not PNG bytes): that is the vision agent's
  "screen changed" signal. The step count must reach the task's steps_min, and then
  evaluate(state, task.check) must pass.
- wrong: perform one plausible wrong action through the UI, confirm it took effect,
  and assert evaluate() fails. This proves the checks are strict.

This is the ONLY place that drives the app with DOM selectors (data-testid and
window.AcmeTestHooks). The agent never does. No page may raise a JS error or open a
native dialog along the way.
"""

from __future__ import annotations

import io
from typing import Callable

import pytest
from PIL import Image, ImageChops

from sandbox.app import TASKS, TASKS_BY_ID
from sandbox.checks import evaluate

VIEW_W, VIEW_H = 1280, 800

# UI steps the oracle takes per task (each one changes the screen). Asserted exactly, and
# each must be >= the task's steps_min, so a path that shrinks or grows is noticed.
ORACLE_STEPS = {
    "text-size-large": 1,
    "download-september-statement": 1,
    "mark-all-read": 1,
    "turn-on-autopay": 1,
    "high-contrast": 3,
    "sms-billing-off": 2,
    "max-usage-month": 3,
    "report-outage-riverside": 2,
    "star-landlord-message": 2,
    "pay-50": 5,
    "message-support-meter": 6,
    "stop-promotions": 6,
}


def _benign_console(text: str) -> bool:
    """Chromium logs every non-2xx fetch (e.g. an intentional 400 from /api/action)."""
    return text.startswith("Failed to load resource")


class Driver:
    """Runs UI steps on one page and asserts each step visibly changes the viewport."""

    def __init__(self, page):
        self.page = page
        self.steps: list[str] = []
        self.dialogs: list[str] = []
        page.on("dialog", self._on_dialog)

    def _on_dialog(self, dialog):
        self.dialogs.append(f"{dialog.type}: {dialog.message}")
        dialog.dismiss()

    # -- screenshots ---------------------------------------------------------------

    def shot(self) -> Image.Image:
        img = Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        assert img.size == (VIEW_W, VIEW_H), img.size
        return img

    def _step(self, label: str, action: Callable[[], None], nav: bool) -> None:
        before = self.shot()
        if nav:
            with self.page.expect_navigation():
                action()
        else:
            action()
        self.page.wait_for_load_state("load")
        after = self.shot()
        n = len(self.steps) + 1
        assert ImageChops.difference(before, after).getbbox() is not None, (
            f"step {n} ({label}) did not change the screen; steps so far: {self.steps}")
        self.steps.append(label)

    # -- geometry helpers ------------------------------------------------------------

    def box(self, testid: str) -> dict:
        b = self.page.get_by_test_id(testid).bounding_box()
        assert b is not None, f"{testid} is not rendered"
        return b

    def in_view(self, testid: str) -> bool:
        b = self.box(testid)
        return b["x"] >= 0 and b["y"] >= 0 and b["x"] + b["width"] <= VIEW_W and b["y"] + b["height"] <= VIEW_H

    def below_fold(self, testid: str) -> bool:
        return self.box(testid)["y"] >= VIEW_H

    def scroll_y(self) -> float:
        return self.page.evaluate("window.scrollY")

    # -- UI steps ------------------------------------------------------------------

    def click(self, testid: str, nav: bool = False, on_screen: bool = False) -> None:
        """Click a visible element. on_screen=True: it must already be fully in the viewport
        (scroll tasks, where Playwright's auto-scroll would hide a missing scroll step)."""
        loc = self.page.get_by_test_id(testid)
        assert loc.count() == 1, f"expected one {testid}, found {loc.count()}"
        assert loc.is_visible(), f"{testid} is not visible"
        if on_screen:
            assert self.in_view(testid), f"{testid} is not fully on screen at scrollY {self.scroll_y()}"
        self._step(f"click {testid}", loc.click, nav)

    def type(self, text: str) -> None:
        self._step(f"type {text!r}", lambda: self.page.keyboard.type(text), nav=False)

    def click_canvas(self, canvas_testid: str, hook: str, target_id: str, nav: bool = False) -> None:
        """Click a canvas hit target. Hook points are canvas-relative CSS px."""
        b = self.box(canvas_testid)
        points = self.page.evaluate(f"window.AcmeTestHooks.{hook}")
        pt = next((p for p in points if p["id"] == target_id), None)
        assert pt is not None, f"{target_id} not in AcmeTestHooks.{hook}: {points}"
        x, y = b["x"] + pt["x"], b["y"] + pt["y"]
        assert 0 <= x < VIEW_W and 0 <= y < VIEW_H, f"{target_id} at ({x}, {y}) is off screen"
        self._step(f"click {canvas_testid}:{target_id}", lambda: self.page.mouse.click(x, y), nav)

    def wheel(self, dy: int, reveal: list[str]) -> None:
        """Scroll with the mouse wheel over the content until `reveal` is fully on screen."""
        for t in reveal:
            assert self.below_fold(t), f"{t} should start below the fold"
        start = self.scroll_y()
        max_y = self.page.evaluate("document.documentElement.scrollHeight - window.innerHeight")
        want = min(start + dy, max_y)

        def act():
            self.page.mouse.move(VIEW_W * 0.6, VIEW_H * 0.6)
            self.page.mouse.wheel(0, dy)
            self.page.wait_for_function("(y) => Math.abs(window.scrollY - y) < 1", arg=want)

        self._step(f"wheel {dy}", act, nav=False)
        for t in reveal:
            assert self.in_view(t), f"{t} still not fully visible after scrolling to {self.scroll_y()}"

    # -- end-of-test checks ------------------------------------------------------------

    def assert_clean(self) -> None:
        errors = [str(e) for e in self.page.page_errors(filter="all")]
        console = [m.text for m in self.page.console_messages(filter="all")
                   if m.type == "error" and not _benign_console(m.text)]
        assert not errors, f"page errors: {errors}"
        assert not console, f"console errors: {console}"
        assert not self.dialogs, f"native dialogs opened: {self.dialogs}"


# ---- oracle solutions ----------------------------------------------------------------


def solve_text_size_large(d: Driver):
    d.click("a11y-text-size-L", nav=True)


def solve_download_september(d: Driver):
    d.click("download-2026-09", nav=True)  # unlabeled icon button in the September row


def solve_mark_all_read(d: Driver):
    d.click("mark-all-read", nav=True)  # unlabeled double-check icon


def solve_turn_on_autopay(d: Driver):
    d.click("autopay-toggle", nav=True)


def solve_high_contrast(d: Driver):
    d.click("topbar-settings", nav=True)  # unlabeled gear
    d.click("tab-accessibility", nav=True)
    d.click("a11y-high-contrast", nav=True)


def solve_sms_billing_off(d: Driver):
    d.click("tab-notifications", nav=True)
    d.click("notif-billing-sms", nav=True)


def _open_usage_from_collapsed_nav(d: Driver):
    assert d.page.get_by_test_id("nav-usage").count() == 0, "the side nav should start collapsed"
    d.click("topbar-menu", nav=True)  # unlabeled hamburger
    d.click("nav-usage", nav=True)


def solve_max_usage_month(d: Driver):
    _open_usage_from_collapsed_nav(d)
    d.click_canvas("usage-chart", "usageBars", "2026-07", nav=True)


def _report_outage(d: Driver, district: str, name: str):
    d.click_canvas("outage-map", "districts", district)  # opens the report dialog
    assert d.page.get_by_test_id("report-title").inner_text() == f"Report an outage in {name}"
    d.click("report-submit", nav=True)


def solve_report_riverside(d: Driver):
    _report_outage(d, "riverside", "Riverside")


def solve_star_landlord(d: Driver):
    d.wheel(400, reveal=["star-msg-landlord"])
    d.click("star-msg-landlord", nav=True, on_screen=True)  # unlabeled star


def _pay(d: Driver, start: str, amount: str | None):
    d.click(start, nav=start == "dash-pay")  # dash-pay goes to /billing?pay=1; the dialog opens on load
    assert d.page.get_by_test_id("pay-dialog").is_visible()
    if amount is not None:
        d.click("pay-amount")  # focusing the field selects "Other amount"
        d.type(amount)
    d.click("pay-continue")
    assert d.page.get_by_test_id("confirm-dialog").is_visible()
    d.click("pay-confirm", nav=True)


def solve_pay_50(d: Driver):
    _pay(d, "dash-pay", "50")
    assert d.page.get_by_test_id("confirm-dialog").is_hidden()


def _send_message(d: Driver, to: str, body: str):
    d.click("quick-message", nav=True)
    d.click("compose-to")  # in-page picker
    d.click(f"compose-to-{to}")
    d.click("compose-body")
    d.type(body)
    d.click("compose-send", nav=True)
    assert d.page.url.endswith("/messages?tab=sent"), d.page.url


def solve_message_support(d: Driver):
    _send_message(d, "support", "My meter is broken")


PROMO = ["notif-promotions-email", "notif-promotions-sms", "notif-promotions-push"]


def _promotions(d: Driver, switches: list[str]):
    d.click("topbar-settings", nav=True)  # unlabeled gear
    d.click("tab-notifications", nav=True)
    d.wheel(400, reveal=PROMO)
    for s in switches:
        d.click(s, nav=True, on_screen=True)  # each reload keeps the scroll position


def solve_stop_promotions(d: Driver):
    _promotions(d, PROMO)


SOLVERS = {
    "text-size-large": solve_text_size_large,
    "download-september-statement": solve_download_september,
    "mark-all-read": solve_mark_all_read,
    "turn-on-autopay": solve_turn_on_autopay,
    "high-contrast": solve_high_contrast,
    "sms-billing-off": solve_sms_billing_off,
    "max-usage-month": solve_max_usage_month,
    "report-outage-riverside": solve_report_riverside,
    "star-landlord-message": solve_star_landlord,
    "pay-50": solve_pay_50,
    "message-support-meter": solve_message_support,
    "stop-promotions": solve_stop_promotions,
}


# ---- wrong actions -------------------------------------------------------------------
# Each entry: (id suffix, UI actions, proof that the wrong action took effect).


def _wrong_max_usage(d: Driver):
    _open_usage_from_collapsed_nav(d)
    d.click_canvas("usage-chart", "usageBars", "2026-08", nav=True)  # the runner-up


def _wrong_star(d: Driver):
    d.wheel(400, reveal=["star-msg-paperless"])
    d.click("star-msg-paperless", nav=True, on_screen=True)  # the message just below the landlord's


def _wrong_high_contrast(d: Driver):
    d.click("topbar-settings", nav=True)
    d.click("tab-accessibility", nav=True)
    d.click("a11y-reduce-motion", nav=True)


def _wrong_sms(d: Driver):
    d.click("tab-notifications", nav=True)
    d.click("notif-billing-email", nav=True)


def inbox(s: dict, mid: str) -> dict:
    return s["messages"]["inbox"][mid]


def notif(s: dict, cat: str) -> dict:
    return s["settings"]["notifications"][cat]


WRONG = {
    "text-size-large": (
        "picks-small",
        lambda d: d.click("a11y-text-size-S", nav=True),
        lambda s: s["settings"]["accessibility"]["text_size"] == "S",
    ),
    "download-september-statement": (
        "downloads-august",
        lambda d: d.click("download-2026-08", nav=True),
        lambda s: s["billing"]["downloaded"] == ["2026-08"],
    ),
    "mark-all-read": (
        "opens-one-message",
        lambda d: d.click("msg-row-msg-outage-hillcrest", nav=True),  # the row centre is the link
        lambda s: inbox(s, "msg-outage-hillcrest")["read"] and not inbox(s, "msg-statement-sep")["read"],
    ),
    "turn-on-autopay": (
        "pays-full-balance",
        lambda d: _pay(d, "pay-open", None),
        lambda s: [p["amount_cents"] for p in s["billing"]["payments"]] == [14237] and not s["billing"]["autopay"],
    ),
    "high-contrast": (
        "toggles-reduce-motion",
        _wrong_high_contrast,
        lambda s: s["settings"]["accessibility"]["reduce_motion"] and not s["settings"]["accessibility"]["high_contrast"],
    ),
    "sms-billing-off": (
        "turns-off-billing-email",
        _wrong_sms,
        lambda s: notif(s, "billing") == {"email": False, "sms": True, "push": False},
    ),
    "max-usage-month": (
        "selects-august",
        _wrong_max_usage,
        lambda s: s["usage"]["selected_month"] == "2026-08",
    ),
    "report-outage-riverside": (
        "reports-old-town",
        lambda d: _report_outage(d, "old_town", "Old Town"),
        lambda s: [r["district"] for r in s["outages"]["reports"]] == ["old_town"],
    ),
    "star-landlord-message": (
        "stars-paperless",
        _wrong_star,
        lambda s: inbox(s, "msg-paperless")["starred"] and not inbox(s, "msg-landlord")["starred"],
    ),
    "pay-50": (
        "pays-full-balance",
        lambda d: _pay(d, "dash-pay", None),
        lambda s: [p["amount_cents"] for p in s["billing"]["payments"]] == [14237],
    ),
    "message-support-meter": (
        "sends-to-billing",
        lambda d: _send_message(d, "billing", "My meter is broken"),
        lambda s: [m["to"] for m in s["messages"]["sent"]] == ["billing"],
    ),
    "stop-promotions": (
        "leaves-push-on",
        lambda d: _promotions(d, PROMO[:2]),
        lambda s: notif(s, "promotions") == {"email": False, "sms": False, "push": True},
    ),
}

TASK_IDS = [t["id"] for t in TASKS]


def test_oracle_covers_every_task():
    assert set(SOLVERS) == set(WRONG) == set(ORACLE_STEPS) == set(TASK_IDS)
    for tid, n in ORACLE_STEPS.items():
        assert n >= TASKS_BY_ID[tid]["steps_min"], tid


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_oracle_solves(task_id, open_page, get_state):
    task = TASKS_BY_ID[task_id]
    page, sid = open_page(task_id)
    d = Driver(page)
    SOLVERS[task_id](d)
    print(f"{task_id}: {len(d.steps)} UI steps: {d.steps}")
    assert len(d.steps) >= task["steps_min"], f"{task_id}: {len(d.steps)} < steps_min {task['steps_min']}"
    assert len(d.steps) == ORACLE_STEPS[task_id], d.steps
    passed, why = evaluate(get_state(sid), task["check"])
    assert passed, f"{task_id} not solved: {why} (steps: {d.steps})"
    d.assert_clean()


@pytest.mark.parametrize("task_id", [pytest.param(t, id=f"{t}--{WRONG[t][0]}") for t in TASK_IDS])
def test_wrong_action_rejected(task_id, open_page, get_state):
    task = TASKS_BY_ID[task_id]
    page, sid = open_page(task_id)
    d = Driver(page)
    _, act, took_effect = WRONG[task_id]
    act(d)
    state = get_state(sid)
    assert took_effect(state), f"{task_id}: the wrong action did not happen (steps: {d.steps})"
    passed, why = evaluate(state, task["check"])
    assert not passed, f"{task_id}: wrong action '{WRONG[task_id][0]}' passed the check: {why}"
    d.assert_clean()
