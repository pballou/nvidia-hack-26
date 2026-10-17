"""DEV TOOL: measure the sandbox and write console/mock_scripts.json for the mock agent.

    python -m console.tools.calibrate_mock [--sandbox-url http://127.0.0.1:8090] [--tasks id,id] [--headed] [--no-replay]

This is the ONLY place in console/ allowed to use DOM selectors (data-testid locators,
elementFromPoint and window.AcmeTestHooks for canvas targets), exactly like the scripted
oracle in sandbox/tests. For each task it resets a fresh session, walks the hand-written
steps below, measures each target's center at that moment, and performs the step the
way the mock agent will: pixel clicks, mouse wheel and keys (console/mock_agent.perform).
Targets below the fold get scroll steps; a toast covering a target gets a dismiss step.
The output holds only pixel coordinates and texts, never selectors, so the mock agent
replays it from pixels alone. Then it checks evaluate() on the sandbox state, replays
every script through mock_agent.run_task (unless --no-replay) and prints a table.

Re-run it whenever the sandbox layout changes; the mock reports a stale script when a
page path doesn't match.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from console.mock_agent import (SCRIPTS_PATH, SCROLL_ORIGIN, VIEWPORT, MockConfig, open_page, perform, run_task,
                                settle)
from sandbox.checks import evaluate

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((REPO_ROOT / "sandbox" / "tasks.json").read_text())
W, H = VIEWPORT["width"], VIEWPORT["height"]

# Where elementFromPoint lands relative to the target: ok | toast | covered | offscreen.
HIT_TEST_JS = """(el, [x, y]) => {
  if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) return 'offscreen';
  const hit = document.elementFromPoint(x, y);
  if (!hit) return 'offscreen';
  if (el === hit || el.contains(hit)) return 'ok';
  const toast = document.getElementById('toast');
  return toast && toast.contains(hit) ? 'toast' : 'covered';
}"""


def click(target: str, testid: str | None = None, *, canvas: tuple[str, str, str] | None = None,
          why: str, screen: str) -> dict:
    """canvas = (canvas testid, AcmeTestHooks key, hit-target id); hook points are canvas-relative CSS px."""
    return {"tool": "click", "target": target, "testid": testid, "canvas": canvas, "why": why, "screen": screen}


def type_into(text: str, target: str, testid: str, *, why: str, screen: str) -> dict:
    return {"tool": "type_text", "text": text, "target": target, "testid": testid, "why": why, "screen": screen}


def done(summary: str, *, why: str, success: bool = True) -> dict:
    return {"tool": "done", "summary": summary, "success": success, "why": why}


TOPBAR = ("Top bar: a menu icon with three lines and the Acme Utilities logo on the left; a bell, a gear, "
          "a question mark and an avatar with the initials JL on the right.")
SIDENAV = "Side menu: Dashboard, Billing, Usage, Outage map, Messages, Settings, Help."
DASHBOARD = (f"Acme Utilities customer portal, Dashboard: Good afternoon, Jordan. {TOPBAR} {SIDENAV} Amount due "
             "$142.37, due by October 28, 2026, with Pay bill and View statements buttons. A Service status card: "
             "Riverside power on, Harbor Point crews on site. Quick actions: Pay bill, Report an outage, Message us, "
             "Get help. Recent activity and Your account below.")
SETTINGS_PROFILE = (f"Settings page, Profile tab. Tabs: Profile, Notifications, Accessibility. {TOPBAR} "
                    "A profile form with Name (Jordan Lee), Email and Phone, and a Save button.")
SETTINGS_A11Y = ("Settings page, Accessibility tab. Rows: High contrast with a switch (off); Text size with S, M "
                 "and L buttons (M selected); Reduce motion with a switch (off).")
SETTINGS_NOTIF = ("Settings page, Notifications tab. A table of notification types, each row with Email, SMS and "
                  "Push switches: Billing alerts, Payment confirmations, Outage alerts, Usage insights, New messages, "
                  "Service updates, and more rows further down.")
BILLING = ("Billing page. Balance $142.37 due October 28, 2026, with a Pay bill button. Autopay is off, with a "
           "switch. A Statements table lists September 2026 down to October 2025; each row shows the date, amount, "
           "status and a small download icon at the right end.")
INBOX = ("Messages page, Inbox. A Compose button and a small icon button with a double check mark above the "
         "list. 14 messages, newest first; unread ones are bold: Power restored in Hillcrest, Your September "
         "statement is ready, Never miss a due date: try Autopay, Scheduled tree trimming near your street. Each "
         "row has a star icon on the left.")

PLANS: dict[str, dict] = {
    "text-size-large": {"start_screen": SETTINGS_A11Y, "steps": [
        click("the L button in the Text size control", "a11y-text-size-L",
              why="The user can't read the text. Text size has S, M and L buttons, and L is the largest.",
              screen="Settings, Accessibility tab. All the text is larger now and the L button is selected. A banner "
                     "at the top says: Text size set to Large."),
        done("I set the text size to Large, so everything on screen is bigger now.",
             why="The banner confirms the text size is Large."),
    ]},
    "download-september-statement": {"start_screen": BILLING, "steps": [
        click("the download icon at the right end of the September 2026 row", "download-2026-09",
              why="Each statement row ends in an unlabeled download icon. September 2026 is the first row.",
              screen="Billing page. A banner says: Downloaded September 2026 statement (PDF). The September 2026 "
                     "row is marked as downloaded."),
        done("I downloaded your September 2026 statement.", why="The banner confirms the September download."),
    ]},
    "mark-all-read": {"start_screen": INBOX, "steps": [
        click("the double check mark icon above the inbox list", "mark-all-read",
              why="The unlabeled icon with two check marks above the inbox should mark every message as read.",
              screen="Messages, Inbox. A banner says: Marked 4 messages as read. No message is bold any more."),
        done("I marked all your messages as read.", why="The banner says 4 messages were marked read."),
    ]},
    "turn-on-autopay": {"start_screen": BILLING, "steps": [
        click("the Autopay switch", "autopay-toggle",
              why="The Billing page has an Autopay switch, and it is off.",
              screen="Billing page. The Autopay switch is on. A banner says: Autopay is on. We'll pay from "
                     "Checking ••4821 on each due date."),
        done("I turned on Autopay, so your bill is paid automatically on each due date.",
             why="The switch is on and the banner confirms it."),
    ]},
    "high-contrast": {"start_screen": DASHBOARD, "steps": [
        click("the gear icon in the top-right toolbar", "topbar-settings",
              why="Display options live in Settings, and the gear icon in the top bar should open it.",
              screen=SETTINGS_PROFILE),
        click("the Accessibility tab", "tab-accessibility",
              why="High contrast is a display option, so it should be on the Accessibility tab.",
              screen=SETTINGS_A11Y),
        click("the High contrast switch", "a11y-high-contrast",
              why="This switch turns on high contrast. It is off now.",
              screen="The page now uses a high-contrast theme: dark background, white text and bright outlines. "
                     "The High contrast switch is on. A banner says: High contrast turned on."),
        done("I turned on high contrast.", why="The theme changed and the switch is on."),
    ]},
    "sms-billing-off": {"start_screen": SETTINGS_PROFILE, "steps": [
        click("the Notifications tab", "tab-notifications",
              why="Text message alerts are notification settings, so I'll open the Notifications tab.",
              screen=SETTINGS_NOTIF),
        click("the SMS switch in the Billing alerts row", "notif-billing-sms",
              why="Billing alerts is the first row; its SMS switch is on and the user wants it off.",
              screen="Settings, Notifications tab. In the Billing alerts row the SMS switch is now off. A banner "
                     "says: SMS for Billing alerts turned off."),
        done("I turned off text messages for billing alerts. Email alerts for billing are still on.",
             why="The banner confirms SMS for Billing alerts is off."),
    ]},
    "max-usage-month": {"start_screen": DASHBOARD.replace(SIDENAV + " ", "The side menu is hidden. "), "steps": [
        click("the menu icon with three lines at the top left", "topbar-menu",
              why="The side menu is hidden. The three-line icon should open it so I can find Usage.",
              screen=f"The side menu is open. {SIDENAV} The dashboard is on the right."),
        click("Usage in the side menu", "nav-usage",
              why="Monthly power use is on the Usage page.",
              screen="Usage page. A bar chart of monthly electricity use from Oct 2025 to Sep 2026, one bar per "
                     "month. The Jul bar is the tallest and Aug is a little shorter. Below: Click a bar to see "
                     "that month's details."),
        click("the tallest bar in the usage chart, July", canvas=("usage-chart", "usageBars", "2026-07"),
              why="The July 2026 bar is the tallest in the chart, so July is the month with the most use.",
              screen="Usage page. The July bar is highlighted and a details panel shows July 2026: 968 kWh."),
        done("July 2026 was your highest month, at 968 kWh. I opened its details.",
             why="The details panel shows July 2026 with 968 kWh."),
    ]},
    "report-outage-riverside": {"start_screen": (
            "Outage map page. A map with six labeled districts: Riverside, Hillcrest, Old Town, Maple Grove, Harbor "
            "Point and Northgate. Harbor Point is marked with an active outage (crews on site). Text: Click a "
            "district to report an outage."), "steps": [
        click("the Riverside district on the outage map", canvas=("outage-map", "districts", "riverside"),
              why="To report an outage in Riverside, I click Riverside on the map.",
              screen="A Report an outage dialog for Riverside is open: What's happening (No power selected), a "
                     "Details box, and Cancel and Submit report buttons."),
        click("the Submit report button in the dialog", "report-submit",
              why="The dialog is for Riverside with No power selected; submitting sends the report.",
              screen="Outage map page. A banner says: Outage reported in Riverside. Reference OUT-1001."),
        done("I reported an outage in Riverside. Your reference number is OUT-1001.",
             why="The banner confirms the report and gives a reference."),
    ]},
    "star-landlord-message": {"start_screen": INBOX, "steps": [
        click("the star next to the message from Dana Whitfield (Landlord)", "star-msg-landlord",
              why="The landlord is Dana Whitfield. Her message is near the bottom of the inbox; I'll click its star.",
              screen="Messages, Inbox. The star next to Dana Whitfield (Landlord), Water heater replacement on "
                     "September 20, is filled in. A banner says: Starred: Water heater replacement on September 20."),
        done("I starred the message from your landlord, Dana Whitfield.", why="The star is filled in."),
    ]},
    "pay-50": {"start_screen": DASHBOARD, "steps": [
        click("the Pay bill button on the amount due card", "dash-pay",
              why="The Amount due card on the dashboard has a Pay bill button.",
              screen="A Pay bill dialog is open. Amount: Full balance $142.37 (selected) or Other amount. Pay "
                     "from Checking ••4821. Cancel and Continue buttons."),
        click("the Other amount option", "pay-other",
              why="The user wants to pay $50, not the full balance.",
              screen="Pay bill dialog. Other amount is selected and an Amount field appeared."),
        type_into("50", "the Amount field", "pay-amount",
                  why="Enter 50 dollars in the Amount field.",
                  screen="Pay bill dialog. The Amount field shows 50."),
        click("the Continue button in the dialog", "pay-continue",
              why="The amount is set, so I continue to the confirmation.",
              screen="Confirm payment dialog: Pay $50.00 from Checking ••4821. Back and Confirm payment buttons."),
        click("the Confirm payment button", "pay-confirm",
              why="The confirmation shows $50.00, which is what the user asked for.",
              screen="Billing page. A banner says: Payment of $50.00 received. Confirmation PAY-1001. The balance "
                     "is now $92.37."),
        done("I paid $50.00 toward your bill. Your balance is now $92.37 (confirmation PAY-1001).",
             why="The banner confirms the payment."),
    ]},
    "message-support-meter": {"start_screen": DASHBOARD, "steps": [
        click("the Message us card in Quick actions", "quick-message",
              why="The Message us quick action, Write to our team, opens a new message.",
              screen="New message page. To (Choose one), Subject and Message fields, and a Send button."),
        click("the To field", "compose-to",
              why="First I choose who gets the message.",
              screen="A list opened under To: Customer Support, Billing Department, Outage Team, Field Services, "
                     "Dana Whitfield (Landlord)."),
        click("Customer Support in the list", "compose-to-support",
              why="The user wants to write to support.",
              screen="New message. To: Customer Support. Subject and Message are empty."),
        type_into("Broken meter", "the Subject field", "compose-subject",
                  why="A short subject helps support.",
                  screen="New message. To: Customer Support. Subject: Broken meter."),
        type_into("Hi, my meter is broken. Can someone take a look?", "the Message box", "compose-body",
                  why="The message says the meter is broken, in the user's words.",
                  screen="New message. To: Customer Support. Subject: Broken meter. Message: Hi, my meter is "
                         "broken. Can someone take a look?"),
        click("the Send button", "compose-send",
              why="Everything is filled in, so I send it.",
              screen="Messages page. A banner says: Message sent to Customer Support."),
        done("I sent Customer Support a message saying your meter is broken.",
             why="The banner confirms the message was sent."),
    ]},
    "stop-promotions": {"start_screen": DASHBOARD, "steps": [
        click("the gear icon in the top-right toolbar", "topbar-settings",
              why="Notification preferences are in Settings, behind the gear icon.",
              screen=SETTINGS_PROFILE),
        click("the Notifications tab", "tab-notifications",
              why="Promotions are a kind of notification.",
              screen=SETTINGS_NOTIF),
        click("the Email switch in the Offers and promotions row", "notif-promotions-email",
              why="Offers and promotions has Email, SMS and Push switches, all on. I'll turn each off.",
              screen="Settings, Notifications. In the Offers and promotions row, Email is now off; SMS and Push "
                     "are still on. A banner says: Email for Offers and promotions turned off."),
        click("the SMS switch in the Offers and promotions row", "notif-promotions-sms",
              why="Next, SMS for Offers and promotions.",
              screen="Offers and promotions: Email off, SMS off, Push on. A banner says: SMS for Offers and "
                     "promotions turned off."),
        click("the Push switch in the Offers and promotions row", "notif-promotions-push",
              why="Last, Push for Offers and promotions.",
              screen="Offers and promotions: Email, SMS and Push are all off. A banner says: Push for Offers and "
                     "promotions turned off."),
        done("I turned off email, text and push notifications for offers and promotions. Your other alerts are "
             "unchanged.", why="All three switches in the Offers and promotions row are off."),
    ]},
}


class CalibrationError(Exception):
    pass


def _path(page) -> str:
    return urlsplit(page.url).path


async def measure(page, step: dict):
    """(locator, (x, y)) for the step's target at this moment, in viewport CSS px."""
    if step.get("canvas"):
        testid, hook, target_id = step["canvas"]
        canvas = page.get_by_test_id(testid)
        await canvas.wait_for(state="visible", timeout=5000)
        box = await canvas.bounding_box()
        points = await page.evaluate("(k) => (window.AcmeTestHooks || {})[k] || null", hook)
        point = next((p for p in points or [] if p.get("id") == target_id), None)
        if box is None or point is None:
            raise CalibrationError(f"no AcmeTestHooks.{hook} point {target_id!r} on {testid}")
        return canvas, (box["x"] + point["x"], box["y"] + point["y"])
    loc = page.get_by_test_id(step["testid"])
    await loc.wait_for(state="visible", timeout=5000)
    box = await loc.bounding_box()
    if box is None:
        raise CalibrationError(f"{step['testid']} has no bounding box")
    return loc, (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


async def _do(page, step: dict) -> dict:
    await perform(page, step)
    await settle(page)
    step["path_after"] = _path(page)
    return step


async def reveal(page, step: dict) -> tuple[tuple[float, float], list[dict]]:
    """Scroll / dismiss the toast until the target's center is clickable. Returns its point and the extra steps."""
    extra: list[dict] = []
    last_y = None
    for _ in range(10):
        loc, (x, y) = await measure(page, step)
        hit = await loc.evaluate(HIT_TEST_JS, [x, y])
        if hit == "ok":
            return (round(x, 1), round(y, 1)), extra
        if hit == "toast":
            _, (tx, ty) = await measure(page, {"testid": "toast-close"})
            extra.append(await _do(page, {
                "tool": "click", "arguments": {"target_description": "the X button on the banner at the top"},
                "x": round(tx, 1), "y": round(ty, 1),
                "reasoning": f"A banner is covering {step['target']}, so I'll close it first.",
                "screen_after": "The banner is closed."}))
            continue
        if last_y is not None and abs(y - last_y) < 1:
            raise CalibrationError(f"scrolling doesn't move {step['target']!r} (y={y:.0f}, hit={hit})")
        last_y = y
        direction = "down" if y >= H / 2 else "up"
        amount = "page" if (y > H + 200 or y < -200) else "small"
        extra.append(await _do(page, {
            "tool": "scroll", "arguments": {"direction": direction, "amount": amount},
            "x": SCROLL_ORIGIN[0], "y": SCROLL_ORIGIN[1],
            "reasoning": f"I can't see {step['target']} on the screen yet, so I'll scroll {direction}.",
            "screen_after": f"The page scrolled {direction}; more of it is visible."}))
    raise CalibrationError(f"could not bring {step['target']!r} into view")


def start_url(sandbox_url: str, task: dict, sid: str) -> str:
    path = task["start_path"] or "/"
    return f"{sandbox_url}{path}{'&' if '?' in path else '?'}sid={sid}"


def reset(sandbox_url: str, sid: str, task_id: str) -> None:
    httpx.post(f"{sandbox_url}/api/reset", json={"sid": sid, "task_id": task_id}, timeout=5,
               trust_env=False).raise_for_status()


def verdict(sandbox_url: str, sid: str, task: dict) -> tuple[bool, str]:
    r = httpx.get(f"{sandbox_url}/api/state", params={"sid": sid}, timeout=5, trust_env=False)
    r.raise_for_status()
    return evaluate(r.json(), task["check"])


async def calibrate(task: dict, plan: dict, sandbox_url: str, headed: bool) -> tuple[dict, bool, str]:
    sid = f"calibrate-{task['id']}-{secrets.token_hex(3)}"
    reset(sandbox_url, sid, task["id"])
    out: list[dict] = []
    async with open_page(headed) as page:
        await page.goto(start_url(sandbox_url, task, sid), wait_until="load")
        await settle(page)
        for i, step in enumerate(plan["steps"], 1):
            if step["tool"] == "done":
                out.append({"tool": "done", "arguments": {"summary": step["summary"], "success": step["success"]},
                            "reasoning": step["why"]})
                break
            try:
                (x, y), extra = await reveal(page, step)
                out += extra
                args = {"target_description": step["target"]}
                if step["tool"] == "type_text":
                    args = {"text": step["text"], **args}
                out.append(await _do(page, {"tool": step["tool"], "arguments": args, "x": x, "y": y,
                                            "reasoning": step["why"], "screen_after": step["screen"]}))
            except Exception as e:
                what = step.get("testid") or step.get("canvas")
                raise CalibrationError(f"step {i} ({what}) on {_path(page)}: {type(e).__name__}: "
                                       f"{str(e).splitlines()[0][:120]}") from e
    passed, why = verdict(sandbox_url, sid, task)
    return {"request": task["request"], "start_screen": plan["start_screen"], "steps": out}, passed, why


async def replay(task: dict, scripts_path: Path, sandbox_url: str, runs_dir: str) -> tuple[bool, str]:
    sid = f"calibrate-replay-{task['id']}-{secrets.token_hex(3)}"
    reset(sandbox_url, sid, task["id"])
    cfg = MockConfig(runs_dir=runs_dir, sandbox_url=sandbox_url, step_delay_s=0, scripts_path=str(scripts_path))
    errors: list[str] = []

    async def on_event(ev):
        if ev.kind == "error":
            errors.append(ev.payload["message"])

    await run_task(task["request"], start_url=start_url(sandbox_url, task, sid), cfg=cfg, on_event=on_event)
    passed, why = verdict(sandbox_url, sid, task)
    return passed and not errors, "; ".join(errors) or why


async def main_async(args) -> int:
    sandbox_url = args.sandbox_url.rstrip("/")
    try:
        httpx.get(f"{sandbox_url}/api/health", timeout=3, trust_env=False).raise_for_status()
    except httpx.HTTPError as e:
        print(f"sandbox not reachable at {sandbox_url}: {e}", file=sys.stderr)
        return 2
    wanted = [t for t in TASKS if not args.tasks or t["id"] in args.tasks.split(",")]
    out_path = Path(args.out)
    try:
        existing = json.loads(out_path.read_text()).get("tasks", {})
    except (OSError, ValueError, AttributeError):
        existing = {}
    scripts = {tid: s for tid, s in existing.items() if tid in PLANS and tid not in {t["id"] for t in wanted}}
    rows = []
    for task in wanted:
        try:
            script, passed, why = await calibrate(task, PLANS[task["id"]], sandbox_url, args.headed)
        except Exception as e:
            script, passed, why = None, False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        if passed:
            scripts[task["id"]] = script
        else:
            scripts.pop(task["id"], None)  # a failing script would only mislead the mock
        rows.append([task["id"], len(script["steps"]) if script else 0, passed, None, why])
        print(f"  {task['id']}: {'PASS' if passed else 'FAIL'}  {why}", flush=True)

    order = [t["id"] for t in TASKS]
    data = {
        "_about": ("Generated by console/tools/calibrate_mock.py; replayed by console/mock_agent.py. Pixel points are "
                   "viewport CSS px at 1280x800, device_scale_factor 1. Texts and coordinates only: no selectors."),
        "viewport": dict(VIEWPORT),
        "tasks": {tid: scripts[tid] for tid in order if tid in scripts},
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(data['tasks'])} scripts to {out_path}")

    if not args.no_replay:
        with tempfile.TemporaryDirectory(prefix="mock-replay-") as tmp:
            for row in rows:
                if row[2]:
                    task = next(t for t in TASKS if t["id"] == row[0])
                    ok, why = await replay(task, out_path, sandbox_url, tmp)
                    row[3] = ok
                    if not ok:
                        row[4] = f"replay: {why}"

    print(f"\n{'task':<30} {'steps':>5}  {'walk':<5} {'replay':<6}  explanation")
    for tid, n, walk, rep, why in rows:
        rep_s = "-" if rep is None else ("PASS" if rep else "FAIL")
        print(f"{tid:<30} {n:>5}  {'PASS' if walk else 'FAIL':<5} {rep_s:<6}  {why}")
    failed = [r for r in rows if not r[2] or r[3] is False]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} tasks OK")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m console.tools.calibrate_mock", description=__doc__.splitlines()[0])
    ap.add_argument("--sandbox-url", default="http://127.0.0.1:8090")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default: all 12); others are kept")
    ap.add_argument("--out", default=str(SCRIPTS_PATH))
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--no-replay", action="store_true", help="skip replaying the scripts through the mock agent")
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
