"""DEV TOOL: measure the sandbox and write console/mock_scripts.json for the mock agent.

    python -m console.tools.calibrate_mock [--sandbox-url http://127.0.0.1:8090] [--tasks id,id] [--headed] [--no-replay]

This is the ONLY place in console/ allowed to use DOM selectors (data-testid locators,
elementFromPoint and window.AcmeTestHooks for canvas targets), exactly like the scripted
oracle in sandbox/tests. Its output holds only pixel coordinates and texts, never
selectors, so the mock agent replays it from pixels alone.

PLANS below is the hand-written path a pixel agent would take through each of the 12
tasks: the tool calls Lightning would make (target descriptions in the grounder's style:
short, visual, positional), its reasoning, and the structured screen description the VLM
would write after each step (common/grounding.py SUMMARY_PROMPT: PAGE / ELEMENTS / STATE /
OVERLAYS). The texts were written from screenshots of the real pages, so keep them in sync
with the sandbox.

For each task the tool:
  1. resets a fresh sid and opens the task's start page (1280x800, dsf 1), recorded as the
     script's start_path: the mock replays a script only from that page;
  2. measures each click target's center with DOM locators at that moment;
  3. checks the center is really clickable (elementFromPoint). If not, it adds a scroll or
     a toast-dismiss step and reports it as "auto" (better: add an explicit step to PLANS);
  4. performs each step the way the mock will (console/mock_agent.perform: pixel clicks,
     mouse wheel and keys) and records the page path after it;
  5. validates the script against the agent's tool schema (agent/prompts.py) and checks
     evaluate() on the sandbox state;
  6. writes mock_scripts.json, then replays every script through mock_agent.run_task
     (unless --no-replay): no errors, done(success=true), every action changed the screen,
     every click has a grounding event and an overlay image, and evaluate() passes.

It prints a summary table and exits 1 if any task fails (2 if the sandbox is down).
Re-run it whenever the sandbox layout changes; the mock reports a stale script when a page
path doesn't match.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from agent.prompts import TOOL_NAMES, tool_parameters
from console.mock_agent import (SCRIPTS_PATH, SCROLL_ORIGIN, VIEWPORT, MockConfig, open_page, perform, run_task,
                                settle)
from sandbox.checks import evaluate

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((REPO_ROOT / "sandbox" / "tasks.json").read_text())
W, H = VIEWPORT["width"], VIEWPORT["height"]
EDGE_PX = 40  # a target centered closer than this to the bottom edge gets scrolled into view
STEP_KEYS = {"tool", "arguments", "x", "y", "reasoning", "screen_after", "path_after"}

# Where elementFromPoint lands relative to the target: ok | toast | covered | offscreen.
HIT_TEST_JS = """(el, [x, y]) => {
  if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) return 'offscreen';
  const hit = document.elementFromPoint(x, y);
  if (!hit) return 'offscreen';
  if (el === hit || el.contains(hit)) return 'ok';
  const toast = document.getElementById('toast');
  return toast && toast.contains(hit) ? 'toast' : 'covered';
}"""


# --- plan helpers -------------------------------------------------------------------------


def click(target: str, testid: str | None = None, *, canvas: tuple[str, str, str] | None = None,
          why: str, screen: str) -> dict:
    """canvas = (canvas testid, AcmeTestHooks key, hit-target id); hook points are canvas-relative CSS px."""
    return {"tool": "click", "target": target, "testid": testid, "canvas": canvas, "why": why, "screen": screen}


def type_into(text: str, target: str, testid: str, *, why: str, screen: str) -> dict:
    return {"tool": "type_text", "text": text, "target": target, "testid": testid, "why": why, "screen": screen}


def scroll(direction: str, amount: str, *, why: str, screen: str) -> dict:
    return {"tool": "scroll", "direction": direction, "amount": amount, "why": why, "screen": screen}


def done(summary: str, *, why: str, success: bool = True) -> dict:
    return {"tool": "done", "summary": summary, "success": success, "why": why}


# --- screen descriptions (what the VLM would write; SUMMARY_PROMPT sections) ------------------


def topbar(bell: str = "a red badge 3") -> list[str]:
    return [
        "icon button: three horizontal lines @ top-left",
        "logo: Acme Utilities, Customer portal @ top-left",
        f"icon button: bell with {bell} @ top-right",
        "icon button: gear @ top-right",
        "icon button: question mark in a circle @ top-right",
        "button: round JL avatar with a down arrow @ top-right",
    ]


def sidenav(current: str, messages_badge: bool = True) -> list[str]:
    msgs = "Messages (badge 4)" if messages_badge else "Messages"
    return [f"side menu links: Dashboard, Billing, Usage, Outage map, {msgs}, Settings, Help @ left "
            f"({current} highlighted)"]


def screen(page: str, elements: list[str], state: str, overlays: str = "none") -> str:
    return "\n".join([f"PAGE: {page}", "ELEMENTS:", *(f"- {e}" for e in elements), f"STATE: {state}",
                      f"OVERLAYS: {overlays}"])


def toast(text: str) -> str:
    return f"toast: {text}, with an X close button @ top"


def dashboard(nav: bool = True) -> str:
    return screen(
        "Acme Utilities customer portal, Dashboard. Headings: Good afternoon, Jordan (Saturday, October 17, "
        "2026); Amount due; Service status; Quick actions; Recent activity; Your account.",
        topbar() + (sidenav("Dashboard") if nav else []) + [
            "button: Pay bill, in the dark Amount due card @ center",
            "button: View statements, in the Amount due card @ center",
            "link: Outage map, in the Service status card @ top-right",
            "card: Pay bill, Make a one-time payment (Quick actions) @ bottom-left",
            "card: Report an outage, Tell us what is happening (Quick actions) @ bottom",
            "card: Message us, Write to our team (Quick actions) @ bottom",
            "card: Get help, FAQs and call-backs (Quick actions) @ bottom-right",
        ],
        ("" if nav else "The side menu is hidden and the dashboard fills the width. ")
        + "Amount due $142.37, due by October 28, 2026, with an Autopay off label. Service status: Riverside "
          "Power on (your area); Harbor Point Crews on site, 214 customers affected; Planned tree trimming near "
          "Maple Court, Oct 21, 2026.")


SETTINGS_PAGE = ("Settings: Manage your profile, how we contact you, and how the portal looks. Tabs: Profile, "
                 "Notifications, Accessibility.")
SETTINGS_TABS = ["tab: Profile @ top", "tab: Notifications @ top", "tab: Accessibility @ top"]

SETTINGS_PROFILE = screen(
    f"{SETTINGS_PAGE} Sections: Personal details, Account details.",
    topbar() + sidenav("Settings") + SETTINGS_TABS + [
        "input: Full name, Jordan Lee @ center",
        "input: Email, jordan.lee@example.com @ center",
        "input: Mobile phone, (919) 555-0142 @ center",
        "button: Discard changes @ bottom",
        "button: Save changes @ bottom",
        "link: Contact us, in the Account details card @ right",
    ],
    "Profile tab selected (underlined). Account number 4417-2093-88 and service address 212 Maple Court, Apt 3B, "
    "Riverside are locked.")


def settings_a11y(*, hc: bool = False, size: str = "M", overlays: str = "none") -> str:
    preview = {"M": "This is Medium text, the standard size.", "L": "This is Large text."}[size]
    theme = ("The whole portal uses a high-contrast theme now: black background, white text, yellow outlines and "
             "a yellow highlight on Settings. " if hc else "")
    bigger = "All text, buttons and menus are larger now. " if size == "L" else ""
    return screen(
        f"{SETTINGS_PAGE} Cards: High contrast (Bold colors and outlines that are easier to see); Text size (Make "
        "words, buttons and menus larger on every page); Reduce motion.",
        topbar() + sidenav("Settings") + SETTINGS_TABS + [
            f"toggle: High contrast, {'On' if hc else 'Off'} @ right (High contrast row)",
            "segmented control: S, M, L buttons @ right (Text size row)",
            "toggle: Reduce motion, Off @ right (Reduce motion row)",
        ],
        f"{theme}{bigger}Accessibility tab selected. High contrast {'on' if hc else 'off'}. Text size {size} "
        f'selected; the preview reads "{preview}" Reduce motion off.',
        overlays)


NOTIF_ROWS = [("Billing alerts", "On, On, Off"), ("Payment confirmations", "On, Off, On"),
              ("Outage alerts", "On, On, On"), ("Usage insights", "On, Off, Off"), ("New messages", "On, Off, On"),
              ("Service updates", "On, Off, Off"), ("Offers and promotions", "On, On, On"),
              ("Surveys and feedback", "On, Off, Off")]


def settings_notif(*, scrolled: bool = False, changes: dict[str, str] | None = None, note: str = "",
                   overlays: str = "none") -> str:
    rows = dict(NOTIF_ROWS, **(changes or {}))
    shown = list(rows)[1:] if scrolled else list(rows)[:4]
    page = ("Settings, Notifications tab, scrolled down. The table's NOTIFICATION / EMAIL / SMS / PUSH header stays "
            "under the top bar, the Billing alerts row is hidden above it and the end of the table is visible."
            if scrolled else
            f"{SETTINGS_PAGE} Text: Choose how we contact you about each kind of update. Changes save as soon as you "
            "flip a switch. A Where we reach you card (email, SMS (919) 555-0142, Acme Utilities app), then a table "
            "with Email, SMS and Push columns.")
    elements = topbar() + sidenav("Settings") + ([] if scrolled else SETTINGS_TABS + [
        "link: Edit contact details (pencil icon) @ top-right"])
    elements += [f"toggles: Email, SMS, Push in the {name} row @ right" for name in shown]
    tail = (" Below the table: Safety alerts about downed lines near your address are always sent by SMS, even "
            "when Outage alerts are off." if scrolled else " More rows continue below the bottom edge.")
    state = ("Notifications tab selected. Switches (Email, SMS, Push): "
             + "; ".join(f"{name} {rows[name]}" for name in shown) + "." + tail + (f" {note}" if note else ""))
    return screen(page, elements, state, overlays)


def billing(*, overlays: str = "none", amount_due: str = "$142.37", autopay: bool = False, extra: str = "") -> str:
    autopay_state = ("Autopay On: 'On: we pay from Checking ••4821 on each due date'; Next automatic payment: "
                     "$142.37 on Oct 28, 2026." if autopay else
                     "Autopay Off: 'Never miss a due date: turn it on and we'll pay your full balance automatically.'")
    return screen(
        "Billing: Pay your bill, manage Autopay and download statements. Sections: Amount due; Autopay; "
        "Statements (The last 12 months, newest first).",
        topbar() + sidenav("Billing") + [
            "button: Pay bill, in the dark Amount due card @ center",
            f"toggle: Autopay, {'On' if autopay else 'Off'}, in the Autopay card @ top-right",
            "table rows (Statement, Issued, Amount, Status), each with a small download icon (arrow over a line) "
            "at the right end @ center: September 2026, Oct 1, 2026, $142.37, Due; August 2026, $151.20, Paid; "
            "July 2026, $163.88, Paid; June 2026, $139.05, Paid; May 2026, $104.12, Paid (cut off at the bottom)",
        ],
        f"Amount due {amount_due}, due by October 28, 2026. {autopay_state}" + (f" {extra}" if extra else ""),
        overlays)


INBOX_TOP = [
    ("Acme Outage Team", "Power restored in Hillcrest", "Oct 16", True),
    ("Acme Billing", "Your September statement is ready", "Oct 1", True),
    ("Acme Billing", "Never miss a due date: try Autopay", "Sep 28", True),
    ("Acme Energy Insights", "Your usage is 22% lower than last month", "Sep 25", False),
    ("Customer Support", "Re: Question about my meter reading", "Sep 22", False),
    ("Acme Outage Team", "Scheduled tree trimming near your street", "Sep 19", True),
    ("Acme Billing", "Payment received. Thank you!", "Sep 18", False),
]
INBOX_BOTTOM = [
    ("Acme Outage Team", "Scheduled tree trimming near your street", "Sep 19", True),
    ("Acme Billing", "Payment received. Thank you!", "Sep 18", False),
    ("Acme Energy Insights", "You qualify for a smart thermostat rebate", "Sep 15", False),
    ("Acme Utilities", "How did we do? 2-minute survey", "Sep 12", False),
    ("Acme Outage Team", "Heat advisory: tips to save energy", "Sep 8", False),
    ("Dana Whitfield (Landlord)", "Water heater replacement on September 20", "Sep 5", False),
    ("Acme Billing", "You're now on paperless billing", "Aug 30", False),
    ("Acme Utilities", "Welcome to the new customer portal", "Aug 21", False),
    ("Acme Field Services", "Smart meter upgrade complete", "Aug 14", False),
]


def inbox(*, scrolled: bool = False, all_read: bool = False, starred: tuple[str, ...] = ("Customer Support",),
          overlays: str = "none") -> str:
    rows = INBOX_BOTTOM if scrolled else INBOX_TOP
    lines = []
    for sender, subject, date, unread in rows:
        marks = [m for m, on in (("unread, bold with a blue dot", unread and not all_read),
                                 ("filled yellow star", sender in starred)) if on]
        lines.append(f"message row: {sender}, {subject}, {date}{' (' + ', '.join(marks) + ')' if marks else ''}, "
                     f"with a star icon at the left")
    if scrolled:
        page = ("Messages, Inbox, scrolled down. The Customer Support row is cut off under the top bar; the list "
                "ends above the footer.")
        elements = topbar() + sidenav("Messages", not all_read) + lines
    else:
        page = ("Messages: Updates from Acme Utilities and the conversations you start. Tabs: Inbox, Sent. A toolbar "
                "above the message list.")
        elements = topbar() + sidenav("Messages", not all_read) + [
            "tab: Inbox" + ("" if all_read else " with a badge 4") + " @ top-left",
            "tab: Sent @ top-left",
            "button: New message (pencil icon) @ top-left",
            "icon button: two check marks, right of New message @ top-left",
        ] + lines + ["more message rows continue below the bottom edge"]
    counts = "All read · 14 messages; no row is bold." if all_read else "4 unread · 14 messages."
    return screen(page, elements, f"Inbox tab selected. {counts} Starred: {', '.join(starred)}.", overlays)


def usage(*, selected: bool = False, overlays: str = "none") -> str:
    details = ("Month details card: July 2026, 968 kWh, 30% above your monthly average; Bill amount $163.88, Paid; "
               "Daily average 31.2 kWh; a View statements button. The Jul bar is dark green with an outline."
               if selected else
               "No month selected. Month details card: Select a bar to see that month's details.")
    return screen(
        "Energy usage: Last 12 months, kWh. Stats: 12-month total 8,933 kWh; Monthly average 744 kWh; Select a "
        "month to see details (Click a bar in the chart below). Monthly usage bar chart, October 2025 to "
        "September 2026, and a Month details card on the right.",
        topbar() + sidenav("Usage") + [
            "chart bars: Oct, Nov, Dec, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, left to right @ center",
            "chart: y axis 0 to 1,000 kWh; the Jul bar is the tallest (just under 1,000), Aug and Jan are next "
            "(about 900 and 875), Apr is the shortest @ center",
        ] + (["button: View statements, in the Month details card @ right"] if selected else []),
        details, overlays)


def outage_map(*, reported: bool = False, overlays: str = "none") -> str:
    reports = ("Your reports (1): Riverside, Received, No power, OUT-1001 · Today; a green pin on Riverside."
               if reported else "Your reports: No reports yet. Click a district on the map to tell us about an outage.")
    return screen(
        "Outage map: Click your district to report an outage. A map of six districts, a legend (Active outage, Your "
        "report, Your address) and cards on the right: Current outages (1 active), Your reports, See a downed "
        "power line?",
        topbar() + sidenav("Outage map") + [
            "map region: Northgate (purple) @ top-left of the map",
            "map region: Hillcrest (light green) @ top-right of the map",
            "map region: Old Town (tan) @ center of the map",
            "map region: Harbor Point (red outline, hatched, lightning icon) @ right of the map",
            "map region: Riverside (light green, with a house icon) @ bottom-left of the map",
            "map region: Maple Grove (pink) @ bottom of the map",
        ],
        f"Current outages: Harbor Point, Crews on site, 214 customers affected, since today. {reports}",
        overlays)


OUTAGE_DIALOG = screen(
    "Outage map with a dialog open: Report an outage in Riverside.",
    [
        "icon button: X @ top-right of the dialog",
        "radio buttons: No power, Flickering lights, Downed power line (What's happening?) @ center",
        "text box: Details (optional), For example: the whole street went dark @ center",
        "button: Cancel @ bottom-right of the dialog",
        "button: Submit report (paper plane icon) @ bottom-right of the dialog",
    ],
    "No power is selected. Details is empty. Info box: No known outages in Riverside right now. Your report goes "
    "straight to our crews.",
    "dialog: Report an outage in Riverside; the page behind it is dimmed")


def pay_dialog(step: str) -> str:
    if step == "confirm":
        return screen(
            "Billing with a dialog open: Confirm your payment.",
            ["icon button: X @ top-right of the dialog", "button: Back (left arrow) @ bottom of the dialog",
             "button: Confirm payment (lock icon) @ bottom-right of the dialog"],
            "Pay $50.00 from Checking ••4821? Amount $50.00; From Checking ••4821; Payment date Today, Oct 17, 2026; "
            "Balance after payment $92.37.",
            "dialog: Confirm your payment; the page behind it is dimmed")
    other = step in ("other", "typed")
    value = "50" if step == "typed" else "0.00"
    return screen(
        "Billing with a dialog open: Pay your bill. Amount due by Oct 28, 2026: $142.37. How much would you like to "
        "pay?",
        ["icon button: X @ top-right of the dialog",
         "radio option: Full balance, $142.37 @ center of the dialog",
         f"radio option: Other amount, with a $ box showing {value} @ center of the dialog",
         "button: Cancel @ bottom of the dialog",
         "button: Continue (right arrow) @ bottom-right of the dialog"],
        (f"Other amount is selected and its box is outlined{', showing 50' if step == 'typed' else ''}."
         if other else "Full balance is selected.") + " Paid today from Checking ••4821.",
        "dialog: Pay your bill; the page behind it is dimmed")


def compose(*, to: str = "", subject: str = "", body: str = "", overlays: str = "none") -> str:
    return screen(
        "New message: Write to our team or one of your contacts. We usually reply within 1 business day. Side "
        "cards: Emergency? (call 1-800-555-0199 for a downed power line or a gas smell); Tips for a faster answer.",
        topbar() + sidenav("Messages") + [
            f"dropdown: To, {to or 'Choose a recipient'} (down arrow at the right) @ center",
            f"input: Subject, {subject or 'Optional'} @ center",
            f"text box: Message, {body or 'How can we help?'} @ center",
            "button: Send (paper plane icon) @ bottom-left",
            "button: Cancel @ bottom-left",
        ],
        f"To: {to or 'empty'}. Subject: {subject or 'empty'}. Message: {body or 'empty'}.",
        overlays)


RECIPIENTS = screen(
    "New message, with the To dropdown open.",
    topbar() + sidenav("Messages") + [
     "dropdown: To, Choose a recipient (outlined, open) @ center",
     "list options: Customer Support, Billing Department, Outage Team, Field Services, Dana Whitfield (Landlord), "
     "top to bottom, under the To field @ center"],
    "No recipient chosen yet. The list covers the Subject and Message fields.",
    "dropdown list: Customer Support, Billing Department, Outage Team, Field Services, Dana Whitfield (Landlord)")

SENT = screen(
    "Messages: Updates from Acme Utilities and the conversations you start. Tabs: Inbox, Sent.",
    topbar() + sidenav("Messages") + [
        "tab: Inbox with a badge 4 @ top-left", "tab: Sent @ top-left", "button: New message (pencil icon) @ top-left",
        "message row: To: Customer Support, Broken meter – Hi, my meter is broken. Can someone take a look?, Today "
        "(highlighted) @ center"],
    "Sent tab selected. 1 sent.",
    toast("Message sent to Customer Support"))

# --- the 12 plans ------------------------------------------------------------------------------

SMS_OFF_NOTE = "SMS for Billing alerts is now Off."

PLANS: dict[str, dict] = {
    "text-size-large": {"start_screen": settings_a11y(), "steps": [
        click("the L button at the right end of the Text size control", "a11y-text-size-L",
              why="The user can't read the text. The Text size control has S, M and L buttons with M selected; L is "
                  "the largest.",
              screen=settings_a11y(size="L", overlays=toast("Text size set to Large"))),
        done("I set the text size to Large, so the words, buttons and menus on every page are bigger now.",
             why="L is selected, everything got bigger and the toast says Text size set to Large."),
    ]},
    "download-september-statement": {"start_screen": billing(), "steps": [
        click("the download icon at the right end of the September 2026 row", "download-2026-09",
              why="Each statement row ends in a small unlabeled download icon. September 2026 is the top row of the "
                  "Statements table.",
              screen=billing(overlays=toast("Downloaded September 2026 statement (PDF)"),
                             extra="The September 2026 row is highlighted and shows a Downloaded badge next to Due.")),
        done("I downloaded your September 2026 statement.",
             why="The toast confirms the download and the September 2026 row shows a Downloaded badge."),
    ]},
    "mark-all-read": {"start_screen": inbox(), "steps": [
        click("the double check mark icon next to the New message button", "mark-all-read",
              why="There's no 'Mark all read' text, but the small icon with two check marks next to New message "
                  "usually does that. The inbox has 4 unread messages.",
              screen=inbox(all_read=True, overlays=toast("Marked 4 messages as read"))),
        done("I marked all your messages as read.",
             why="The toast says 4 messages were marked as read and the inbox shows All read."),
    ]},
    "turn-on-autopay": {"start_screen": billing(), "steps": [
        click("the Autopay switch at the top right of the Autopay card", "autopay-toggle",
              why="The Autopay card at the top right has a switch, and it is Off.",
              screen=billing(autopay=True, overlays=toast(
                  "Autopay is on. We'll pay from Checking ••4821 on each due date."))),
        done("I turned on Autopay. Your bill will be paid from Checking ••4821 on each due date; the next payment "
             "is $142.37 on Oct 28.", why="The switch is On and the toast confirms Autopay is on."),
    ]},
    "high-contrast": {"start_screen": dashboard(), "steps": [
        click("the gear icon in the top-right toolbar", "topbar-settings",
              why="High contrast is a display setting. The top bar has no Settings label, but the gear icon there "
                  "usually opens Settings.",
              screen=SETTINGS_PROFILE),
        click("the Accessibility tab", "tab-accessibility",
              why="Settings opened on the Profile tab. High contrast should be on the Accessibility tab.",
              screen=settings_a11y()),
        click("the High contrast switch at the right of the High contrast row", "a11y-high-contrast",
              why="The High contrast row has a switch, and it is Off.",
              screen=settings_a11y(hc=True, overlays=toast("High contrast turned on"))),
        done("I turned on high contrast. The portal now uses a black background with white text and yellow "
             "highlights.", why="The High contrast switch is On, the theme changed and the toast confirms it."),
    ]},
    "sms-billing-off": {"start_screen": SETTINGS_PROFILE, "steps": [
        click("the Notifications tab", "tab-notifications",
              why="Text-message alerts are notification settings, so I'll open the Notifications tab.",
              screen=settings_notif()),
        click("the SMS switch in the Billing alerts row", "notif-billing-sms",
              why="Billing alerts is the first row of the table. Its SMS switch is On and the user wants it off; "
                  "Email and Push stay as they are.",
              screen=settings_notif(changes={"Billing alerts": "On, Off, Off"}, note=SMS_OFF_NOTE,
                                    overlays=toast("SMS for Billing alerts turned off"))),
        done("I turned off text messages for billing alerts. You still get billing alerts by email.",
             why="The SMS switch in the Billing alerts row is Off and the toast confirms it."),
    ]},
    "max-usage-month": {"start_screen": dashboard(nav=False), "steps": [
        click("the three-line menu icon at the top left", "topbar-menu",
              why="The side menu is hidden. The menu icon with three lines at the top left should open it, so I can "
                  "find the usage page.",
              screen=dashboard()),
        click("Usage in the side menu", "nav-usage",
              why="The side menu is open. Power use per month is on the Usage page.",
              screen=usage()),
        click("the tallest bar in the usage chart (Jul)", canvas=("usage-chart", "usageBars", "2026-07"),
              why="The chart has one bar per month from Oct 2025 to Sep 2026. The Jul bar is the tallest, so July is "
                  "the month with the most use. Clicking it opens its details.",
              screen=usage(selected=True, overlays=toast("Showing July 2026"))),
        done("July 2026 was your highest month: 968 kWh, 30% above your monthly average. I opened its details.",
             why="The Month details card shows July 2026 with 968 kWh."),
    ]},
    "report-outage-riverside": {"start_screen": outage_map(), "steps": [
        click("the Riverside district in the lower left of the map", canvas=("outage-map", "districts", "riverside"),
              why="Riverside is labeled on the map in the lower left, next to the house icon. Clicking a district "
                  "starts a report.",
              screen=OUTAGE_DIALOG),
        click("the Submit report button at the bottom right of the dialog", "report-submit",
              why="The dialog is for Riverside and No power is already selected, which fits an outage. Submitting "
                  "sends the report.",
              screen=outage_map(reported=True, overlays=toast("Outage reported in Riverside. Reference OUT-1001."))),
        done("I reported a power outage in Riverside. Your reference number is OUT-1001.",
             why="The toast confirms the report, and Your reports lists Riverside as Received."),
    ]},
    "star-landlord-message": {"start_screen": inbox(), "steps": [
        scroll("down", "page",
               why="The messages I can see are from Acme teams and Customer Support. The landlord's message must be "
                   "further down the inbox, so I'll scroll down.",
               screen=inbox(scrolled=True)),
        click("the star icon at the left of the Dana Whitfield (Landlord) row", "star-msg-landlord",
              why="Dana Whitfield (Landlord) is the landlord. The star at the left of her row is empty; clicking it "
                  "stars the message.",
              screen=inbox(scrolled=True, starred=("Customer Support", "Dana Whitfield (Landlord)"),
                           overlays=toast("Starred: Water heater replacement on September 20"))),
        done("I starred the message from your landlord, Dana Whitfield: Water heater replacement on September 20.",
             why="The star in Dana Whitfield's row is filled and the toast confirms it."),
    ]},
    "pay-50": {"start_screen": dashboard(), "steps": [
        click("the Pay bill button in the Amount due card", "dash-pay",
              why="The Amount due card shows $142.37 and has a Pay bill button.",
              screen=pay_dialog("start")),
        click("the Other amount option in the dialog", "pay-other",
              why="The user wants to pay $50, not the full $142.37 balance.",
              screen=pay_dialog("other")),
        type_into("50", "the amount box next to the $ sign in the Other amount row", "pay-amount",
                  why="Enter 50 in the Other amount box.",
                  screen=pay_dialog("typed")),
        click("the Continue button at the bottom right of the dialog", "pay-continue",
              why="The amount is $50, so I continue to the confirmation step.",
              screen=pay_dialog("confirm")),
        click("the Confirm payment button", "pay-confirm",
              why="The confirmation shows $50.00 from Checking ••4821, which is what the user asked for.",
              screen=billing(amount_due="$92.37", overlays=toast("Payment of $50.00 received. Confirmation PAY-1001."),
                             extra="Last payment $50.00 on Oct 17, 2026 · Confirmation PAY-1001.")),
        done("I paid $50.00 toward your bill from Checking ••4821. Your balance is now $92.37 (confirmation "
             "PAY-1001).", why="The toast confirms the payment and the amount due is now $92.37."),
    ]},
    "message-support-meter": {"start_screen": dashboard(), "steps": [
        click("the Message us card in Quick actions", "quick-message",
              why="The Quick actions row has a Message us card, Write to our team. It should open a new message.",
              screen=compose()),
        click("the To dropdown that says Choose a recipient", "compose-to",
              why="First I choose who gets the message.",
              screen=RECIPIENTS),
        click("Customer Support at the top of the recipient list", "compose-to-support",
              why="The user wants to write to support.",
              screen=compose(to="Customer Support")),
        type_into("Broken meter", "the Subject field", "compose-subject",
                  why="A short subject helps support see what it's about.",
                  screen=compose(to="Customer Support", subject="Broken meter")),
        type_into("Hi, my meter is broken. Can someone take a look?", "the Message box", "compose-body",
                  why="The message says the meter is broken, in the user's words.",
                  screen=compose(to="Customer Support", subject="Broken meter",
                                 body="Hi, my meter is broken. Can someone take a look?")),
        click("the Send button at the bottom left of the form", "compose-send",
              why="Recipient, subject and message are filled in, so I'll send it.",
              screen=SENT),
        done("I sent Customer Support a message saying your meter is broken. It's in Messages under Sent.",
             why="The toast says Message sent to Customer Support and the Sent tab lists it."),
    ]},
    "stop-promotions": {"start_screen": dashboard(), "steps": [
        click("the gear icon in the top-right toolbar", "topbar-settings",
              why="Notification preferences live in Settings. The gear icon in the top bar should open it.",
              screen=SETTINGS_PROFILE),
        click("the Notifications tab", "tab-notifications",
              why="Promotions are a kind of notification, so they should be on the Notifications tab.",
              screen=settings_notif()),
        scroll("down", "page",
               why="The rows I can see are alerts about my account. There's no promotions row yet, so I'll scroll "
                   "down the table.",
               screen=settings_notif(scrolled=True)),
        click("the Email switch in the Offers and promotions row", "notif-promotions-email",
              why="Offers and promotions has Email, SMS and Push all On. The user wants none of them, so I'll turn "
                  "each one off, starting with Email.",
              screen=settings_notif(scrolled=True, changes={"Offers and promotions": "Off, On, On"},
                                    overlays=toast("Email for Offers and promotions turned off"))),
        click("the SMS switch in the Offers and promotions row", "notif-promotions-sms",
              why="Email is off now. Next, SMS for Offers and promotions.",
              screen=settings_notif(scrolled=True, changes={"Offers and promotions": "Off, Off, On"},
                                    overlays=toast("SMS for Offers and promotions turned off"))),
        click("the Push switch in the Offers and promotions row", "notif-promotions-push",
              why="Last, Push for Offers and promotions.",
              screen=settings_notif(scrolled=True, changes={"Offers and promotions": "Off, Off, Off"},
                                    note="The Offers and promotions row shows an All off badge.",
                                    overlays=toast("Push for Offers and promotions turned off"))),
        done("I turned off email, text and push notifications for offers and promotions. Your other notifications "
             "are unchanged.", why="All three switches in the Offers and promotions row are Off, with an All off badge."),
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


def _scroll_step(direction: str, amount: str, reasoning: str, screen_after: str) -> dict:
    return {"tool": "scroll", "arguments": {"direction": direction, "amount": amount},
            "x": SCROLL_ORIGIN[0], "y": SCROLL_ORIGIN[1], "reasoning": reasoning, "screen_after": screen_after}


async def reveal(page, step: dict) -> tuple[tuple[float, float], list[dict]]:
    """Make sure the target's center is clickable; scroll / dismiss the toast if not. Returns the point and any
    auto-added steps (PLANS should make these explicit)."""
    extra: list[dict] = []
    last_y = None
    for _ in range(10):
        loc, (x, y) = await measure(page, step)
        hit = await loc.evaluate(HIT_TEST_JS, [x, y])
        if hit == "ok" and y > H - EDGE_PX:
            hit = "edge"
        if hit == "ok":
            return (round(x, 1), round(y, 1)), extra
        if hit == "toast":
            _, (tx, ty) = await measure(page, {"testid": "toast-close"})
            extra.append(await _do(page, {
                "tool": "click", "arguments": {"target_description": "the X button on the toast at the top"},
                "x": round(tx, 1), "y": round(ty, 1),
                "reasoning": f"A toast is covering {step['target']}, so I'll close it first.",
                "screen_after": "The toast is closed."}))
            continue
        if last_y is not None and abs(y - last_y) < 1:
            raise CalibrationError(f"scrolling doesn't move {step['target']!r} (y={y:.0f}, hit={hit})")
        last_y = y
        direction = "down" if y >= H / 2 else "up"
        amount = "page" if (y > H + 200 or y < -200) else "small"
        extra.append(await _do(page, _scroll_step(
            direction, amount, f"I can't see {step['target']} well yet, so I'll scroll {direction}.",
            f"The page scrolled {direction}; more of it is visible.")))
    raise CalibrationError(f"could not bring {step['target']!r} into view")


def validate_script(script: dict) -> None:
    """The script must look like what Lightning + the grounder could produce: tool calls that fit the agent's
    tool schema (agent/prompts.py), pixel points inside the viewport, a final done, and no selectors."""
    start = script.get("start_path", "/")
    if not isinstance(start, str) or not start.startswith("/"):
        raise CalibrationError(f"start_path must be a page path like /billing, not {start!r}")
    steps = script.get("steps") or []
    if not steps or steps[-1].get("tool") != "done":
        raise CalibrationError("script must end with done")
    for i, step in enumerate(steps, 1):
        extra = set(step) - STEP_KEYS
        if extra:
            raise CalibrationError(f"step {i} has non-replay keys {sorted(extra)} (selectors must not leak)")
        tool, args = step.get("tool"), step.get("arguments") or {}
        if tool not in TOOL_NAMES:
            raise CalibrationError(f"step {i}: unknown tool {tool!r}")
        params = tool_parameters(tool)
        unknown = set(args) - set(params["properties"])
        missing = set(params.get("required", [])) - set(args)
        if unknown or missing:
            raise CalibrationError(f"step {i} {tool}: unknown args {sorted(unknown)}, missing {sorted(missing)}")
        for key, spec in params["properties"].items():
            if key in args and "enum" in spec and args[key] not in spec["enum"]:
                raise CalibrationError(f"step {i} {tool}: {key}={args[key]!r} not in {spec['enum']}")
        if tool == "done":
            if i != len(steps):
                raise CalibrationError(f"step {i}: done before the last step")
            continue
        if not step.get("reasoning") or not step.get("screen_after"):
            raise CalibrationError(f"step {i} {tool}: missing reasoning or screen_after")
        if tool == "click" or (tool == "type_text" and args.get("target_description")):
            x, y = step.get("x"), step.get("y")
            if not all(isinstance(v, (int, float)) for v in (x, y)) or not (0 <= x < W and 0 <= y < H):
                raise CalibrationError(f"step {i} {tool}: point ({x}, {y}) is outside the {W}x{H} viewport")


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


async def calibrate(task: dict, plan: dict, sandbox_url: str, headed: bool) -> tuple[dict, bool, str, int]:
    """Walk the plan with pixel actions. Returns (script, passed, explanation, number of auto-added steps)."""
    sid = f"calibrate-{task['id']}-{secrets.token_hex(3)}"
    reset(sandbox_url, sid, task["id"])
    out: list[dict] = []
    auto = 0
    async with open_page(headed) as page:
        await page.goto(start_url(sandbox_url, task, sid), wait_until="load")
        await settle(page)
        for i, step in enumerate(plan["steps"], 1):
            if step["tool"] == "done":
                out.append({"tool": "done", "arguments": {"summary": step["summary"], "success": step["success"]},
                            "reasoning": step["why"]})
                break
            try:
                if step["tool"] == "scroll":
                    out.append(await _do(page, _scroll_step(step["direction"], step["amount"], step["why"],
                                                            step["screen"])))
                    continue
                (x, y), extra = await reveal(page, step)
                out += extra
                auto += len(extra)
                args = {"target_description": step["target"]}
                if step["tool"] == "type_text":
                    args = {"text": step["text"], **args}
                out.append(await _do(page, {"tool": step["tool"], "arguments": args, "x": x, "y": y,
                                            "reasoning": step["why"], "screen_after": step["screen"]}))
            except Exception as e:
                what = step.get("testid") or step.get("canvas") or step["tool"]
                raise CalibrationError(f"step {i} ({what}) on {_path(page)}: {type(e).__name__}: "
                                       f"{str(e).splitlines()[0][:120]}") from e
    script = {"request": task["request"], "start_path": task["start_path"] or "/", "start_screen": plan["start_screen"],
              "steps": out}
    validate_script(script)
    passed, why = verdict(sandbox_url, sid, task)
    return script, passed, why, auto


async def replay(task: dict, scripts_path: Path, sandbox_url: str, runs_dir: str) -> tuple[bool, str]:
    """Run the script through the real mock agent from a fresh reset and check what the console relies on."""
    sid = f"calibrate-replay-{task['id']}-{secrets.token_hex(3)}"
    reset(sandbox_url, sid, task["id"])
    cfg = MockConfig(runs_dir=runs_dir, sandbox_url=sandbox_url, step_delay_s=0, scripts_path=str(scripts_path))
    events: list = []

    async def on_event(ev):
        events.append(ev)

    result = await run_task(task["request"], start_url=start_url(sandbox_url, task, sid), cfg=cfg, on_event=on_event)
    problems = [f"error: {e.payload['message']}" for e in events if e.kind == "error"]
    if not result["success_claimed"]:
        problems.append(f"success_claimed false ({result['summary']})")
    for e in events:
        if e.kind == "action" and e.payload.get("name") != "done" and e.payload.get("screen_changed") is not True:
            problems.append(f"step {e.step} {e.payload.get('name')}: the screen did not change")
        if e.kind == "tool_call" and (e.payload["name"] == "click" or (
                e.payload["name"] == "type_text" and e.payload["arguments"].get("target_description"))):
            g = next((g for g in events if g.kind == "grounding" and g.step == e.step), None)
            if g is None or not g.overlay_path or not Path(g.overlay_path).is_file():
                problems.append(f"step {e.step}: no grounding event with an overlay image")
    passed, why = verdict(sandbox_url, sid, task)
    if not passed:
        problems.append(f"verdict: {why}")
    return not problems, "; ".join(problems) or why


async def main_async(args) -> int:
    sandbox_url = args.sandbox_url.rstrip("/")
    try:
        httpx.get(f"{sandbox_url}/api/health", timeout=3, trust_env=False).raise_for_status()
    except httpx.HTTPError as e:
        print(f"sandbox not reachable at {sandbox_url}: {e}", file=sys.stderr)
        return 2
    wanted_ids = [t for t in args.tasks.split(",") if t] if args.tasks else []
    unknown = sorted(set(wanted_ids) - {t["id"] for t in TASKS})
    if unknown:
        print(f"unknown task id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    missing_plans = sorted({t["id"] for t in TASKS} - set(PLANS))
    if missing_plans:
        print(f"warning: no plan for task(s) {', '.join(missing_plans)}", file=sys.stderr)
    wanted = [t for t in TASKS if not wanted_ids or t["id"] in wanted_ids]
    out_path = Path(args.out)
    try:
        existing = json.loads(out_path.read_text()).get("tasks", {})
    except (OSError, ValueError, AttributeError):
        existing = {}
    scripts = {tid: s for tid, s in existing.items() if tid in PLANS and tid not in {t["id"] for t in wanted}}
    rows: list[dict] = []
    for task in wanted:
        t0 = time.perf_counter()
        row = {"id": task["id"], "steps": 0, "auto": 0, "walk": False, "replay": None, "why": ""}
        try:
            if task["id"] not in PLANS:
                raise CalibrationError("no plan in PLANS")
            script, row["walk"], row["why"], row["auto"] = await calibrate(task, PLANS[task["id"]], sandbox_url,
                                                                           args.headed)
            row["steps"] = len(script["steps"])
        except Exception as e:
            script, row["why"] = None, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        if row["walk"]:
            scripts[task["id"]] = script
        else:
            scripts.pop(task["id"], None)  # a failing script would only mislead the mock
        row["secs"] = time.perf_counter() - t0
        rows.append(row)
        auto = f" ({row['auto']} auto-added step(s): add them to PLANS)" if row["auto"] else ""
        print(f"  walk {task['id']}: {'PASS' if row['walk'] else 'FAIL'}  {row['why']}{auto}", flush=True)

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
                if row["walk"]:
                    t0 = time.perf_counter()
                    task = next(t for t in TASKS if t["id"] == row["id"])
                    try:
                        row["replay"], why = await replay(task, out_path, sandbox_url, tmp)
                    except Exception as e:
                        row["replay"], why = False, f"{type(e).__name__}: {e}"
                    row["secs"] += time.perf_counter() - t0
                    if not row["replay"]:
                        row["why"] = f"replay: {why}"

    print(f"\n{'task':<30} {'steps':>5} {'auto':>4}  {'walk':<5} {'replay':<6} {'secs':>5}  explanation")
    for r in rows:
        rep = "-" if r["replay"] is None else ("PASS" if r["replay"] else "FAIL")
        print(f"{r['id']:<30} {r['steps']:>5} {r['auto']:>4}  {'PASS' if r['walk'] else 'FAIL':<5} {rep:<6} "
              f"{r['secs']:>5.1f}  {r['why']}")
    failed = [r for r in rows if not r["walk"] or r["replay"] is False]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} tasks OK" + (f"; FAILED: {', '.join(r['id'] for r in failed)}"
                                                               if failed else ""))
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
