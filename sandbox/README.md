# Sandbox: "Acme Utilities" customer portal

The target app the GUI Assist agent operates in, both in the NeMo Gym `web_agent`
env and in the demo. It's a realistic utility account portal with verifiable
per-session state, and with UI that accessibility-tree agents can't handle: icon-only
buttons with no accessible name, and `<canvas>` widgets with no DOM element per target.

```bash
uvicorn sandbox.app:app --host 127.0.0.1 --port 8090      # or scripts/run_sandbox.sh
python -m pytest sandbox/tests -q                          # unit + oracle + determinism + no-egress
python -m sandbox.tools.make_gallery                       # regenerate sandbox/screenshots/*.png
```

Validation (`sandbox/tests`):

- **`test_oracle.py`:** a scripted oracle (DOM selectors, allowed only here) solves all
  12 tasks from fresh resets through the UI, and asserts the screen changes after
  every step. It also shows that one plausible wrong action per task fails its check.
- **`test_a11y_gap.py`:** the deliberately unlabeled controls have no accessible
  name, the canvases have no per-target DOM and no keyboard path, the fold rules
  hold, and there are no native popups.
- **`test_determinism.py`:** 22 page states give pixel-identical screenshots in
  fresh sessions. Determinism is only guaranteed on one machine, since fonts differ
  across OSes.
- **`test_network.py`:** no request leaves 127.0.0.1.

[Gallery](screenshots/) (1280×800): dashboard, billing and the pay confirmation,
usage with July selected, the outage report dialog, inbox, compose, notification and
accessibility settings, high contrast, large text, and help.

## HTTP contract (docs/tasks/contracts.md §1)

| Route | Use |
| --- | --- |
| `GET /<page>?sid=<sid>` | First request sets the `sid` cookie; later navigation uses it. No sid and no cookie means sid `default`. |
| `POST /api/reset {"sid", "task_id"}` | Resets that session to the default state plus the task's `initial_state_overrides` → `{"ok": true}`. Unknown task: 404. |
| `GET /api/state?sid=<sid>` | Full session state. **Verifiers only, never the agent.** |
| `POST /api/action {"type", "payload"}` | The UI's only mutation endpoint (session from the cookie). Called by page JS. |
| `GET /api/health` | Liveness. |

- Tasks: [`tasks.json`](tasks.json) (12, schema in contracts.md §1).
- Checks: [`checks.py`](checks.py) `evaluate(state, check) -> (passed, explanation)`.
  Paths are dotted, with integer list indices (`messages.sent.0.to`, negative allowed).
  `exists` takes an optional `"value": false` to assert a path is absent (for
  example "exactly one payment": `billing.payments.0` exists and `billing.payments.1`
  doesn't).
- `initial_state_overrides`: `{"dotted.path": value}`. Each path must already exist in
  the default state.

## State (sandbox/state.py)

Fixed seed data; "today" is **2026-10-17**. Money is integer cents.

| Path | What |
| --- | --- |
| `session.signed_in` | Sign out from the avatar menu |
| `ui.nav_collapsed`, `ui.flash` | Hamburger nav state; the pending toast text |
| `billing.balance_cents`, `billing.autopay`, `billing.statements[]`, `billing.downloaded[]` (statement ids, deduped), `billing.payments[]` | Billing |
| `usage.months[]` (oldest first), `usage.selected_month` | Canvas bar chart; July 2026 is the unique max |
| `outages.districts[]`, `outages.active[]`, `outages.reports[]` | Canvas map |
| `messages.inbox.<id>` (`read`, `starred`), `messages.sent[]` | Messages |
| `settings.profile`, `settings.notifications.<category>.<email\|sms\|push>`, `settings.accessibility.{high_contrast,text_size,reduce_motion}` | Settings |
| `notifications.items[]`, `help.contact_requests[]`, `activity[]`, `history[]`, `meta` | The rest |

Action types (`state.ACTIONS`): `nav.toggle`, `billing.download {statement_id}`,
`billing.autopay {enabled}`, `billing.pay {amount_cents}`,
`usage.select_month {month}`, `outages.report {district, type?, details?}`,
`messages.star {id, starred}`, `messages.mark_all_read`,
`messages.send {to, subject?, body}`,
`settings.notification {category, channel, enabled}`,
`settings.accessibility {key, value}`, `settings.profile {name, email, phone}`,
`help.contact {topic, message}`, `notifications.mark_read`, `session.sign_out`,
`session.sign_in`. Opening `/messages/<id>` marks that message read.

## UI rules (for anyone editing templates/static)

1. **Viewport 1280×800, `device_scale_factor=1`.** Everything works at that size.
2. **Deterministic rendering:** no animations, transitions, smooth scrolling, timers
   or auto-dismissing toasts. JS never calls `Date`, `Math.random` or
   `performance.now`. Dates come from the server (`TODAY`).
3. **Fully local:** no CDNs, web fonts or remote images. Icons are inline SVG from
   `templates/_icons.html`. The favicon is `data:,`.
4. **Nothing outside the page's pixels:** no native `<select>`, date pickers,
   `alert/confirm/prompt` or `title` tooltips. A headless screenshot can't show them,
   so a vision agent couldn't use them. Use the in-page dropdown, modal and segmented
   control components.
5. **Text size S/M/L rescales the whole UI.** The root font size is 14/16/20 px; all
   sizes are in `rem`, and canvases scale with the root font size. **High contrast** is
   a real theme (`html.hc`).
6. **Deliberately unlabeled icon-only buttons:** hamburger, gear, per-statement
   download, mark-all-read and message stars have no text, no `aria-label`, no
   `title` and no visually-hidden text. That's the accessibility gap the vision agent
   closes. Other controls are properly labeled.
   - These controls have **no accessible name**. That's the claim, and it's tested
     in `test_a11y_gap.py`.
   - The raw HTML still carries the test hooks (rule 8) and ordinary wiring: the gear
     is a link to `/settings`, and labeled routes to the same pages exist (side nav,
     bell menu).
   - The download buttons and stars sit in rows with named neighbours.
   - So a DOM agent reading raw HTML could still find its way. Word the demo claims as
     "works from pixels; these controls have no accessible name", not "DOM agents
     fail".
   - **The canvases are mouse-only on purpose** (no tabindex, no key handling), like
     many real chart and map widgets, so keyboard input can't select a bar or a
     district blind.
7. **Every state change is visible:** a toast after each action, plus a persistent
   visual change (switch position, badge, highlighted row, details panel).
8. **Test hooks:** `data-testid` attributes, plus `window.AcmeTestHooks` for canvas
   hit targets. They exist **only for the scripted oracle in `sandbox/tests`**; the
   agent never reads the DOM.
