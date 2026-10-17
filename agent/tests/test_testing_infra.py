"""Test-only infrastructure: the oracle guard, FakeLightning, and the DOM oracle on real Chromium.

Tests may read the DOM and /api/state to check results; the agent itself never does.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from agent.browser import BrowserSession
from agent.prompts import SYSTEM_PROMPT, chat_tools
from agent.testing import oracle_grounder
from agent.testing.fake_lightning import FakeLightning, Reply, click, done, say, type_text
from agent.testing.oracle_grounder import OracleGrounder


def _state(base_url: str, sid: str) -> dict:
    return httpx.get(f"{base_url}/api/state", params={"sid": sid}, timeout=5).json()


async def _box(session: BrowserSession, selector: str) -> dict:
    box = await session.page.locator(selector).bounding_box()
    assert box, f"{selector} not visible"
    return box


def _inside(x: float | None, y: float | None, box: dict) -> bool:
    return (x is not None and y is not None and box["x"] <= x <= box["x"] + box["width"]
            and box["y"] <= y <= box["y"] + box["height"])


# --- guard ---------------------------------------------------------------------------------

def test_oracle_guard_raises_outside_pytest(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(oracle_grounder, "_debug_enabled", False)
    with pytest.raises(RuntimeError, match="test-only"):
        oracle_grounder.assert_oracle_allowed()
    with pytest.raises(RuntimeError, match="test-only"):
        OracleGrounder(SimpleNamespace(page=None))
    oracle_grounder.enable_debug_oracle()  # what --use-oracle-grounder does
    oracle_grounder.assert_oracle_allowed()
    OracleGrounder(SimpleNamespace(page=None))


def test_oracle_guard_allows_pytest(monkeypatch):
    monkeypatch.setattr(oracle_grounder, "_debug_enabled", False)
    oracle_grounder.assert_oracle_allowed()
    assert OracleGrounder(SimpleNamespace(page=None)).page is None


# --- FakeLightning -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fake_lightning_tool_calls_parse_with_openai_sdk():
    fake = FakeLightning([
        Reply([click("the gear icon in the top-right toolbar"), ("click", '{"target_description": ')],
              reasoning="The settings are behind the gear."),
        type_text("Alex Kim", "the Display name field"),
    ])
    resp = await fake.client.chat.completions.create(
        model="lightning", messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "hi"}],
        tools=chat_tools(), temperature=1.0, top_p=0.95, parallel_tool_calls=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content is None
    assert choice.message.reasoning_content == "The settings are behind the gear."
    first, second = choice.message.tool_calls
    assert first.type == "function" and first.function.name == "click"
    assert json.loads(first.function.arguments) == {"target_description": "the gear icon in the top-right toolbar"}
    assert second.function.arguments == '{"target_description": '  # verbatim, malformed
    with pytest.raises(json.JSONDecodeError):
        json.loads(second.function.arguments)
    assert first.id != second.id
    assert resp.usage.total_tokens > 0

    req = fake.last_request
    assert req["model"] == "lightning" and req["temperature"] == 1.0 and req["top_p"] == 0.95
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert [t["function"]["name"] for t in req["tools"]] == ["click", "type_text", "press_key", "scroll", "done"]
    assert req["messages"][0]["content"] == SYSTEM_PROMPT and req["parallel_tool_calls"] is False

    resp = await fake.client.chat.completions.create(model="lightning", messages=[{"role": "user", "content": "x"}])
    (call,) = resp.choices[0].message.tool_calls
    assert call.function.name == "type_text"
    assert json.loads(call.function.arguments) == {"text": "Alex Kim", "target_description": "the Display name field"}
    assert fake.exhausted and len(fake.requests) == 2


@pytest.mark.asyncio
async def test_fake_lightning_text_replies_exhaustion_callable_and_errors():
    fake = FakeLightning([say("All done, high contrast is on."), Reply(status=503, content="overloaded")])
    resp = await fake.client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "stop" and msg.content == "All done, high contrast is on."
    assert not msg.tool_calls and msg.reasoning_content is None

    with pytest.raises(openai.APIStatusError) as err:  # max_retries=0: one request per script entry
        await fake.client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert err.value.status_code == 503 and len(fake.requests) == 2

    resp = await fake.client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    (call,) = resp.choices[0].message.tool_calls
    assert call.function.name == "done"
    assert json.loads(call.function.arguments) == {"summary": "script exhausted", "success": False}

    seen = []
    scripted = FakeLightning(lambda body: seen.append(len(body["messages"])) or Reply([done("ok")]))
    for n in (1, 3):
        resp = await scripted.client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}] * n)
        assert resp.choices[0].message.tool_calls[0].function.name == "done"
    assert seen == [1, 3] and not scripted.exhausted


# --- OracleGrounder on the fake sandbox ----------------------------------------------------

@pytest.mark.asyncio
async def test_oracle_locates_gear_icon_on_dashboard(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(f"{fake_sandbox_url}/?sid={sid}")
    oracle = OracleGrounder(browser_session)
    img = await browser_session.screenshot()
    loc = await oracle.locate(img, "the gear icon in the top-right toolbar")
    assert loc.error is None and loc.model == "oracle" and loc.raw.startswith("oracle: <button#gear")
    assert _inside(loc.x, loc.y, await _box(browser_session, "#gear"))
    assert loc.point_1000 == pytest.approx((loc.x / 1280 * 1000, loc.y / 800 * 1000))

    bell = await oracle.locate(img, "the bell icon")
    assert _inside(bell.x, bell.y, await _box(browser_session, "#bell"))

    await browser_session.click(loc.x, loc.y)
    await browser_session.settle()
    assert (await browser_session.url()).split("?")[0].endswith("/settings")


@pytest.mark.asyncio
async def test_oracle_locates_settings_controls_and_clicks_work(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(f"{fake_sandbox_url}/settings?sid={sid}")
    oracle = OracleGrounder(browser_session)
    img = await browser_session.screenshot()

    hc = await oracle.locate(img, "the High contrast toggle")
    assert hc.error is None and _inside(hc.x, hc.y, await _box(browser_session, "#hc"))
    large = await oracle.locate(img, "the L text size button")
    assert _inside(large.x, large.y, await _box(browser_session, "button[data-size=L]"))
    name = await oracle.locate(img, "the Display name field")
    assert _inside(name.x, name.y, await _box(browser_session, "#name"))

    await browser_session.click(hc.x, hc.y)
    await browser_session.click(large.x, large.y)
    await browser_session.settle()
    a11y = _state(fake_sandbox_url, sid)["settings"]["accessibility"]
    assert a11y == {"high_contrast": True, "text_size": "L"}


@pytest.mark.asyncio
async def test_oracle_summary_sections_state_and_overlays(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(f"{fake_sandbox_url}/settings?sid={sid}")
    oracle = OracleGrounder(browser_session)
    summary = await oracle.summarize(await browser_session.screenshot())
    text = summary.text
    assert summary.error is None and summary.model == "oracle"
    lines = text.splitlines()
    assert len(lines) <= 40
    assert lines[0].startswith("PAGE: Settings - Acme Utilities")
    assert [ln.split(":")[0] for ln in lines if not ln.startswith("- ")] == ["PAGE", "ELEMENTS", "STATE", "OVERLAYS"]
    assert "High contrast" in text and "gear icon" in text and "bell icon" in text
    assert "- High contrast toggle: off" in text
    assert '- Display name field: "Sam Rivera"' in text
    assert "OVERLAYS: none" in text

    hc = await oracle.locate(None, "the High contrast toggle")
    await browser_session.click(hc.x, hc.y)
    await browser_session.settle()
    assert "- High contrast toggle: on" in (await oracle.summarize(None)).text

    await browser_session.goto(f"{fake_sandbox_url}/?sid={sid}")
    bell = await oracle.locate(None, "the bell icon in the header")
    await browser_session.click(bell.x, bell.y)
    await browser_session.settle()
    text = (await oracle.summarize(None)).text
    overlays = text.split("OVERLAYS:")[1]
    assert "Notifications" in overlays and "September bill" in overlays


@pytest.mark.asyncio
async def test_oracle_errors_for_unmatched_targets_and_canvas(browser_session, fake_sandbox_url, sid):
    await browser_session.goto(f"{fake_sandbox_url}/?sid={sid}")
    oracle = OracleGrounder(browser_session)
    miss = await oracle.locate(None, "the Riverside district on the map")
    assert miss.x is None and miss.y is None and miss.error.startswith("oracle: no element matches")

    await browser_session.page.evaluate("""() => {
        const c = document.createElement('canvas');
        c.width = 400; c.height = 300; c.id = 'outage-map'; c.setAttribute('aria-label', 'Outage map');
        document.querySelector('main').appendChild(c);
    }""")
    on_canvas = await oracle.locate(None, "the Riverside district on the outage map")
    assert on_canvas.x is None and "canvas" in on_canvas.error
    assert "(canvas: chart or map)" in (await oracle.summarize(None)).text


_GENERIC_PAGE = """<!doctype html><html><head><title>Billing - Test</title>
<style>body{margin:0;font-family:Arial} table{margin:40px} td{padding:8px 16px} button{padding:6px}</style></head>
<body><h1>Billing</h1>
<table>
 <tr><td>August 2026</td><td>$80.10</td><td><button class="dl" aria-label="Download statement">
   <svg width="16" height="16"><path d="M0 0h16v16H0z"/></svg></button></td></tr>
 <tr><td>September 2026</td><td>$84.20</td><td><button class="dl" aria-label="Download statement">
   <svg width="16" height="16"><path d="M0 0h16v16H0z"/></svg></button></td></tr>
</table>
<label style="margin:40px"><input type="checkbox" id="autopay"> Autopay</label>
<svg width="320" height="120" style="display:block;margin:40px">
 <rect data-name="Riverside" onclick="this.dataset.hit = '1'" x="0" y="0" width="140" height="120" fill="#9c9"/>
 <rect data-name="Hillcrest" onclick="this.dataset.hit = '1'" x="160" y="0" width="140" height="120" fill="#99c"/>
</svg>
<ul>
 <li onclick="this.dataset.hit = '1'">Landlord: Rent reminder
   <button class="star" aria-label="Star" onclick="event.stopPropagation(); this.dataset.hit='1'">&#9734;</button></li>
 <li onclick="this.dataset.hit = '1'">Support: Ticket update
   <button class="star" aria-label="Star" onclick="event.stopPropagation(); this.dataset.hit='1'">&#9734;</button></li>
</ul>
<button id="prev">Previous</button> <button id="next">Next</button></body></html>"""


@pytest.mark.asyncio
async def test_oracle_generic_page_rows_labels_svg_regions_and_dialogs(browser_session):
    page = browser_session.page
    await page.set_content(_GENERIC_PAGE)
    oracle = OracleGrounder(browser_session)

    sept = await oracle.locate(None, "the Download button in the September row")
    rows = page.locator("button.dl")
    assert _inside(sept.x, sept.y, await rows.nth(1).bounding_box())
    assert not _inside(sept.x, sept.y, await rows.nth(0).bounding_box())

    autopay = await oracle.locate(None, "the Autopay checkbox")
    await browser_session.click(autopay.x, autopay.y)
    assert await page.locator("#autopay").is_checked()

    riverside = await oracle.locate(None, "the Riverside district on the map")
    await browser_session.click(riverside.x, riverside.y)
    assert await page.locator("rect[data-name=Riverside]").get_attribute("data-hit") == "1"
    assert await page.locator("rect[data-name=Hillcrest]").get_attribute("data-hit") is None

    star = await oracle.locate(None, "the star next to the message from the landlord")
    await browser_session.click(star.x, star.y)
    assert await page.locator("button.star").first.get_attribute("data-hit") == "1"
    assert await page.locator("li").first.get_attribute("data-hit") is None
    message = await oracle.locate(None, "the message from the landlord")
    assert _inside(message.x, message.y, await page.locator("li").first.bounding_box())
    assert not _inside(message.x, message.y, await page.locator("button.star").first.bounding_box())
    nxt = await oracle.locate(None, "the Next button")
    assert _inside(nxt.x, nxt.y, await page.locator("#next").bounding_box())

    text = (await oracle.summarize(None)).text
    assert "PAGE: Billing - Test | headings: Billing" in text
    assert "- Autopay checkbox: on" in text
    assert "- region: Riverside" in text and "- region: Hillcrest" in text
    assert text.count("Download statement") == 2 and "September 2026" in text  # duplicates get their row

    await page.evaluate("""() => {
        const back = document.createElement('div');
        back.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.3)';
        const dlg = document.createElement('div');
        dlg.setAttribute('role', 'dialog');
        dlg.style.cssText = 'position:fixed;left:400px;top:250px;width:400px;height:200px;background:#fff';
        dlg.innerHTML = 'Delete this statement? <button id="cancel">Cancel</button> <button id="del">Delete</button>';
        document.body.append(back, dlg);
    }""")
    covered = await oracle.locate(None, "the Download button in the September row")
    assert covered.x is None and covered.error.startswith("oracle: no element matches")
    delete = await oracle.locate(None, "the Delete button in the dialog")
    assert _inside(delete.x, delete.y, await page.locator("#del").bounding_box())
    text = (await oracle.summarize(None)).text
    elements = text.split("ELEMENTS:\n")[1].splitlines()
    assert elements[0].startswith("- button: Cancel") and elements[1].startswith("- button: Delete")
    assert "- dialog: Delete this statement? Cancel Delete @ center" in text


@pytest.mark.asyncio
async def test_oracle_summary_is_capped_at_40_lines(browser_session):
    buttons = "".join(f"<button>Item {i}</button>" for i in range(80))
    await browser_session.page.set_content(f"<title>Crowded</title><div>{buttons}</div>")
    lines = (await OracleGrounder(browser_session).summarize(None)).text.splitlines()
    assert len(lines) == 40
    assert lines[-3].startswith("- (+") and lines[-2:] == ["STATE: none", "OVERLAYS: none"]
