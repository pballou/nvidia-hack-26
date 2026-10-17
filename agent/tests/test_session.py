"""AgentSession (the one tool executor) against an in-memory FakeBrowser and fake VLM roles,
plus a few cases in real Chromium (the `browser_session` fixture).

The fakes here are reused by agent/tests/test_planner.py.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest
from PIL import Image, ImageChops, ImageDraw

from agent.observation import NO_CHANGE_NOTE
from agent.session import ALREADY_DONE, DONE_NOTE, AgentSession, Located, Summary, normalize_key, validate_args

# name -> (left, top, right, bottom) in viewport px
BUTTONS = {"gear": (1200, 10, 1250, 50), "bell": (1140, 10, 1190, 50), "name field": (400, 300, 700, 340)}


class FakeBrowser:
    """1280x800 in-memory page. A click inside a BUTTONS rect toggles that button's color, so the
    screen changes; a click anywhere else changes nothing. Typed text grows a bar at the bottom."""

    def __init__(self, url: str = "http://sandbox.test/settings?sid=t", fail_on: set[str] | None = None):
        self._url = url
        self.fail_on = fail_on or set()
        self.toggled: set[str] = set()
        self.text = ""
        self.log: list[tuple] = []
        self.closed = False

    def _maybe_fail(self, op: str) -> None:
        if op in self.fail_on:
            raise RuntimeError(f"browser {op} crashed")

    def render(self) -> Image.Image:
        img = Image.new("RGB", (1280, 800), "white")
        d = ImageDraw.Draw(img)
        for name, rect in BUTTONS.items():
            d.rectangle(rect, fill=(30, 160, 60) if name in self.toggled else (120, 120, 120))
        if self.text:
            d.rectangle([0, 780, min(1279, 10 * len(self.text)), 799], fill=(0, 0, 200))
        return img

    async def goto(self, url: str) -> None:
        self.log.append(("goto", url))
        self._url = url

    async def screenshot(self) -> Image.Image:
        self._maybe_fail("screenshot")
        return self.render()

    async def click(self, x: float, y: float) -> None:
        self._maybe_fail("click")
        self.log.append(("click", x, y))
        for name, (l, t, r, b) in BUTTONS.items():
            if l <= x <= r and t <= y <= b:
                self.toggled ^= {name}

    async def type(self, text: str) -> None:
        self.log.append(("type", text))
        self.text += text

    async def press(self, key: str) -> None:
        self.log.append(("press", key))

    async def scroll(self, direction: str, amount: str) -> None:
        self.log.append(("scroll", direction, amount))

    async def select_all(self) -> None:
        self.log.append(("select_all",))
        self.text = ""

    async def settle(self) -> None:
        pass

    async def url(self) -> str:
        return self._url

    async def title(self) -> str:
        return "Settings - Fake"

    def screen_changed(self, before: Image.Image, after: Image.Image) -> bool:
        return ImageChops.difference(before.convert("RGB"), after.convert("RGB")).getbbox() is not None

    async def close(self) -> None:
        self.closed = True

    @property
    def clicks(self) -> list[tuple]:
        return [e for e in self.log if e[0] == "click"]


class FakeGrounder:
    """Finds a BUTTONS name inside the target text and returns its center; 'nowhere' -> blank area."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[str] = []

    async def locate(self, img: Image.Image, target: str) -> Located:
        self.calls.append(target)
        if self.fail:
            return Located(None, None, raw="I can't see it", error="no point in reply", model="base")
        for name, (l, t, r, b) in BUTTONS.items():
            if name in target.lower():
                x, y = (l + r) / 2, (t + b) / 2
                break
        else:
            x, y = 640.0, 600.0
        p1000 = (round(x / 1280 * 1000), round(y / 800 * 1000))
        return Located(x, y, raw=f'{{"point_2d": [{p1000[0]}, {p1000[1]}]}}', point_1000=p1000, model="base",
                       latency_ms=1.0)


class FakeSummarizer:
    def __init__(self):
        self.calls = 0

    async def summarize(self, img: Image.Image) -> Summary:
        self.calls += 1
        return Summary(text="PAGE: Settings\nELEMENTS:\n- icon button: gear @ top-right\n- input: Display name @ center",
                       model="base", latency_ms=1.0)


class FakePool:
    """Stand-in for agent.browser.BrowserPool: hands out one prepared FakeBrowser."""

    def __init__(self, browser: FakeBrowser | None = None):
        self.browser = browser or FakeBrowser()
        self.viewports: list = []
        self.kwargs: dict = {}

    async def new_session(self, viewport, **kwargs) -> FakeBrowser:
        self.viewports.append(viewport)
        self.kwargs = kwargs
        return self.browser


async def _session(**browser_kwargs) -> tuple[AgentSession, FakeBrowser]:
    b = FakeBrowser(**browser_kwargs)
    s = AgentSession(b, FakeGrounder(), FakeSummarizer())
    await s.reset()
    return s, b


# --- validate_args ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,args,ok,fragment", [
    ("click", {"target_description": "the gear icon"}, True, None),
    ("click", {}, False, "missing required"),
    ("click", {"target_description": None}, False, "missing required"),
    ("click", {"target_description": "  "}, False, "must not be empty"),
    ("click", {"target_description": "x", "x": 3}, False, "unknown argument"),
    ("click", {"target_description": 7}, False, "must be a string"),
    ("type_text", {"text": ""}, True, None),
    ("type_text", {"text": "Alex Kim", "target_description": "the Display name field"}, True, None),
    ("press_key", {"key": "Enter"}, True, None),
    ("scroll", {"direction": "down", "amount": "page"}, True, None),
    ("scroll", {"direction": "sideways", "amount": "page"}, False, "must be one of"),
    ("done", {"summary": "ok", "success": True}, True, None),
    ("done", {"summary": "ok", "success": "maybe"}, False, "true or false"),
    ("done", {"summary": "ok"}, False, "missing required"),
    ("open_settings", {}, False, "unknown tool"),
    ("click", ["the gear"], False, "JSON object"),
])
def test_validate_args(name, args, ok, fragment):
    clean, err = validate_args(name, args)
    assert (err is None) == ok, err
    if ok:
        assert isinstance(clean, dict)
    else:
        assert clean is None and fragment in err


def test_validate_args_coerces_string_bool():
    clean, err = validate_args("done", {"summary": "ok", "success": "true"})
    assert err is None and clean["success"] is True


@pytest.mark.parametrize("blank", ["", "  "])
def test_empty_optional_string_means_not_given(blank):
    # schema-valid (no minLength), and Lightning builds often fill optional fields with ""
    clean, err = validate_args("type_text", {"text": "Alex Kim", "target_description": blank})
    assert err is None and clean == {"text": "Alex Kim"}
    assert validate_args("press_key", {"key": blank})[1] is not None  # required ones must not be empty


@pytest.mark.parametrize("key,expected", [
    ("enter", "Enter"), ("ESC", "Escape"), ("Tab", "Tab"), ("F5", "F5"),
    ("ctrl+a", "ControlOrMeta+a"), ("Ctrl+A", "ControlOrMeta+A"), ("cmd+a", "ControlOrMeta+a"),
    ("command+v", "ControlOrMeta+v"), ("meta+c", "ControlOrMeta+c"), ("ctrl+shift+z", "ControlOrMeta+Shift+z"),
    ("ctrl+enter", "Control+Enter"), ("cmd+left", "Meta+ArrowLeft"), ("alt+tab", "Alt+Tab"),
    ("shift+tab", "Shift+Tab"), ("option+a", "Alt+a"),
])
def test_normalize_key(key, expected):
    assert normalize_key(key) == expected


# --- execute ---------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_observation():
    s, _ = await _session()
    obs = s.last_observation
    assert obs.step == 0 and obs.screen_changed is None
    text = obs.render()
    assert "URL: /settings" in text and "sid=" not in text  # query string hidden
    assert "Display name" in text


@pytest.mark.asyncio
async def test_unknown_tool():
    s, b = await _session()
    out = await s.execute("open_settings", {})
    assert not out.ok and "unknown tool" in out.error
    assert b.clicks == [] and out.observation is not None
    assert "open_settings failed" in out.observation.render()


@pytest.mark.asyncio
async def test_missing_args():
    s, b = await _session()
    out = await s.execute("click", {})
    assert not out.ok and "missing required argument" in out.error
    assert b.clicks == []


@pytest.mark.asyncio
async def test_click_changes_screen():
    s, b = await _session()
    out = await s.execute("click", {"target_description": "the gear icon in the top-right toolbar"})
    assert out.ok and out.screen_changed is True and not out.done
    assert b.clicks == [("click", 1225.0, 30.0)] and "gear" in b.toggled
    assert out.click["x"] == 1225.0 and out.click["point_1000"] == (957, 38)
    assert out.before is not None and out.after is not None and out.before.tobytes() != out.after.tobytes()
    assert {"ground", "act", "settle", "screenshot", "summary"} <= set(out.timings_ms)
    text = out.observation.render()
    assert "The screen changed." in text and NO_CHANGE_NOTE not in text
    assert out.observation.step == 1 and s.n_clicks == 1


@pytest.mark.asyncio
async def test_click_on_nothing_reports_no_change():
    s, b = await _session()
    out = await s.execute("click", {"target_description": "nowhere in particular"})
    assert out.ok and out.screen_changed is False
    assert len(b.clicks) == 1 and not b.toggled
    assert NO_CHANGE_NOTE in out.observation.render()
    assert "NO change" in out.one_line()


@pytest.mark.asyncio
async def test_grounder_error_does_not_click():
    b = FakeBrowser()
    s = AgentSession(b, FakeGrounder(fail=True), FakeSummarizer())
    await s.reset()
    out = await s.execute("click", {"target_description": "the gear icon"})
    assert not out.ok and "could not locate" in out.error
    assert b.clicks == []
    assert out.click["error"] == "no point in reply" and out.click["x"] is None
    assert "click" not in out.tool_payload()


@pytest.mark.asyncio
async def test_type_text_with_target_clicks_selects_and_types():
    s, b = await _session()
    out = await s.execute("type_text", {"text": "Alex Kim", "target_description": "the Display name field"})
    assert out.ok
    ops = [e for e in b.log if e[0] != "goto"]
    assert [e[0] for e in ops] == ["click", "select_all", "type"]
    assert ops[2] == ("type", "Alex Kim")
    assert 'typed "Alex Kim" into "the Display name field"' in out.observation.render()


@pytest.mark.asyncio
async def test_type_text_without_target_only_types():
    s, b = await _session()
    out = await s.execute("type_text", {"text": "hello"})
    assert out.ok and out.click is None
    assert [e[0] for e in b.log] == ["type"]


@pytest.mark.asyncio
async def test_press_key_normalized_and_scroll():
    s, b = await _session()
    assert (await s.execute("press_key", {"key": "enter"})).ok
    assert (await s.execute("scroll", {"direction": "down", "amount": "small"})).ok
    assert b.log == [("press", "Enter"), ("scroll", "down", "small")]


@pytest.mark.asyncio
async def test_done_then_any_tool_is_refused():
    s, b = await _session()
    out = await s.execute("done", {"summary": "Turned it on.", "success": True})
    assert out.ok and out.done and s.finished == {"summary": "Turned it on.", "success": True}
    assert DONE_NOTE in out.observation.render()
    after = await s.execute("click", {"target_description": "the gear icon"})
    assert not after.ok and after.error == ALREADY_DONE
    assert b.clicks == []
    again = await s.execute("done", {"summary": "again", "success": False})
    assert not again.ok and again.error == ALREADY_DONE and s.finished["success"] is True


@pytest.mark.asyncio
async def test_browser_exception_is_an_outcome():
    s, b = await _session(fail_on={"click"})
    out = await s.execute("click", {"target_description": "the gear icon"})
    assert not out.ok and "RuntimeError" in out.error and "crashed" in out.error
    text = out.observation.render()
    assert "Last action: click failed: RuntimeError: browser click crashed" in text
    assert "click failed: click failed" not in text
    assert s.summarizer.calls == 2  # re-observed: the screen may have changed anyway
    assert out.after is not None and s.last_screenshot is out.after


class ShotFailsAfterClick(FakeBrowser):
    """The screenshot right after a click raises (`fails` times), e.g. a navigation mid-capture."""

    def __init__(self, fails: int = 1):
        super().__init__()
        self.fails, self.pending = fails, 0

    async def click(self, x: float, y: float) -> None:
        await super().click(x, y)
        self.pending = self.fails

    async def screenshot(self) -> Image.Image:
        if self.pending:
            self.pending -= 1
            raise TimeoutError("Timeout 15000ms exceeded.\nCall log:\n  - taking page screenshot")
        return await super().screenshot()


@pytest.mark.asyncio
async def test_action_that_ran_but_could_not_be_captured_is_not_a_failure():
    b = ShotFailsAfterClick()
    s = AgentSession(b, FakeGrounder(), FakeSummarizer())
    await s.reset()
    out = await s.execute("click", {"target_description": "the gear icon"})
    assert b.toggled == {"gear"}  # it really happened, so the model must not be told to retry it
    assert out.ok and out.error is None and out.screen_changed is None and s.n_failed == 0
    text = out.observation.render()
    assert 'clicked "the gear icon"' in text and "could not be captured" in text and "TimeoutError" in text
    assert "Call log" not in text and NO_CHANGE_NOTE not in text
    assert out.after is not None and out.after.getpixel((1225, 30)) == (30, 160, 60)  # the fresh screen


@pytest.mark.asyncio
async def test_screen_unavailable_after_an_action():
    b = ShotFailsAfterClick(fails=2)
    s = AgentSession(b, FakeGrounder(), FakeSummarizer())
    await s.reset()
    out = await s.execute("click", {"target_description": "the gear icon"})
    assert out.ok and "(screen unavailable: TimeoutError" in out.observation.render()
    assert s.last_screenshot is None and out.after is None and s.last_observation is out.observation


class ToastBrowser(FakeBrowser):
    """Shows a toast until `toast` is cleared (as if it faded while Lightning was planning)."""
    toast = True

    def render(self) -> Image.Image:
        img = super().render()
        if self.toast:
            ImageDraw.Draw(img).rectangle([500, 700, 800, 760], fill=(20, 20, 20))
        return img


@pytest.mark.asyncio
async def test_change_is_judged_against_the_live_screen_not_the_last_observation():
    b = ToastBrowser()
    s = AgentSession(b, FakeGrounder(), FakeSummarizer())
    await s.reset()
    b.toast = False
    out = await s.execute("click", {"target_description": "nowhere in particular"})
    assert out.ok and out.screen_changed is False and NO_CHANGE_NOTE in out.observation.render()
    assert out.before.getpixel((600, 730)) == (255, 255, 255)  # grounded on the screen as it is now


class FocusBrowser(FakeBrowser):
    """Like a real page: a click outside the name field leaves nothing focused, so select-all
    highlights the whole page and typing goes nowhere."""

    def __init__(self):
        super().__init__()
        self.focused = self.page_selected = False

    async def click(self, x: float, y: float) -> None:
        self.log.append(("click", x, y))
        l, t, r, b = BUTTONS["name field"]
        self.focused, self.page_selected = l <= x <= r and t <= y <= b, False

    async def select_all(self) -> None:
        self.log.append(("select_all",))
        self.page_selected = not self.focused

    async def type(self, text: str) -> None:
        self.log.append(("type", text))
        if self.focused:
            self.text = text

    def render(self) -> Image.Image:
        img = super().render()
        if self.page_selected:
            ImageDraw.Draw(img).rectangle([40, 80, 1240, 760], fill=(180, 205, 255))
        return img


@pytest.mark.asyncio
async def test_type_text_into_a_missed_field_reports_no_change():
    b = FocusBrowser()
    s = AgentSession(b, FakeGrounder(), FakeSummarizer())
    await s.reset()
    miss = await s.execute("type_text", {"text": "Alex Kim", "target_description": "nowhere near the field"})
    assert b.page_selected and b.text == ""
    assert miss.ok and miss.screen_changed is False
    text = miss.observation.render()
    assert "the text did not appear" in text and NO_CHANGE_NOTE in text
    hit = await s.execute("type_text", {"text": "Alex Kim", "target_description": "the name field"})
    assert b.text == "Alex Kim" and hit.screen_changed is True and "did not appear" not in hit.observation.render()
    assert {"act", "screenshot"} <= set(hit.timings_ms)


# --- real Chromium ---------------------------------------------------------------------------------

class FixedGrounder:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    async def locate(self, img: Image.Image, target: str) -> Located:
        return Located(self.x, self.y, raw="fixed", model="test")


FORM = ("data:text/html;charset=utf-8," + quote(
    "<body style='margin:40px;font-family:sans-serif'><p>Some paragraph text so that a select-all shows.</p>"
    "<div><span id=lbl>Display name</span> <input id=name value='Sam Rivera'></div></body>"))


async def _center(session, selector: str) -> tuple[float, float]:
    box = await session.page.locator(selector).bounding_box()  # test-only DOM access, to aim
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


@pytest.mark.asyncio
async def test_type_text_near_miss_in_chromium_is_no_change(browser_session):
    await browser_session.goto(FORM)
    s = AgentSession(browser_session, FixedGrounder(*await _center(browser_session, "#lbl")), FakeSummarizer())
    await s.reset()
    miss = await s.execute("type_text", {"text": "Alex Kim", "target_description": "the Display name field"})
    assert await browser_session.page.input_value("#name") == "Sam Rivera"
    assert miss.ok and miss.screen_changed is False

    s.grounder = FixedGrounder(*await _center(browser_session, "#name"))
    hit = await s.execute("type_text", {"text": "Alex Kim", "target_description": "the Display name field"})
    assert await browser_session.page.input_value("#name") == "Alex Kim"
    assert hit.screen_changed is True


@pytest.mark.asyncio
async def test_ctrl_a_selects_all_on_every_os(browser_session):
    await browser_session.goto(FORM)
    await browser_session.click(*await _center(browser_session, "#name"))
    s = AgentSession(browser_session, FakeGrounder(), FakeSummarizer())
    await s.reset()
    for key in ("ctrl+a", "Backspace"):  # how Lightning writes "clear this field"
        assert (await s.execute("press_key", {"key": key})).ok
    assert await browser_session.page.input_value("#name") == ""


@pytest.mark.asyncio
async def test_tool_payload_shape():
    s, _ = await _session()
    ok = (await s.execute("click", {"target_description": "the gear icon"})).tool_payload()
    assert set(ok) == {"ok", "observation", "click"}
    assert ok["ok"] is True and ok["click"] == {"x": 1225, "y": 30} and isinstance(ok["observation"], str)
    bad = (await s.execute("scroll", {"direction": "left", "amount": "page"})).tool_payload()
    assert set(bad) == {"ok", "observation", "error"} and bad["ok"] is False
    keys = (await s.execute("press_key", {"key": "Tab"})).tool_payload()
    assert set(keys) == {"ok", "observation"}
