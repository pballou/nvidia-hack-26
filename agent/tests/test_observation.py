"""agent/observation.py: URL display, summary clipping, rendering."""

from agent.observation import MAX_LINE_CHARS, NO_CHANGE_NOTE, build_observation, clip_summary, display_url


def test_display_url_keeps_query_except_sid():
    assert display_url("http://127.0.0.1:8090/settings?tab=accessibility&sid=abc") == "/settings?tab=accessibility"
    assert display_url("http://127.0.0.1:8090/billing?sid=abc") == "/billing"
    assert display_url("http://h/messages?sid=a#top") == "/messages#top"
    assert display_url("http://h") == "/"
    assert display_url("about:blank") == "about:blank" and display_url("") == ""


def test_clip_summary_drops_repeats_and_blanks():
    looped = "PAGE: Billing\n- row: Autopay toggle @ right\n\n" + "- row: Autopay toggle @ right\n" * 11 + "STATE: none"
    assert clip_summary(looped).splitlines() == ["PAGE: Billing", "- row: Autopay toggle @ right", "STATE: none"]


def test_clip_summary_caps_lines_and_length():
    text = "\n".join(f"- item {i}" for i in range(50))
    out = clip_summary(text).splitlines()
    assert len(out) == 41 and out[-1] == "(+10 more lines omitted)"
    long = clip_summary("x" * (MAX_LINE_CHARS + 50))
    assert len(long) == MAX_LINE_CHARS and long.endswith("…")


def test_render_no_change_note_and_url():
    obs = build_observation(step=2, url="http://h/settings?tab=accessibility&sid=s", title="Settings",
                            screen_changed=False, last_action_result='clicked "x"', screen_summary="PAGE: Settings")
    text = obs.render()
    assert text.startswith("Step 2 | URL: /settings?tab=accessibility | Title: Settings")
    assert NO_CHANGE_NOTE in text and "sid" not in text
