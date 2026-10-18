"""agent/observation.py: URL display, summary clipping, rendering."""

from agent.observation import (MAX_LINE_CHARS, NO_CHANGE_NOTE, build_observation, clip_summary, display_url,
                               flatten_json_summary)


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


# The merged grounder on the Mac (:8011) answered the summary prompt on /billing with one JSON line
# (runs 20261018-022251-09cacf step 4, 20261018-022950-0d561e step 4), so the cap kept only PAGE and the
# start of NAV, and the confirm-payment dialog never reached Lightning. The first 240 characters below
# are from those traces; the rest is illustrative (the full reply, ~217 tokens, wasn't recorded).
MERGED_GROUNDER_JSON = (
    '{"PAGE": "Billing", "NAV": "top bar: hamburger menu icon, Acme Utilities logo, customer portal text, '
    'notification badge with 3, settings gear icon, help question mark icon, user avatar with initials JL and '
    'dropdown arrow; side menu: Dashboard, Billing, Usage, Messages, Settings, Help", '
    '"MAIN": ["top: Pay your bill @ left", "middle: Other amount $50 @ right"], '
    '"STATE": "Other amount selected", '
    '"OVERLAYS": {"Confirm payment dialog": "Pay $50.00 from Checking 4821?", "buttons": ["Back @ left", '
    '"Confirm payment @ right"]}}')


def test_json_description_becomes_lines_and_keeps_the_dialog():
    out = clip_summary(MERGED_GROUNDER_JSON).splitlines()
    assert out[0] == "PAGE: Billing"
    assert out[1].startswith("NAV: top bar: hamburger menu icon") and len(out[1]) <= MAX_LINE_CHARS
    assert "MAIN: top: Pay your bill @ left" in out and "STATE: Other amount selected" in out
    assert "OVERLAYS - Confirm payment dialog: Pay $50.00 from Checking 4821?" in out
    assert "OVERLAYS - buttons: Confirm payment @ right" in out


def test_json_description_cut_off_mid_object_is_split_at_its_keys():
    cut = MERGED_GROUNDER_JSON[:MERGED_GROUNDER_JSON.index('"STATE"') + 30]
    out = flatten_json_summary(cut).splitlines()
    assert out[0] == "PAGE: Billing" and out[-1].startswith("STATE: Other amount")
    assert any(ln.startswith("MAIN: top: Pay your bill") for ln in out)


def test_fenced_json_and_non_json_descriptions():
    assert flatten_json_summary("```json\n" + MERGED_GROUNDER_JSON + "\n```").startswith("PAGE: Billing\nNAV:")
    base = "PAGE: Pay your bill\nNAV: hamburger menu icon @ top\nMAIN - bottom-right: Continue → @ right"
    assert flatten_json_summary(base) is None and clip_summary(base) == base   # the base model's format: unchanged
    assert flatten_json_summary("") is None and flatten_json_summary("{not json at all") is None
