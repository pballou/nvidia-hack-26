"""Unit tests for common/grounding.py. Run from the repo root: python -m pytest common/tests"""

import base64
import io

import pytest
from PIL import Image

from common import grounding as g


def _area_ok(w, h):
    return w % g.FACTOR == 0 and h % g.FACTOR == 0 and g.MIN_PIXELS <= w * h <= g.MAX_PIXELS


# --- preprocess_screenshot / smart_resize_dims -------------------------------------

@pytest.mark.parametrize("size", [
    (1280, 800),     # our Playwright viewport: already compliant, untouched
    (1920, 1080),    # desktop
    (2560, 1440),    # big desktop: must shrink
    (2880, 1800),    # retina macOS
    (3840, 2160),    # 4K
    (1080, 2400),    # tall phone
    (1170, 2532),    # iPhone
    (720, 1280),     # small phone
    (100, 60),       # tiny: must grow to MIN_PIXELS
    (33, 17),        # tinier, odd sizes
    (8000, 6000),    # huge
    (5000, 200),     # very wide strip (ratio 25)
    (200, 5000),     # very tall strip
])
def test_resize_constraints(size):
    w, h = size
    out = g.preprocess_screenshot(Image.new("RGB", size, (10, 20, 30)))
    W, H = out.size
    assert out.mode == "RGB"
    assert _area_ok(W, H), (size, out.size)
    # aspect preserved up to the rounding to multiples of 32
    assert abs(W / H - w / h) / (w / h) < 0.1
    assert out.size == g.smart_resize_dims(w, h)


def test_viewport_passes_through_unchanged():
    img = Image.new("RGB", (1280, 800), (1, 2, 3))
    out = g.preprocess_screenshot(img)
    assert out.size == (1280, 800)
    assert out is not img  # always a new image
    assert out.getpixel((0, 0)) == (1, 2, 3)


def test_resize_is_idempotent():
    for size in [(2560, 1440), (1170, 2532), (100, 60), (1366, 768)]:
        once = g.preprocess_screenshot(Image.new("RGB", size))
        assert g.smart_resize_dims(*once.size) == once.size


def test_matches_reference_smart_resize():
    """Same math as transformers' Qwen2-VL smart_resize with factor 32 (copied inline)."""
    import math

    def ref(height, width, factor=32, min_pixels=g.MIN_PIXELS, max_pixels=g.MAX_PIXELS):
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    for w, h in [(1280, 800), (1920, 1080), (2560, 1440), (1080, 2400), (300, 200), (4000, 3000), (1366, 768), (15, 3000), (3000, 15), (10, 10)]:
        rh, rw = ref(h, w)
        assert g.smart_resize_dims(w, h) == (rw, rh)


def test_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        g.smart_resize_dims(0, 100)
    with pytest.raises(ValueError):
        g.smart_resize_dims(10_000, 20)  # aspect ratio 500 > 200


def test_alpha_composited_on_white():
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))  # fully transparent red
    img.putpixel((0, 0), (255, 0, 0, 255))
    out = g.preprocess_screenshot(img)
    assert out.mode == "RGB"
    # resized from 64x64 up to 256x256 (MIN_PIXELS); far corner is white
    assert out.getpixel((out.width - 1, out.height - 1)) == (255, 255, 255)


@pytest.mark.parametrize("mode", ["L", "LA", "P", "I;16", "CMYK"])
def test_other_modes_become_rgb(mode):
    out = g.preprocess_screenshot(Image.new(mode, (1280, 800)))
    assert out.mode == "RGB" and out.size == (1280, 800)


def test_palette_transparency():
    img = Image.new("P", (1280, 800), 0)
    img.info["transparency"] = 0
    out = g.preprocess_screenshot(img)
    assert out.getpixel((5, 5)) == (255, 255, 255)


def test_input_not_mutated():
    img = Image.new("RGBA", (3000, 2000), (0, 0, 0, 128))
    g.preprocess_screenshot(img)
    assert img.mode == "RGBA" and img.size == (3000, 2000)


# --- to_data_url ---------------------------------------------------------------------

def test_data_url_roundtrip():
    img = Image.new("RGB", (64, 32), (7, 8, 9))
    url = g.to_data_url(img)
    assert url.startswith("data:image/png;base64,")
    back = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert back.size == (64, 32) and back.convert("RGB").getpixel((3, 3)) == (7, 8, 9)


# --- coordinates ---------------------------------------------------------------------

@pytest.mark.parametrize("w,h", [(1280, 800), (1080, 2400), (1920, 1080), (37, 91)])
def test_pixels_grid_roundtrip(w, h):
    for px in [(0, 0), (w, h), (w / 2, h / 3), (1.5, h - 0.5)]:
        x, y = g.to_pixels(g.to_grid(px, w, h), w, h)
        assert x == pytest.approx(px[0]) and y == pytest.approx(px[1])
    for pt in [(0, 0), (1000, 1000), (500, 500), (123.4, 987.6)]:
        x, y = g.to_grid(g.to_pixels(pt, w, h), w, h)
        assert x == pytest.approx(pt[0]) and y == pytest.approx(pt[1])


def test_to_pixels_values():
    assert g.to_pixels((500, 500), 1280, 800) == (640.0, 400.0)
    assert g.to_pixels((1000, 0), 1280, 800) == (1280.0, 0.0)


def test_grid_is_resize_invariant():
    """The same 0-1000 point lands on the same relative spot in the original and resized image."""
    w, h = 2560, 1440
    W, H = g.smart_resize_dims(w, h)
    pt = (731.0, 222.0)
    x0, y0 = g.to_pixels(pt, w, h)
    x1, y1 = g.to_pixels(pt, W, H)
    assert x0 / w == pytest.approx(x1 / W) and y0 / h == pytest.approx(y1 / H)


# --- parse_point -----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"point_2d": [512, 300]}', (512.0, 300.0)),
    ('{"point_2d": [512.5, 300.25]}', (512.5, 300.25)),
    ('```json\n{"point_2d": [10, 20]}\n```', (10.0, 20.0)),
    ('[{"point_2d": [1, 2], "label": "gear"}]', (1.0, 2.0)),
    ('Sure! The gear icon is here: {"point_2d": [940, 35]}. Click it.', (940.0, 35.0)),
    ("{'point_2d': [7, 8]}", (7.0, 8.0)),
    ('point_2d: [ 100 , 200 ]', (100.0, 200.0)),
    ('{"bbox_2d": [100, 200, 300, 400]}', (200.0, 300.0)),
    ('The box is "bbox_2d": [0, 0, 10, 20] I think', (5.0, 10.0)),
    ("<point>(412,88)</point>", (412.0, 88.0)),
    ("<point>412 88</point>", (412.0, 88.0)),
    ("Click at (512, 300).", (512.0, 300.0)),
    ("[333.3, 666.6]", (333.3, 666.6)),
    ('{"point_2d": [1000, 1000]}', (1000.0, 1000.0)),
    ('{"point_2d": [0, 0]}', (0.0, 0.0)),
])
def test_parse_point_ok(text, expected):
    assert g.parse_point(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "",
    "   ",
    None,
    "I cannot find that element.",
    '{"point_2d": []}',
    '{"point_2d": [512]}',
    '{"point_2d": [1200, 300]}',   # off the 0-1000 grid
    '{"point_2d": [-5, 300]}',
    "point_2d: [nan, 3]",
    "version 3.5 of the app",
])
def test_parse_point_bad(text):
    assert g.parse_point(text) is None


def test_parse_point_prefers_key_over_earlier_pair():
    assert g.parse_point('Screen 1280, 800. {"point_2d": [10, 20]}') == (10.0, 20.0)


# --- messages ----------------------------------------------------------------------------

def test_grounding_messages_shape():
    msgs = g.build_grounding_messages("data:image/png;base64,AAAA", "  the gear icon in the top bar ")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    img, txt = msgs[0]["content"]
    assert img == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert txt["type"] == "text"
    assert '"the gear icon in the top bar"' in txt["text"]
    assert '{"point_2d": [x, y]}' in txt["text"]


def test_grounding_family_switch():
    a = g.build_grounding_messages("u", "x", family="qwen3_vl")
    b = g.build_grounding_messages("u", "x", family="qwen3_5")
    assert a[0]["content"][0] == b[0]["content"][0]
    with pytest.raises(ValueError):
        g.build_grounding_messages("u", "x", family="llava")


def test_summary_messages_shape():
    msgs = g.build_summary_messages("file:///tmp/x.png")
    img, txt = msgs[0]["content"]
    assert img["image_url"]["url"] == "file:///tmp/x.png"
    for section in ("PAGE:", "ELEMENTS:", "STATE:", "OVERLAYS:"):
        assert section in txt["text"]
    assert "40 lines" in txt["text"]
