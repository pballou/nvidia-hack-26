"""Single owner of the grounding-model interface (docs/tasks/contracts.md §2).

Everything that talks to the grounding VLM imports from here: data prep, the Gym
`gui_grounding` and `web_agent` envs, TRL rewards, the agent's click tool and the
eval scripts. Keep these functions pure (no network, no file I/O) so training and
eval can't drift apart.

Conventions:
- Points are on Qwen3-VL's relative 0-1000 grid (x right, y down), so they don't
  change when the image is resized. Convert to pixels of whatever image (or viewport)
  you started from with `to_pixels`.
- `preprocess_screenshot` reproduces the Qwen3-VL processor's `smart_resize`
  (factor 32 = 16 px patch x 2x2 merge), so a preprocessed image passes through
  vLLM's processor unchanged as long as vLLM's cap equals MAX_PIXELS / MIN_PIXELS.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re

from PIL import Image

# Placeholder until gate D1 freezes it. MUST equal GROUNDER_MAX_PIXELS /
# GROUNDER_MIN_PIXELS in serving.env (docs/gpu-box.md §1) and the TRL processor cap.
MAX_PIXELS: int = 2_073_600
MIN_PIXELS: int = 65_536
FACTOR: int = 32  # Qwen3-VL / Qwen3.5: patch 16 x spatial merge 2
MAX_ASPECT_RATIO: float = 200.0  # the processor rejects anything more extreme

FAMILIES = ("qwen3_vl", "qwen3_5")

# Charlie's prompt from scripts/base_grounding_smoke.py. Qwen3.5 is assumed to share
# Qwen3-VL's 0-1000 point convention until stream A verifies it at the M1 baseline;
# change only its entry if it needs a different native prompt.
_GROUNDING_PROMPTS = {
    "qwen3_vl": (
        'This is a screenshot. Where should I click to accomplish this instruction: "{instruction}"?\n'
        'Output the click point in JSON format as {{"point_2d": [x, y]}}.'
    ),
    "qwen3_5": (
        'This is a screenshot. Where should I click to accomplish this instruction: "{instruction}"?\n'
        'Output the click point in JSON format as {{"point_2d": [x, y]}}.'
    ),
}

# Screen description for the text-only planner (agent/observation.py). Concrete example
# lines matter: with a "<type>" placeholder Qwen3.5-2B copied it literally, and Qwen3-VL-2B
# skipped icon-only buttons (checked on the fake sandbox, 2026-10-17).
SUMMARY_PROMPT = """\
Describe this app screenshot for someone who cannot see it, so they can tell a helper what to click.
Plain text, at most 40 lines, exactly these four sections:

PAGE: the app or page title and the main headings, top to bottom, on one line.
ELEMENTS: every visible clickable or editable element, one per line, top to bottom. Include \
icon-only buttons (describe the icon's shape), links, buttons, tabs, inputs, toggles, \
checkboxes, menu items, list rows and chart bars. Format: "- <kind>: <its text, or the icon \
shape if it has no text> (<the label or row it belongs to>) @ <position>", where position is \
top-left, top, top-right, left, center, right, bottom-left, bottom or bottom-right. Examples:
- icon button: bell icon @ top-right
- toggle: off (High contrast) @ center
- button: Download (September row) @ right
- input: "Sam Rivera" (Display name) @ center
STATE: what is selected, checked, switched on, or filled in; "none" if nothing.
OVERLAYS: any open dialog, menu, dropdown, toast or banner and its text; "none" if none.

Describe only what is visible. Never invent text."""

# Greedy decoding alone makes Qwen3-VL-2B loop on sparse screens (the same 5 lines until
# max_tokens); a mild frequency penalty fixes it without hurting Qwen3.5-2B.
SUMMARY_SAMPLING = {"temperature": 0.0, "frequency_penalty": 0.3, "max_tokens": 600}

# "point_2d": [x, y]  (also tolerates single quotes and missing quotes)
_POINT_KEY_RE = re.compile(
    r"point(?:_2d)?['\"]?\s*[:=]\s*[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]",
    re.I,
)
# "bbox_2d": [x1, y1, x2, y2] -> center (Qwen3-VL sometimes answers with a box)
_BBOX_KEY_RE = re.compile(
    r"bbox(?:_2d)?['\"]?\s*[:=]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
    re.I,
)
# <point>(x, y)</point> or <point>x y</point> (Gym gui_coordinate style)
_POINT_TAG_RE = re.compile(
    r"<point>\s*[\[(]?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[\])]?\s*</point>", re.I
)
# Last resort: the first "x, y" pair anywhere, e.g. "(512, 300)" or "[512.5, 300]".
_PAIR_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?!\w|\.\d)")


def smart_resize_dims(width: int, height: int) -> tuple[int, int]:
    """Return (W, H) after Qwen's smart_resize: multiples of FACTOR, area in [MIN, MAX]."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio of {width}x{height} exceeds {MAX_ASPECT_RATIO}")
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return w_bar, h_bar


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB, compositing any transparency onto white."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        rgba = img.convert("RGBA")
        out = Image.new("RGB", rgba.size, (255, 255, 255))
        out.paste(rgba, mask=rgba.getchannel("A"))
        return out
    return img.convert("RGB")


def preprocess_screenshot(img: Image.Image) -> Image.Image:
    """RGB, aspect-preserving resize so W, H are multiples of 32 and MIN <= W*H <= MAX.

    No padding, cropping or normalization; the processor normalizes pixels. Returns a
    new image (the input is not modified), unchanged in size when already compliant
    (e.g. a 1280x800 viewport screenshot).
    """
    rgb = _to_rgb(img)
    w, h = smart_resize_dims(*rgb.size)
    if (w, h) != rgb.size:
        rgb = rgb.resize((w, h), Image.Resampling.BICUBIC)
    elif rgb is img:
        rgb = rgb.copy()
    return rgb


def to_data_url(img: Image.Image) -> str:
    """PNG base64 data URL (for chat-completions `image_url`)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _check_family(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f"unknown model family {family!r}; expected one of {FAMILIES}")


def grounding_prompt(instruction: str, family: str = "qwen3_vl") -> str:
    """The text half of the grounding prompt (for callers that build their own message)."""
    _check_family(family)
    return _GROUNDING_PROMPTS[family].format(instruction=instruction.strip())


def build_grounding_messages(image_url: str, instruction: str, family: str = "qwen3_vl") -> list[dict]:
    """OpenAI chat-completions messages: screenshot + instruction -> point_2d on 0-1000."""
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": grounding_prompt(instruction, family)},
        ],
    }]


def build_summary_messages(image_url: str) -> list[dict]:
    """OpenAI chat-completions messages asking for a compact structured screen description."""
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": SUMMARY_PROMPT},
        ],
    }]


def _valid(x: float, y: float) -> tuple[float, float] | None:
    if math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
        return x, y
    return None


def _from_json(text: str) -> tuple[float, float] | None:
    """Parse a JSON answer like {"point_2d": [x, y]} or [{"point_2d": ...}], if the text is one."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    try:
        if isinstance(obj.get("point_2d"), list):
            x, y = obj["point_2d"][:2]
            return float(x), float(y)
        if isinstance(obj.get("bbox_2d"), list):
            x1, y1, x2, y2 = obj["bbox_2d"][:4]
            return (float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2
    except (TypeError, ValueError):
        return None
    return None


def parse_point(text: str) -> tuple[float, float] | None:
    """Extract the click point (x, y) on the 0-1000 grid from a model reply.

    Accepts {"point_2d": [x, y]} (also inside extra text or code fences), a
    {"bbox_2d": [x1, y1, x2, y2]} box (returns its center), <point>(x, y)</point>,
    or a bare "x, y" pair. Returns None if nothing parses or the point is outside
    [0, 1000] (a point off the screen is not a usable click).
    """
    if not isinstance(text, str) or not text.strip():
        return None
    pt = _from_json(text)
    if pt is None:
        if m := _POINT_KEY_RE.search(text):
            pt = float(m.group(1)), float(m.group(2))
        elif m := _BBOX_KEY_RE.search(text):
            x1, y1, x2, y2 = (float(g) for g in m.groups())
            pt = (x1 + x2) / 2, (y1 + y2) / 2
        elif m := _POINT_TAG_RE.search(text):
            pt = float(m.group(1)), float(m.group(2))
        elif m := _PAIR_RE.search(text):
            pt = float(m.group(1)), float(m.group(2))
    return _valid(*pt) if pt is not None else None


def to_pixels(pt1000: tuple[float, float], w: int, h: int) -> tuple[float, float]:
    """0-1000 grid point -> pixel coordinates in a w x h image (or CSS px of a viewport)."""
    return pt1000[0] / 1000.0 * w, pt1000[1] / 1000.0 * h


def to_grid(pt_px: tuple[float, float], w: int, h: int) -> tuple[float, float]:
    """Pixel coordinates in a w x h image -> 0-1000 grid point (inverse of to_pixels)."""
    return pt_px[0] / w * 1000.0, pt_px[1] / h * 1000.0
