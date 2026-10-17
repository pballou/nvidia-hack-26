"""Pure metadata helpers for the Lightning rewrite pipeline: source classes, selection, styles, the ScreenSpot/GroundUI
contamination check, and the bucketed geometry/neighbor metadata Lightning sees instead of the screenshot.

OS-Atlas Linux/macOS are training data under eval option C (AGENTS.md): the 28 ScreenSpot-v2 desktop rows captured on
the same machines are excluded from the test set instead (benchmarks/screenspot_v2/excluded_rows.json)."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RAW_LABEL_SOURCES = frozenset({"os_atlas/linux", "os_atlas/macos", "os_atlas/androidworld", "os_atlas/fineweb",
                               "os_atlas/windows"})
NATURAL_SOURCES = frozenset({"os_atlas/uibert", "os_atlas/widget_captioning", "groundcua"})
SELECT_PCT = 60
STYLE_SPLIT = (("functional", 50), ("descriptive", 30), ("positional", 20))
SIZE_SMALL_MAX = 0.005
SIZE_MEDIUM_MAX = 0.03
MAX_NEIGHBORS = 8
LABEL_MAX_CHARS = 60
REQUIRED_KEYS = ("id", "source", "image_path", "img_w", "img_h", "instruction", "bbox", "platform")
ROWS = ("top", "middle", "bottom")
COLS = ("left", "center", "right")
BENCHMARK_MARKERS = ("screenspot", "groundui")  # substrings of source/id/image_path that are never training data


class ContaminationError(ValueError):
    """Manifest contains ScreenSpot (held-out test) or GroundUI-18K rows."""


class ManifestError(ValueError):
    """Manifest is malformed or has a source we don't know how to treat."""


def stable_pct(key: str, salt: str = "") -> int:
    return int(hashlib.sha256((salt + key).encode("utf-8")).hexdigest(), 16) % 100


def _names(values) -> set[str]:
    if isinstance(values, str):
        values = values.split(",")
    return {v.strip() for v in values or () if v and v.strip()}


def classify_source(source: str, extra_raw=(), extra_natural=()) -> str:
    """'raw' (bare a11y/HTML labels, ~60% get rewritten) or 'natural' (already instructions, pass through)."""
    raw, natural = _names(extra_raw), _names(extra_natural)
    if source in raw and source in natural:
        raise ManifestError(f"source {source!r} is listed in both --raw-sources and --natural-sources")
    if source in raw:
        return "raw"
    if source in natural:
        return "natural"
    if source in RAW_LABEL_SOURCES:
        return "raw"
    if source in NATURAL_SOURCES or source.startswith("groundcua/"):
        return "natural"
    raise ManifestError(f"unknown source {source!r}: pass --raw-sources {source} if its instructions are bare labels "
                        f"(~{SELECT_PCT}% get rewritten) or --natural-sources {source} if they are already natural "
                        f"instructions (passed through unchanged)")


def is_selected(row: dict, extra_raw=(), extra_natural=()) -> bool:
    return classify_source(row["source"], extra_raw, extra_natural) == "raw" and stable_pct(row["id"]) < SELECT_PCT


def assign_style(row_id: str) -> str:
    p, edge = stable_pct(row_id, "style:"), 0
    for style, share in STYLE_SPLIT:
        edge += share
        if p < edge:
            return style
    return STYLE_SPLIT[-1][0]


def _contamination_reasons(row: dict) -> list[str]:
    source, rid, image_path = (str(row.get(k) or "").lower() for k in ("source", "id", "image_path"))
    return [name for name in BENCHMARK_MARKERS if any(name in v for v in (source, rid, image_path))]


def assert_clean_manifest(rows) -> None:
    """Raise ContaminationError if any row could come from ScreenSpot (the held-out test) or GroundUI-18K."""
    counts, examples = Counter(), []
    for row in rows:
        reasons = _contamination_reasons(row)
        counts.update(reasons)
        if reasons and len(examples) < 5:
            examples.append(str(row.get("id")))
    if counts:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        raise ContaminationError(
            f"contaminated manifest ({detail}; e.g. {', '.join(examples)}). AGENTS.md: never train on ScreenSpot-v2/v1 "
            f"(ScreenSpot-v2-clean is the held-out test), Wave-UI source=='screenspot' rows or GroundUI-18K (both "
            f"contain ScreenSpot). Fix the sampler, not this check.")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_manifest(path: Path, extra_raw=(), extra_natural=()) -> list[dict]:
    """Read and check a canonical manifest JSONL (keys, unique ids, bbox, contamination, sources). No filtering."""
    path = Path(path)
    rows, seen = [], {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError(f"{where}: invalid JSON ({e})") from None
            if not isinstance(row, dict):
                raise ManifestError(f"{where}: expected a JSON object")
            missing = [k for k in REQUIRED_KEYS if k not in row]
            if missing:
                raise ManifestError(f"{where}: missing keys {missing}")
            not_str = [k for k in ("id", "source", "image_path", "platform", "instruction")
                       if not isinstance(row[k], str) or (k != "instruction" and not row[k])]
            if not_str:
                raise ManifestError(f"{where}: {not_str} must be strings (non-empty except instruction)")
            bbox = row["bbox"]
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(_is_number(v) for v in bbox)):
                raise ManifestError(f"{where}: bbox must be 4 numbers [x1, y1, x2, y2], got {bbox!r}")
            if not all(_is_number(row[k]) and row[k] > 0 for k in ("img_w", "img_h")):
                raise ManifestError(f"{where}: img_w/img_h must be positive numbers")
            if row["id"] in seen:
                raise ManifestError(f"{where}: duplicate id {row['id']!r} (first seen on line {seen[row['id']]})")
            seen[row["id"]] = lineno
            rows.append(row)
    assert_clean_manifest(rows)
    for row in rows:
        classify_source(row["source"], extra_raw, extra_natural)
    return rows


def index_by_image(rows) -> dict[str, list[dict]]:
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_path"]].append(row)
    return dict(by_image)


def control_type(row) -> str | None:
    """Control type from role/data_type, last dotted segment ('android.widget.ImageButton' -> 'ImageButton')."""
    for key in ("role", "data_type"):
        value = row.get(key)
        if value is None or str(value).strip().lower() in ("", "text", "icon"):
            continue
        segments = [s for s in str(value).strip().split(".") if s]
        if segments:
            return segments[-1]
    return None


def element_kind(row) -> str | None:
    """ScreenSpot-style 'text'/'icon' if the row carries it (data_type, else role), otherwise None."""
    for key in ("data_type", "role"):
        value = str(row.get(key) or "").strip().lower()
        if value in ("text", "icon"):
            return value
    return None


def _center(bbox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _third(v: float, size: float) -> int:
    return min(2, max(0, math.floor(3 * v / max(size, 1e-9) + 1e-9)))


def position_bucket(bbox, img_w, img_h) -> str:
    """3x3 grid cell of the bbox center; centers on a 1/3 boundary go to the later cell."""
    cx, cy = _center(bbox)
    row, col = ROWS[_third(cy, img_h)], COLS[_third(cx, img_w)]
    return "center" if (row, col) == ("middle", "center") else f"{row}-{col}"


def size_bucket(bbox, img_w, img_h) -> str:
    frac = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]) / max(img_w * img_h, 1e-9)
    return "small" if frac < SIZE_SMALL_MAX else "medium" if frac < SIZE_MEDIUM_MAX else "large"


def _inside(point, bbox) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def relative_direction(target_bbox, other_bbox) -> str:
    """Where OTHER lies relative to TARGET (image y grows downward); ties go horizontal."""
    tc, oc = _center(target_bbox), _center(other_bbox)
    if _inside(tc, other_bbox) or _inside(oc, target_bbox):
        return "overlapping"
    dx, dy = oc[0] - tc[0], oc[1] - tc[1]
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "below" if dy > 0 else "above"


def normalize_label(text) -> str:
    return " ".join(str(text).split())


def _original(row) -> str:
    return normalize_label(row.get("instruction_original") or row["instruction"])


def neighbors(row, image_rows, k=MAX_NEIGHBORS) -> list[dict]:
    """Up to k other targets on the same screenshot, nearest center first (ties by label, id). Same-label rows kept."""
    tc = _center(row["bbox"])
    ranked = []
    for other in image_rows:
        if other["id"] == row["id"]:
            continue
        oc = _center(other["bbox"])
        label = _original(other)[:LABEL_MAX_CHARS]
        ranked.append((math.hypot(oc[0] - tc[0], oc[1] - tc[1]), label, str(other["id"]), other))
    ranked.sort(key=lambda t: t[:3])
    return [{"label": label, "direction": relative_direction(row["bbox"], other["bbox"])}
            for _, label, _, other in ranked[:max(0, k)]]


def row_meta(row, by_image, style=None) -> dict:
    """Everything Lightning is told about one target. It never sees the image."""
    meta = {"label": _original(row), "platform": row["platform"]}
    control, kind = control_type(row), element_kind(row)
    if control is not None:
        meta["control"] = control
    if kind is not None:
        meta["kind"] = kind
    meta["position"] = position_bucket(row["bbox"], row["img_w"], row["img_h"])
    meta["size"] = size_bucket(row["bbox"], row["img_w"], row["img_h"])
    meta["neighbors"] = neighbors(row, by_image.get(row["image_path"], ()))
    meta["style"] = style or assign_style(row["id"])
    return meta
