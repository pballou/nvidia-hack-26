"""BLADE pair analysis for two gui_grounding dev rollout files (stdlib only).

Runbook: docs/blade/README.md. Typical use after night-loop round R (all reads, no writes on the box):

    python3 scripts/blade_pair.py fetch --round R            # scp artifacts -> runs/blade/rR/night/...
    python3 scripts/blade_pair.py analyze --round R          # -> runs/blade/rR/{analysis.md,analysis.json,rows.jsonl,
                                                             #    knobs_proposed.json,knobs_diff.json}
    python3 scripts/blade_pair.py images --round R --n 12    # optional: overlays of failure screens -> rR/images/

`analyze` also takes explicit paths (--before/--after/--ref/--profile/--pool-stats/--train-file/--knobs), e.g.
the round-0 base-vs-SFT pair:

    python3 scripts/blade_pair.py analyze --out runs/blade/r0 --names base,sft \
        --before runs/blade/base/dev_base_qwen35.jsonl --after runs/blade/r0/night/round_0/dev_r0/rollouts.jsonl

Failure taxonomy (one primary label per failed row of the AFTER model, first match wins):
  FMT_unparsed                 no parsable point                                   (BI)
  DA_ungroundable_instruction  instruction is an AX handle, D-Bus name, pointer,
                               value literal, filename slug, no letters             (DA)
  DA_sub_precision_box         box short side < 6 px or < 0.6% of the image side    (DA/TI)
  DA_confirmed                 row or screenshot listed in --da-confirmed (checked
                               by eye with `images`)                                (DA)
  SCREEN_cluster               >= 3 failures and >= 60% of the screenshot's rows,
                               screenshot not yet reviewed (no --da-confirmed entry) (verify: DA or KG)
  CONSENSUS_miss               the reference model also misses, within 1.5% of the
                               image diagonal of this point (label or duplicate)    (DA? / TI)
  KG_wrong_annotated_element   point inside another annotated box on the screenshot (KG)
  KG_near_miss                 point within 2% of the image diagonal of the box     (KG, precision)
  KG_far_miss                  everything else                                      (KG)
Cross-cut flags: broken (BEFORE passed), attractor (AFTER clicks a repeated x or y value that the reference
model does not, and mostly misses there).

Knob proposal (only source_weights and exclude_ids; every other knob is copied unchanged):
  source_weights[s] = clip(round_to_0.25(err_s / err_all), 0.5, 2.0), + 0.25 if s has >= 5 broken rows (cap 2.0),
      err = model-attributable failures (FMT_ + KG_ labels) / rows not flagged as DA, on the AFTER model's dev run.
      Weights are absolute (recomputed each round), not multiplied onto the previous round's.
  exclude_ids = current exclude_ids + train-file rows whose instruction is ungroundable or whose box is
      sub-precision (same rules as above) + (with --profile) the not-yet-profiled train rows of screenshots
      whose >= 2 profiled rows were all never-pass. Dev ids are never added: dev rows never enter the GRPO pool.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import math
import re
import shutil
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "blade"
HOST = "permanent-tan-marmot"
SSH_OPTS = ["-T", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes"]
NIGHT_DIR = "/home/ubuntu/night"
BASE_DEV = "/home/ubuntu/kyle/nvidia-hack-26/results/dev_base_qwen35.jsonl"
DA_CONFIRMED = RUNS / "da_confirmed_dev.txt"  # dev rows/screens confirmed as DA by eye; dev is fixed, so it carries over
SOURCES = ("os_atlas/linux", "os_atlas/macos", "os_atlas/androidworld", "os_atlas/uibert", "os_atlas/fineweb")
PLATFORMS = ("desktop", "mobile", "web")

NEAR_FRAC = 0.02  # of the image diagonal
CONSENSUS_FRAC = 0.015
SUB_PRECISION_PX, SUB_PRECISION_REL = 6, 0.006
SCREEN_MIN_FAILS, SCREEN_MIN_FRAC = 3, 0.6
W_MIN, W_MAX, BROKEN_BUMP_MIN = 0.5, 2.0, 5

_JUNK = (
    ("ax_handle", re.compile(r"<AXUIElement|\{pid=\d+\}")),
    ("dbus_name", re.compile(r"^:\d+\.\d+/")),
    ("pointer_or_uuid", re.compile(r"0x[0-9a-fA-F]{6,}|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-")),
    ("value_literal", re.compile(r"^(true|false|none|null|nan|undefined)$", re.I)),
)
_SLUG = re.compile(r"^[\w.-]+$")


# --------------------------------------------------------------------------- io


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def reply_text(r: dict) -> str:
    out = []
    for o in (r.get("response") or {}).get("output", []) or []:
        if o.get("type") == "message":
            out += [c.get("text", "") for c in o.get("content", []) or [] if isinstance(c, dict)]
    return "\n".join(out).strip()


def image_url(r: dict) -> str | None:
    for m in (r.get("responses_create_params") or {}).get("input", []) or []:
        for c in m.get("content", []) if isinstance(m.get("content"), list) else []:
            if c.get("type") == "input_image":
                return c.get("image_url")
    return None


def load_rollouts(path: Path) -> dict[str, dict]:
    """id -> normalized row (dev runs have one repeat; with more, reward is the mean, the point the first's)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(path):
        groups[r["id"]].append(r)
    out = {}
    for rid, rs in groups.items():
        r = rs[0]
        w, h = r["img_w"], r["img_h"]
        pt = None
        if r.get("parsed") and r.get("pred_x") is not None and r.get("pred_y") is not None:
            pt = (r["pred_x"] / 1000.0 * w, r["pred_y"] / 1000.0 * h)
        out[rid] = {
            "id": rid, "screen": rid.rsplit(":", 1)[0], "source": r.get("source"), "platform": r.get("platform"),
            "role": r.get("data_type") or "any", "instruction": r.get("instruction", ""), "bbox": r["bbox"],
            "w": w, "h": h, "reward": sum(float(x.get("reward") or 0) for x in rs) / len(rs), "repeats": len(rs),
            "pt": pt, "pred1000": (r.get("pred_x"), r.get("pred_y")), "text": reply_text(r)[:300],
            "image": image_url(r), "masked": any(bool(x.get("mask_sample")) for x in rs),
        }
    return out


# --------------------------------------------------------------------------- rules


def junk_kind(instruction: str, source: str | None) -> str | None:
    s = (instruction or "").strip()
    if not re.search(r"[^\W\d_]", s):
        return "no_letters"
    for name, rx in _JUNK:
        if rx.search(s):
            return name
    if source == "os_atlas/fineweb" and _SLUG.match(s) and re.search(r"[-_]", s) and re.search(r"\d{3,}|\d+x\d+", s):
        return "filename_slug"  # image alt text such as "20220112-SMA-0862": not visible on the page
    return None


def sub_precision(bbox, w, h) -> bool:
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return min(bw, bh) < SUB_PRECISION_PX or min(bw / w, bh / h) < SUB_PRECISION_REL


def dist_to_box(pt, b) -> float:
    dx = max(b[0] - pt[0], 0.0, pt[0] - b[2])
    dy = max(b[1] - pt[1], 0.0, pt[1] - b[3])
    return math.hypot(dx, dy)


def inside(pt, b) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def passed(row: dict | None) -> bool:
    return bool(row) and row["reward"] >= 0.5


# --------------------------------------------------------------------------- stats


def acc(rows) -> float | None:
    rows = list(rows)
    return round(100.0 * sum(r["reward"] for r in rows) / len(rows), 2) if rows else None


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def slices(ids, rows_by_model: dict[str, dict], key: str, min_n: int = 1) -> list[dict]:
    groups = defaultdict(list)
    first = next(iter(rows_by_model.values()))
    for i in ids:
        groups[first[i][key]].append(i)
    out = []
    for val, gids in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        if len(gids) < min_n:
            continue
        d = {"slice": val, "n": len(gids)}
        for name, rows in rows_by_model.items():
            d[name] = acc(rows[i] for i in gids)
        names = list(rows_by_model)
        d["delta"] = round(d[names[1]] - d[names[0]], 2)
        out.append(d)
    return out


def attractors(after: dict, ref: dict, ids: list[str]) -> list[dict]:
    """Repeated click coordinates of the AFTER model that the reference does not share and that mostly miss."""
    found = []
    for src in sorted({after[i]["source"] for i in ids}):
        sid = [i for i in ids if after[i]["source"] == src and after[i]["pt"]]
        for axis in (0, 1):
            buckets = defaultdict(list)
            for i in sid:
                buckets[round(after[i]["pred1000"][axis] / 4)].append(i)
            for b, bids in buckets.items():
                fails = [i for i in bids if not passed(after[i])]
                ref_same = sum(1 for i in bids if ref.get(i) and ref[i]["pt"]
                               and round(ref[i]["pred1000"][axis] / 4) == b)
                if len(bids) >= 6 and len(fails) >= 0.5 * len(bids) and ref_same <= len(bids) / 3:
                    found.append({"source": src, "axis": "xy"[axis], "value_1000": b * 4, "rows": len(bids),
                                  "fails": len(fails), "ref_same_value": ref_same, "fail_ids": sorted(fails)})
    return sorted(found, key=lambda d: -d["fails"])


# --------------------------------------------------------------------------- analysis


def classify(rid, after, ref, by_screen, da_screens, screen_cluster) -> str:
    a = after[rid]
    if a["pt"] is None:
        return "FMT_unparsed"
    if junk_kind(a["instruction"], a["source"]):
        return "DA_ungroundable_instruction"
    if sub_precision(a["bbox"], a["w"], a["h"]):
        return "DA_sub_precision_box"
    if a["screen"] in da_screens or rid in da_screens:
        return "DA_confirmed"
    if a["screen"] in screen_cluster:
        return "SCREEN_cluster"
    diag = math.hypot(a["w"], a["h"])
    r = ref.get(rid)
    if r and r["pt"] and not passed(r) and math.dist(a["pt"], r["pt"]) <= CONSENSUS_FRAC * diag:
        return "CONSENSUS_miss"
    if any(o != rid and inside(a["pt"], after[o]["bbox"]) for o in by_screen[a["screen"]]):
        return "KG_wrong_annotated_element"
    if dist_to_box(a["pt"], a["bbox"]) <= NEAR_FRAC * diag:
        return "KG_near_miss"
    return "KG_far_miss"


def da_row(row: dict, da_screens: set[str]) -> bool:
    return bool(junk_kind(row["instruction"], row["source"]) or sub_precision(row["bbox"], row["w"], row["h"])
                or row["screen"] in da_screens or row["id"] in da_screens)


def analyze_pair(before, after, ref, names, da_screens) -> tuple[dict, list[dict]]:
    ids = sorted(set(before) & set(after))
    nb, na = names
    by_screen = defaultdict(list)
    for i in ids:
        by_screen[after[i]["screen"]].append(i)
    screen_cluster = set()
    screens = []
    for s, sids in by_screen.items():
        fa = sum(not passed(after[i]) for i in sids)
        fb = sum(not passed(before[i]) for i in sids)
        fr = sum(not passed(ref.get(i)) for i in sids if i in ref)
        if fa >= SCREEN_MIN_FAILS and fa >= SCREEN_MIN_FRAC * len(sids):
            if not (s in da_screens or any(i in da_screens for i in sids)):  # reviewed screens: rows classify normally
                screen_cluster.add(s)
            screens.append({"screen": s, "source": after[sids[0]]["source"], "rows": len(sids),
                            f"{na}_fails": fa, f"{nb}_fails": fb, "ref_fails": fr,
                            "image": after[sids[0]]["image"],
                            "confirmed_da_rows": sum(s in da_screens or i in da_screens for i in sids)})
    screens.sort(key=lambda d: -d[f"{na}_fails"])

    outcome = Counter()
    rows = []
    for i in ids:
        a, b = after[i], before[i]
        oc = {(True, True): "still_pass", (False, True): "fixed", (True, False): "broken",
              (False, False): "still_fail"}[(passed(b), passed(a))]
        outcome[oc] += 1
        label = None if passed(a) else classify(i, after, ref, by_screen, da_screens, screen_cluster)
        diag = math.hypot(a["w"], a["h"])
        r = ref.get(i)
        rows.append({
            "id": i, "screen": a["screen"], "source": a["source"], "platform": a["platform"], "role": a["role"],
            "instruction": a["instruction"], "bbox": a["bbox"], "img": [a["w"], a["h"]], "outcome": oc,
            "label": label, "da_row": da_row(a, da_screens), "junk": junk_kind(a["instruction"], a["source"]),
            f"{nb}_pred1000": b["pred1000"], f"{na}_pred1000": a["pred1000"],
            "ref_pred1000": r["pred1000"] if r else None,
            f"{na}_dist_diag": round(dist_to_box(a["pt"], a["bbox"]) / diag, 4) if a["pt"] else None,
            f"{na}_vs_ref_diag": round(math.dist(a["pt"], r["pt"]) / diag, 4) if a["pt"] and r and r["pt"] else None,
            f"{nb}_reply": b["text"], f"{na}_reply": a["text"], "image": a["image"],
        })
    by_id = {r["id"]: r for r in rows}
    att = attractors(after, ref, ids)
    for d in att:
        for i in d["fail_ids"]:
            by_id[i].setdefault("attractor", f'{d["axis"]}={d["value_1000"]}')

    fails = [r for r in rows if r["label"]]
    tax = Counter(r["label"] for r in fails)
    tax_src = defaultdict(Counter)
    for r in fails:
        tax_src[r["source"]][r["label"]] += 1
    broken_tax = Counter(r["label"] for r in fails if r["outcome"] == "broken")

    models = {nb: before, na: after}
    clean = [i for i in ids if not by_id[i]["da_row"]]
    per_source = slices(ids, models, "source")
    for d in per_source:
        sids = [i for i in clean if after[i]["source"] == d["slice"]]
        srows = [by_id[i] for i in ids if after[i]["source"] == d["slice"]]
        d[f"{nb}_da_adjusted"] = acc(before[i] for i in sids)
        d[f"{na}_da_adjusted"] = acc(after[i] for i in sids)
        d["da_rows"] = d["n"] - len(sids)
        d["broken"] = sum(r["outcome"] == "broken" for r in srows)
        d["fixed"] = sum(r["outcome"] == "fixed" for r in srows)
        model_err = sum(1 for r in srows if r["label"] and r["label"].startswith(("FMT_", "KG_")))
        d["model_err_rate"] = round(model_err / max(1, len(sids)), 4)
    summary = {
        "names": {"before": nb, "after": na}, "rows": len(ids),
        "missing": {"before_only": len(set(before) - set(after)), "after_only": len(set(after) - set(before))},
        "masked": {nb: sum(before[i]["masked"] for i in ids), na: sum(after[i]["masked"] for i in ids)},
        "unparsed": {nb: sum(before[i]["pt"] is None for i in ids), na: sum(after[i]["pt"] is None for i in ids)},
        "overall": {nb: acc(before[i] for i in ids), na: acc(after[i] for i in ids),
                    "delta": round(acc(after[i] for i in ids) - acc(before[i] for i in ids), 2),
                    f"{nb}_da_adjusted": acc(before[i] for i in clean), f"{na}_da_adjusted": acc(after[i] for i in clean),
                    "da_rows": len(ids) - len(clean)},
        "outcomes": dict(outcome), "mcnemar_p": mcnemar_p(outcome["fixed"], outcome["broken"]),
        "per_source": per_source, "per_platform": slices(ids, models, "platform"),
        "per_role": slices(ids, models, "role", min_n=10),
        "taxonomy": dict(tax.most_common()), "taxonomy_by_source": {s: dict(c) for s, c in sorted(tax_src.items())},
        "taxonomy_broken": dict(broken_tax.most_common()),
        "junk_rows": dict(Counter(r["junk"] for r in rows if r["junk"])),
        "screen_clusters": screens, "attractors": att,
        # A confirmed-DA row that BEFORE or the reference passes probably has a correct box (a screen id in
        # --da-confirmed flags every row of the screen): re-check it and list the bad rows by id instead.
        "da_confirmed_but_passed": sorted(r["id"] for r in fails if r["label"] == "DA_confirmed"
                                          and (passed(before[r["id"]]) or passed(ref.get(r["id"])))),
    }
    return summary, rows


# --------------------------------------------------------------------------- knobs


def default_knobs() -> dict:
    """DEFAULT_KNOBS from training/night_loop.py, parsed (not imported: that module needs the GPU venv)."""
    tree = ast.parse((REPO / "training/night_loop.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFAULT_KNOBS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("DEFAULT_KNOBS not found in training/night_loop.py")


def q(x: float) -> float:
    return round(x * 4) / 4


def propose_knobs(summary: dict, current: dict, train_rows: list[dict] | None, profile: list[dict] | None,
                  dev_ids: set[str]) -> tuple[dict, dict]:
    knobs = json.loads(json.dumps(current))
    ps = {d["slice"]: d for d in summary["per_source"]}
    tot_err = sum(d["model_err_rate"] * (d["n"] - d["da_rows"]) for d in ps.values())
    tot_n = sum(d["n"] - d["da_rows"] for d in ps.values())
    err_all = tot_err / max(1, tot_n)
    weights, why = {}, {}
    for src in SOURCES:
        d = ps.get(src)
        if not d or err_all <= 0:
            weights[src] = current["source_weights"].get(src, 1.0)
            why[src] = "no dev rows / no model errors: unchanged"
            continue
        w = min(W_MAX, max(W_MIN, q(d["model_err_rate"] / err_all)))
        note = f"model-attributable error {100 * d['model_err_rate']:.1f}% vs {100 * err_all:.1f}% overall"
        if d["broken"] >= BROKEN_BUMP_MIN:
            w = min(W_MAX, w + 0.25)
            note += f"; +0.25 for {d['broken']} broken rows"
        weights[src], why[src] = w, note
    knobs["source_weights"] = {**current.get("source_weights", {}), **weights}

    excl = set(current.get("exclude_ids", []))
    added = Counter()
    if train_rows:
        for t in train_rows:
            if t.get("split", "train") != "train" or t["id"] in dev_ids or t["id"] in excl:
                continue
            ins = t.get("instruction_original") or t.get("instruction", "")
            kind = junk_kind(ins, t.get("source"))
            if kind or sub_precision(t["bbox"], t["img_w"], t["img_h"]):
                excl.add(t["id"])
                added[f"{t['source']}:{kind or 'sub_precision_box'}"] += 1
        if profile:
            prof_rate = {p["sample"]["id"]: float(p["mean/reward"]) for p in profile if int(p.get("num_rollouts", 0)) >= 4}
            by_screen = defaultdict(list)
            for rid, rate in prof_rate.items():
                by_screen[rid.rsplit(":", 1)[0]].append(rate)
            bad = {s for s, rates in by_screen.items() if len(rates) >= 2 and max(rates) <= 0}
            for t in train_rows:
                s = t["id"].rsplit(":", 1)[0]
                if s in bad and t["id"] not in prof_rate and t["id"] not in excl and t["id"] not in dev_ids:
                    excl.add(t["id"])
                    added[f"{t['source']}:screen_never_pass"] += 1
    knobs["exclude_ids"] = sorted(excl)
    diff = {"source_weights": {s: {"from": current["source_weights"].get(s, 1.0), "to": w, "why": why[s]}
                               for s, w in weights.items() if w != current["source_weights"].get(s, 1.0)},
            "exclude_ids": {"from_count": len(current.get("exclude_ids", [])), "to_count": len(excl),
                            "added_by_rule": dict(sorted(added.items()))},
            "unchanged_keys": sorted(k for k in current if k not in ("source_weights", "exclude_ids"))}
    return knobs, diff


# --------------------------------------------------------------------------- markdown


def table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)


def markdown(summary: dict, rows: list[dict], extras: dict) -> str:
    nb, na = summary["names"]["before"], summary["names"]["after"]
    o = summary["overall"]
    lines = [f"# BLADE pair: {nb} -> {na} (auto-generated by scripts/blade_pair.py)", ""]
    for k, v in extras.get("inputs", {}).items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Headline", "",
              f"- rows {summary['rows']} (missing {summary['missing']}, masked {summary['masked']}, unparsed "
              f"{summary['unparsed']})",
              f"- overall {nb} {o[nb]} -> {na} {o[na]} (delta {o['delta']:+.2f}); DA-adjusted ({o['da_rows']} rows "
              f"dropped) {o[f'{nb}_da_adjusted']} -> {o[f'{na}_da_adjusted']}",
              f"- outcomes {summary['outcomes']}; McNemar exact p = {summary['mcnemar_p']:.2g}", "",
              "### Per source", "",
              table(summary["per_source"], ["slice", "n", nb, na, "delta", "fixed", "broken", "da_rows",
                                            f"{nb}_da_adjusted", f"{na}_da_adjusted", "model_err_rate"]),
              "", "### Per platform", "", table(summary["per_platform"], ["slice", "n", nb, na, "delta"]),
              "", "### Per element role (n >= 10)", "", table(summary["per_role"], ["slice", "n", nb, na, "delta"]),
              "", "## Failure taxonomy of " + na, "",
              table([{"label": k, "rows": v} for k, v in summary["taxonomy"].items()], ["label", "rows"]),
              "", "By source:", ""]
    labels = list(summary["taxonomy"])
    lines.append(table([{"source": s, **c} for s, c in summary["taxonomy_by_source"].items()], ["source"] + labels))
    lines += ["", f"Broken rows ({nb} pass -> {na} fail) by label: {summary['taxonomy_broken']}", ""]
    if summary.get("da_confirmed_but_passed"):
        lines += [f"**Check:** DA_confirmed rows that {nb} or the reference passes (box probably right; list bad rows "
                  f"by id instead of the screen): {summary['da_confirmed_but_passed']}", ""]
    lines += [
              "## Screen clusters (look at these first)", "",
              table(summary["screen_clusters"], ["screen", "source", "rows", f"{na}_fails", f"{nb}_fails",
                                                 "ref_fails", "confirmed_da_rows"]),
              "", "## Attractors (repeated click coordinate, mostly misses, not shared by the reference)", ""]
    for d in summary["attractors"]:
        lines.append(f"- {d['source']} {d['axis']}={d['value_1000']}: {d['rows']} rows, {d['fails']} fail, "
                     f"ref same value {d['ref_same_value']}")
    lines += ["", "## Examples per label (up to 3)", ""]
    for lab in labels:
        ex = [r for r in rows if r["label"] == lab][:3]
        lines.append(f"**{lab}**")
        for r in ex:
            b1000 = [round(r["bbox"][0] / r["img"][0] * 1000), round(r["bbox"][1] / r["img"][1] * 1000),
                     round(r["bbox"][2] / r["img"][0] * 1000), round(r["bbox"][3] / r["img"][1] * 1000)]
            lines.append(f"- `{r['id']}` \"{r['instruction'][:80]}\" box1000 {b1000}; {nb} {r[f'{nb}_pred1000']} "
                         f"{na} {r[f'{na}_pred1000']} ({r['outcome']}{', ' + r['attractor'] if r.get('attractor') else ''})")
        lines.append("")
    if extras.get("pool_stats"):
        lines += ["## Profile / pool (this round)", "", "```json", json.dumps(extras["pool_stats"], indent=1), "```", ""]
    if extras.get("knobs_diff"):
        lines += ["## Proposed knobs.json diff", "", "```json", json.dumps(extras["knobs_diff"], indent=1), "```", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- commands


def run(cmd: list[str], timeout: float, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, timeout=timeout, **kw)


def remote_tar(host: str, root: str, rels: list[str], dest: Path, timeout: float = 600) -> None:
    """Read-only: stream `tar -c` of root/rels over ssh into dest (missing files are skipped with a warning)."""
    dest.mkdir(parents=True, exist_ok=True)
    quoted = " ".join("'" + r.replace("'", "'\\''") + "'" for r in rels)
    remote = f"cd '{root}' && tar -cf - --ignore-failed-read {quoted} 2>/dev/null; true"
    ssh = subprocess.Popen(["ssh", *SSH_OPTS, host, remote], stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-xf", "-", "-C", str(dest)], stdin=ssh.stdout, timeout=timeout)
    ssh.stdout.close()
    ssh.wait(timeout=60)
    if tar.returncode != 0:
        print(f"warning: local tar exit {tar.returncode}", file=sys.stderr)


def cmd_fetch(a) -> None:
    out = a.out or RUNS / f"r{a.round}"
    r = a.round
    rels = [f"round_{r}/summary.json", f"round_{r}/knobs_used.json", f"round_{r}/pool_stats.json",
            f"round_{r}/compare/compare_report.json", f"round_{r}/compare/compare_report.md",
            f"round_{r}/dev_r{r}/rollouts.jsonl", f"round_{r}/dev_r{r}/rollouts_aggregate_metrics.json",
            f"round_{r}/dev_r{r}/rollouts_failures.jsonl", f"round_{r}/profile_r{r}/rollouts_reward_profiling.jsonl",
            *([f"round_{r}/DONE"] if r >= 1 else []), "knobs.json", "STATUS.json", "state.json", "loop.json"]
    if r >= 1:
        rels += [f"round_{r - 1}/dev_r{r - 1}/rollouts.jsonl", f"round_{r - 1}/dev_r{r - 1}/rollouts_aggregate_metrics.json"]
    remote_tar(a.host, a.night_dir, rels, out / "night")
    base = RUNS / "base"
    if not (base / Path(BASE_DEV).name).exists():
        stem = BASE_DEV.removesuffix(".jsonl")
        remote_tar(a.host, str(Path(BASE_DEV).parent),
                   [Path(BASE_DEV).name, Path(stem + "_aggregate_metrics.json").name], base)
    loop = out / "night" / "loop.json"
    if loop.exists():
        tf = json.loads(loop.read_text())["args"]["train_file"]
        if not (RUNS / "data" / Path(tf).name).exists():
            remote_tar(a.host, str(Path(tf).parent), [Path(tf).name], RUNS / "data")
    have = [p for p in rels if (out / "night" / p).exists()]
    missing = [p for p in rels if p not in have]
    print(f"fetched {len(have)}/{len(rels)} into {out / 'night'}" + (f"; missing: {missing}" if missing else ""))
    if r >= 1 and not (out / "night" / f"round_{r}/DONE").exists():
        print(f"warning: round_{r}/DONE not there yet: the round may still be running", file=sys.stderr)
    for name in (f"round_{r}/summary.json", "STATUS.json"):
        p = out / "night" / name
        if p.exists():
            d = json.loads(p.read_text())
            keep = ("round", "kept", "dev_acc_prev", "dev_acc_new", "dev_gain", "method", "pool_size", "grpo_steps",
                    "stop_reason", "phase", "updated", "hint")
            print(f"{name}: " + json.dumps({k: d[k] for k in keep if k in d}))
    print("the loop waits for knobs only if STATUS.json phase is 'round R done: waiting for BLADE ...'")


def round_paths(a) -> None:
    """Fill analyze's inputs from the fetch layout runs/blade/r<R>/night/... when --round is given."""
    if a.round is None:
        return
    out = a.out or RUNS / f"r{a.round}"
    n = out / "night"
    r = a.round
    a.out = out
    a.after = a.after or n / f"round_{r}/dev_r{r}/rollouts.jsonl"
    a.before = a.before or (n / f"round_{r - 1}/dev_r{r - 1}/rollouts.jsonl" if r >= 1 else RUNS / "base" / Path(BASE_DEV).name)
    a.ref = a.ref or RUNS / "base" / Path(BASE_DEV).name
    if a.profile is None and (n / f"round_{r}/profile_r{r}/rollouts_reward_profiling.jsonl").exists():
        a.profile = n / f"round_{r}/profile_r{r}/rollouts_reward_profiling.jsonl"
    if a.pool_stats is None and (n / f"round_{r}/pool_stats.json").exists():
        a.pool_stats = n / f"round_{r}/pool_stats.json"
    if a.knobs is None and r >= 1 and (n / "knobs.json").exists():
        a.knobs = n / "knobs.json"
    if a.train_file is None and (n / "loop.json").exists():
        tf = RUNS / "data" / Path(json.loads((n / "loop.json").read_text())["args"]["train_file"]).name
        a.train_file = tf if tf.exists() else None
    if a.names is None:
        a.names = "base,sft" if r == 0 else f"r{r - 1},r{r}"
    if a.da_confirmed is None and DA_CONFIRMED.exists():
        a.da_confirmed = [str(DA_CONFIRMED)]


def cmd_analyze(a) -> None:
    round_paths(a)
    if not (a.before and a.after and a.out):
        raise SystemExit("need --round, or --before, --after and --out")
    names = tuple((a.names or "before,after").split(","))
    before, after = load_rollouts(Path(a.before)), load_rollouts(Path(a.after))
    ref = load_rollouts(Path(a.ref)) if a.ref else before
    da_screens = set()  # screen ids and/or row ids
    for s in a.da_confirmed or []:
        if Path(s).is_file():
            da_screens |= {ln.split("#")[0].strip() for ln in Path(s).read_text().splitlines()} - {""}
        else:
            da_screens.add(s)
    summary, rows = analyze_pair(before, after, ref, names, da_screens)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    extras = {"inputs": {"before": a.before, "after": a.after, "ref": a.ref or a.before, "profile": a.profile,
                         "pool_stats": a.pool_stats, "train_file": a.train_file, "knobs": a.knobs,
                         "da_confirmed": len(da_screens)}}
    if a.pool_stats and Path(a.pool_stats).exists():
        extras["pool_stats"] = json.loads(Path(a.pool_stats).read_text())
    use_file = a.knobs and str(a.knobs) != "default" and Path(a.knobs).exists()
    current = json.loads(Path(a.knobs).read_text()) if use_file else default_knobs()
    if "knobs" in current and isinstance(current["knobs"], dict):  # a knobs_used.json was passed
        current = current["knobs"]
    train_rows = read_jsonl(Path(a.train_file)) if a.train_file else None
    profile = read_jsonl(Path(a.profile)) if a.profile else None
    knobs, diff = propose_knobs(summary, current, train_rows, profile, set(after))
    if not train_rows:
        diff["exclude_ids"]["note"] = "no --train-file: exclude_ids left as is"
    extras["knobs_diff"] = diff
    summary["inputs"] = extras["inputs"]
    write_json(out / "analysis.json", summary)
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    write_json(out / "knobs_proposed.json", knobs)
    write_json(out / "knobs_diff.json", diff)
    (out / "analysis.md").write_text(markdown(summary, rows, extras))
    o = summary["overall"]
    print(f"{names[0]} {o[names[0]]} -> {names[1]} {o[names[1]]} ({o['delta']:+.2f}); outcomes {summary['outcomes']}")
    print("taxonomy", summary["taxonomy"])
    print("screen clusters", [d["screen"] for d in summary["screen_clusters"]])
    if summary["da_confirmed_but_passed"]:
        print("CHECK: DA_confirmed rows that the BEFORE/reference model passes (box probably right):",
              summary["da_confirmed_but_passed"])
    print("knob diff", json.dumps(diff))
    print(f"wrote {out}/analysis.md, analysis.json, rows.jsonl, knobs_proposed.json, knobs_diff.json")


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", head[16:24])


def cmd_images(a) -> None:
    out_root = a.out or RUNS / f"r{a.round}"
    rows = read_jsonl(out_root / "rows.jsonl")
    summary = json.loads((out_root / "analysis.json").read_text())
    nb, na = summary["names"]["before"], summary["names"]["after"]
    by_screen = defaultdict(list)
    for r in rows:
        by_screen[r["screen"]].append(r)
    pick = [d["screen"] for d in summary["screen_clusters"]]
    rank = {"CONSENSUS_miss": 0, "DA_ungroundable_instruction": 1, "DA_sub_precision_box": 1, "KG_far_miss": 2,
            "KG_wrong_annotated_element": 3, "KG_near_miss": 4}
    for r in sorted((r for r in rows if r["label"]), key=lambda r: (r["outcome"] != "broken", rank.get(r["label"], 9))):
        if r["screen"] not in pick:
            pick.append(r["screen"])
    pick = (a.screens or []) + [s for s in pick[: a.n] if s not in (a.screens or [])] if not a.only else list(a.screens)
    paths = {s: by_screen[s][0]["image"].removeprefix("file://") for s in pick if by_screen[s][0]["image"]}
    img_dir = out_root / "images"
    root = "/home/ubuntu/data"
    rels = [p.removeprefix(root + "/") for p in paths.values() if p.startswith(root + "/")]
    rels = [p for p in rels if not (img_dir / "raw" / p).exists()]
    if rels:
        remote_tar(a.host, root, rels, img_dir / "raw")
    qlmanage = shutil.which("qlmanage")
    for s, p in paths.items():
        local = img_dir / "raw" / p.removeprefix(root + "/")
        if not local.exists():
            print(f"missing image for {s}: {p}", file=sys.stderr)
            continue
        iw, ih = png_size(local)
        rs = by_screen[s]
        sx, sy = iw / rs[0]["img"][0], ih / rs[0]["img"][1]
        lw = max(2, round(iw / 400))
        side = max(iw, ih)  # square canvas: QuickLook thumbnails crop SVGs to a top-left square
        el = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{side}" '
              f'height="{side}" viewBox="0 0 {side} {side}"><rect width="{side}" height="{side}" fill="white"/>'
              f'<image x="0" y="0" width="{iw}" height="{ih}" xlink:href="data:image/png;base64,'
              f'{base64.b64encode(local.read_bytes()).decode()}"/>']
        legend = [f"{s}  (green box = pass for {na}, orange = fail; red dot = {na}, blue = {nb}, purple = ref)"]
        for k, r in enumerate(sorted(rs, key=lambda r: r["id"])):
            x1, y1, x2, y2 = r["bbox"]
            col = "lime" if r["outcome"] in ("still_pass", "fixed") else "orange"
            el.append(f'<rect x="{x1 * sx:.1f}" y="{y1 * sy:.1f}" width="{(x2 - x1) * sx:.1f}" height="{(y2 - y1) * sy:.1f}" '
                      f'fill="none" stroke="{col}" stroke-width="{lw}"/><text x="{x1 * sx + 2:.1f}" y="{y1 * sy - 3:.1f}" '
                      f'font-size="{lw * 7}" fill="{col}" stroke="black" stroke-width="0.6">{k}</text>')
            for key, c in ((f"{nb}_pred1000", "dodgerblue"), ("ref_pred1000", "purple"), (f"{na}_pred1000", "red")):
                pt = r.get(key)
                if pt and pt[0] is not None and (key != "ref_pred1000" or pt != r.get(f"{nb}_pred1000")):
                    el.append(f'<circle cx="{pt[0] / 1000 * iw:.1f}" cy="{pt[1] / 1000 * ih:.1f}" r="{lw * 3}" '
                              f'fill="{c}" stroke="white" stroke-width="1"/><text x="{pt[0] / 1000 * iw + lw * 4:.1f}" '
                              f'y="{pt[1] / 1000 * ih:.1f}" font-size="{lw * 6}" fill="{c}" stroke="white" '
                              f'stroke-width="0.5">{k}</text>')
            legend.append(f"{k}: {r['outcome']:10s} {r['label'] or '':28s} {r['id']}  \"{r['instruction'][:90]}\"")
        el.append("</svg>")
        stem = re.sub(r"[^\w.-]", "_", s)
        svg = img_dir / f"{stem}.svg"
        svg.write_text("".join(el))
        (img_dir / f"{stem}.txt").write_text("\n".join(legend) + "\n")
        if qlmanage:
            run([qlmanage, "-t", "-s", str(min(2000, side)), "-o", str(img_dir), str(svg)], timeout=120,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"{len(paths)} overlays in {img_dir} (*.svg.png if qlmanage exists; legends in *.txt)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="read-only scp (ssh tar) of one round's artifacts")
    f.add_argument("--round", type=int, required=True)
    f.add_argument("--night-dir", default=NIGHT_DIR)
    f.add_argument("--host", default=HOST)
    f.add_argument("--out", type=Path)
    z = sub.add_parser("analyze", help="pair analysis + knob proposal")
    z.add_argument("--round", type=int)
    z.add_argument("--before", type=Path)
    z.add_argument("--after", type=Path)
    z.add_argument("--ref", type=Path, help="independent reference for CONSENSUS_miss (default: --before)")
    z.add_argument("--names", help="before,after display names")
    z.add_argument("--profile", type=Path, help="round's rollouts_reward_profiling.jsonl")
    z.add_argument("--pool-stats", type=Path)
    z.add_argument("--train-file", type=Path, help="the loop's train file (exclude_ids candidates)")
    z.add_argument("--knobs", type=Path, help="current knobs.json (or a knobs_used.json); 'default' = DEFAULT_KNOBS of training/night_loop.py")
    z.add_argument("--da-confirmed", nargs="*",
                   help="row ids / screen ids confirmed as DA by eye, or files listing them (one per line, # comments)")
    z.add_argument("--out", type=Path)
    i = sub.add_parser("images", help="fetch failure screenshots and draw overlays (after analyze)")
    i.add_argument("--round", type=int)
    i.add_argument("--out", type=Path)
    i.add_argument("--n", type=int, default=12)
    i.add_argument("--screens", nargs="*", help="screen ids to draw first (the id without its trailing :k)")
    i.add_argument("--only", action="store_true", help="draw only --screens")
    i.add_argument("--host", default=HOST)
    a = ap.parse_args()
    {"fetch": cmd_fetch, "analyze": cmd_analyze, "images": cmd_images}[a.cmd](a)


if __name__ == "__main__":
    main()
