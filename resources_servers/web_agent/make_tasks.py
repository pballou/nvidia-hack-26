"""Build NeMo Gym rows for the `web_agent` env from sandbox tasks (sandbox/tasks.json).

A row is `responses_create_params` (agent/prompts.py's system prompt, the first user
message and the five tools) plus the flat task fields that seed_session and verify read:
task_id, start_path, check, tags, steps_min, initial_observation, initial_screen_sha,
initial_observation_source, prompt_version.

Initial observation: Gym's legacy /run path discards seed_session's response
(responses_api_agents/simple_agent/app.py:430-444), so the server can't hand the policy the
start screen. It is precomputed here with the interactive agent's own code path (fresh sid
-> POST /api/reset -> BrowserSession.goto(start_url) -> AgentSession.reset(): settle,
screenshot, one VLM summary) and baked into the first user message as
prompts.initial_user_message(request, obs.render()). The row also keeps the Observation dict
(`initial_observation`, which seed_session uses to prime the session without a VLM call) and
`initial_screen_sha` (screen_sha() of the start screenshot; seed_session compares it with the
live start screen and reports `initial_obs_stale`).

--placeholder-observation: no browser, sandbox or VLM. The summary is PLACEHOLDER_SUMMARY,
initial_screen_sha is null (seed_session skips the stale check) and
initial_observation_source is "placeholder". Plumbing only: scripts/agent_eval.sh refuses
these rows unless FORCE=1.

Open data (AGENTS.md): start-screen summaries are model output, so every output gets a
sidecar <out>.meta.json with the exact command, config, served models and prompt hashes.

Run from the repo root with .venv-gym/bin/python:
  # committed example: 4 easy + 1 medium task, placeholder start screens
  python resources_servers/web_agent/make_tasks.py --placeholder-observation \\
      --ids text-size-large,download-september-statement,mark-all-read,turn-on-autopay,high-contrast \\
      --out resources_servers/web_agent/data/example.jsonl
  # full set (not committed): T1 sandbox on :8090 + VLM on :8001. Use the summary_model the eval will use.
  python resources_servers/web_agent/make_tasks.py --summary-model base --out /home/ubuntu/data/web_agent/tasks.jsonl
  # plumbing on the fake sandbox (started in-process; still needs the VLM)
  python resources_servers/web_agent/make_tasks.py --fake --out /tmp/web_agent_fake.jsonl
  # validate rows with Gym's own row model plus this env's invariants
  python resources_servers/web_agent/make_tasks.py --check resources_servers/web_agent/data/example.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import platform
import re
import shlex
import sys
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:  # first: importing nemo_gym puts $NEMO_GYM_EXTRA_ROOTS (maybe another checkout) in front of sys.path
    import nemo_gym  # noqa: F401
except ImportError:  # optional here (--placeholder-observation); --check reports it
    pass

# Run as a file: put the repo root (agent/, common/, sandbox/) first on sys.path, so this checkout wins.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [str(REPO_ROOT), *(p for p in sys.path if p != str(REPO_ROOT))]

from PIL import Image  # noqa: E402

from agent.config import SERVED_NAMES, AgentConfig, load_config  # noqa: E402
from agent.observation import Observation, build_observation  # noqa: E402
from agent.prompts import PROMPT_VERSION, SYSTEM_PROMPT, initial_user_message, responses_tools  # noqa: E402

DEFAULT_TASKS = REPO_ROOT / "sandbox" / "tasks.json"
FAKE_SOURCE = "agent.testing.fake_sandbox.FAKE_TASKS"
PLACEHOLDER_SUMMARY = "(screen description not generated: VLM unavailable at generation time)"
PLACEHOLDER_SOURCE = "placeholder"
ROW_KEYS = ("responses_create_params", "task_id", "start_path", "check", "tags", "steps_min",
            "initial_observation", "initial_screen_sha", "initial_observation_source", "prompt_version")
RCP_KEYS = frozenset({"input", "tools", "parallel_tool_calls", "temperature", "top_p", "max_output_tokens"})
# Owned by Gym's collector or by the VerifyResponse; rows must not carry them (nor any _ng_* key).
RESERVED_KEYS = frozenset({"reward", "mask_sample", "failure_kind", "failure_reason", "response", "agent_ref",
                           "task_source", "task_input"})
SCREEN_SHA_SIZE = (160, 100)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def screen_sha(img: Image.Image) -> str:
    """Fingerprint of a screenshot: sha256 of its 160x100 grayscale BOX downsample.

    seed_session recomputes it on the live start screen (initial_obs_stale). Import this
    function there rather than re-implementing it, so both sides hash identically.
    """
    small = img.convert("L").resize(SCREEN_SHA_SIZE, Image.Resampling.BOX)
    return hashlib.sha256(small.tobytes()).hexdigest()


def start_url(sandbox_url: str, start_path: str, sid: str) -> str:
    """The sandbox URL for `start_path` with `sid` (start_path may carry its own query)."""
    return sandbox_url.rstrip("/") + start_path + ("&" if "?" in start_path else "?") + "sid=" + sid


def _without_sid(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "sid"]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _sha256(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# --- tasks ---------------------------------------------------------------------------------

def load_tasks(path: Path | None, fake: bool) -> tuple[list[dict], str]:
    """(tasks, source label)."""
    if fake:
        from agent.testing.fake_sandbox import FAKE_TASKS

        return [dict(t) for t in FAKE_TASKS], FAKE_SOURCE
    path = Path(path or DEFAULT_TASKS)
    return json.loads(path.read_text()), _display_path(path)


def select_tasks(tasks: list[dict], ids: str | None, tags: str | None, limit: int | None) -> list[dict]:
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        by_id = {t["id"]: t for t in tasks}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"unknown task id(s): {', '.join(missing)}")
        tasks = [by_id[i] for i in wanted]
    if tags:
        wanted_tags = {t.strip() for t in tags.split(",") if t.strip()}
        tasks = [t for t in tasks if wanted_tags & set(t.get("tags", []))]
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise SystemExit("no tasks selected")
    return tasks


# --- rows ----------------------------------------------------------------------------------

def make_row(task: dict, obs: Observation, *, screen: str | None, source: str, cfg: AgentConfig) -> dict:
    s = cfg.sampling
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": initial_user_message(task["request"], obs.render())},
            ],
            # strict=True would make every property required (type_text.target_description is optional).
            "tools": responses_tools(strict=False),
            "parallel_tool_calls": False,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "max_output_tokens": s.max_tokens,  # vllm_model sends it as max_tokens
        },
        "task_id": str(task["id"]),
        "start_path": task["start_path"],
        "check": task["check"],
        "tags": list(task.get("tags", [])),
        "steps_min": task.get("steps_min"),
        "initial_observation": obs.to_dict(),
        "initial_screen_sha": screen,
        "initial_observation_source": source,
        "prompt_version": PROMPT_VERSION,
    }


def placeholder_observation(task: dict, sandbox_url: str) -> Observation:
    return build_observation(step=0, url=sandbox_url.rstrip("/") + task["start_path"], title="",
                             screen_changed=None, last_action_result="", screen_summary=PLACEHOLDER_SUMMARY)


def make_vlm(cfg: AgentConfig):
    """The start-screen summarizer: the real VLM client only (tests may swap in a stub client)."""
    from agent.vlm import VLMClient

    return VLMClient(cfg)


class _LastSummary:
    """Summarizer wrapper that keeps the last Summary, so a VLM error aborts generation."""

    def __init__(self, inner: Any):
        self.inner, self.last = inner, None

    async def summarize(self, img: Image.Image):
        self.last = await self.inner.summarize(img)
        return self.last


async def observe_start(task: dict, cfg: AgentConfig, pool: Any, vlm: Any, http: Any) -> tuple[Observation, str]:
    """The start screen's Observation and screen_sha, exactly as the interactive agent sees step 0."""
    from agent.session import AgentSession

    sid = f"make-tasks-{uuid.uuid4().hex[:12]}"
    try:
        r = await http.post(cfg.sandbox_url.rstrip("/") + "/api/reset", json={"sid": sid, "task_id": task["id"]})
        ok = r.status_code == 200 and r.json().get("ok") is True
    except Exception as e:  # sandbox down, or not the sandbox
        raise RuntimeError(f"sandbox /api/reset at {cfg.sandbox_url} failed: {type(e).__name__}: {e}") from e
    if not ok:
        raise RuntimeError(f"sandbox /api/reset failed for {task['id']!r}: HTTP {r.status_code} {r.text[:200]}")
    bs = await pool.new_session(cfg.viewport, navigation_timeout_s=cfg.timeouts.navigation_s)
    summarizer = _LastSummary(vlm)
    try:
        await bs.goto(start_url(cfg.sandbox_url, task["start_path"], sid))
        obs, img, _ = await AgentSession(bs, vlm, summarizer).reset()
    finally:
        await bs.close()
    if summarizer.last is None or summarizer.last.error:
        err = summarizer.last.error if summarizer.last else "no summary"
        raise RuntimeError(f"start-screen summary failed for {task['id']!r}: {err}. "
                           "For plumbing-only rows use --placeholder-observation.")
    return dataclasses.replace(obs, url=_without_sid(obs.url)), screen_sha(img)


async def generate_live(tasks: list[dict], cfg: AgentConfig) -> list[dict]:
    import httpx

    from agent.browser import close_pool, get_pool

    vlm = make_vlm(cfg)
    rows = []
    try:
        pool = await get_pool(cfg.browser)
        async with httpx.AsyncClient(timeout=cfg.timeouts.navigation_s, trust_env=False) as http:
            for task in tasks:
                obs, sha = await observe_start(task, cfg, pool, vlm, http)
                rows.append(make_row(task, obs, screen=sha, source=f"vlm:{cfg.summary_model}", cfg=cfg))
                print(f"  {task['id']}: {len(obs.screen_summary.splitlines())} summary lines", file=sys.stderr)
    finally:
        await close_pool()
        if hasattr(vlm, "aclose"):
            await vlm.aclose()
    return rows


async def served_models(base_url: str) -> dict[str, str | None]:
    """{served name: root checkpoint} from an OpenAI-compatible /models (best effort, for the meta file)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as http:
            r = await http.get(base_url.rstrip("/") + "/models")
            return {m["id"]: m.get("root") for m in r.json().get("data", [])}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


# --- meta sidecar (open-data rule) -----------------------------------------------------------

def _display_path(p: str | Path) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _command() -> str:
    return shlex.join(["python", _display_path(sys.argv[0]), *sys.argv[1:]])


def _versions(*names: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_meta(out: Path, rows: list[dict], cfg: AgentConfig, *, source: str, tasks_source: str,
               tasks: list[dict], sandbox_note: str | None, vlm_models: dict | None) -> dict:
    from common.grounding import MAX_PIXELS, SUMMARY_PROMPT

    config = cfg.public_dict()
    config.pop("lightning", None)  # Lightning is not used to build rows
    config.update(browser=dataclasses.asdict(cfg.browser), timeouts=dataclasses.asdict(cfg.timeouts),
                  sampling=dataclasses.asdict(cfg.sampling))
    placeholder = source == PLACEHOLDER_SOURCE
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": _command(),
        "env_overrides": {k: v for k, v in sorted(os.environ.items()) if k.startswith("AGENT_") and "KEY" not in k},
        "output": _display_path(out),
        "rows": len(rows),
        "rows_sha256": _sha256(out.read_bytes()),
        "task_ids": [r["task_id"] for r in rows],
        "tasks_source": tasks_source,
        "tasks_sha256": _sha256(_canonical(tasks)),
        "initial_observation_source": source,
        "summary_model": None if placeholder else cfg.summary_model,
        "vlm_served_models": vlm_models,
        "sandbox": sandbox_note,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _sha256(SYSTEM_PROMPT),
        "tools_sha256": _sha256(_canonical(responses_tools(strict=False))),
        "summary_prompt_sha256": None if placeholder else _sha256(SUMMARY_PROMPT),
        "max_pixels": None if placeholder else MAX_PIXELS,
        "config": config,
        "python": platform.python_version(),
        "packages": _versions("playwright", "openai", "pillow", "nemo-gym"),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


# --- self-check ------------------------------------------------------------------------------

def check_rows(path: Path, tasks_by_id: dict[str, dict], *, require_gym: bool = True) -> list[str]:
    """Problems in a rows file ([] = ok): Gym's own row model, then this env's invariants."""
    problems: list[str] = []
    try:
        from nemo_gym.base_resources_server import BaseRunRequest  # what `gym dataset collate` validates
        from pydantic import ValidationError
    except ImportError as e:
        BaseRunRequest = None
        if require_gym:
            problems.append(f"nemo_gym is not importable ({e}); run with .venv-gym/bin/python")
    tools = responses_tools(strict=False)
    seen: set[str] = set()
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        problems.append(f"{path}: no rows")
    for n, line in enumerate(lines, 1):
        where = f"{path.name}:{n}"

        def bad(msg: str, where: str = where) -> None:
            problems.append(f"{where}: {msg}")

        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            bad(f"not JSON ({e})")
            continue
        if not isinstance(row, dict):
            bad("not a JSON object")
            continue
        if BaseRunRequest is not None:
            try:
                BaseRunRequest.model_validate(row)
            except ValidationError as e:
                bad(f"Gym rejects the row: {e.errors()[:3]}")
        missing = [k for k in ROW_KEYS if k not in row]
        if missing:
            bad(f"missing {missing}")
            continue
        extra = sorted(k for k in row if k in RESERVED_KEYS or k.startswith("_ng_"))
        if extra:
            bad(f"reserved keys {extra}")
        tid = row["task_id"]
        if not isinstance(tid, str) or not tid:
            bad("task_id must be a non-empty string")
            continue
        if tid in seen:
            bad(f"duplicate task_id {tid!r}")
        seen.add(tid)
        if not isinstance(row["start_path"], str) or not row["start_path"].startswith("/"):
            bad("start_path must start with '/'")
        if not isinstance(row["check"], dict) or not (row["check"].get("all") or row["check"].get("any")):
            bad("check must be {'all': [...]} (or 'any') with at least one condition")
        if not isinstance(row["tags"], list) or not all(isinstance(t, str) for t in row["tags"]):
            bad("tags must be a list of strings")

        rcp = row["responses_create_params"]
        if not isinstance(rcp, dict):
            bad("responses_create_params must be an object")
            continue
        if "tool_choice" in rcp:
            # simple_agent ends an episode only on a reply without a tool call, so "required" would run
            # every task to max_steps. Keep the default "auto" (the interactive planner uses "required").
            bad(f"tool_choice must not be set (got {rcp['tool_choice']!r}): Gym rows use the default 'auto'")
        if set(rcp) != RCP_KEYS:
            bad(f"responses_create_params keys {sorted(rcp)} != {sorted(RCP_KEYS)}")
        if rcp.get("parallel_tool_calls") is not False:
            bad("parallel_tool_calls must be false")
        msgs = rcp.get("input")
        if not (isinstance(msgs, list) and len(msgs) == 2 and [m.get("role") for m in msgs] == ["system", "user"]):
            bad("input must be [system, user]")
            continue
        if msgs[0].get("content") != SYSTEM_PROMPT:
            bad("system prompt differs from agent/prompts.py (stale row: regenerate)")
        if rcp.get("tools") != tools:
            bad("tools differ from agent/prompts.py responses_tools(strict=False) (stale row: regenerate)")
        if row["prompt_version"] != PROMPT_VERSION:
            bad(f"prompt_version {row['prompt_version']!r} != {PROMPT_VERSION!r} (stale row: regenerate)")

        try:
            obs = Observation(**row["initial_observation"])
        except TypeError as e:
            bad(f"initial_observation is not an agent.observation.Observation: {e}")
            continue
        task = tasks_by_id.get(tid)
        if task is None:
            bad(f"task {tid!r} is not in the tasks source")
        else:
            if msgs[1].get("content") != initial_user_message(task["request"], obs.render()):
                bad("first user message != initial_user_message(request, initial_observation.render())")
            for key in ("start_path", "check", "tags", "steps_min"):
                if row[key] != task.get(key):
                    bad(f"{key} differs from the tasks source (stale row: regenerate)")
        if not str(msgs[1].get("content", "")).endswith(obs.render()):
            bad("first user message does not end with initial_observation.render()")

        source, sha = row["initial_observation_source"], row["initial_screen_sha"]
        if source == PLACEHOLDER_SOURCE:
            if sha is not None:
                bad("placeholder rows must have initial_screen_sha null")
            if obs.screen_summary != PLACEHOLDER_SUMMARY:
                bad("placeholder rows must carry PLACEHOLDER_SUMMARY")
        elif isinstance(source, str) and source.startswith("vlm:") and source[4:] in SERVED_NAMES:
            if not (isinstance(sha, str) and _SHA_RE.match(sha)):
                bad("VLM rows need a sha256 initial_screen_sha")
            if obs.screen_summary == PLACEHOLDER_SUMMARY:
                bad("VLM rows must not carry the placeholder summary")
        else:
            bad(f"initial_observation_source must be 'placeholder' or 'vlm:<{'|'.join(SERVED_NAMES)}>', got {source!r}")
    return problems


def _check_tasks(args: argparse.Namespace) -> dict[str, dict]:
    """Tasks to compare rows against: the chosen source, or by default sandbox/tasks.json + FAKE_TASKS."""
    if args.fake or args.tasks:
        return {t["id"]: t for t in load_tasks(args.tasks, args.fake)[0]}
    tasks = load_tasks(DEFAULT_TASKS, False)[0] + load_tasks(None, True)[0]  # fake ids are prefixed fake_
    return {t["id"]: t for t in tasks}


# --- CLI -------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--tasks", type=Path, help=f"tasks json (default: {_display_path(DEFAULT_TASKS)})")
    src.add_argument("--fake", action="store_true", help="FAKE_TASKS on the in-process fake sandbox")
    p.add_argument("--ids", help="comma-separated task ids (kept in this order)")
    p.add_argument("--tags", help="comma-separated tags (a task with any of them is kept)")
    p.add_argument("--limit", type=int)
    p.add_argument("--config", help="agent yaml (default: $AGENT_CONFIG or configs/agent.yaml)")
    p.add_argument("--sandbox-url", help="sandbox base URL (default: sandbox_url from the agent config)")
    p.add_argument("--summary-model", choices=SERVED_NAMES, help="served name for the start-screen summary")
    p.add_argument("--placeholder-observation", action="store_true",
                   help="no browser/VLM: placeholder start screens (plumbing only, never reported)")
    p.add_argument("--out", type=Path, help="output jsonl; <out>.meta.json is written next to it")
    p.add_argument("--check", type=Path, metavar="FILE", help="validate a rows file and exit")
    args = p.parse_args(argv)
    if args.check is None and args.out is None:
        p.error("--out is required (or use --check FILE)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check is not None:
        problems = check_rows(args.check, _check_tasks(args))
        for msg in problems:
            print(f"FAIL {msg}", file=sys.stderr)
        n = sum(1 for ln in args.check.read_text().splitlines() if ln.strip())
        print(f"{args.check}: {n} rows, {'OK' if not problems else f'{len(problems)} problem(s)'}")
        return 1 if problems else 0

    overrides: dict[str, Any] = {}
    if args.sandbox_url:
        overrides["sandbox_url"] = args.sandbox_url
    if args.summary_model:
        overrides["summary_model"] = args.summary_model
    cfg = load_config(args.config, overrides)
    all_tasks, tasks_source = load_tasks(args.tasks, args.fake)
    tasks = select_tasks(all_tasks, args.ids, args.tags, args.limit)

    vlm_models = sandbox_note = None
    if args.placeholder_observation:
        source = PLACEHOLDER_SOURCE
        rows = [make_row(t, placeholder_observation(t, cfg.sandbox_url), screen=None, source=source, cfg=cfg)
                for t in tasks]
    else:
        source = f"vlm:{cfg.summary_model}"
        fake_server = None
        if args.fake:
            from agent.testing.fake_sandbox import FakeSandboxServer

            fake_server = FakeSandboxServer()
            cfg = cfg.with_options(sandbox_url=fake_server.start())
            sandbox_note = "in-process agent.testing.fake_sandbox"
        try:
            print(f"start screens: {len(tasks)} task(s), sandbox {cfg.sandbox_url}, "
                  f"summary_model {cfg.summary_model} @ {cfg.vlm.base_url}", file=sys.stderr)
            vlm_models = asyncio.run(served_models(cfg.vlm.base_url))
            rows = asyncio.run(generate_live(tasks, cfg))
        except RuntimeError as e:  # sandbox reset or VLM summary failed: nothing is written
            print(f"error: {e}", file=sys.stderr)
            return 1
        finally:
            if fake_server is not None:
                fake_server.stop()

    write_jsonl(args.out, rows)
    meta = build_meta(args.out, rows, cfg, source=source, tasks_source=tasks_source, tasks=tasks,
                      sandbox_note=sandbox_note, vlm_models=vlm_models)
    meta_path = args.out.with_name(args.out.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({source}); meta {meta_path}")

    problems = check_rows(args.out, {t["id"]: t for t in all_tasks}, require_gym=False)
    for msg in problems:
        print(f"FAIL {msg}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
