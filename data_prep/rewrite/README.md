# Lightning data engine (T3)

Nemotron 3.5 Lightning rewrites raw OS-Atlas labels ("More options") into short click instructions ("open more
options"). It only sees metadata, never the image. The spec is
[docs/tasks/T3-lightning-data-engine.md](../../docs/tasks/T3-lightning-data-engine.md).

Which rows get rewritten: 60% of the raw-label rows (`os_atlas/linux`, `macos`, `androidworld`, `fineweb`, `windows`),
chosen by hash of `id`. Natural-instruction sources (`uibert`, `widget_captioning`, `groundcua`) pass through unchanged.
The manifest load aborts if it finds any ScreenSpot or GroundUI row. OS-Atlas Linux/macOS are training data (eval option
C in AGENTS.md): the 28 ScreenSpot-v2 desktop rows captured on the same machines are excluded from the test set instead
(`benchmarks/screenspot_v2/excluded_rows.json`).

## Setup

Run every command from the repo root. Python 3.13; the only dependencies are httpx, pillow and pytest.

```bash
uv venv --python 3.13.14 data_prep/rewrite/.venv
uv pip install --python data_prep/rewrite/.venv/bin/python -r data_prep/rewrite/requirements.txt
data_prep/rewrite/.venv/bin/python -m pytest data_prep/rewrite/tests -q   # no network, no real key
```

The hosted endpoint reads `NVIDIA_API_KEY` from the environment or `./.env`, which is gitignored. Never paste the key
into a command line or a chat.

## Workflow

`M` is Charlie's `manifest.jsonl`. `OUT` is a directory **outside the repo**, or under the gitignored `/runs/`.

```bash
PY=data_prep/rewrite/.venv/bin/python
M=~/t3/os_atlas_20k/manifest.jsonl

# 1. Look at exactly what is sent (no network, writes nothing)
$PY data_prep/rewrite/rewrite.py --manifest $M --out runs/rewrite_v3 --show-prompt

# 2. 200-row dry run (implies --limit 200), then the review sheet
$PY data_prep/rewrite/rewrite.py --manifest $M --out runs/rewrite_v3 --dry-run
$PY data_prep/rewrite/review_sample.py --out runs/rewrite_v3          # open runs/rewrite_v3/review.html
```

The dry run passes the gate if all three hold:

- at least 90% of reviewed rows are faithful (the review sheet keeps the tally). The gate measures the rewrite only:
  rows judged "source error: box ≠ label" (the source boxed the wrong element) don't count against it;
- at least 95% of the rows sent to the model are valid on the first try (`stats.json` `quality.first_try_valid_rate`;
  numeric rule drops are excluded because they are never sent);
- the drop rate is under 25% (`quality.drop_rate_raw`: every drop except `llm_error`, rule drops included;
  `quality.drop_rate_model` is the model's own share).

If it fails, edit `prompts.py`, bump `PROMPT_VERSION` and rerun. Do this at most twice.

```bash
# 3. Full run. Using the same OUT reuses the 200 cached rows.
$PY data_prep/rewrite/rewrite.py --manifest $M --out runs/rewrite_v3

# 4. Training manifests for Kyle, written next to the manifest: train_lightning.jsonl (mixed) and
#    train_template.jsonl (same rows, raw labels), plus apply_stats.json
$PY data_prep/rewrite/apply.py --manifest $M --rewrites runs/rewrite_v3/rewrites.jsonl
```

`apply.py` always writes into the manifest's directory. `image_path` is relative to that directory, and
`training/sft.py` resolves it against the train file's own directory, so a train file anywhere else would point at
missing images. `--out` is optional; any value that resolves to a different directory is refused.

**Numeric labels are dropped by rule.** Before batching, a target whose label satisfies `validate.is_numeric_label`
(only numbers, prices, percentages, counters, clock times, dates or relative times such as "73 minutes ago" or "now")
is cached as `keep=false`, `drop_reason=numeric`, `model=rule:is_numeric_label`, `attempts=0`, and never sent.
`METHOD.md` quotes the rule verbatim and counts these rows (`stats.json` `rewrites.rule_drops`).

**Resume.** Every finished row is cached in `OUT/cache.sqlite`, keyed by `(PROMPT_VERSION, id)`. Rerunning the same
command after a crash, Ctrl-C, outage or rate limit only sends the rows still pending. The exit code tells you where
you are:

- `0`: all rows done.
- `2`: some rows pending (rerun to finish them). That includes the output breaker below.
- `1`: a fatal error, such as a bad key (401), a bad URL or model (404), or a rejected manifest.
- `3`: refused: the local server is not serving the official checkpoint (see below).

Rows that failed validation three times in one invocation are cached as `drop_reason=llm_error`. `--retry-errors`
sends them again. If a manifest label changes for an id, the cached row is redone.

- **Attempts count across invocations.** Every failed attempt is stored per row (`row_history` in the cache), so a row
  that failed before an abort, or that is re-sent with `--retry-errors`, keeps its earlier attempts and errors. It is
  never counted as valid on the first try. The 3-attempt budget is per invocation.
- **Output breaker.** If 5 replies in a row have no valid item, the run stops with exit code 2. That means unparseable
  or truncated replies, or batches where every item fails. Tune it with `--max-consecutive-empty-replies`. Rows whose
  last attempt failed during that streak are not written as `llm_error`; they stay pending for the rerun. A
  single-row retry whose one item fails doesn't start a streak, because stubborn rows are expected among retries.
- **Run history.** Each invocation that sends at least one request is logged in the cache's `runs` table.
  `stats.json` `throughput` (and `METHOD.md`) sums rows and wall time over all of those invocations and lists them in
  `runs`. A resume that sends nothing rewrites the outputs from the cache and leaves these numbers unchanged.

## Endpoints

| `--endpoint` | URL | Model | Notes |
| --- | --- | --- | --- |
| `hosted` (default) | `https://integrate.api.nvidia.com/v1` | `nvidia/nemotron-3.5-lightning-30b-a3b` | needs `NVIDIA_API_KEY` |
| `local` | `http://127.0.0.1:8000/v1` | `lightning` | Patrick's Mac via vllm-metal ([docs/local-mac.md](../../docs/local-mac.md) §2) |
| `h200` | `http://127.0.0.1:8000/v1` | `lightning` | `ssh -N -L 8000:localhost:8000 permanent-tan-marmot`, only if the GPU is idle and Kyle agrees |

- `--base-url` and `--model` override the defaults.
- **Provenance check.** Only the official `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` checkpoint, or NVIDIA's
  own hosted endpoint, may produce data we report. Before sending anything, `rewrite.py` reads `GET /models` and records
  the `root` of the served name (`stats.json` `run.served_model_root`). On `local`/`h200` it refuses with exit code 3
  unless that root starts with `nvidia/` and has no community/quantization marker (`mlx-community`, `4bit`, `8bit`,
  `gguf`, `awq`, `gptq`, `-q4`, `-q8`). Serve the checkpoint by its HF id, not a local path, or the check can't see it.
- `--allow-unofficial-model` is for **plumbing only**. The rows are flagged in the cache, `METHOD.md` and `review.html`
  carry a NOT FOR PUBLICATION banner, and a later run without the flag regenerates those rows instead of reusing them.
  Use a separate `OUT` for plumbing anyway.
- `--model-label` defaults to the served checkpoint (else the model name). It is the `model` written in each row.
- Every request uses temperature 0.7, top_p 0.95, thinking off, and `--response-format json_object` by default. If the
  server rejects the response format, the client steps down `json_schema` -> `json_object` -> `none`.
- `--batch-size 20` is the default. `--concurrency` defaults per endpoint: hosted 16, local 4, h200 8.
- **The shared Mac server takes only 2 concurrent requests, and T1/T2 use it too.** On `--endpoint local` there, pass
  `--concurrency 1`.
- Hosted is slow and queue-bound. Measured 2026-10-17 ~18:05 UTC: 25-126 s per call even for 8-token replies, about 12%
  of calls over 120 s, no 429 at up to 40 concurrent, no rate-limit headers, and both `json_object` and `json_schema`
  accepted. So the per-request timeout is 300 s and hosted gets more slots.
- 429/5xx responses, timeouts, connection errors and undecodable bodies (corrupt gzip) back off with jitter and honor
  `Retry-After`. The run stops after 5 consecutive transport failures; tune this with
  `--max-consecutive-transport-errors`. Rate-limit headers the server sends appear in the summary and in `stats.json`.

## Outputs (in OUT)

| File | What | Publish? |
| --- | --- | --- |
| `rewrites.jsonl` | `{id, source, original, rewritten, style, keep, drop_reason, prompt_version, model, ts}` in manifest order. Byte-identical across reruns from the same cache. | yes |
| `stats.json` | Run settings, command, git sha, manifest sha256, and counts by source, style and drop reason (rule drops separate). Also quality gate, tokens, requests, latency, throughput over all invocations, and the run list. | yes |
| `METHOD.md` | Verbatim prompt, the numeric rule, who generated the rows (from the records), parameters, selection and style rules, metadata given, validator codes, counts, limitations, citation | yes |
| `cache.sqlite` | Full records (attempts, warnings, errors), per-row attempt history, every request log and the run history; the resume state | no (keep it on the box) |
| `review.html` | Contact sheet for the manual audit | no |

**Never commit anything in OUT.** Patrick uploads `rewrites.jsonl`, `stats.json` and `METHOD.md` to
`patrickballou/gui-assist-data` ([docs/submission.md](../../docs/submission.md)). Screenshots aren't re-hosted;
`id`/`image_path` refer to OS-Atlas files by name. Log the run in `docs/runs.md`: the command, `stats.json` totals,
throughput and tokens.
