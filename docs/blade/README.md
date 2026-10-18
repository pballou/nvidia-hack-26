# BLADE between night-loop rounds: runbook

This is how the orchestrator (Patrick's Mac) analyses a finished night-loop round, then changes
`/home/ubuntu/night/knobs.json` or releases the loop. It takes **under 10 minutes**.

- **Tool:** `scripts/blade_pair.py` (stdlib only).
- **Example report:** [round_0_base_vs_sft.md](round_0_base_vs_sft.md).
- **Local outputs** go to `runs/blade/` (gitignored by `/runs/`). Never commit them.

**Box rules:**

- Host alias: `permanent-tan-marmot`.
- Every ssh/scp uses `-o ConnectTimeout=15` and a `timeout`.
- Everything below is read-only on the box except **two writes the orchestrator is allowed**:
  `/home/ubuntu/night/knobs.json` and `/home/ubuntu/night/round_<r>/BLADE_DONE`.
- Never create `STOP` unless you mean to stop the loop. Never start or kill anything.

## 1. When the loop listens (from `training/night_loop.py`)

**Tonight's loop** (docs/runs.md) started Sat 19:38 with:

```bash
training/night_loop.sh --init-adapter /home/ubuntu/ckpt/sft-lightning/adapter \
  --train-file /home/ubuntu/data/os_atlas_20k/train_lightning.jsonl \
  --last-round-start 04:15 --no-new-step 05:25 --hard-end 05:40
```

It started from `DEFAULT_KNOBS`, with at most 3 rounds. Round 1 already runs on the defaults. The first BLADE
window is the wait after round 1, and only if round 1 is kept. Times below use tonight's flags; the code
defaults are 01:30 / 02:40 / 02:50.

- **Knobs are read once per round,** at the start of `round(r)`, via `load_knobs()`. What was actually used
  is in `round_<r>/knobs_used.json` (`knobs` plus `notes`).
- **There is no wait between round 0 and round 1.** `run()` calls `baseline()` (round-0 dev eval only, no
  `DONE`) and then `round(1)`.
  - Knobs meant for round 1 must be in `knobs.json` **before launch**. In the rehearsal, round 0 took 2 min
    (launched 17:52:22, round-0 `summary.json` at 17:54:24).
  - The loop writes `DEFAULT_KNOBS` only if `knobs.json` is missing, and never moves it.
- **After round r, `wait_for_blade(r)` runs only if all of these hold:**
  - round r was **kept** (dev gain ≥ +0.5);
  - it set no `stop_reason`;
  - r < 3.

  Otherwise the loop goes straight to `finish()` (`FINAL.json`), and knob changes no longer matter.
- **While it waits:**
  - `STATUS.json` has phase `"round <r> done: waiting for BLADE / knobs.json until <Day HH:MM>"`, with a
    `hint` naming the `BLADE_DONE` path.
  - The wait lasts until **min(now + 10 min, `--last-round-start`)**, i.e. 04:15 EDT tonight (01:30 by code
    default). Every 10 s the loop checks, in this order:
    1. `STOP` → stop;
    2. `round_<r>/BLADE_DONE` exists → continue now;
    3. the sha256 of `knobs.json` differs from its value **when the wait started** → continue after 30 s.

    At the end of the wait it continues with whatever `knobs.json` says.
  - A knobs edit made *before* the wait started doesn't release it (the baseline digest already includes the
    edit). So after writing knobs, always touch `BLADE_DONE`.
  - `round(r)` deletes `round_<r>/BLADE_DONE` and `DONE` at its own start. Touch `BLADE_DONE` only after
    `round_<r>/DONE` exists. Touching it earlier pre-releases the wait before you've looked.
- **After the wait:** tonight no round starts at or after 04:15 EDT, and no GRPO step starts after 05:25.
  Every step is capped at the 05:40 hard end, then `finish()` re-serves the best adapter and writes
  `FINAL.json` last, a few minutes after 05:40 at the latest. If the wait ends at 04:15, BLADE results only
  feed the morning report.
- **`load_knobs` semantics:**
  - Each key present in the file overrides the previous round's value.
  - `source_weights` is **merged per source**. Values are floats, negatives become 0, and **0 drops the
    source from the pool**.
  - `exclude_ids` is **replaced wholesale** (list of strings), so always write the cumulative list.
  - Scalar knobs are clamped.
  - Unknown keys are ignored, with a note.
  - **Unreadable JSON keeps the previous knobs silently,** with only a note in `knobs_used.json`. Validate
    before uploading.
- **Our policy:** only `source_weights` and `exclude_ids` change. `pool_max`, `num_generations`,
  `grpo_max_steps`, `round_time_budget_min` and `format_reward` stay byte-identical. `blade_pair.py` copies
  them unchanged.
- **What the knobs act on:**
  - Both touch only the **train-profile pool**: rows with 0 < pass rate < 1 and at least 4 rollouts, then
    weighted and capped at `pool_max`. They also filter the rejection-sampled SFT fallback.
  - If candidates exceed `pool_max`, weights act as sampling priorities. Otherwise each row appears int(w)
    times plus one more with probability frac(w): 1.5 means half the rows twice; 0.5 drops about half.
  - Dev ids never reach the pool (`load_train_rows` refuses them), so **never put dev ids in `exclude_ids`**.
    They would do nothing.
  - Each round profiles 4,000 *fresh* train rows (`used_ids`). So exclusions must be train ids that haven't
    been profiled yet: rule-based ids over the whole train file, which the script produces.

## 2. Remote paths (`/home/ubuntu/night` = N)

| What | Path on the box | Needed for |
|---|---|---|
| This round's dev rollouts | `N/round_<r>/dev_r<r>/rollouts.jsonl` (+ `rollouts_aggregate_metrics.json`, `rollouts_failures.jsonl`) | AFTER |
| Previous kept round's dev rollouts | `N/round_<r-1>/dev_r<r-1>/rollouts.jsonl` (+ `_aggregate_metrics.json`). For r = 1 that's `N/round_0/dev_r0/`. When the loop waits, round r-1 was always kept | BEFORE |
| Gym compare report (previous kept → this round) | `N/round_<r>/compare/compare_report.{md,json}` | cross-check |
| Pool stats | `N/round_<r>/pool_stats.json` (`profile_by_source`, `candidates`, `pool_by_source`) | pool context |
| Train profile (pass rate per task) | `N/round_<r>/profile_r<r>/rollouts_reward_profiling.jsonl`. Skip `profile_r<r>/rollouts.jsonl` (about 90 MB at 4k × 8) | screenshot-level exclusions |
| Round summary / knobs used | `N/round_<r>/summary.json`, `N/round_<r>/knobs_used.json`, `N/round_<r>/DONE` | kept? stop_reason? |
| Loop state | `N/STATUS.json`, `N/state.json`, `N/loop.json` (`args.train_file`), `N/knobs.json` | is it waiting? current knobs |
| Base reference (once) | `/home/ubuntu/kyle/nvidia-hack-26/results/dev_base_qwen35.jsonl` (+ `_aggregate_metrics.json`) | consensus check |
| Train file (once) | `loop.json` → `args.train_file`; tonight `/home/ubuntu/data/os_atlas_20k/train_lightning.jsonl` (≈ 8 MB; the local copy in `runs/blade/data/` matches its sha256 `7039f731…`) | `exclude_ids` |
| Screenshots (only a few) | the `file:///home/ubuntu/data/cache/preprocessed/2073600/…png` URL in each rollout | eye checks |

`fetch` pulls all of these in a single `ssh … tar -c` stream (about 15–45 s). Manual fallback, same files:

```bash
H=permanent-tan-marmot; N=/home/ubuntu/night; R=1; L=runs/blade/r$R/night
mkdir -p $L/round_$R/dev_r$R $L/round_$((R-1))/dev_r$((R-1)) $L/round_$R/compare $L/round_$R/profile_r$R
timeout 120 scp -o ConnectTimeout=15 "$H:$N/round_$R/dev_r$R/rollouts*.jsonl" "$H:$N/round_$R/dev_r$R/rollouts_aggregate_metrics.json" $L/round_$R/dev_r$R/
timeout 120 scp -o ConnectTimeout=15 "$H:$N/round_$((R-1))/dev_r$((R-1))/rollouts.jsonl" "$H:$N/round_$((R-1))/dev_r$((R-1))/rollouts_aggregate_metrics.json" $L/round_$((R-1))/dev_r$((R-1))/
timeout 120 scp -o ConnectTimeout=15 "$H:$N/round_$R/compare/compare_report.*" $L/round_$R/compare/
timeout 120 scp -o ConnectTimeout=15 "$H:$N/round_$R/profile_r$R/rollouts_reward_profiling.jsonl" $L/round_$R/profile_r$R/
timeout 120 scp -o ConnectTimeout=15 "$H:$N/round_$R/{summary,knobs_used,pool_stats}.json" $L/round_$R/
timeout 120 scp -o ConnectTimeout=15 "$H:$N/{knobs,STATUS,state,loop}.json" $L/
```

## 3. Per-round procedure (r = the round that just finished)

Run from the repo root on the Mac:

```bash
cd /Users/pballou/projects/nvidia-hack-26
R=1                                                           # the round whose DONE just appeared

# 0-1 min: fetch. Prints summary.json (kept, dev_acc_prev -> dev_acc_new, stop_reason) and STATUS.json (phase).
python3 -I scripts/blade_pair.py fetch --round $R
#   If the phase isn't "round $R done: waiting for BLADE ...", the loop isn't waiting.
#   Still write the report, but don't touch the box.

# 1-2 min: analyze. BEFORE = round_{R-1} dev, AFTER = round_R dev, consensus reference = base run.
# Also uses the round's profile + pool_stats, the current knobs.json, the train file and
# runs/blade/da_confirmed_dev.txt.
python3 -I scripts/blade_pair.py analyze --round $R
#   -> runs/blade/r$R/analysis.md (read this), analysis.json, rows.jsonl,
#      knobs_proposed.json, knobs_diff.json

# 2-6 min, optional: look at screenshots the script couldn't classify (SCREEN_cluster rows), or a few broken rows.
python3 -I scripts/blade_pair.py images --round $R --n 8      # at most a few dozen images, please
#   -> runs/blade/r$R/images/<screen>.svg.png (red = AFTER point, blue = BEFORE, purple = base ref,
#      green/orange box = pass/fail) plus <screen>.txt legends. Open them with the Read tool.
#   If a screenshot's boxes visibly don't match its pixels, append the bad row ids to
#   runs/blade/da_confirmed_dev.txt and re-run `analyze` (under 1 s). Use a screen id only if every box on it
#   is wrong; a screen id also flags correctly boxed rows. Round 0 had two of these, both real regressions.
#   `analyze` prints "CHECK: DA_confirmed rows that the BEFORE/reference model passes": re-check those rows.
```

**6–8 min: sanity-check `runs/blade/r$R/knobs_diff.json`** before applying:

- Only `source_weights` and `exclude_ids` appear. `unchanged_keys` lists the other five.
- Every weight is in [0.5, 2.0], and none is 0.
- `exclude_ids.to_count` ≥ `from_count`: the list is cumulative and never shrinks.
- No dev ids: the script never adds them (it skips every id in the AFTER dev run). It copies the current list
  unchanged, though, so a dev id written by hand would stay.
- If you override a weight by hand, edit `runs/blade/r$R/knobs_proposed.json` and say why in the round report.
  Weights are absolute values recomputed each round from that round's dev run, not multipliers.

**8–9 min: apply.** Atomic write (`mv` on the same filesystem), then release the wait:

```bash
python3 -I -m json.tool runs/blade/r$R/knobs_proposed.json > /dev/null && \
timeout 60 scp -o ConnectTimeout=15 runs/blade/r$R/knobs_proposed.json permanent-tan-marmot:/home/ubuntu/night/knobs.json.tmp && \
timeout 60 ssh -o ConnectTimeout=15 permanent-tan-marmot \
  "mv /home/ubuntu/night/knobs.json.tmp /home/ubuntu/night/knobs.json && touch /home/ubuntu/night/round_$R/BLADE_DONE"
```

No change wanted? Just release it:

```bash
timeout 60 ssh -o ConnectTimeout=15 permanent-tan-marmot "touch /home/ubuntu/night/round_$R/BLADE_DONE"
```

**After the next round starts, check that the knobs took:**

```bash
timeout 60 ssh -o ConnectTimeout=15 permanent-tan-marmot "cat /home/ubuntu/night/round_$((R+1))/knobs_used.json"
```

The values should match and `"notes"` should be `[]`. A note such as "knobs.json unreadable" means the previous
knobs were used.

**Report (optional, after the loop moves on):** copy the key tables from `runs/blade/r$R/analysis.md` into
`docs/blade/round_<R>_r<R-1>_vs_r<R>.md`, using round 0's report as the template. Commit only small
docs, never `runs/`.

### Round 0 (before launch) is different

**Tonight this window has passed.** The loop launched at 19:38, before this re-run was done. Keep this section
for a future launch.

- **Use the winning SFT's own dev eval, before the loop starts.** Round 1 reads `knobs.json` about 2 minutes
  after launch, so there's no time to analyse `round_0/dev_r0`.
- **Where that eval lives:** `training/gymctl.py eval` writes
  `/home/ubuntu/night/evals/<name>/dev_<name>/rollouts.jsonl` when run with its default `--out-root`.
  Otherwise use the path recorded in `docs/runs.md`.
- **BEFORE is the base run. Knobs start from `DEFAULT_KNOBS`,** parsed from `training/night_loop.py`.
- **Write to `runs/blade/r0_$NAME`, not `runs/blade/r0`.** `runs/blade/r0` holds the template-rehearsal
  copies that [round_0_base_vs_sft.md](round_0_base_vs_sft.md) cites.

```bash
NAME=sft-lightning                                    # whichever won on dev (tonight: sft-lightning)
TRAIN=train_lightning.jsonl                           # the winner's train file (same ids in both files)
OUT=runs/blade/r0_$NAME
python3 -I scripts/blade_pair.py fetch --round 0 --out $OUT   # gets the base run once; round_0/ is missing before launch, which is fine
mkdir -p $OUT/$NAME && timeout 120 scp -o ConnectTimeout=15 \
  "permanent-tan-marmot:/home/ubuntu/night/evals/$NAME/dev_$NAME/rollouts.jsonl" $OUT/$NAME/
[ -f runs/blade/data/$TRAIN ] || timeout 120 scp -o ConnectTimeout=15 \
  "permanent-tan-marmot:/home/ubuntu/data/os_atlas_20k/$TRAIN" runs/blade/data/
python3 -I scripts/blade_pair.py analyze --out $OUT --names base,$NAME --knobs default \
  --before runs/blade/base/dev_base_qwen35.jsonl --after $OUT/$NAME/rollouts.jsonl \
  --ref runs/blade/base/dev_base_qwen35.jsonl --train-file runs/blade/data/$TRAIN \
  --da-confirmed runs/blade/da_confirmed_dev.txt
```

- **Apply** `$OUT/knobs_proposed.json` with the same scp + `mv`, but without `touch BLADE_DONE`: there's no
  wait and no `round_0/` yet. The loop keeps an existing `knobs.json`.
- **Template adapter:** its recommendation is already computed in
  [round_0_base_vs_sft.md](round_0_base_vs_sft.md). It's superseded tonight, because the loop runs from
  sft-lightning.
- **To analyse the live loop's own round 0,** use `fetch --round 0 --out runs/blade/r0_live` and then
  `analyze --round 0 --out runs/blade/r0_live`. The default `--out` would overwrite the rehearsal copies in
  `runs/blade/r0`.
- **Expect close numbers, not identical ones,** to the standalone eval. T = 0 dev runs still vary by about
  ±0.2 points: sft-template scored 85.66 standalone and 85.86 in the loop.

## 4. What the script does (so you can trust or override it)

- **`fetch`:**
  - read-only `ssh <host> "cd <night> && tar -cf - …"` into `runs/blade/r<R>/night/`, mirroring the box
    layout;
  - fetches the base run (into `runs/blade/base/`) and the loop's train file (into `runs/blade/data/`) once;
  - prints round status.
  - Flags: `--night-dir` (e.g. `/home/ubuntu/night_rehearsal`) and `--out`.
- **`analyze`:**
  - Aligns the two dev runs on `id`.
  - Reports overall, per-source, per-platform and per-role accuracy; fixed, broken, still-fail; and exact
    McNemar.
  - Labels every AFTER failure with one of the taxonomy labels in the script docstring:
    - `DA_*`: ungroundable instruction, sub-precision box, or confirmed by eye;
    - `SCREEN_cluster`: ≥ 3 and ≥ 60% of a screenshot's rows fail and nobody has reviewed it yet;
    - `CONSENSUS_miss`: the base model misses at the same spot, within 1.5% of the diagonal;
    - `KG_wrong_annotated_element`, `KG_near_miss` (within 2% of the diagonal), `KG_far_miss`.
  - Flags "attractors": a repeated x or y value the AFTER model clicks and the base model doesn't, mostly
    missing. Round 0 found androidworld x = 568.
- **Knob rule:**
  - `w = clip(round_0.25(err_s / err_all), 0.5, 2.0)`, plus 0.25 if the source has ≥ 5 broken rows. Here
    `err` is model-attributable error (`FMT_` and `KG_` labels) on rows that aren't artifacts.
  - `exclude_ids` = current list + train rows with ungroundable instructions (AX handles, D-Bus names,
    pointers, value literals, fineweb file-name slugs) or sub-precision boxes (< 6 px or < 0.6% of the image
    side).
  - With a profile, it also adds the not-yet-profiled train rows of screenshots whose ≥ 2 profiled rows were
    all never-pass.
- **Rehearsal check:**

  ```bash
  python3 -I scripts/blade_pair.py fetch --round 1 --night-dir /home/ubuntu/night_rehearsal --out runs/blade/rehearsal/r1
  python3 -I scripts/blade_pair.py analyze --round 1 --out runs/blade/rehearsal/r1
  ```

  This reproduces the rehearsal's dev 85.86 → 85.96 (5 fixed / 4 broken, not kept).
