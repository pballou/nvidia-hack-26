"""GRPO on the grounder with TRL and the NeMo Gym gui_grounding /verify reward (docs/tasks/B-night-loop.md).

    python -m training.grpo --init-adapter /home/ubuntu/ckpt/<run>/adapter --pool /home/ubuntu/night/round_1/pool.jsonl \
        --out /home/ubuntu/ckpt/night/round_1/grpo --verify-url http://127.0.0.1:12010/verify [--max-steps 200]

- Pool rows are Gym rows (data_prep/gym_rows.py), so the prompt text and the pre-resized
  screenshot are exactly what Gym evals send; verify() gets the row's canonical fields.
- Reward = the Gym resources server's /verify (1 if the point lands in the box), plus
  `--format-reward` x (reply parses as a point). One vLLM instance is colocated with the
  trainer (TRL `vllm_mode="colocate"`); the LoRA continues from --init-adapter.
- Gate D2 runs in-process: after `--smoke-steps` steps, the median wall time per step
  (excluding the first two) must be <= `--gate-s-per-step`, otherwise training stops,
  no adapter is saved and the exit code is 10 (the loop then falls back to
  rejection-sampled SFT).
- Training stops early (adapter saved, exit 0) at --deadline (no step starts after it),
  at --max-minutes, or when --stop-file appears. <out>/grpo_summary.json records why.

TRL 1.16.0 + vLLM 0.31.0 workarounds (found by reading the installed sources, see
docs/runs.md "night loop"):
1. Colocate passes HF-tokenized prompt ids, with the image already expanded into
   <|image_pad|> runs, together with the image; vLLM expands the image again. Collapse
   each run to one pad so vLLM's expansion reproduces TRL's ids exactly (asserted).
2. The processor must pad on the left: TRL left-pads prompt ids itself but takes
   mm_token_type_ids from the processor, and Qwen3.5's image positions use those.
3. Colocated vLLM gets the same image caps as the :8001 server (TRL passes none).
4. Weight sync is merge -> push -> unmerge in bf16, which drifts the base weights a
   little every step; restore the exact base weights after each unmerge.
"""

import argparse
import concurrent.futures as cf
import json
import math
import os
import statistics
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import torch
from datasets import Dataset, Image, List
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import trl.generation.vllm_generation as vg
from common import grounding
from training.lora import MODELS
from training.sft import check_format

MODEL_ID = MODELS["qwen3.5"]
GATE_EXIT = 10
ROW_KEYS = ("id", "source", "instruction", "platform", "data_type", "bbox", "img_w", "img_h")


# --------------------------------------------------------------------------- TRL / vLLM patches


def patch_vllm_generation(image_pad_id: int) -> None:
    """Workarounds 1 and 3 above."""
    orig_generate = vg.VLLMGeneration.generate

    def generate(self, prompts, images, num_generations, profiler=None):
        if images is None:
            return orig_generate(self, prompts, images, num_generations, profiler)
        want = [len(p) for p in prompts]
        collapsed = [[t for i, t in enumerate(p) if not (t == image_pad_id and i and p[i - 1] == image_pad_id)]
                     for p in prompts]
        out = orig_generate(self, collapsed, images, num_generations, profiler)
        got = [len(p) for p in out[0]]
        if got != want:
            raise RuntimeError(f"vLLM re-expanded prompts to {got[:4]} tokens, trainer has {want[:4]}")
        return out

    vg.VLLMGeneration.generate = generate

    orig_llm = vg.LLM

    def llm(*args, **kwargs):
        kwargs.setdefault("limit_mm_per_prompt", {"image": 1, "video": 0})
        kwargs.setdefault("mm_processor_kwargs", {"size": {"longest_edge": grounding.MAX_PIXELS,
                                                           "shortest_edge": grounding.MIN_PIXELS}})
        kwargs["max_num_batched_tokens"] = 8192  # TRL hard-codes 4096: two 2 MP prompts per prefill step
        return orig_llm(*args, **kwargs)

    vg.LLM = llm


def patch_exact_unmerge() -> None:
    """Workaround 4: unmerge restores a snapshot instead of subtracting the delta again."""
    snapshot: dict[int, tuple] = {}
    orig_merge, orig_unmerge = BaseTuner.merge_adapter, BaseTuner.unmerge_adapter

    def merge_adapter(self, *args, **kwargs):
        for m in self.model.modules():
            if isinstance(m, BaseTunerLayer):
                snapshot[id(m)] = (m, m.get_base_layer().weight.detach().clone())
        return orig_merge(self, *args, **kwargs)

    def unmerge_adapter(self, *args, **kwargs):
        out = orig_unmerge(self, *args, **kwargs)
        with torch.no_grad():
            for m, w in snapshot.values():
                m.get_base_layer().weight.copy_(w)
        snapshot.clear()
        return out

    BaseTuner.merge_adapter, BaseTuner.unmerge_adapter = merge_adapter, unmerge_adapter


# --------------------------------------------------------------------------- rewards


class GymVerify:
    """Reward function: POST each completion to the Gym resources server's /verify."""

    __name__ = "gym_verify"  # TRL logs rewards/<__name__>/mean

    def __init__(self, url: str, max_fail_frac: float = 0.05):
        self.url = url
        self.max_fail_frac = max_fail_frac
        self.pool = cf.ThreadPoolExecutor(32)
        self.calls = self.failures = 0

    def _post(self, row: dict, text: str) -> float:
        body = {k: row.get(k) for k in ROW_KEYS} | {
            "responses_create_params": {"input": []},
            "response": {"id": "resp_grpo", "created_at": time.time(), "model": "grounder", "object": "response",
                         "parallel_tool_calls": False, "tool_choice": "none", "tools": [],
                         "output": [{"type": "message", "id": "msg_grpo", "role": "assistant", "status": "completed",
                                     "content": [{"type": "output_text", "text": text, "annotations": []}]}]},
        }
        data = json.dumps(body).encode()
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    out = json.load(r)
                return float(out["reward"])
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)

    def __call__(self, prompts, completions, row_json, **kwargs):
        futures = [self.pool.submit(self._post, json.loads(rj), c[0]["content"]) for rj, c in zip(row_json, completions)]
        rewards, fails = [], 0
        for f in futures:
            try:
                rewards.append(f.result())
            except Exception:
                rewards.append(0.0)
                fails += 1
        self.calls += len(rewards)
        self.failures += fails
        if fails > self.max_fail_frac * len(rewards):
            raise RuntimeError(f"{fails}/{len(rewards)} /verify calls failed ({self.url})")
        return rewards


def format_ok(prompts, completions, **kwargs):
    return [float(grounding.parse_point(c[0]["content"]) is not None) for c in completions]


# --------------------------------------------------------------------------- data


def load_pool(path: Path) -> Dataset:
    examples = []
    for line in path.open():
        row = json.loads(line)
        image, text = row["responses_create_params"]["input"][0]["content"]
        assert image["type"] == "input_image" and text["type"] == "input_text", row["id"]
        examples.append({
            "prompt": [{"role": "user", "content": text["text"]}],
            "images": [image["image_url"].removeprefix("file://")],
            "row_json": json.dumps({k: row.get(k) for k in ROW_KEYS}),
        })
    assert examples, f"empty pool: {path}"
    return Dataset.from_list(examples).cast_column("images", List(Image()))


# --------------------------------------------------------------------------- control


class Control(TrainerCallback):
    """Gate D2, deadline / time budget / STOP file, and per-step wall times."""

    def __init__(self, a, t_start: float):
        self.a = a
        self.t_start = t_start
        self.times: list[float] = []
        self._t = None
        self.gate: dict | None = None
        self.stop_reason = "max_steps"

    def on_train_begin(self, args, state, control, **kwargs):
        self._t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        self.times.append(now - self._t)
        self._t = now
        n = len(self.times)
        if self.gate is None and n >= min(self.a.smoke_steps, self.a.max_steps):
            steady = self.times[2:] or self.times
            s = statistics.median(steady)
            self.gate = {"steps": n, "median_s_per_step": round(s, 2), "limit": self.a.gate_s_per_step,
                         "passed": s <= self.a.gate_s_per_step}
            print("GATE_D2 " + json.dumps(self.gate), flush=True)
            if not self.gate["passed"]:
                self.stop_reason = "gate_d2_failed"
                control.should_training_stop = True
                return
        est = statistics.median(self.times[-10:])
        if self.a.deadline and now + 0.5 * est >= self.a.deadline:
            self.stop_reason, control.should_training_stop = "deadline", True
        elif self.a.max_minutes and now - self.t_start + 0.5 * est >= 60 * self.a.max_minutes:
            self.stop_reason, control.should_training_stop = "time_budget", True
        elif self.a.stop_file and Path(self.a.stop_file).exists():
            self.stop_reason, control.should_training_stop = "stop_file", True


def _mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return round(sum(xs) / len(xs), 4) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-adapter", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True, help="Gym rows (pool.jsonl from training/night_loop.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify-url", required=True)
    ap.add_argument("--num-generations", type=int, default=6)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-minutes", type=float, help="stop once this much wall time has passed (incl. loading)")
    ap.add_argument("--deadline", type=float, help="unix time; no step starts after it")
    ap.add_argument("--stop-file", type=Path)
    ap.add_argument("--format-reward", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-completion-length", type=int, default=32)
    ap.add_argument("--vllm-gpu-util", type=float, default=0.25)
    ap.add_argument("--smoke-steps", type=int, default=20)
    ap.add_argument("--gate-s-per-step", type=float, default=35.0)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))
    summary_path = a.out / "grpo_summary.json"
    summary_path.unlink(missing_ok=True)

    with urllib.request.urlopen(a.verify_url.rsplit("/", 1)[0] + "/health", timeout=10) as r:
        assert r.status == 200, f"verify server not healthy: {a.verify_url}"

    processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")  # workaround 2
    processor.image_processor.size.longest_edge = grounding.MAX_PIXELS
    processor.image_processor.size.shortest_edge = grounding.MIN_PIXELS
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    patch_vllm_generation(image_pad_id)
    patch_exact_unmerge()

    ds = load_pool(a.pool)
    # Same prompt as SFT and serving, and the processor doesn't resize the screenshot again.
    check_format(processor, {"prompt": ds[0]["prompt"], "images": ds[0]["images"],
                             "completion": [{"role": "assistant", "content": '{"point_2d": [500, 500]}'}]}, "qwen3.5")

    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(a.init_adapter), is_trainable=True)

    reward_funcs, weights = [GymVerify(a.verify_url)], [1.0]
    if a.format_reward > 0:
        reward_funcs.append(format_ok)
        weights.append(a.format_reward)
    g = a.num_generations
    config = GRPOConfig(
        output_dir=str(a.out),
        seed=a.seed,
        max_steps=a.max_steps,
        # prompts_per_step prompts x g completions per optimizer step, in g micro-batches
        per_device_train_batch_size=a.prompts_per_step,
        gradient_accumulation_steps=g,
        num_generations=g,
        max_completion_length=a.max_completion_length,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        loss_type="dapo",
        scale_rewards="group",
        num_iterations=1,
        learning_rate=a.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        max_grad_norm=1.0,
        disable_dropout=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_gpu_util,
        vllm_max_model_length=3072,  # <= ~2,025 image tokens + prompt + completion; fails loudly if images double-expand
        vllm_enable_sleep_mode=False,
        vllm_tensor_parallel_size=1,
        vllm_importance_sampling_correction=True,
        chat_template_kwargs={"enable_thinking": False},
        reward_weights=weights,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=a.save_steps,
        save_total_limit=2,
        log_completions=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )
    control = Control(a, t_start)
    trainer = GRPOTrainer(model=model, reward_funcs=reward_funcs, args=config, train_dataset=ds,
                          processing_class=processor, callbacks=[control])
    t_train = time.time()
    trainer.train()

    gate_failed = control.gate is not None and not control.gate["passed"]
    adapter = None
    if not gate_failed:
        adapter = a.out / "adapter"
        trainer.save_model(str(adapter))

    hist = [h for h in trainer.state.log_history if "rewards/gym_verify/mean" in h]
    verify = [h["rewards/gym_verify/mean"] for h in hist]
    summary = {
        "init_adapter": str(a.init_adapter),
        "adapter": str(adapter) if adapter else None,
        "pool": str(a.pool),
        "pool_rows": len(ds),
        "grpo_steps": len(control.times),
        "s_per_step": round(statistics.median(control.times[2:] or control.times), 2) if control.times else None,
        "gate_d2": control.gate,
        "stop_reason": control.stop_reason,
        "reward_mean_first": _mean(verify[:10]),
        "reward_mean_last": _mean(verify[-10:]),
        "frac_reward_zero_std_mean": _mean([h.get("frac_reward_zero_std") for h in hist]),
        "verify_calls": reward_funcs[0].calls,
        "verify_failures": reward_funcs[0].failures,
        "load_s": round(t_train - t_start, 1),
        "train_s": round(time.time() - t_train, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "config": {"num_generations": g, "prompts_per_step": a.prompts_per_step, "lr": a.lr,
                   "format_reward": a.format_reward, "max_completion_length": a.max_completion_length,
                   "vllm_gpu_util": a.vllm_gpu_util, "seed": a.seed},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    (a.out / "log_history.json").write_text(json.dumps(trainer.state.log_history))
    print("SUMMARY " + json.dumps(summary), flush=True)
    return GATE_EXIT if gate_failed else 0


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as e:  # argparse errors
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:  # incl. KeyboardInterrupt from the loop's SIGINT
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # skip interpreter teardown on every path: the colocated vLLM engine can hang it
