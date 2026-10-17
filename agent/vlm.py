"""OpenAI-compatible client for the grounding VLM on :8001 (vLLM; served names `base`, `grounder`).

One VLMClient is both the Grounder and the Summarizer of agent/session.py:
  locate(img, target)  -> Located   (cfg.grounder_model, point on the 0-1000 grid -> viewport CSS px)
  summarize(img)       -> Summary   (cfg.summary_model, the structured screen description)

Prompts, preprocessing, point parsing and the summary sampling (SUMMARY_SAMPLING: greedy
plus a frequency penalty, so Qwen3-VL doesn't loop) come only from common/grounding.py. Which
served name each role uses is config-only (`grounder_model` / `summary_model`), since
`base` and `grounder` are one vLLM process with the LoRA adapter switched per request.

Neither method raises: transport and API problems get one retry, then come back as
`error` strings. The screenshot is encoded once per call and never logged.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

import openai
from openai import AsyncOpenAI
from PIL import Image

from agent.config import AgentConfig
from agent.observation import clip_summary
from agent.session import Located, Summary
from common.grounding import (SUMMARY_SAMPLING, build_grounding_messages, build_summary_messages, parse_point,
                              preprocess_screenshot, to_data_url, to_pixels)

log = logging.getLogger(__name__)

# Summary sampling is owned by common/grounding.py (SUMMARY_SAMPLING); a point needs few tokens.
GROUNDING_SAMPLING = {"temperature": 0.0, "max_tokens": 64}

# APITimeoutError is a subclass of APIConnectionError; InternalServerError is any 5xx.
_RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, openai.RateLimitError)
_DATA_URL_RE = re.compile(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+")


def strip_thinking(text: str | None) -> str:
    """Drop a <think>...</think> block: keep what follows `</think>`, or what precedes an unclosed `<think>`."""
    text = text or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:  # cut off by max_tokens before the answer
        text = text.split("<think>", 1)[0]
    return text.strip()


def _short(msg: object, limit: int = 240) -> str:
    """Error text safe for observations and traces: no image data, bounded length."""
    s = _DATA_URL_RE.sub("data:image/...;base64,<omitted>", str(msg)).strip()
    return s if len(s) <= limit else s[:limit] + "..."


class VLMClient:
    """Grounder + Summarizer backed by the OpenAI-compatible VLM server (cfg.vlm)."""

    def __init__(self, cfg: AgentConfig, client: AsyncOpenAI | None = None, *, retry_backoff_s: float = 1.0):
        self.cfg = cfg
        if client is None:
            client = AsyncOpenAI(base_url=cfg.vlm.base_url, api_key=cfg.vlm.api_key,
                                 timeout=cfg.timeouts.vlm_s, max_retries=0)
        else:  # our single retry is the only one, whatever the injected client was built with
            client = client.with_options(max_retries=0, timeout=cfg.timeouts.vlm_s)
        self.client = client
        self.retry_backoff_s = retry_backoff_s

    @property
    def base_url(self) -> str:
        return str(self.client.base_url).rstrip("/")

    async def locate(self, img: Image.Image, target: str) -> Located:
        """Ask cfg.grounder_model where `target` is; return the click point in `img`'s pixels (viewport CSS px)."""
        t0 = time.perf_counter()
        model = self.cfg.grounder_model
        try:
            messages = build_grounding_messages(to_data_url(preprocess_screenshot(img)), target, self.cfg.model_family)
        except Exception as e:
            return Located(None, None, error=f"could not prepare the request: {_short(e)}", model=model,
                           latency_ms=_ms(t0))
        raw, err = await self._complete(model, messages, GROUNDING_SAMPLING)
        if err:
            return Located(None, None, raw=raw, error=err, model=model, latency_ms=_ms(t0))
        pt = parse_point(strip_thinking(raw))
        if pt is None:
            return Located(None, None, raw=raw, error=f"could not parse a point from: {raw[:120]!r}",
                           model=model, latency_ms=_ms(t0))
        # The grid is resize-invariant, so map onto the ORIGINAL screenshot (= viewport CSS px
        # at device_scale_factor=1), not the preprocessed image, and keep the point on screen.
        x, y = to_pixels(pt, img.width, img.height)
        x, y = min(max(x, 0.0), img.width - 1.0), min(max(y, 0.0), img.height - 1.0)
        return Located(x, y, raw=raw, point_1000=pt, model=model, latency_ms=_ms(t0))

    async def summarize(self, img: Image.Image) -> Summary:
        """Structured text description of the screen from cfg.summary_model (at most 40 lines)."""
        t0 = time.perf_counter()
        model = self.cfg.summary_model
        try:
            url = to_data_url(preprocess_screenshot(img))
        except Exception as e:
            return Summary("", error=f"could not prepare the screenshot: {_short(e)}", model=model, latency_ms=_ms(t0))
        raw, err = await self._complete(model, build_summary_messages(url), SUMMARY_SAMPLING)
        if err:
            return Summary("", error=err, model=model, latency_ms=_ms(t0))
        text = clip_summary(strip_thinking(raw))
        if not text:
            return Summary("", error=f"{model} returned an empty screen description", model=model, latency_ms=_ms(t0))
        return Summary(text, model=model, latency_ms=_ms(t0))

    async def _complete(self, model: str, messages: list[dict], sampling: dict) -> tuple[str, str | None]:
        """One chat completion with one retry on transport/5xx/429 errors. Returns (content, error)."""
        kwargs: dict = {"model": model, "messages": messages, **sampling}
        if self.cfg.model_family == "qwen3_5":  # its chat template thinks by default
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        err = ""
        for attempt in (1, 2):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                if not resp.choices:
                    return "", f"{model} returned no choices"
                return resp.choices[0].message.content or "", None
            except _RETRYABLE as e:
                err = self._describe(e, model)
                if attempt == 1:
                    log.warning("VLM call to %s failed (%s); retrying once", model, err)
                    await asyncio.sleep(self.retry_backoff_s)
            except openai.APIStatusError as e:  # 4xx: retrying won't help (e.g. adapter not loaded)
                return "", self._describe(e, model)
            except Exception as e:
                return "", f"VLM call failed: {type(e).__name__}: {_short(e)}"
        return "", err

    def _describe(self, e: Exception, model: str) -> str:
        if isinstance(e, openai.APITimeoutError):
            return f"VLM timed out after {self.cfg.timeouts.vlm_s:g}s at {self.base_url}"
        if isinstance(e, openai.APIConnectionError):
            cause = e.__cause__ or e
            return f"VLM unreachable at {self.base_url}: {type(cause).__name__}: {_short(cause)}"
        if isinstance(e, openai.APIStatusError):  # body is the server's {"message", "type", ...} when it sent one
            detail = e.body.get("message") if isinstance(e.body, dict) else None
            return f"VLM error {e.status_code} from {self.base_url} (model {model!r}): {_short(detail or e.message)}"
        return f"VLM call failed: {type(e).__name__}: {_short(e)}"

    async def aclose(self) -> None:
        """Close the underlying HTTP client (shared with an injected client)."""
        await self.client.close()


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
