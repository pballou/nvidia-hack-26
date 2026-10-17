"""Async OpenAI-compatible chat client for Nemotron 3.5 Lightning: concurrency cap, backoff, response_format downgrade.

The caller loads the key explicitly, so nothing here reads .env implicitly:
  ep = ENDPOINTS["hosted"]
  client = LightningClient(ep, api_key=load_api_key(ep.api_key_env) if ep.api_key_env else None)
"""

import asyncio
import copy
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    model: str
    api_key_env: str | None


ENDPOINTS = {
    "hosted": Endpoint("hosted", "https://integrate.api.nvidia.com/v1", "nvidia/nemotron-3.5-lightning-30b-a3b", "NVIDIA_API_KEY"),
    "local": Endpoint("local", "http://127.0.0.1:8000/v1", "lightning", None),
    "h200": Endpoint("h200", "http://127.0.0.1:8000/v1", "lightning", None),  # box :8000 via ssh tunnel
}
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False}}
RESPONSE_FORMATS = ("json_schema", "json_object", "none")  # downgrade order
DOWNGRADE_STATUS = frozenset({400, 422})
DOWNGRADE_HINTS = ("response_format", "json_schema", "guided", "json_object")
RETRY_AFTER_MAX = 300.0
_DOTENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)$")


class TransportError(RuntimeError):
    """Retries exhausted on 408/429/5xx, timeouts, connection errors or unparseable 200 bodies."""

    def __init__(self, message: str, attempts: int = 0, last_error: str = ""):
        super().__init__(message)
        self.attempts, self.last_error = attempts, last_error


class FatalHTTPError(RuntimeError):
    """Non-retryable HTTP error (401/403/404/422, ...). The message never contains the API key."""

    def __init__(self, status: int, body: str, url: str, hint: str = ""):
        super().__init__(f"HTTP {status} from {url}: {body}" + (f" [{hint}]" if hint else ""))
        self.status, self.body, self.url = status, body, url


@dataclass
class Completion:
    text: str
    finish_reason: str | None
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    http_attempts: int
    response_format: str


def load_api_key(env_name: str, dotenv: Path = Path(".env")) -> str | None:
    """Return $env_name, else its value in `dotenv` ("export K=V", quotes, inline " # comment"); empty -> None.

    Never log or print the result."""
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    path = Path(dotenv)
    if not path.is_file():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _DOTENV_LINE.match(line)
        if m and m.group(1) == env_name:
            found = _dotenv_value(m.group(2))  # last definition wins
    return found or None


def _dotenv_value(raw: str) -> str:
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        return (raw[1:end] if end > 0 else raw[1:]).strip()
    return re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()


def _retry_after(value: str | None) -> float | None:
    """Retry-After as seconds (delta or HTTP date), clamped to [0, RETRY_AFTER_MAX]."""
    if not value:
        return None
    try:
        secs = float(value)
    except ValueError:
        try:
            secs = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return None
    return min(max(secs, 0.0), RETRY_AFTER_MAX)


def _retryable(status: int) -> bool:
    return status in (408, 429) or status >= 500


class LightningClient:
    """One instance per run: shares the semaphore, the stats and the (possibly downgraded) response_format."""

    def __init__(self, endpoint: Endpoint | str, *, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, concurrency: int = 6, max_attempts: int = 6, backoff_base: float = 2.0,
                 backoff_max: float = 60.0, timeout_s: float = 180.0, response_format: str = "json_object",
                 transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep):
        if response_format not in RESPONSE_FORMATS:
            raise ValueError(f"response_format must be one of {RESPONSE_FORMATS}, got {response_format!r}")
        if concurrency < 1 or max_attempts < 1:
            raise ValueError("concurrency and max_attempts must be >= 1")
        self.endpoint = ENDPOINTS[endpoint] if isinstance(endpoint, str) else endpoint
        self.base_url = (base_url or self.endpoint.base_url).rstrip("/")
        self.model = model or self.endpoint.model
        self.url = f"{self.base_url}/chat/completions"
        self.response_format = response_format
        self.concurrency, self.max_attempts = concurrency, max_attempts
        self.backoff_base, self.backoff_max = backoff_base, backoff_max
        self._secret = api_key or None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(timeout_s), headers=headers)
        self._sem = asyncio.Semaphore(concurrency)
        self._sleep = sleep
        self._rng = random.Random()
        self.stats = {"requests": 0, "http_attempts": 0, "retries": 0, "by_status": {}, "prompt_tokens": 0,
                      "completion_tokens": 0, "downgrades": [], "transport_errors": 0, "fatal_errors": 0, "rate_limit": {}}

    def __repr__(self) -> str:
        return (f"LightningClient(endpoint={self.endpoint.name!r}, base_url={self.base_url!r}, model={self.model!r}, "
                f"response_format={self.response_format!r}, concurrency={self.concurrency})")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(self, messages: list[dict], *, json_schema: dict | None = None) -> Completion:
        """POST one chat completion; the concurrency slot is held through retries so backoff also throttles."""
        async with self._sem:
            return await self._complete(messages, json_schema)

    async def _complete(self, messages, json_schema) -> Completion:
        self.stats["requests"] += 1
        start, attempts, failures = time.monotonic(), 0, 0
        while True:
            mode = self._mode(json_schema)
            attempts += 1
            self.stats["http_attempts"] += 1
            retry_after = None
            try:
                resp = await self._http.post(self.url, json=self._body(messages, json_schema, mode))
            except httpx.TransportError as e:  # timeouts and connection errors
                self._count(type(e).__name__)
                error = f"{type(e).__name__}: {self._scrub(str(e))[:300]}"
            else:
                status = resp.status_code
                self._count(str(status))
                self._note_rate_limit(resp.headers)
                if status == 200:
                    done = self._parse(resp, mode, start, attempts)
                    if done is not None:
                        return done
                    error = f"HTTP 200 with unparseable body: {self._scrub(resp.text)[:300]}"
                elif status in DOWNGRADE_STATUS and mode != "none" and any(h in resp.text.lower() for h in DOWNGRADE_HINTS):
                    self._downgrade(mode, status)
                    continue
                elif _retryable(status):
                    error = f"HTTP {status}: {self._scrub(resp.text)[:300]}"
                    retry_after = _retry_after(resp.headers.get("retry-after"))
                else:
                    self.stats["fatal_errors"] += 1
                    raise FatalHTTPError(status, self._scrub(resp.text)[:300], self.url, self._hint(status))
            failures += 1
            if failures >= self.max_attempts:
                self.stats["transport_errors"] += 1
                raise TransportError(f"{self.url}: gave up after {failures} attempts; last error: {error}", failures, error)
            delay = self._rng.uniform(0, min(self.backoff_max, self.backoff_base * 2 ** (failures - 1)))
            if retry_after is not None:
                delay = max(delay, retry_after)
            self.stats["retries"] += 1
            await self._sleep(delay)

    def _mode(self, json_schema) -> str:
        mode = self.response_format
        return "json_object" if mode == "json_schema" and json_schema is None else mode

    def _body(self, messages, json_schema, mode) -> dict:
        body = {"model": self.model, "messages": messages, **copy.deepcopy(SAMPLING)}
        if mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "rewrites", "schema": json_schema}}
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def _parse(self, resp: httpx.Response, mode: str, start: float, attempts: int) -> Completion | None:
        try:
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"].get("content")
            if isinstance(content, list):
                content = "".join(p.get("text") or "" for p in content if isinstance(p, dict))
            usage = data.get("usage") or {}
            pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
            model = data.get("model") or self.model
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return None
        self.stats["prompt_tokens"] += pt
        self.stats["completion_tokens"] += ct
        return Completion(text=content if isinstance(content, str) else "", finish_reason=choice.get("finish_reason"),
                          model=str(model), prompt_tokens=pt, completion_tokens=ct,
                          latency_s=round(time.monotonic() - start, 3), http_attempts=attempts, response_format=mode)

    def _downgrade(self, sent: str, status: int) -> None:
        """Step one level below the format this request sent; concurrent 400s never skip a level."""
        target = RESPONSE_FORMATS[max(RESPONSE_FORMATS.index(self.response_format), RESPONSE_FORMATS.index(sent) + 1)]
        if target != self.response_format:
            self.stats["downgrades"].append({"from": self.response_format, "to": target, "status": status})
            self.response_format = target

    def _hint(self, status: int) -> str:
        if status in (401, 403):
            env = self.endpoint.api_key_env
            return f"check {env} in .env" if env else "server rejected the request; was it started with --api-key?"
        if status == 404:
            return f"check --base-url ({self.base_url}) and --model ({self.model})"
        return ""

    def _scrub(self, text: str) -> str:
        return text.replace(self._secret, "***") if self._secret else text

    def _count(self, key: str) -> None:
        self.stats["by_status"][key] = self.stats["by_status"].get(key, 0) + 1

    def _note_rate_limit(self, headers: httpx.Headers) -> None:
        for k, v in headers.items():
            if "ratelimit" in k.lower() or k.lower() == "retry-after":
                self.stats["rate_limit"][k.lower()] = v
