"""GUI click grounding: screenshot + instruction -> click point; reward 1 if it lands in the target box.

Adapted from NeMo Gym v0.7.0 (tag commit 1f8afc71d4) resources_servers/gui_coordinate,
Apache-2.0; that path last changed 2026-09-29 (016bb6b811). Changes: a binary
point-in-bbox reward on pixel boxes instead of a distance reward, output parsing via
common/grounding.py, and per-slice metrics.

Rows come from data_prep/gym_rows.py. The same /verify endpoint scores OS-Atlas dev
(checkpoint selection), ScreenSpot-v2-clean (the held-out test), and GRPO completions.
"""

import re
import sys
from pathlib import Path
from typing import Any, Optional

# No root pyproject.toml on purpose (AGENTS.md), so put the repo root on sys.path to
# import common/grounding.py, the single owner of output parsing and coordinates.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.grounding import parse_point, to_pixels  # noqa: E402
from nemo_gym.base_resources_server import (  # noqa: E402
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_subset_metrics  # noqa: E402

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)
SLICE_KEYS = ("platform", "data_type", "platform_data_type", "source")


class GuiGroundingResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GuiGroundingRunRequest(BaseRunRequest):
    id: str
    source: str
    instruction: str
    platform: str  # mobile | desktop | web
    data_type: Optional[str] = None  # text | icon (ScreenSpot); None for OS-Atlas
    bbox: list[float]  # [x1, y1, x2, y2], pixels of the original img_w x img_h screenshot
    img_w: int
    img_h: int


class GuiGroundingVerifyRequest(GuiGroundingRunRequest, BaseVerifyRequest):
    pass


class GuiGroundingVerifyResponse(GuiGroundingVerifyRequest, BaseVerifyResponse):
    platform_data_type: str
    parsed: bool
    pred_x: Optional[float] = None  # 0-1000 grid
    pred_y: Optional[float] = None


def _last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                texts.extend(c.text for c in content if isinstance(getattr(c, "text", None), str))
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


def _inside(pt_px: tuple[float, float], bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= pt_px[0] <= x2 and y1 <= pt_px[1] <= y2  # inclusive, as in ScreenSpot


def score(text: str, bbox: list[float], img_w: int, img_h: int) -> tuple[float, Optional[tuple[float, float]]]:
    """(reward, parsed 0-1000 point or None) for one model reply."""
    pt = parse_point(_THINK_RE.sub("", text))
    hit = pt is not None and _inside(to_pixels(pt, img_w, img_h), bbox)
    return float(hit), pt


class GuiGroundingResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: GuiGroundingResourcesServerConfig

    async def verify(self, body: GuiGroundingVerifyRequest) -> GuiGroundingVerifyResponse:
        fields = body.model_dump()
        fields["platform_data_type"] = f"{body.platform}/{body.data_type or 'any'}"
        try:
            reward, pt = score(_last_assistant_text(body), body.bbox, body.img_w, body.img_h)
        except Exception as e:  # an exception in verify() aborts the whole gym eval run
            return GuiGroundingVerifyResponse(
                **fields, reward=0.0, parsed=False, mask_sample=True, failure_reason=f"verify error: {e!r}"
            )
        return GuiGroundingVerifyResponse(
            **fields,
            reward=reward,
            parsed=pt is not None,
            pred_x=pt[0] if pt else None,
            pred_y=pt[1] if pt else None,
            failure_reason=None if pt else "no parsable click point",
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Accuracy per platform, text/icon, platform x text/icon and source."""
        metrics: dict[str, Any] = {}
        for key in SLICE_KEYS:
            metrics.update(compute_subset_metrics(tasks, key))
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key = super().get_key_metrics(agent_metrics)
        key.update({k: v for k, v in agent_metrics.items() if re.search(r"pass@1(\[avg-of-\d+\])?/accuracy$", k)})
        return key


if __name__ == "__main__":
    GuiGroundingResourcesServer.run_webserver()
