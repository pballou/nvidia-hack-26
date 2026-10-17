from unittest.mock import MagicMock

import pytest
from app import (
    GuiGroundingResourcesServer,
    GuiGroundingResourcesServerConfig,
    GuiGroundingVerifyRequest,
    _inside,
    score,
)

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient

# A 2000 x 1000 screenshot with the target at pixels x 400-600, y 100-200,
# i.e. x 200-300 and y 100-200 on the 0-1000 grid.
ROW = {"id": "t:0", "source": "os_atlas/linux", "instruction": "open settings", "platform": "desktop",
       "data_type": None, "bbox": [400.0, 100.0, 600.0, 200.0], "img_w": 2000, "img_h": 1000}


def _server() -> GuiGroundingResourcesServer:
    config = GuiGroundingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return GuiGroundingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(text: str, **overrides) -> GuiGroundingVerifyRequest:
    response = NeMoGymResponse(
        id="response", created_at=0.0, model="model", object="response", parallel_tool_calls=False,
        tool_choice="none", tools=[],
        output=[NeMoGymResponseOutputMessage(
            id="message", role="assistant", status="completed", type="message",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        )],
    )
    params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "x"}])
    return GuiGroundingVerifyRequest(**{**ROW, **overrides}, responses_create_params=params, response=response)


class TestApp:
    def test_sanity(self) -> None:
        _server()

    @pytest.mark.parametrize(
        ("pt", "expected"),
        [((500, 150), True), ((400, 100), True), ((600, 200), True), ((399.9, 150), False), ((500, 201), False)],
    )
    def test_inside_is_inclusive(self, pt, expected) -> None:
        assert _inside(pt, ROW["bbox"]) is expected

    @pytest.mark.parametrize(
        ("text", "reward"),
        [
            ('{"point_2d": [250, 150]}', 1.0),
            ('```json\n[{"point_2d": [250, 150], "label": "Settings"}]\n```', 1.0),  # Qwen3.5's base format
            ('<think>maybe {"point_2d": [900, 900]}</think>{"point_2d": [250, 150]}', 1.0),
            ('{"point_2d": [350, 150]}', 0.0),  # right row, wrong column
        ],
    )
    def test_score(self, text, reward) -> None:
        assert score(text, ROW["bbox"], ROW["img_w"], ROW["img_h"])[0] == reward

    @pytest.mark.asyncio
    async def test_verify_hit(self) -> None:
        out = await _server().verify(_request('{"point_2d": [250, 150]}'))
        assert (out.reward, out.parsed, out.pred_x, out.pred_y) == (1.0, True, 250.0, 150.0)
        assert out.platform_data_type == "desktop/any" and not out.mask_sample

    @pytest.mark.asyncio
    async def test_verify_unparsed_counts_as_miss(self) -> None:
        out = await _server().verify(_request("I can't find it."))
        assert (out.reward, out.parsed, out.mask_sample) == (0.0, False, False)
        assert out.failure_reason == "no parsable click point"

    @pytest.mark.asyncio
    async def test_verify_error_masks_instead_of_raising(self) -> None:
        out = await _server().verify(_request('{"point_2d": [250, 150]}', bbox=[1.0, 2.0]))
        assert (out.reward, out.mask_sample) == (0.0, True)
        assert out.failure_reason.startswith("verify error")

    def test_compute_metrics_slices(self) -> None:
        tasks = [
            [{"reward": 1.0, "platform": "mobile", "data_type": "icon", "platform_data_type": "mobile/icon", "source": "s"}],
            [{"reward": 0.0, "platform": "mobile", "data_type": "text", "platform_data_type": "mobile/text", "source": "s"}],
        ]
        metrics = _server().compute_metrics(tasks)
        assert metrics["mobile/pass@1/accuracy"] == 50.0
        assert metrics["mobile/icon/pass@1/accuracy"] == 100.0
        assert metrics["text/pass@1/accuracy"] == 0.0
