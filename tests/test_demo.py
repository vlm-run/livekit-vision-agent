"""The demo-only behavior: the model picker and the overlay frame."""

from __future__ import annotations

import json
from typing import Any

import agent
import demo
from test_agent import fake_gateway  # noqa: F401  (autouse)
from test_agent import FakeRoom, FakeSession, camera, run_loop, sent_count, stop, wait_for

class _Reader:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    async def read_all(self) -> str:
        return json.dumps(self._body)

async def test_each_result_comes_with_the_frame_it_was_read_on():
    """The page draws a result on the JPEG it belongs to, so the frame goes first, same id."""
    room = FakeRoom()
    vision = demo.DemoAgent(room)  # type: ignore[arg-type]
    feed = camera(vision)
    task = await run_loop(vision)
    await wait_for(lambda: len(room.local_participant.texts) >= 2)
    await stop(task)
    feed.cancel()

    files = room.local_participant.files
    assert room.local_participant.order[:2] == ["bytes:frame", "text:vision"]
    for file, (_, text) in zip(files, room.local_participant.texts, strict=False):
        payload = json.loads(text)
        assert file["mime_type"] == "image/jpeg" and file["data"][:3] == b"\xff\xd8\xff"
        assert file["name"] == str(payload["_vlmrun"]["seq"])
        assert file["attributes"] == {"frame": str(payload["_vlmrun"]["seq"])}
async def test_control_message_switches_the_model():
    room = FakeRoom()
    vision = demo.DemoAgent(room)  # type: ignore[arg-type]
    room.register_text_stream_handler("control", vision._on_control)
    feed = camera(vision)
    task = await run_loop(vision)
    await wait_for(lambda: sent_count() >= 1)
    await vision._apply_control(_Reader({"model": "geopavlakos/hamer", "method": "pose"}))  # type: ignore[arg-type]
    await wait_for(lambda: FakeSession.instances[-1].model == "geopavlakos/hamer")
    await wait_for(lambda: len(FakeSession.instances[-1].sent) >= 1)
    await stop(task)
    feed.cancel()

    assert FakeSession.instances[0].closed, "the old model's session was closed"
    payload = json.loads(room.local_participant.texts[-1][1])
    assert payload["_vlmrun"]["model"] == "geopavlakos/hamer"
async def test_control_message_without_a_model_is_ignored():
    vision = demo.DemoAgent(FakeRoom())  # type: ignore[arg-type]
    before = vision._want
    await vision._apply_control(_Reader({"method": "pose"}))  # type: ignore[arg-type]
    assert vision._want == before


def test_entrypoints_can_be_pickled():
    """LiveKit sends the entrypoint to a job process, so a closure or lambda would fail there."""
    import pickle

    for fn in (agent.entrypoint, demo.entrypoint):
        assert pickle.loads(pickle.dumps(fn)) is fn
