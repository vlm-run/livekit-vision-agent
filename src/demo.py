"""The agent plus what the demo page needs: the model picker and the overlay frame.

    uv run src/demo.py dev

``agent.py`` alone posts results on the ``vision`` topic. This adds a ``control``
stream the page uses to switch model, and the JPEG each result was read on, so the
page can draw on the frame the model saw.
"""

from __future__ import annotations

import asyncio
import json
import logging

from livekit import rtc
from livekit.agents import WorkerOptions, cli

import agent

logger = logging.getLogger("vlmrun-vision")


class DemoAgent(agent.VisionAgent):
    async def start(self) -> None:
        self.room.register_text_stream_handler("control", self._on_control)
        await super().start()

    def _on_control(self, reader: rtc.TextStreamReader, _participant: str) -> None:
        self._track_task(asyncio.create_task(self._apply_control(reader)))

    async def _apply_control(self, reader: rtc.TextStreamReader) -> None:
        try:
            body = json.loads(await reader.read_all())
        except Exception:
            logger.exception("control message")
            return
        model = str(body.get("model") or "").strip()
        if not model:
            return
        self._want = (model, str(body.get("method") or ""), str(body.get("prompt") or ""))
        logger.info("model request %s method=%s", model, self._want[1] or "(default)")

    async def _publish(self, jpeg: bytes, event: dict, model: str, latency_ms: int) -> None:
        # The frame goes first, so the page draws the result on the JPEG the model saw.
        frame_id = str(self._seq + 1)
        await self.room.local_participant.send_bytes(
            jpeg, topic="frame", name=frame_id, mime_type="image/jpeg",
            attributes={"frame": frame_id}, compress=False,
        )
        await super()._publish(jpeg, event, model, latency_ms)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=agent.make_entrypoint(DemoAgent)))
