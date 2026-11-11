"""Watch a LiveKit room and read each new frame on the VLM Run gateway.

The page publishes a camera or a screen. This worker keeps the newest frame,
JPEG-encodes it, and sends those bytes on one ``WS /v1/ws`` session. Each
``result`` is published back into the room on the ``vision`` text stream.

The gateway bills the seconds a session is open, so the session opens on the
first frame, closes when the video track goes away or no new frame arrives for
``IDLE_CLOSE_S``, and each frame is sent once.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from PIL import Image

from gateway import GatewaySession, observation, sample_interval

load_dotenv(".env.local")
load_dotenv()

logger = logging.getLogger("vlmrun-vision")

GATEWAY_WS = os.environ.get("VLMRUN_GATEWAY_WS", "wss://gateway.vlm.run/v1/ws")
MODEL = os.environ.get("VLMRUN_MODEL", "usyd-community/vitpose-plus-large")
METHOD = os.environ.get("VLMRUN_METHOD", "pose")
PROMPT = os.environ.get("VLMRUN_PROMPT", "")
# JSON for the method, e.g. {"bbox_xywh": [0.1, 0.1, 0.4, 0.5]} for SAM segment_box.
PARAMS = os.environ.get("VLMRUN_PARAMS", "")
VISION_FPS = float(os.environ.get("VISION_FPS", "8"))
RESULT_TOPIC = "vision"
# Close the session after this long without a new frame (a paused or muted camera).
IDLE_CLOSE_S = 5.0
# Wait between failed attempts doubles from RETRY_S up to MAX_RETRY_S, so a model at
# capacity ("capacity_error", which has no retry_after) isn't hammered.
RETRY_S = 1.0
MAX_RETRY_S = 30.0
# A frame with no reply in this long means the session is stuck (seen live: one reply
# took 19 s); the session is closed and reopened.
FRAME_TIMEOUT_S = 5.0


def _jpeg(frame: rtc.VideoFrame, max_width: int = 640) -> bytes:
    # The reader already asks VideoStream for RGB24, so the buffer is packed RGB.
    image = Image.frombytes("RGB", (frame.width, frame.height), bytes(frame.data))
    if image.width > max_width:
        image = image.resize((max_width, int(image.height * max_width / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()


class VisionAgent:
    def __init__(self, room: rtc.Room) -> None:
        self.room = room
        self.latest: rtc.VideoFrame | None = None
        self._new_frame = asyncio.Event()
        self._track_sid: str | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stream: rtc.VideoStream | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._want = (MODEL, METHOD, PROMPT, PARAMS)
        self._seq = 0

    async def start(self) -> None:
        self.room.on("track_subscribed", self._on_track)
        self.room.on("track_unsubscribed", self._on_track_gone)
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                    self._watch(publication.track)
        await self._analysis_loop()

    def _track_task(self, task: asyncio.Task[None]) -> None:
        # Keep a reference so the task isn't garbage-collected mid-run.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def push(self, frame: rtc.VideoFrame) -> None:
        """Offer a new frame; it replaces any frame not yet sent."""
        self.latest = frame
        self._new_frame.set()

    def _on_track(
        self,
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        logger.info("watching %s", participant.identity)
        self._watch(track)

    def _on_track_gone(
        self,
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        _participant: rtc.RemoteParticipant,
    ) -> None:
        if track.sid != self._track_sid:
            return
        logger.info("video track left")
        self._release_reader()
        self._track_sid = None
        self.latest = None
        self._new_frame.set()  # wakes the loop so it closes the session

    def _watch(self, track: rtc.Track) -> None:
        self._release_reader()
        self._track_sid = track.sid
        self._stream = rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.RGB24)
        self._reader = asyncio.create_task(self._read_frames(self._stream))

    def _release_reader(self) -> None:
        # A separate task closes the stream: a reader cancelled before it ever ran would
        # skip its own cleanup and leak the stream.
        reader, stream = self._reader, self._stream
        self._reader, self._stream = None, None
        if reader is None and stream is None:
            return

        async def close() -> None:
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            if stream is not None:
                await stream.aclose()

        self._track_task(asyncio.create_task(close()))

    async def _read_frames(self, stream: rtc.VideoStream) -> None:
        async for event in stream:
            if stream is not self._stream:
                return  # released; a late frame must not reopen a session
            self.push(event.frame)

    async def _next_frame(self, timeout: float | None) -> rtc.VideoFrame | None:
        """The newest unsent frame, or None after ``timeout`` without one."""
        try:
            await asyncio.wait_for(self._new_frame.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        self._new_frame.clear()
        frame, self.latest = self.latest, None
        return frame

    async def _open(self, api_key: str, want: tuple[str, str, str, str]) -> GatewaySession:
        model, method, prompt, params = want
        session = GatewaySession(GATEWAY_WS, api_key, model, method, prompt, json.loads(params or "{}"))
        await session.open()
        logger.info("gateway open model=%s method=%s max_fps=%s", model, method or "(default)", session.max_fps)
        return session

    def _event(self, kind: str, **data: object) -> None:
        """A session lifecycle event. The demo posts these to the page; here it is a hook."""

    async def _publish(self, jpeg: bytes, event: dict, model: str, latency_ms: int, seq: int) -> None:
        """Post the result as JSON on the ``vision`` text stream."""
        payload = observation(event, model=model, latency_ms=latency_ms)
        payload["_vlmrun"]["seq"] = seq
        await self.room.local_participant.send_text(json.dumps(payload), topic=RESULT_TOPIC)

    async def _analysis_loop(self) -> None:
        api_key = os.environ.get("VLMRUN_API_KEY", "")
        if not api_key:
            raise RuntimeError("VLMRUN_API_KEY is not set")
        session: GatewaySession | None = None
        opened_with: tuple[str, str, str, str] | None = None
        interval = sample_interval(VISION_FPS, 0)
        failures = 0

        async def drop_session(reason: str) -> None:
            nonlocal session
            if session is not None:
                await session.aclose()
                self._event("session.close", reason=reason)
                session = None

        try:
            while True:
                frame = await self._next_frame(IDLE_CLOSE_S if session is not None else None)
                if frame is None:
                    # The camera left or went idle: close the billed session until the next frame.
                    if session is not None:
                        logger.info("closing gateway session (no frames)")
                        await drop_session("no frames for %gs" % IDLE_CLOSE_S)
                    continue
                if session is not None and opened_with != self._want:
                    await drop_session("model changed")
                if session is None:
                    opened_with = self._want
                    self._event("session.start", model=opened_with[0], method=opened_with[1])
                    try:
                        session = await self._open(api_key, opened_with)
                    except Exception as e:
                        failures += 1
                        wait = min(MAX_RETRY_S, RETRY_S * 2 ** (failures - 1))
                        logger.warning("gateway session failed (%s); retrying in %.0fs", e, wait)
                        self._event("session.failed", error=str(e), retry_s=wait)
                        await asyncio.sleep(wait)
                        continue
                    interval = sample_interval(VISION_FPS, session.max_fps)
                    self._event("session.open", model=session.model, max_fps=session.max_fps)
                started = time.perf_counter()
                try:
                    jpeg = await asyncio.to_thread(_jpeg, frame)
                    event = await session.send_jpeg(jpeg, timeout=FRAME_TIMEOUT_S)
                    failures = 0
                    if event is not None:
                        latency_ms = round((time.perf_counter() - started) * 1000)
                        # Off the critical path: the next frame doesn't wait for the room.
                        self._seq += 1  # numbered here, not in the task, so concurrent publishes can't share an id
                        self._track_task(
                            asyncio.create_task(self._publish(jpeg, event, session.model, latency_ms, self._seq))
                        )
                        logger.info("result %sms model=%s", latency_ms, session.model)
                except Exception:
                    logger.exception("frame failed")
                    failures += 1
                    self._event("frame.failed", retry_s=min(MAX_RETRY_S, RETRY_S * 2 ** (failures - 1)))
                    await drop_session("frame failed")
                    await asyncio.sleep(min(MAX_RETRY_S, RETRY_S * 2 ** (failures - 1)))
                    continue
                await asyncio.sleep(max(0.0, interval - (time.perf_counter() - started)))
        finally:
            await drop_session("worker stopped")


async def run(ctx: JobContext, agent_cls: type[VisionAgent]) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)
    logger.info("joined %s", ctx.room.name)
    await agent_cls(ctx.room).start()


# Module level: LiveKit pickles the entrypoint to start each job in its own process.
async def entrypoint(ctx: JobContext) -> None:
    await run(ctx, VisionAgent)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
