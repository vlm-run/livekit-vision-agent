"""Baseline for realtime sessions on the VLM Run gateway socket.

    uv run python bench/ws_baseline.py --models google/gemma4-e2b-it:realtime qwen/qwen3.5-0.8b:realtime

Per model and target fps: open time, round-trip p50/p95, gateway-side latency,
drop and error counts, and a sample reply. Cold models answer "starting up";
the open is retried and the wait is reported as ``cold_wait_s``.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gateway import GatewaySession, message_content  # noqa: E402

BUS = "https://ultralytics.com/images/bus.jpg"


def jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def frames() -> list[tuple[str, bytes]]:
    req = urllib.request.Request(BUS, headers={"User-Agent": "bench"})
    with urllib.request.urlopen(req, timeout=30) as r:
        bus = Image.open(io.BytesIO(r.read()))
    shapes = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(shapes)
    d.rectangle((80, 80, 280, 280), fill="red")
    d.ellipse((340, 160, 560, 380), fill="blue")
    return [
        ("bus_640", jpeg(bus.resize((640, 480)))),
        ("bus_1280", jpeg(bus.resize((1280, 960)))),
        ("shapes", jpeg(shapes)),
        ("blank", jpeg(Image.new("RGB", (640, 480), "white"))),
    ]


def summarize(ev) -> str:
    text = message_content(ev.get("response"))
    try:
        body = json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError:
        return str(text)[:160]
    if isinstance(body, dict) and isinstance(body.get("items"), list):
        return f"{len(body['items'])} items"
    return str(text)[:160]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))] if xs else float("nan")


async def open_session(url, key, model, prompt, method="chat", max_wait=420):
    start = time.perf_counter()
    cold = 0.0
    while True:
        s = GatewaySession(url, key, model, method, prompt)
        t0 = time.perf_counter()
        try:
            await s.open()
            return s, time.perf_counter() - t0, cold
        except RuntimeError as e:
            m = re.search(r"Retry in about (\d+) s", str(e))
            if not m or time.perf_counter() - start > max_wait:
                raise
            wait = min(int(m.group(1)), 30)
            print(f"  {model} cold, retry in {wait}s", flush=True)
            await asyncio.sleep(wait)
            cold = time.perf_counter() - start


async def run_fps(session, fps, n, frame_set):
    rtts, gw, drops, errors, sample = [], [], 0, 0, {}
    interval = 1.0 / min(fps, session.max_fps or fps)
    for i in range(n):
        name, data = frame_set[i % len(frame_set)]
        t0 = time.perf_counter()
        try:
            ev = await session.send_jpeg(data, timeout=30)
        except (RuntimeError, TimeoutError) as e:
            errors += 1
            sample.setdefault("error", str(e))
            if isinstance(e, TimeoutError):
                break
            continue
        rtt = (time.perf_counter() - t0) * 1000
        if ev is None:
            drops += 1
        else:
            rtts.append(rtt)
            if ev.get("latency_ms") is not None:
                gw.append(float(ev["latency_ms"]))
            sample.setdefault(name, summarize(ev))
        await asyncio.sleep(max(0.0, interval - (time.perf_counter() - t0)))
    return {
        "fps": fps, "sent": n, "ok": len(rtts), "drops": drops, "errors": errors,
        "rtt_p50": round(pct(rtts, .5)), "rtt_p95": round(pct(rtts, .95)),
        "rtt_mean": round(statistics.mean(rtts)) if rtts else None,
        "gateway_p50": round(pct(gw, .5)) if gw else None,
        "samples": sample,
    }


async def bench_model(url, key, spec, default_prompt, fps_list, n, frame_set):
    model, _, rest = spec.partition("@")
    method, _, prompt = rest.partition("@")
    method = method or "chat"
    prompt = prompt or (default_prompt if method == "chat" else "")
    out = {"model": spec}
    try:
        s, open_s, cold = await open_session(url, key, model, prompt, method)
    except Exception as e:
        out["error"] = str(e)
        return out
    out.update(open_s=round(open_s, 2), cold_wait_s=round(cold, 1), max_fps=s.max_fps, runs=[])
    try:
        for fps in fps_list:
            print(f"  {spec} @ {fps} fps", flush=True)
            out["runs"].append(await run_fps(s, fps, n, frame_set))
    finally:
        await s.aclose()
    # second open: warm reconnect cost
    t0 = time.perf_counter()
    s2, _, _ = await open_session(url, key, model, prompt, method)
    out["reopen_s"] = round(time.perf_counter() - t0, 2)
    await s2.aclose()
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="model[@method[@prompt]]")
    ap.add_argument("--fps", nargs="+", type=float, default=[1, 5, 10])
    ap.add_argument("--frames", type=int, default=20, help="frames per fps level")
    ap.add_argument("--prompt", default="Describe what you see in one sentence.")
    ap.add_argument("--out", default="bench/results.json")
    a = ap.parse_args()
    url = os.environ.get("VLMRUN_GATEWAY_WS", "wss://gateway.vlm.run/v1/ws")
    key = os.environ["VLMRUN_API_KEY"]
    fs = frames()
    results = [await bench_model(url, key, m, a.prompt, a.fps, a.frames, fs) for m in a.models]
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\n{'model':34} {'fps':>4} {'ok':>3} {'drop':>4} {'err':>3} {'p50':>6} {'p95':>6} {'gw50':>6}")
    for r in results:
        if "error" in r:
            print(f"{r['model'][:34]:34} ERROR {r['error']}")
            continue
        print(f"{r['model'][:34]:34} open={r['open_s']}s cold={r['cold_wait_s']}s reopen={r['reopen_s']}s max_fps={r['max_fps']}")
        for x in r["runs"]:
            print(f"{'':34} {x['fps']:>4g} {x['ok']:>3} {x['drops']:>4} {x['errors']:>3} {x['rtt_p50']:>6} {x['rtt_p95']:>6} {str(x['gateway_p50']):>6}")


asyncio.run(main())
