"""Single-process asyncio server. Serves index.html + streams pipeline frames.

Run:
    python visualizer/web/websocket_server.py [--channels 8|16|32] [--file PATH]

Then open http://localhost:8765 in a browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Allow running from project root or from this dir
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aiohttp import web                                        # noqa: E402
from visualizer.runner import Runner                            # noqa: E402

log = logging.getLogger(__name__)
INDEX_PATH = Path(__file__).parent / "index.html"

# State shared by producer + websocket connections
_clients: set[web.WebSocketResponse] = set()
_runner: Runner | None = None


def _frame_to_json(frame) -> str:
    """Subsample channels for the wire — never more than 16 lines actually plotted."""
    n_show = min(frame.n_ch, 8)  # display first 8 channels max in line panel
    payload = {
        "type":     "frame",
        "n_ch":     frame.n_ch,
        "raw":      frame.raw[:n_show].tolist(),
        "filtered": frame.filtered[:n_show].tolist(),
        "bands":    frame.band_powers.mean(axis=0).tolist(),  # mean across channels
        "metrics": {
            "latency_ms":     round(frame.latency_ms, 3),
            "n_channels":     frame.n_ch,
            "ops_per_w":      frame.ops_per_w,
            "avg_ops_per_w":  frame.avg_ops_per_w,
            "rel_throughput": frame.relative_throughput,
            "power_w":        frame.power_w,
        },
    }
    return json.dumps(payload)


async def producer():
    """Drives the runner; broadcasts each frame to all connected clients."""
    assert _runner is not None
    loop = asyncio.get_event_loop()
    frames = iter(_runner)
    while True:
        # Hop off the loop so other coroutines can run between windows.
        frame = await loop.run_in_executor(None, next, frames)
        if not _clients:
            await asyncio.sleep(0.05)
            continue
        msg = _frame_to_json(frame)
        await asyncio.gather(*(c.send_str(msg) for c in list(_clients)),
                             return_exceptions=True)


async def index(request: web.Request) -> web.Response:
    return web.FileResponse(INDEX_PATH)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    _clients.add(ws)
    log.info("client connected (total: %d)", len(_clients))
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if cmd.get("type") == "set_channels":
                    n = int(cmd["n"])
                    if n in (8, 16, 32) and _runner is not None:
                        log.info("client requested reconfigure → %d ch", n)
                        _runner.reconfigure(n)
    finally:
        _clients.discard(ws)
        log.info("client disconnected (total: %d)", len(_clients))
    return ws


async def on_startup(app: web.Application):
    app["producer"] = asyncio.create_task(producer())


async def on_cleanup(app: web.Application):
    app["producer"].cancel()
    try:
        await app["producer"]
    except asyncio.CancelledError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=8, choices=[8, 16, 32])
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--fs", type=float, default=250.0)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    global _runner
    _runner = Runner(n_ch=args.channels, fs=args.fs, source_path=args.file)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    log.info("serving http://localhost:%d", args.port)
    web.run_app(app, port=args.port, print=None)


if __name__ == "__main__":
    main()
