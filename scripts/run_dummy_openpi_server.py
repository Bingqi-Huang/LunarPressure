#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import http
import logging
from typing import Any

import numpy as np

from lunar_pressure.observation_contract import build_hold_action
from lunar_pressure.wire import MsgpackNumpyCodec

logger = logging.getLogger(__name__)

try:
    import websockets.exceptions as _ws_exc
except ImportError:
    _ws_exc = None


async def handler(websocket: Any) -> None:
    codec = MsgpackNumpyCodec()
    await websocket.send(codec.pack({"server": "dummy-openpi"}))
    while True:
        try:
            obs = codec.unpack(await websocket.recv())
        except Exception:
            if _ws_exc is not None:
                logger.debug("dummy server recv closed")
            return
        try:
            action = build_hold_action(obs)
        except Exception as exc:
            logger.exception("dummy server failed to build action")
            try:
                await websocket.send(
                    codec.pack({"error": str(exc), "server": "dummy-openpi"})
                )
            except Exception:
                pass
            await websocket.close(code=1011, reason="dummy server error")
            return
        await websocket.send(
            codec.pack(
                {
                    "actions": np.tile(action[None, :], (1, 1)),
                    "server_timing": {"infer_ms": 0.0},
                    "dummy_prompt": obs.get("prompt"),
                }
            )
        )


def health_check(connection: Any, request: Any) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def run(host: str, port: int) -> None:
    import websockets.asyncio.server as ws_server

    async with ws_server.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=health_check,
    ) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dummy OpenPI backend that returns hold actions.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(run(args.host, args.port))


if __name__ == "__main__":
    main()

