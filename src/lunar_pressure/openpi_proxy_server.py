from __future__ import annotations

import argparse
import asyncio
import http
import logging
from typing import Any

from .config import LunarPressureConfig, load_config
from .gemini_gauge_reader import GeminiGaugeReader
from .logging import JsonlRunLogger
from .observation_contract import build_hold_response
from .openpi_backend_client import RetryingOpenPIBackendClient
from .orchestrator import LunarPressureOrchestrator
from .wire import MsgpackNumpyCodec

try:
    import websockets.exceptions as _ws_exc
except ImportError:
    _ws_exc = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class OpenPICompatibleOrchestratorServer:
    def __init__(
        self,
        orchestrator: LunarPressureOrchestrator,
        host: str,
        port: int,
        metadata: dict[str, Any] | None = None,
    ):
        self.orchestrator = orchestrator
        self.host = host
        self.port = port
        self.metadata = metadata or {"server": "lunar-pressure", "protocol": "openpi-websocket"}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        try:
            import websockets.asyncio.server as ws_server
        except ImportError as exc:
            raise RuntimeError("websockets is required to run the orchestrator server") from exc

        async with ws_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("LunarPressure server listening on ws://%s:%s", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket: Any) -> None:
        codec = MsgpackNumpyCodec()
        await websocket.send(codec.pack(self.metadata))

        while True:
            # --- receive ---
            try:
                raw = await websocket.recv()
            except Exception as exc:
                # Clean disconnect from the client — just exit the handler.
                if _ws_exc is not None and isinstance(exc, (_ws_exc.ConnectionClosedOK, _ws_exc.ConnectionClosed)):
                    return
                # Any other recv error (e.g. network reset): also exit silently.
                logger.debug("recv error, closing handler: %s", exc)
                return

            try:
                obs = codec.unpack(raw)
            except Exception:
                logger.exception("failed to decode incoming frame; closing")
                await websocket.close(code=1011, reason="bad frame")
                return

            # --- infer ---
            try:
                response = self.orchestrator.infer(obs)
            except Exception:
                logger.exception("orchestrator handler error")
                # Wire-protocol contract: RoboCOIN's _extract_actions() only
                # understands {"action": ...} / {"actions": ...} responses, so
                # we never send a bare error dict. Instead, if it is safe to
                # do so (latch confirmed AND a hold action can be built from
                # the current observation), send one final hold action; else
                # close cleanly without sending a frame the client cannot
                # parse. Either way, never re-raise.
                hold_response = self._safe_hold_fallback(obs)
                if hold_response is not None:
                    try:
                        await websocket.send(codec.pack(hold_response))
                    except Exception:
                        logger.debug("failed to send fallback hold; closing anyway")
                await websocket.close(code=1011, reason="internal server error")
                return

            await websocket.send(codec.pack(response))

    def _safe_hold_fallback(self, observation: Any) -> dict[str, Any] | None:
        """Return a hold-action response if it is safe to construct one, else None."""
        if not getattr(self.orchestrator.config, "hold_action_safety_confirmed", False):
            return None
        try:
            return build_hold_response(observation, reason="server_error_fallback_hold")
        except Exception:
            logger.debug("could not construct fallback hold action", exc_info=True)
            return None


def _health_check(connection: Any, request: Any) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def build_orchestrator(config: LunarPressureConfig, run_name: str | None = None) -> LunarPressureOrchestrator:
    """Build an orchestrator from a pre-loaded config object."""
    gauge_reader = GeminiGaugeReader(config)
    backend = RetryingOpenPIBackendClient(config.openpi_backend_host, config.openpi_backend_port)
    run_logger = JsonlRunLogger(config, run_name=run_name)
    return LunarPressureOrchestrator(config, gauge_reader, backend, run_logger)


def build_orchestrator_from_config(config_path: str, run_name: str | None = None) -> LunarPressureOrchestrator:
    """Build an orchestrator by loading config from *config_path*."""
    return build_orchestrator(load_config(config_path), run_name=run_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the LunarPressure OpenPI-compatible orchestrator server.")
    parser.add_argument("--config", default="configs/lunar_pressure.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    orchestrator = build_orchestrator(config, run_name=args.run_name)
    server = OpenPICompatibleOrchestratorServer(
        orchestrator,
        host=config.lunarpressure_server_host,
        port=config.lunarpressure_server_port,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
