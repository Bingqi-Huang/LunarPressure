from __future__ import annotations

import time
from typing import Any, Mapping

from .wire import MsgpackNumpyCodec

try:
    import websockets.exceptions as _ws_exceptions
    _WEBSOCKET_RETRY_EXC = (OSError, TimeoutError, _ws_exceptions.WebSocketException)
except ImportError:
    _ws_exceptions = None  # type: ignore[assignment]
    _WEBSOCKET_RETRY_EXC = (OSError, TimeoutError)


class OpenPIBackendError(RuntimeError):
    pass


class OpenPIBackendClient:
    """Synchronous OpenPI websocket client matching openpi-client's protocol."""

    def __init__(self, host: str, port: int | None = None, api_key: str | None = None, *, connect: bool = True):
        if host.startswith("ws"):
            self.uri = host
        else:
            self.uri = f"ws://{host}"
        if port is not None:
            self.uri += f":{port}"
        self.api_key = api_key
        self._packer = None
        self._ws = None
        self.metadata: dict[str, Any] = {}
        if connect:
            self.connect()

    def connect(self) -> None:
        try:
            import websockets.sync.client
        except ImportError as exc:
            raise OpenPIBackendError(
                "websockets is required for OpenPIBackendClient"
            ) from exc

        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        self._packer = MsgpackNumpyCodec()
        self._ws = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
        )
        self.metadata = self._packer.unpack(self._ws.recv())

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if self._ws is None or self._packer is None:
            self.connect()
        assert self._ws is not None
        assert self._packer is not None
        try:
            self._ws.send(self._packer.pack(dict(observation)))
            response = self._ws.recv()
            if isinstance(response, str):
                raise OpenPIBackendError(f"OpenPI backend returned an error string:\n{response}")
            return self._packer.unpack(response)
        except Exception:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            raise

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None


class RetryingOpenPIBackendClient(OpenPIBackendClient):
    def __init__(
        self,
        host: str,
        port: int | None = None,
        api_key: str | None = None,
        *,
        retry_sleep_s: float = 5.0,
    ):
        self.retry_sleep_s = retry_sleep_s
        super().__init__(host, port, api_key, connect=False)
        self.connect()

    def connect(self) -> None:
        while True:
            try:
                return super().connect()
            except _WEBSOCKET_RETRY_EXC:
                time.sleep(self.retry_sleep_s)
