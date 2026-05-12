from __future__ import annotations

import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import LunarPressureConfig, dump_config_snapshot
from .schemas import RunSummary, StepLogRecord


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlRunLogger:
    """Small append-only run logger for replay/debugging.

    All file I/O is offloaded to a single-threaded executor so the hot path
    (orchestrator infer loop) is never blocked by disk writes.  Callers still
    receive the destination path immediately — only the bytes hit disk later.
    """

    def __init__(self, config: LunarPressureConfig, run_name: str | None = None):
        if run_name is None:
            run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_name = run_name
        self.root = Path(config.run_dir) / run_name
        self.images_dir = self.root / "images"
        self.actions_dir = self.root / "actions"
        self.gemini_raw_dir = self.root / "gemini_raw"
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.actions_dir.mkdir(exist_ok=True)
        self.gemini_raw_dir.mkdir(exist_ok=True)
        (self.root / "config_snapshot.yaml").write_text(dump_config_snapshot(config), encoding="utf-8")
        self.steps_path = self.root / "steps.jsonl"
        self._step_index = 0
        self.start_time: str = utc_timestamp()
        self._pending_summary: RunSummary | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lunar-logger"
        )

    # ------------------------------------------------------------------
    # Public write methods — return path immediately, I/O runs in worker
    # ------------------------------------------------------------------

    def write_step(self, record: StepLogRecord) -> str:
        path = str(self.steps_path)
        line = json.dumps(_json_safe(record.model_dump(mode="json")), ensure_ascii=False) + "\n"
        self._step_index += 1
        self._executor.submit(_append_text, path, line)
        return path

    def write_raw_gemini(self, raw_text: str) -> str:
        path = self.gemini_raw_dir / f"step_{self._step_index:06d}.txt"
        self._executor.submit(_write_text, str(path), raw_text)
        return str(path)

    def write_action(self, action_payload: Any) -> str:
        path = self.actions_dir / f"step_{self._step_index:06d}.json"
        text = json.dumps(_json_safe(action_payload), ensure_ascii=False, indent=2)
        self._executor.submit(_write_text, str(path), text)
        return str(path)

    def write_summary(self, summary: RunSummary) -> None:
        if summary.end_time is None:
            summary = summary.model_copy(update={"end_time": utc_timestamp()})
        self._pending_summary = summary
        text = json.dumps(_json_safe(summary.model_dump(mode="json")), ensure_ascii=False, indent=2)
        self._executor.submit(_write_text, str(self.root / "summary.json"), text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "JsonlRunLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class NullRunLogger:
    def write_step(self, record: StepLogRecord) -> None:
        return None

    def write_raw_gemini(self, raw_text: str) -> str | None:
        return None

    def write_action(self, action_payload: Any) -> str | None:
        return None

    def write_summary(self, summary: RunSummary) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "NullRunLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


# ------------------------------------------------------------------
# Worker helpers (plain module-level functions; no closure captures)
# ------------------------------------------------------------------

def _append_text(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
