#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lunar_pressure.config import LunarPressureConfig, load_config
from lunar_pressure.gemini_gauge_reader import GaugeReadResult, GaugeReaderError, GeminiGaugeReader
from lunar_pressure.local_planner import LocalPressurePlanner
from lunar_pressure.observation_contract import get_image, replace_prompt
from lunar_pressure.schemas import ContactState, GaugeReading
from lunar_pressure.wire import MsgpackNumpyCodec

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenPI-compatible websocket server that receives RoboCOIN "
            "observations, computes the prompt LunarPressure would send to OpenPI, "
            "and intentionally sends no robot action."
        )
    )
    parser.add_argument("--config", default="configs/lunar_pressure.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--target-pressure-mpa", type=float, default=None)
    parser.add_argument(
        "--contact-state",
        choices=["from_observation_pose", "already_grasping_valve"],
        default=None,
    )
    parser.add_argument(
        "--fake-pressure-mpa",
        type=float,
        default=None,
        help="Skip Gemini and use this pressure reading. Useful for prompt-only wire tests.",
    )
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument(
        "--keep-open-after-frame",
        action="store_true",
        help=(
            "Keep the websocket open after processing max frames without sending a response. "
            "RoboCOIN will normally block waiting for policy.infer() to return."
        ),
    )
    parser.add_argument(
        "--expected-camera-serial",
        default=None,
        help="Optional note recorded in JSONL; RoboCOIN observations do not usually include this.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.fake_pressure_mpa is None and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --fake-pressure-mpa.", file=sys.stderr)
        sys.exit(2)

    config = load_config(args.config)
    if args.target_pressure_mpa is not None:
        config = config.model_copy(update={"target_pressure_mpa": args.target_pressure_mpa})
    host = args.host or config.lunarpressure_server_host
    port = args.port or config.lunarpressure_server_port
    contact_state: ContactState = args.contact_state or config.initial_contact_state

    output_dir = Path(args.output_dir or default_run_dir("robocoin_prompt_monitor"))
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    if not args.no_save_images:
        image_dir.mkdir(exist_ok=True)

    monitor = PromptMonitor(
        config=config,
        contact_state=contact_state,
        fake_pressure_mpa=args.fake_pressure_mpa,
        output_dir=output_dir,
        image_dir=image_dir,
        save_images=not args.no_save_images,
        max_frames=args.max_frames,
        keep_open_after_frame=args.keep_open_after_frame,
        expected_camera_serial=args.expected_camera_serial,
    )

    print(f"Listening on ws://{host}:{port}")
    print(f"Writing records to {output_dir / 'records.jsonl'}")
    print("No per-observation action response will be sent.")
    print("For RoboCOIN, a block or websocket close after the first observation is expected.")
    asyncio.run(monitor.run(host, port))


class PromptMonitor:
    def __init__(
        self,
        *,
        config: LunarPressureConfig,
        contact_state: ContactState,
        fake_pressure_mpa: float | None,
        output_dir: Path,
        image_dir: Path,
        save_images: bool,
        max_frames: int,
        keep_open_after_frame: bool,
        expected_camera_serial: str | None,
    ):
        self.config = config
        self.contact_state = contact_state
        self.fake_pressure_mpa = fake_pressure_mpa
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.save_images = save_images
        self.max_frames = max_frames
        self.keep_open_after_frame = keep_open_after_frame
        self.expected_camera_serial = expected_camera_serial
        self.reader = GeminiGaugeReader(config)
        self.records_path = output_dir / "records.jsonl"

    async def run(self, host: str, port: int) -> None:
        import websockets.asyncio.server as ws_server

        async with ws_server.serve(
            self.handler,
            host,
            port,
            compression=None,
            max_size=None,
            process_request=health_check,
        ) as server:
            await server.serve_forever()

    async def handler(self, websocket: Any) -> None:
        codec = MsgpackNumpyCodec()
        await websocket.send(
            codec.pack(
                {
                    "server": "lunar-pressure-prompt-monitor",
                    "mode": "receive_only_no_action",
                    "target_pressure_mpa": self.config.target_pressure_mpa,
                }
            )
        )

        frame_index = 0
        while True:
            try:
                raw = await websocket.recv()
            except Exception:
                logger.info("client disconnected")
                return
            try:
                observation = codec.unpack(raw)
            except Exception:
                logger.exception("failed to decode observation")
                await websocket.close(code=1011, reason="bad observation frame")
                return

            frame_index += 1
            self.process_observation(observation, frame_index)

            if frame_index >= self.max_frames:
                if self.keep_open_after_frame:
                    logger.info("processed max frames; keeping websocket open without action response")
                    while True:
                        await asyncio.sleep(3600)
                await websocket.close(code=1000, reason="prompt monitor complete; no action sent")
                return

    def process_observation(self, observation: dict[str, Any], frame_index: int) -> None:
        image_path = self.save_observation_image(observation, frame_index) if self.save_images else None
        record: dict[str, Any] = {
            "timestamp": utc_now(),
            "frame_index": frame_index,
            "target_pressure_mpa": self.config.target_pressure_mpa,
            "contact_state": self.contact_state,
            "expected_camera_serial": self.expected_camera_serial,
            "observation_keys": list(observation.keys()),
            "incoming_task_prompt": observation.get("prompt"),
            "image_path": image_path,
        }

        try:
            result = self.read_gauge(observation)
            record["gemini_raw_response"] = result.raw_response
            record["image_key"] = result.image_key
            record["reading"] = result.reading.model_dump(mode="json")
            record["reading_usable"] = result.reading.is_usable(self.config.gemini_confidence_threshold)

            planner = LocalPressurePlanner(self.config)
            planner.validate_reading_bounds(result.reading)
            plan = planner.plan(result.reading, self.contact_state)
            record["plan"] = plan.model_dump(mode="json")
            if plan.should_act and plan.canonical_prompt is not None:
                forwarded = replace_prompt(observation, plan.canonical_prompt)
                record["would_forward_to_openpi"] = True
                record["forwarded_prompt"] = forwarded["prompt"]
            else:
                record["would_forward_to_openpi"] = False
                record["forwarded_prompt"] = None
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            if isinstance(exc, GaugeReaderError) and exc.raw_response is not None:
                record["gemini_raw_response"] = exc.raw_response

        append_jsonl(self.records_path, record)
        print_record(record)

    def read_gauge(self, observation: dict[str, Any]) -> GaugeReadResult:
        if self.fake_pressure_mpa is not None:
            reading = GaugeReading(
                line_id=self.config.line_id,
                gauge_id=self.config.gauge_id,
                value_mpa=self.fake_pressure_mpa,
                confidence=1.0,
                raw_text="fake pressure supplied by --fake-pressure-mpa",
                need_retry=False,
                risk_flags=[],
            )
            return GaugeReadResult(
                reading=reading,
                raw_response=reading.model_dump_json(),
                image_key=self.config.gemini_primary_image,
            )
        return self.reader.read(observation)

    def save_observation_image(self, observation: dict[str, Any], frame_index: int) -> str | None:
        try:
            image_key, image = get_image(
                observation,
                self.config.gemini_primary_image,
                self.config.gemini_fallback_image,
            )
        except Exception:
            return None
        try:
            array = normalize_image_array(image)
            path = self.image_dir / f"frame_{frame_index:06d}_{sanitize_key(image_key)}.jpg"
            Image.fromarray(array).save(path, quality=95)
            return str(path)
        except Exception:
            logger.debug("failed to save observation image", exc_info=True)
            return None


def health_check(connection: Any, request: Any) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def normalize_image_array(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim != 3:
        raise ValueError(f"Expected image rank 2 or 3, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def print_record(record: dict[str, Any]) -> None:
    prefix = f"[frame {record['frame_index']}] target={record['target_pressure_mpa']:.4f}"
    if record["status"] != "ok":
        print(f"{prefix} ERROR {record['error']}")
        return
    reading = record["reading"]
    plan = record["plan"]
    prompt = record["forwarded_prompt"] or "<STOP: no runtime OpenPI prompt>"
    print(
        f"{prefix} value={reading['value_mpa']:.4f} confidence={reading['confidence']:.2f} "
        f"plan={plan['plan_kind']} prompt={prompt}"
    )


def sanitize_key(key: str) -> str:
    return key.replace("/", "_").replace(".", "_")


def default_run_dir(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return str(Path("runs") / f"{prefix}_{stamp}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
