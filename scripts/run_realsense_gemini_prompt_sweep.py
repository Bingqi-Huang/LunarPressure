#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from lunar_pressure.observation_contract import replace_prompt
from lunar_pressure.schemas import ContactState, GaugeReading


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture RealSense RGB frames, ask Gemini to read the gauge, and "
            "sweep target pressures to verify the OpenPI prompt replacement path. "
            "This script never connects to OpenPI and never emits robot actions."
        )
    )
    parser.add_argument("--config", default="configs/lunar_pressure.yaml")
    parser.add_argument("--serial", default=None, help="RealSense device serial number.")
    parser.add_argument("--list-devices", action="store_true", help="List RealSense devices and exit.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--samples", type=int, default=1, help="Number of screen/gauge images to capture.")
    parser.add_argument(
        "--targets",
        default="auto",
        help=(
            "Comma-separated target pressures in MPa, or 'auto' for "
            "visual_min,target,visual_max from config."
        ),
    )
    parser.add_argument(
        "--contact-states",
        default="from_observation_pose,already_grasping_valve",
        help="Comma-separated contact states to test.",
    )
    parser.add_argument(
        "--fake-pressure-mpa",
        type=float,
        default=None,
        help="Skip Gemini and use this pressure reading. Useful for prompt-only dry runs.",
    )
    parser.add_argument(
        "--no-pause-before-capture",
        action="store_true",
        help="Do not wait for Enter before each sample capture.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    rs = import_realsense()
    if args.list_devices:
        list_devices(rs)
        return

    if args.fake_pressure_mpa is None and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --fake-pressure-mpa.", file=sys.stderr)
        sys.exit(2)

    config = load_config(args.config)
    targets = parse_targets(args.targets, config)
    contact_states = parse_contact_states(args.contact_states)
    output_dir = Path(args.output_dir or default_run_dir("realsense_gemini_prompt_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    records_path = output_dir / "records.jsonl"

    reader = GeminiGaugeReader(config)

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if args.serial:
        rs_config.enable_device(args.serial)
    rs_config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)

    print(f"Writing results to {output_dir}")
    print("This script does not start OpenPI and does not send any action.")

    try:
        pipeline.start(rs_config)
        warmup_camera(pipeline, args.warmup_frames)
        for sample_index in range(args.samples):
            if not args.no_pause_before_capture:
                input(
                    f"\nShow gauge image {sample_index + 1}/{args.samples} on the screen, "
                    "then press Enter to capture..."
                )
            rgb = capture_rgb_frame(pipeline)
            image_path = image_dir / f"sample_{sample_index:03d}.jpg"
            Image.fromarray(rgb).save(image_path, quality=95)

            observation = {
                "prompt": config.task_label,
                config.gemini_primary_image: rgb,
                "observation.state": np.ones(8, dtype=np.float32),
            }
            result = read_gauge(reader, observation, args.fake_pressure_mpa, config)
            print_reading(sample_index, result, image_path)

            for target in targets:
                target_config = config.model_copy(update={"target_pressure_mpa": target})
                for contact_state in contact_states:
                    record = evaluate_prompt(
                        target_config,
                        result.reading,
                        observation,
                        contact_state,
                        sample_index=sample_index,
                        image_path=str(image_path),
                        raw_response=result.raw_response,
                    )
                    append_jsonl(records_path, record)
                    print_plan(record)
    finally:
        pipeline.stop()

    print(f"\nDone. JSONL records: {records_path}")


def import_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed. Install hardware dependencies with:\n"
            "  uv sync --extra hardware\n"
            "or install pyrealsense2 in the active environment."
        ) from exc
    return rs


def list_devices(rs: Any) -> None:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("No RealSense devices found.")
        return
    for device in devices:
        print(f"{device.get_info(rs.camera_info.serial_number)}  {device.get_info(rs.camera_info.name)}")


def warmup_camera(pipeline: Any, frames: int) -> None:
    for _ in range(max(frames, 0)):
        pipeline.wait_for_frames()


def capture_rgb_frame(pipeline: Any) -> np.ndarray:
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    color = frames.get_color_frame()
    if not color:
        raise RuntimeError("RealSense color frame was not available")
    return np.asanyarray(color.get_data()).copy()


def parse_targets(value: str, config: LunarPressureConfig) -> list[float]:
    if value == "auto":
        candidates = [
            config.visual_hard_min_mpa,
            config.target_pressure_mpa,
            config.visual_hard_max_mpa,
        ]
    else:
        candidates = [float(item.strip()) for item in value.split(",") if item.strip()]
    seen: list[float] = []
    for target in candidates:
        if target not in seen:
            seen.append(target)
    return seen


def parse_contact_states(value: str) -> list[ContactState]:
    states: list[ContactState] = []
    allowed = {"from_observation_pose", "already_grasping_valve"}
    for item in value.split(","):
        state = item.strip()
        if not state:
            continue
        if state not in allowed:
            raise ValueError(f"Unknown contact state {state!r}; expected one of {sorted(allowed)}")
        states.append(state)  # type: ignore[arg-type]
    if not states:
        raise ValueError("At least one contact state is required")
    return states


def read_gauge(
    reader: GeminiGaugeReader,
    observation: dict[str, Any],
    fake_pressure_mpa: float | None,
    config: LunarPressureConfig,
) -> GaugeReadResult:
    if fake_pressure_mpa is not None:
        reading = GaugeReading(
            line_id=config.line_id,
            gauge_id=config.gauge_id,
            value_mpa=fake_pressure_mpa,
            confidence=1.0,
            raw_text="fake pressure supplied by --fake-pressure-mpa",
            need_retry=False,
            risk_flags=[],
        )
        return GaugeReadResult(
            reading=reading,
            raw_response=reading.model_dump_json(),
            image_key=config.gemini_primary_image,
        )
    try:
        return reader.read(observation)
    except GaugeReaderError as exc:
        print(f"GaugeReaderError: {exc}", file=sys.stderr)
        if exc.raw_response is not None:
            print(f"raw_response: {exc.raw_response}", file=sys.stderr)
        raise


def evaluate_prompt(
    config: LunarPressureConfig,
    reading: GaugeReading,
    observation: dict[str, Any],
    contact_state: ContactState,
    *,
    sample_index: int,
    image_path: str,
    raw_response: str,
) -> dict[str, Any]:
    planner = LocalPressurePlanner(config)
    record: dict[str, Any] = {
        "timestamp": utc_now(),
        "sample_index": sample_index,
        "image_path": image_path,
        "target_pressure_mpa": config.target_pressure_mpa,
        "contact_state": contact_state,
        "original_task_prompt": config.task_label,
        "gemini_raw_response": raw_response,
        "reading": reading.model_dump(mode="json"),
        "reading_usable": reading.is_usable(config.gemini_confidence_threshold),
    }
    try:
        planner.validate_reading_bounds(reading)
        plan = planner.plan(reading, contact_state)
    except Exception as exc:
        record.update({"status": "planner_error", "error": str(exc)})
        return record

    record["status"] = "ok"
    record["plan"] = plan.model_dump(mode="json")
    if plan.should_act and plan.canonical_prompt is not None:
        forwarded = replace_prompt(observation, plan.canonical_prompt)
        record["would_forward_to_openpi"] = True
        record["forwarded_prompt"] = forwarded["prompt"]
    else:
        record["would_forward_to_openpi"] = False
        record["forwarded_prompt"] = None
    return record


def print_reading(sample_index: int, result: GaugeReadResult, image_path: Path) -> None:
    reading = result.reading
    print(
        f"\n[sample {sample_index}] image={image_path} "
        f"value={reading.value_mpa:.4f} MPa confidence={reading.confidence:.2f} "
        f"need_retry={reading.need_retry}"
    )


def print_plan(record: dict[str, Any]) -> None:
    if record["status"] != "ok":
        print(
            f"  target={record['target_pressure_mpa']:.4f} "
            f"contact={record['contact_state']} ERROR {record['error']}"
        )
        return
    plan = record["plan"]
    prompt = record["forwarded_prompt"] or "<STOP: no runtime OpenPI prompt>"
    print(
        f"  target={record['target_pressure_mpa']:.4f} contact={record['contact_state']} "
        f"plan={plan['plan_kind']} error={plan['error_mpa']:.4f} prompt={prompt}"
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_run_dir(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return str(Path("runs") / f"{prefix}_{stamp}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
