#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from lunar_pressure.config import load_config
from lunar_pressure.gemini_gauge_reader import GaugeReaderError, GeminiGaugeReader


def main() -> None:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print(
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running the smoke test.",
            file=sys.stderr,
        )
        sys.exit(2)

    parser = argparse.ArgumentParser(description="Run a Gemini gauge-reading smoke test on a local image.")
    parser.add_argument("image", help="Path to a pressure-gauge image")
    parser.add_argument("--config", default="configs/lunar_pressure.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    reader = GeminiGaugeReader(config)
    observation = {
        "prompt": config.task_label,
        config.gemini_primary_image: args.image,
        "observation.state": np.ones(8, dtype=np.float32),
    }
    try:
        result = reader.read(observation)
    except GaugeReaderError as exc:
        print(f"GaugeReaderError: {exc}\nraw_response: {exc.raw_response!r}", file=sys.stderr)
        sys.exit(1)
    print(result.reading.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

