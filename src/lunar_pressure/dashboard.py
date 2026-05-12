from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def latest_run(run_dir: str | Path = "runs") -> Path | None:
    root = Path(run_dir)
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_steps(run_path: Path) -> list[dict[str, Any]]:
    steps_path = run_path / "steps.jsonl"
    if not steps_path.exists():
        return []
    steps: list[dict[str, Any]] = []
    for line in steps_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            steps.append(json.loads(line))
    return steps


def render_latest_summary(run_dir: str | Path = "runs") -> str:
    run_path = latest_run(run_dir)
    if run_path is None:
        return "No LunarPressure runs found."
    steps = load_steps(run_path)
    lines = [f"Run: {run_path.name}", f"Path: {run_path}", f"Steps: {len(steps)}"]
    if steps:
        last = steps[-1]
        lines.extend(
            [
                f"Last state: {last.get('state')}",
                f"Outcome: {last.get('outcome')}",
                f"Stop reason: {last.get('stop_reason')}",
                f"Prompt: {last.get('canonical_prompt')}",
            ]
        )
        reading = last.get("gauge_reading")
        if reading:
            lines.append(f"Gauge: {reading.get('value_mpa')} MPa @ confidence {reading.get('confidence')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print a summary of the latest LunarPressure run.")
    parser.add_argument("--run-dir", default="runs")
    args = parser.parse_args(argv)
    print(render_latest_summary(args.run_dir))


if __name__ == "__main__":
    main()

