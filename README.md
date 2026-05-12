# LunarPressure

LunarPressure is a small orchestration server for the RM75 pressure-gauge demo.
It sits between RoboCOIN and an OpenPI policy server:

```text
RoboCOIN robot_client_openpi.py
  -> LunarPressure websocket server
  -> OpenPI policy server
  -> RoboCOIN send_action(...)
```

Gemini only reads the analog pressure gauge. Local deterministic code computes
the pressure error and selects a fixed VLA prompt. OpenPI receives the original
observation with only `prompt` replaced.

## Setup

Use Python 3.10+ on the machine that will run LunarPressure.

```bash
cd LunarPressure
uv sync --extra dev --extra hardware
```

Set a Gemini key before any script that calls Gemini:

```bash
export GEMINI_API_KEY="..."
# or
export GOOGLE_API_KEY="..."
```

The repo does not require local `openpi-lunarbot/` or `RoboCOIN/` folders. Those
folders may exist next to this repo during development only for source lookup.

Main configuration lives in:

```text
configs/lunar_pressure.yaml
configs/prompts/gemini_gauge_reading.md
```

Keep these two safety latches false until they are explicitly validated:

```yaml
hold_action_safety_confirmed: false
auto_mark_grasping_after_act: false
```

## Experiment 1: RealSense + Gemini + Prompt Sweep

This test does not start RoboCOIN, the robot, or OpenPI. It connects directly to
a RealSense camera, captures gauge images from your computer screen, asks Gemini
to read pressure, then sweeps target pressures to verify the planner and prompt
replacement path.

List RealSense devices:

```bash
uv run python scripts/run_realsense_gemini_prompt_sweep.py --list-devices
```

Run one sample:

```bash
uv run python scripts/run_realsense_gemini_prompt_sweep.py \
  --serial "<REAL_REALSENSE_SERIAL>" \
  --targets 0.02,0.10,0.18 \
  --samples 3
```

For each sample, put a different pressure-gauge image on the screen, then press
Enter. The script writes images and JSONL records under `runs/`.

Prompt-only dry run without Gemini:

```bash
uv run python scripts/run_realsense_gemini_prompt_sweep.py \
  --serial "<REAL_REALSENSE_SERIAL>" \
  --fake-pressure-mpa 0.10 \
  --targets 0.02,0.10,0.18
```

## Experiment 2: RoboCOIN Observation Prompt Monitor

This test starts the robot and RoboCOIN camera pipeline, but does not start
OpenPI. LunarPressure acts like an OpenPI-compatible websocket server only long
enough to receive real RoboCOIN observations, read the gauge, and print the
prompt it would forward to OpenPI.

It intentionally sends no per-observation action response. RoboCOIN will usually
block or close after the first `policy.infer()` call; that is expected and keeps
the test action-free.

Start the monitor on the GPU/LunarPressure machine:

```bash
uv run python scripts/run_robocoin_observation_prompt_monitor.py \
  --host 0.0.0.0 \
  --port 8001 \
  --target-pressure-mpa 0.10 \
  --max-frames 1
```

Then start RoboCOIN with its policy server host/port pointed at this monitor,
using the real camera config you intend to use in the demo. The monitor writes
records and any decoded images under `runs/`.

Prompt-only wire dry run:

```bash
uv run python scripts/run_robocoin_observation_prompt_monitor.py \
  --host 0.0.0.0 \
  --port 8001 \
  --fake-pressure-mpa 0.10 \
  --max-frames 1
```

## Real Closed-Loop Run

After the two action-free tests pass, the remaining true hardware gates are:

- confirm RoboCOIN `delta_with`, action key order, units, and gripper semantics;
- validate that hold actions built from `observation.state` keep the robot still;
- verify the real OpenPI server accepts the original RoboCOIN observation keys;
- run a small-motion test with manual e-stop ready.

Only then set:

```yaml
hold_action_safety_confirmed: true
```

Start the real OpenPI policy server separately, for example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_rm75_pick_place \
  --policy.dir=/path/to/checkpoint/
```

Start LunarPressure:

```bash
uv run python scripts/run_orchestrator_server.py --config configs/lunar_pressure.yaml
```

Run tests before changing hardware-facing behavior:

```bash
uv run pytest -q
uv run python -m compileall -q src scripts
```
