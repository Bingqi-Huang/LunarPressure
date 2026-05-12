import json

import numpy as np

from lunar_pressure.config import load_config
from lunar_pressure.logging import JsonlRunLogger, utc_timestamp
from lunar_pressure.schemas import OrchestratorState, StepLogRecord


def test_jsonl_run_logger_writes_config_step_and_action(tmp_path):
    cfg = load_config("configs/lunar_pressure.yaml").model_copy(update={"run_dir": str(tmp_path)})
    logger = JsonlRunLogger(cfg, run_name="test_run")

    action_path = logger.write_action({"action": np.arange(8, dtype=np.float32)})
    logger.write_step(
        StepLogRecord(
            timestamp=utc_timestamp(),
            state=OrchestratorState.STOP,
            observation_keys=["prompt"],
            action_path=action_path,
            outcome="stop",
        )
    )
    # Flush the async thread-pool writer before asserting on disk state.
    logger.close()

    run_path = tmp_path / "test_run"
    assert (run_path / "config_snapshot.yaml").exists()
    assert (run_path / "steps.jsonl").exists()
    assert (run_path / "actions" / "step_000000.json").exists()

    line = (run_path / "steps.jsonl").read_text(encoding="utf-8").strip()
    decoded = json.loads(line)
    assert decoded["state"] == "STOP"
    assert decoded["outcome"] == "stop"

