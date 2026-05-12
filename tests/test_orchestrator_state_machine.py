from dataclasses import dataclass

import numpy as np
import pytest
from pydantic import ValidationError

from lunar_pressure.canonical_prompts import CanonicalPromptCompiler
from lunar_pressure.config import load_config
from lunar_pressure.gemini_gauge_reader import GaugeReadResult, GaugeReaderError
from lunar_pressure.logging import JsonlRunLogger
from lunar_pressure.orchestrator import LunarPressureOrchestrator, OrchestratorSafetyError
from lunar_pressure.schemas import GaugeReading, LocalPlan, OrchestratorState


def make_reading(value, confidence=0.95, need_retry=False):
    return GaugeReading(
        line_id="Line-A",
        gauge_id="G1",
        value_mpa=value,
        confidence=confidence,
        raw_text="needle",
        need_retry=need_retry,
        risk_flags=[],
    )


@dataclass
class FakeGaugeReader:
    readings: list[GaugeReading]

    def read(self, observation):
        reading = self.readings.pop(0) if self.readings else make_reading(0.0)
        return GaugeReadResult(reading=reading, raw_response=reading.model_dump_json(), image_key="observation.scene_image")


class FakeBackend:
    def __init__(self):
        self.prompts = []
        self.calls = 0

    def infer(self, observation):
        self.calls += 1
        self.prompts.append(observation["prompt"])
        return {
            "actions": np.ones((1, 8), dtype=np.float32) * self.calls,
            "server_timing": {"infer_ms": 1.0},
        }


def obs():
    return {
        "prompt": "Keep Line-A pressure at 0.1 MPa",
        "observation.scene_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.state": np.arange(1, 9, dtype=np.float32),
    }


def test_within_tolerance_returns_hold_action_not_zero():
    cfg = load_config("configs/lunar_pressure.yaml").model_copy(update={"hold_action_safety_confirmed": True})
    backend = FakeBackend()
    orchestrator = LunarPressureOrchestrator(cfg, FakeGaugeReader([make_reading(0.1)]), backend)

    response = orchestrator.infer(obs())

    assert backend.calls == 0
    assert orchestrator.state == OrchestratorState.STOP
    np.testing.assert_allclose(response["action"], np.arange(1, 9, dtype=np.float32))
    assert not np.allclose(response["action"], 0)


def test_low_confidence_retries_with_hold_and_does_not_call_backend():
    cfg = load_config("configs/lunar_pressure.yaml").model_copy(update={"hold_action_safety_confirmed": True})
    backend = FakeBackend()
    orchestrator = LunarPressureOrchestrator(
        cfg,
        FakeGaugeReader([make_reading(0.1, confidence=0.2, need_retry=True)]),
        backend,
    )

    response = orchestrator.infer(obs())

    assert backend.calls == 0
    assert orchestrator.state == OrchestratorState.OBSERVE
    np.testing.assert_allclose(response["action"], np.arange(1, 9, dtype=np.float32))


def test_act_phase_uses_canonical_prompt_and_completes_after_max_calls():
    cfg = load_config("configs/lunar_pressure.yaml").model_copy(update={"hold_action_safety_confirmed": True})
    backend = FakeBackend()
    orchestrator = LunarPressureOrchestrator(cfg, FakeGaugeReader([make_reading(0.0)]), backend)

    first = orchestrator.infer(obs())
    second = orchestrator.infer(obs())
    third = orchestrator.infer(obs())

    assert first["actions"].shape == (1, 8)
    assert second["actions"].shape == (1, 8)
    assert third["actions"].shape == (1, 8)
    assert backend.calls == 3
    assert orchestrator.state == OrchestratorState.OBSERVE
    # auto_mark_grasping_after_act defaults to False, so contact_state stays at
    # the initial pose until an explicit grip-confirmation signal flips it.
    assert orchestrator.contact_state == cfg.initial_contact_state
    assert backend.prompts == [
        "Move from the observation pose, grasp valve V1, and turn it clockwise slightly.",
        "Move from the observation pose, grasp valve V1, and turn it clockwise slightly.",
        "Move from the observation pose, grasp valve V1, and turn it clockwise slightly.",
    ]


def test_auto_mark_grasping_flips_contact_state_when_enabled():
    """When auto_mark_grasping_after_act=True the orchestrator flips contact_state on completion.

    This is opt-in because today's `completed` signal can be a timeout, not a
    real grasp confirmation; the safe default is False.
    """
    cfg = load_config("configs/lunar_pressure.yaml").model_copy(
        update={"hold_action_safety_confirmed": True, "auto_mark_grasping_after_act": True}
    )
    backend = FakeBackend()
    orchestrator = LunarPressureOrchestrator(cfg, FakeGaugeReader([make_reading(0.0)]), backend)

    for _ in range(cfg.act_completion.max_policy_calls_per_act):
        orchestrator.infer(obs())

    assert orchestrator.contact_state == "already_grasping_valve"


def test_next_plan_after_first_act_uses_already_grasping_prompt():
    """With auto_mark_grasping_after_act enabled, the next plan flips to the
    'already grasping' prompt set; otherwise stays on the from-observation set."""
    cfg = load_config("configs/lunar_pressure.yaml").model_copy(
        update={"hold_action_safety_confirmed": True, "auto_mark_grasping_after_act": True}
    )
    backend = FakeBackend()
    orchestrator = LunarPressureOrchestrator(
        cfg,
        FakeGaugeReader([make_reading(0.0), make_reading(0.2)]),
        backend,
    )

    for _ in range(3):
        orchestrator.infer(obs())
    orchestrator.infer(obs())

    assert backend.prompts[-1] == "While holding valve V1, turn it counterclockwise slightly."


# ---------------------------------------------------------------------------
# New safety / edge-case tests
# ---------------------------------------------------------------------------


def test_hold_safety_latch_raises_when_false():
    """(i) hold_action_safety_confirmed=False => STOP path raises OrchestratorSafetyError."""
    cfg = load_config("configs/lunar_pressure.yaml")
    # Ensure the latch is actually off (the YAML default is false, but be explicit)
    assert cfg.hold_action_safety_confirmed is False
    backend = FakeBackend()
    orchestrator = LunarPressureOrchestrator(cfg, FakeGaugeReader([make_reading(0.1)]), backend)

    with pytest.raises(OrchestratorSafetyError):
        orchestrator.infer(obs())

    # After the raise, state must be ERROR
    assert orchestrator.state == OrchestratorState.ERROR


def test_observe_retry_exhaustion_stops_and_resets_contact_state():
    """(ii) Exhausting gemini_max_retries_per_observe => STOP; contact_state resets."""
    cfg = load_config("configs/lunar_pressure.yaml").model_copy(
        update={"hold_action_safety_confirmed": True}
    )
    retries = cfg.gemini_max_retries_per_observe
    # Feed exactly `retries` low-confidence readings so we exhaust the budget
    low_readings = [make_reading(0.1, confidence=0.1, need_retry=True) for _ in range(retries)]
    backend = FakeBackend()
    orchestrator = LunarPressureOrchestrator(cfg, FakeGaugeReader(low_readings), backend)

    # Drive the orchestrator until STOP
    for _ in range(retries):
        try:
            response = orchestrator.infer(obs())
        except OrchestratorSafetyError:
            pass
        if orchestrator.state == OrchestratorState.STOP:
            break

    assert orchestrator.state == OrchestratorState.STOP
    assert orchestrator.contact_state == cfg.initial_contact_state


def test_max_act_seconds_budget_triggers_completion():
    """(iii) elapsed >= max_act_seconds => _act_completed returns True even with few calls."""
    cfg = load_config("configs/lunar_pressure.yaml").model_copy(update={"hold_action_safety_confirmed": True})
    backend = FakeBackend()
    orchestrator = LunarPressureOrchestrator(cfg, FakeGaugeReader([make_reading(0.0)]), backend)

    # First infer: OBSERVE->PLAN->ACT, policy_calls becomes 1, act_phase created
    orchestrator.infer(obs())
    assert orchestrator.state == OrchestratorState.ACT

    # Wind back the start time so elapsed >> max_act_seconds
    orchestrator.act_phase.started_monotonic -= cfg.act_completion.max_act_seconds + 100.0

    # Next infer should complete the ACT phase due to time budget
    orchestrator.infer(obs())

    assert orchestrator.state == OrchestratorState.OBSERVE


def test_contact_state_resets_on_stop():
    """(iv) contact_state reverts to initial_contact_state on a STOP transition."""
    cfg = load_config("configs/lunar_pressure.yaml").model_copy(
        update={"hold_action_safety_confirmed": True, "auto_mark_grasping_after_act": True}
    )
    backend = FakeBackend()
    orchestrator = LunarPressureOrchestrator(cfg, FakeGaugeReader([make_reading(0.0)]), backend)

    # Run one full ACT cycle to advance contact_state to already_grasping_valve
    for _ in range(cfg.act_completion.max_policy_calls_per_act):
        orchestrator.infer(obs())
    assert orchestrator.contact_state == "already_grasping_valve"

    # Now supply a reading within tolerance to trigger STOP
    orchestrator.gauge_reader = FakeGaugeReader([make_reading(0.1)])
    orchestrator.infer(obs())

    assert orchestrator.state == OrchestratorState.STOP
    assert orchestrator.contact_state == cfg.initial_contact_state


def test_gemini_parse_failure_raw_response_logged(tmp_path):
    """(v) GaugeReaderError.raw_response is written under gemini_raw/ when reader raises."""

    class ErrorGaugeReader:
        def read(self, observation):
            raise GaugeReaderError("bad parse", raw_response="raw text from gemini")

    cfg = load_config("configs/lunar_pressure.yaml").model_copy(
        update={"hold_action_safety_confirmed": True, "run_dir": str(tmp_path)}
    )
    backend = FakeBackend()
    with JsonlRunLogger(cfg, run_name="err_run") as run_logger:
        orchestrator = LunarPressureOrchestrator(
            cfg, ErrorGaugeReader(), backend, run_logger=run_logger
        )
        orchestrator.infer(obs())
    # Logger is closed (flushed) by context manager __exit__

    gemini_raw_dir = tmp_path / "err_run" / "gemini_raw"
    raw_files = list(gemini_raw_dir.iterdir())
    assert raw_files, "Expected at least one file in gemini_raw/"
    content = raw_files[0].read_text(encoding="utf-8")
    assert "raw text from gemini" in content


def test_canonical_prompt_compiler_raises_on_stop_plan_kind():
    """(vi) CanonicalPromptCompiler.compile('stop', ...) raises ValueError."""
    cfg = load_config("configs/lunar_pressure.yaml")
    compiler = CanonicalPromptCompiler(cfg.canonical_prompts)
    with pytest.raises(ValueError):
        compiler.compile("stop", "from_observation_pose")


def test_local_plan_rejects_action_kind_with_no_direction():
    """(vii) LocalPlan with plan_kind != 'stop' and direction=None raises ValidationError."""
    with pytest.raises(ValidationError):
        LocalPlan(
            plan_kind="increase_pressure",
            error_mpa=0.05,
            direction=None,
            canonical_prompt=None,
            contact_state="from_observation_pose",
            reason="test",
        )

