from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import LunarPressureConfig
from .gemini_gauge_reader import GaugeReader, GaugeReaderError
from .local_planner import LocalPressurePlanner
from .logging import NullRunLogger, utc_timestamp
from .observation_contract import (
    ObservationContractError,
    build_hold_response,
    extract_action_payload,
    get_high_level_task,
    observation_metadata,
    replace_prompt,
)
from .schemas import ContactState, GaugeReading, LocalPlan, OrchestratorState, StepLogRecord


class OrchestratorSafetyError(RuntimeError):
    pass


@dataclass
class ActPhase:
    plan: LocalPlan
    started_monotonic: float
    policy_calls: int = 0
    previous_last_action: np.ndarray | None = None


class LunarPressureOrchestrator:
    def __init__(
        self,
        config: LunarPressureConfig,
        gauge_reader: GaugeReader,
        backend_client: Any,
        run_logger: Any | None = None,
    ):
        self.config = config
        self.gauge_reader = gauge_reader
        self.backend_client = backend_client
        self.run_logger = run_logger or NullRunLogger()
        self.planner = LocalPressurePlanner(config)
        self.state = OrchestratorState.OBSERVE
        self.contact_state: ContactState = config.initial_contact_state
        self.observe_failures = 0
        self.act_phase: ActPhase | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if self.state == OrchestratorState.ACT and self.act_phase is not None:
            return self._act(observation)
        return self._observe_plan_act(observation)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_action_state(self) -> None:
        """Reset transient action state when entering STOP or ERROR."""
        self.contact_state = self.config.initial_contact_state
        self.act_phase = None
        self.observe_failures = 0

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _observe_plan_act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        step_start = time.perf_counter()
        self.state = OrchestratorState.OBSERVE
        reading: GaugeReading | None = None
        plan: LocalPlan | None = None
        raw_path: str | None = None
        image_key: str | None = None
        outcome = "observe_retry"
        stop_reason: str | None = None

        try:
            result = self.gauge_reader.read(observation)
            reading = result.reading
            image_key = result.image_key
            raw_path = self.run_logger.write_raw_gemini(result.raw_response)
        except (GaugeReaderError, ObservationContractError, ValueError) as exc:
            self.observe_failures += 1
            stop_reason = f"gauge_read_failed: {exc}"
            if hasattr(exc, "raw_response") and exc.raw_response is not None:
                raw_path = self.run_logger.write_raw_gemini(exc.raw_response)
            if self.observe_failures >= self.config.gemini_max_retries_per_observe:
                self.state = OrchestratorState.STOP
                self._reset_action_state()
                outcome = "stop_gauge_read_failed"
                response = self._hold(observation, reason=stop_reason)
            else:
                response = self._hold(observation, reason="gauge_retry")
            self._log_step(observation, step_start, reading, plan, None, response, outcome, stop_reason, raw_path, image_path=image_key)
            return response

        if not reading.is_usable(self.config.gemini_confidence_threshold):
            self.observe_failures += 1
            stop_reason = "gauge_low_confidence_or_retry"
            if self.observe_failures >= self.config.gemini_max_retries_per_observe:
                self.state = OrchestratorState.STOP
                self._reset_action_state()
                outcome = "stop_low_confidence"
                response = self._hold(observation, reason=stop_reason)
            else:
                response = self._hold(observation, reason="gauge_retry")
            self._log_step(observation, step_start, reading, plan, None, response, outcome, stop_reason, raw_path, image_path=image_key)
            return response

        self.observe_failures = 0
        try:
            self.planner.validate_reading_bounds(reading)
        except ValueError as exc:
            self.state = OrchestratorState.STOP
            self._reset_action_state()
            stop_reason = f"visual_hard_bound_violation: {exc}"
            response = self._hold(observation, reason=stop_reason)
            self._log_step(observation, step_start, reading, plan, None, response, "stop_hard_bound", stop_reason, raw_path, image_path=image_key)
            return response

        self.state = OrchestratorState.PLAN
        plan = self.planner.plan(reading, self.contact_state)
        if not plan.should_act:
            self.state = OrchestratorState.STOP
            self._reset_action_state()
            response = self._hold(observation, reason=plan.reason)
            self._log_step(observation, step_start, reading, plan, None, response, "stop", plan.reason, raw_path, image_path=image_key)
            return response

        self.state = OrchestratorState.ACT
        self.act_phase = ActPhase(plan=plan, started_monotonic=time.monotonic())
        response = self._act(observation, step_start=step_start, reading=reading, raw_path=raw_path, image_key=image_key)
        return response

    def _act(
        self,
        observation: Mapping[str, Any],
        *,
        step_start: float | None = None,
        reading: GaugeReading | None = None,
        raw_path: str | None = None,
        image_key: str | None = None,
    ) -> dict[str, Any]:
        if self.act_phase is None:
            if step_start is None:
                step_start = time.perf_counter()
            self.state = OrchestratorState.ERROR
            self._reset_action_state()
            stop_reason = "missing_act_phase"
            hold_response = self._hold(observation, reason=stop_reason)
            self._log_step(
                observation, step_start, reading, None, None, hold_response,
                "error_hold", stop_reason, raw_path, image_path=image_key,
            )
            return hold_response

        if step_start is None:
            step_start = time.perf_counter()
        plan = self.act_phase.plan

        if plan.canonical_prompt is None:
            self.state = OrchestratorState.ERROR
            self._reset_action_state()
            raise OrchestratorSafetyError(
                "plan.canonical_prompt is None; refusing to forward to OpenPI"
            )

        forwarded = replace_prompt(observation, plan.canonical_prompt)
        try:
            response = self.backend_client.infer(forwarded)
            payload = extract_action_payload(response)
        except Exception as exc:
            self.state = OrchestratorState.ERROR
            self._reset_action_state()
            stop_reason = f"openpi_backend_failed: {exc}"
            hold_response = self._hold(observation, reason=stop_reason)
            self._log_step(observation, step_start, reading, plan, plan.canonical_prompt, hold_response, "error_hold", stop_reason, raw_path, image_path=image_key)
            return hold_response

        self.act_phase.policy_calls += 1
        completed = self._act_completed(payload.actions, observation)
        action_path = self.run_logger.write_action(response)
        if completed:
            if self.config.auto_mark_grasping_after_act:
                self.contact_state = "already_grasping_valve"
            self.act_phase = None
            self.state = OrchestratorState.OBSERVE
            outcome = "act_complete"
        else:
            self.state = OrchestratorState.ACT
            outcome = "act_continue"
        self._log_step(
            observation,
            step_start,
            reading,
            plan,
            plan.canonical_prompt,
            response,
            outcome,
            None,
            raw_path,
            action_path=action_path,
            image_path=image_key,
        )
        return response

    def _act_completed(self, actions: np.ndarray, observation: Mapping[str, Any]) -> bool:
        assert self.act_phase is not None
        cfg = self.config.act_completion
        elapsed = time.monotonic() - self.act_phase.started_monotonic
        last_action = actions[-1].astype(np.float32, copy=True)

        residual_done = False
        if cfg.use_joint_residual_check and cfg.joint_residual_threshold is not None:
            try:
                from .observation_contract import get_state_vector

                state = get_state_vector(observation)
                dims = min(len(state), len(last_action))
                residual_done = float(np.linalg.norm(last_action[:dims] - state[:dims])) <= cfg.joint_residual_threshold
            except ObservationContractError:
                residual_done = False

        delta_done = False
        if (
            cfg.use_action_delta_check
            and cfg.action_delta_threshold is not None
            and self.act_phase.previous_last_action is not None
        ):
            dims = min(len(self.act_phase.previous_last_action), len(last_action))
            delta_done = (
                float(np.linalg.norm(last_action[:dims] - self.act_phase.previous_last_action[:dims]))
                <= cfg.action_delta_threshold
            )

        self.act_phase.previous_last_action = last_action
        return (
            self.act_phase.policy_calls >= cfg.max_policy_calls_per_act
            or elapsed >= cfg.max_act_seconds
            or residual_done
            or delta_done
        )

    def _hold(self, observation: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
        if not self.config.hold_action_safety_confirmed:
            self.state = OrchestratorState.ERROR
            self._reset_action_state()
            raise OrchestratorSafetyError(
                "hold_action_safety_confirmed is false; refusing to emit hold action until "
                "delta_with / action key order / gripper semantics are validated. "
                "See ARCHITECTURE.md §14."
            )
        try:
            return build_hold_response(observation, reason=reason)
        except ObservationContractError as exc:
            self.state = OrchestratorState.ERROR
            raise OrchestratorSafetyError(f"Cannot build safe hold action: {exc}") from exc

    def _log_step(
        self,
        observation: Mapping[str, Any],
        step_start: float,
        reading: GaugeReading | None,
        plan: LocalPlan | None,
        canonical_prompt: str | None,
        response: Mapping[str, Any] | None,
        outcome: str,
        stop_reason: str | None,
        raw_path: str | None,
        *,
        action_path: str | None = None,
        image_path: str | None = None,
    ) -> None:
        timing = {"total_ms": (time.perf_counter() - step_start) * 1000}
        response_metadata = {}
        if isinstance(response, Mapping):
            response_metadata = {key: type(value).__name__ for key, value in response.items()}
        record = StepLogRecord(
            timestamp=utc_timestamp(),
            state=self.state,
            high_level_task=get_high_level_task(observation),
            observation_keys=list(observation.keys()),
            observation_metadata=observation_metadata(observation),
            image_path=image_path,
            gemini_raw_path=raw_path,
            gauge_reading=reading,
            local_plan=plan,
            canonical_prompt=canonical_prompt,
            openpi_response_metadata=response_metadata,
            action_path=action_path,
            timing=timing,
            stop_reason=stop_reason,
            outcome=outcome,
        )
        self.run_logger.write_step(record)
