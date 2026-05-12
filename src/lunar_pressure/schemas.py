from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Direction = Literal["clockwise", "counterclockwise"]
PlanKind = Literal["increase_pressure", "decrease_pressure", "stop"]
ContactState = Literal["from_observation_pose", "already_grasping_valve"]


class OrchestratorState(str, Enum):
    OBSERVE = "OBSERVE"
    PLAN = "PLAN"
    ACT = "ACT"
    STOP = "STOP"
    ERROR = "ERROR"


class PromptPair(BaseModel):
    increase_pressure: str
    decrease_pressure: str


class CanonicalPromptSet(BaseModel):
    from_observation_pose: PromptPair
    already_grasping_valve: PromptPair
    stop_data_label: str

    def select(self, contact_state: ContactState, plan_kind: PlanKind) -> str:
        if plan_kind == "stop":
            return self.stop_data_label
        group = getattr(self, contact_state)
        return getattr(group, plan_kind)


class ActCompletionConfig(BaseModel):
    max_policy_calls_per_act: int = Field(ge=1)
    max_act_seconds: float = Field(gt=0)
    use_joint_residual_check: bool = True
    use_action_delta_check: bool = True
    joint_residual_threshold: float | None = Field(default=None, gt=0)
    action_delta_threshold: float | None = Field(default=None, gt=0)


class StopProtocolConfig(BaseModel):
    patch_robot_client_for_lunar_control_stop: bool = False
    stop_response_mode: Literal["hold_send_action"] = "hold_send_action"
    stop_behavior: Literal["hold"] = "hold"


class PressureTask(BaseModel):
    line_id: str
    gauge_id: str
    valve_id: str
    target_pressure_mpa: float
    tolerance_mpa: float = Field(gt=0)
    visual_hard_min_mpa: float
    visual_hard_max_mpa: float
    pressure_unit: Literal["MPa"] = "MPa"

    @field_validator("visual_hard_max_mpa")
    @classmethod
    def max_must_exceed_min(cls, value: float, info: Any) -> float:
        min_value = info.data.get("visual_hard_min_mpa")
        if min_value is not None and value <= min_value:
            raise ValueError("visual_hard_max_mpa must exceed visual_hard_min_mpa")
        return value


class GaugeReading(BaseModel):
    line_id: str
    gauge_id: str
    value_mpa: float
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str = ""
    need_retry: bool = False
    risk_flags: list[str] = Field(default_factory=list)

    def is_usable(self, confidence_threshold: float) -> bool:
        return not self.need_retry and self.confidence >= confidence_threshold


class LocalPlan(BaseModel):
    plan_kind: PlanKind
    error_mpa: float
    direction: Direction | None = None
    canonical_prompt: str | None = None
    contact_state: ContactState
    reason: str

    @property
    def should_act(self) -> bool:
        return self.plan_kind in ("increase_pressure", "decrease_pressure")

    @model_validator(mode="after")
    def _direction_required_for_action(self) -> "LocalPlan":
        if self.plan_kind != "stop":
            if self.direction is None or self.canonical_prompt is None:
                raise ValueError(
                    "LocalPlan with plan_kind != 'stop' must set both direction and canonical_prompt"
                )
        return self


class LunarControl(BaseModel):
    stop: bool = False
    reason: str = ""
    mode: str = "hold_send_action"


class OpenPIActionResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: Any | None = None
    actions: Any | None = None
    server_timing: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class StepLogRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: str
    state: OrchestratorState
    high_level_task: str | None = None
    observation_keys: list[str] = Field(default_factory=list)
    observation_metadata: dict[str, Any] = Field(default_factory=dict)
    image_path: str | None = None
    gemini_raw_path: str | None = None
    gauge_reading: GaugeReading | None = None
    local_plan: LocalPlan | None = None
    canonical_prompt: str | None = None
    openpi_response_metadata: dict[str, Any] = Field(default_factory=dict)
    action_path: str | None = None
    timing: dict[str, float] = Field(default_factory=dict)
    stop_reason: str | None = None
    outcome: str | None = None


class RunSummary(BaseModel):
    run_name: str
    state: OrchestratorState
    total_steps: int
    final_reason: str | None = None
    start_time: str | None = None
    end_time: str | None = None

