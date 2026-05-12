from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .schemas import ActCompletionConfig, CanonicalPromptSet, ContactState, Direction, StopProtocolConfig


class LunarPressureConfig(BaseModel):
    line_id: str
    gauge_id: str
    valve_id: str
    target_pressure_mpa: float
    tolerance_mpa: float = Field(gt=0)
    visual_hard_min_mpa: float
    visual_hard_max_mpa: float
    pressure_unit: str = "MPa"

    increase_pressure_direction: Direction = "clockwise"
    decrease_pressure_direction: Direction = "counterclockwise"
    operator_camera_view_definition: str = ""
    initial_contact_state: ContactState = "from_observation_pose"

    use_tiny_action: bool = False
    canonical_prompts: CanonicalPromptSet
    act_completion: ActCompletionConfig
    stop_protocol: StopProtocolConfig
    hold_action_safety_confirmed: bool = False
    auto_mark_grasping_after_act: bool = False

    gemini_model: str
    gemini_confidence_threshold: float = Field(ge=0.0, le=1.0)
    gemini_max_retries_per_observe: int = Field(ge=1)
    gemini_temperature: float = Field(default=0.0, ge=0.0)
    gemini_primary_image: str
    gemini_fallback_image: str | None = None
    gemini_prompt_file: str
    image_preprocess_owner: str
    resize_policy: str
    forward_to_openpi: str

    lunarpressure_server_host: str
    lunarpressure_server_port: int
    openpi_backend_host: str
    openpi_backend_port: int
    robot_pc_ip: str
    gpu_workstation_ip: str

    run_dir: str = "runs"

    @property
    def task_label(self) -> str:
        return f"Keep {self.line_id} pressure at {self.target_pressure_mpa:g} MPa"


ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "LUNARPRESSURE_SERVER_HOST": ("lunarpressure_server_host", str),
    "LUNARPRESSURE_SERVER_PORT": ("lunarpressure_server_port", int),
    "OPENPI_BACKEND_HOST": ("openpi_backend_host", str),
    "OPENPI_BACKEND_PORT": ("openpi_backend_port", int),
    "GEMINI_MODEL": ("gemini_model", str),
    "GEMINI_CONFIDENCE_THRESHOLD": ("gemini_confidence_threshold", float),
    "GEMINI_MAX_RETRIES_PER_OBSERVE": ("gemini_max_retries_per_observe", int),
    "LUNARPRESSURE_RUN_DIR": ("run_dir", str),
}


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    for env_name, (field_name, caster) in ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value is not None and value != "":
            merged[field_name] = caster(value)
    return merged


def load_config(path: str | Path = "configs/lunar_pressure.yaml") -> LunarPressureConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return LunarPressureConfig.model_validate(_apply_env_overrides(data))


def dump_config_snapshot(config: LunarPressureConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

