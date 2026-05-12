from __future__ import annotations

from .canonical_prompts import CanonicalPromptCompiler
from .config import LunarPressureConfig
from .schemas import ContactState, GaugeReading, LocalPlan


class LocalPressurePlanner:
    """Deterministic pressure-error planner.

    Gemini provides only a visual gauge reading; this planner owns the decision
    from pressure error to canonical OpenPI prompt.
    """

    def __init__(self, config: LunarPressureConfig):
        self.config = config
        self.prompts = CanonicalPromptCompiler(config.canonical_prompts)

    def plan(self, reading: GaugeReading, contact_state: ContactState) -> LocalPlan:
        error = self.config.target_pressure_mpa - reading.value_mpa
        if abs(error) <= self.config.tolerance_mpa:
            return LocalPlan(
                plan_kind="stop",
                error_mpa=error,
                direction=None,
                canonical_prompt=None,
                contact_state=contact_state,
                reason="target_reached",
            )

        if error > 0:
            plan_kind = "increase_pressure"
            direction = self.config.increase_pressure_direction
        else:
            plan_kind = "decrease_pressure"
            direction = self.config.decrease_pressure_direction

        return LocalPlan(
            plan_kind=plan_kind,
            error_mpa=error,
            direction=direction,
            canonical_prompt=self.prompts.compile(plan_kind, contact_state),
            contact_state=contact_state,
            reason="pressure_below_target" if error > 0 else "pressure_above_target",
        )

    def validate_reading_bounds(self, reading: GaugeReading) -> None:
        if reading.value_mpa < self.config.visual_hard_min_mpa:
            raise ValueError(
                f"Gauge reading {reading.value_mpa} MPa is below hard visual minimum "
                f"{self.config.visual_hard_min_mpa} MPa"
            )
        if reading.value_mpa > self.config.visual_hard_max_mpa:
            raise ValueError(
                f"Gauge reading {reading.value_mpa} MPa is above hard visual maximum "
                f"{self.config.visual_hard_max_mpa} MPa"
            )

