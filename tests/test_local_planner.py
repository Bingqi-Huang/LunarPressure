from lunar_pressure.config import load_config
from lunar_pressure.local_planner import LocalPressurePlanner
from lunar_pressure.schemas import GaugeReading


def reading(value):
    return GaugeReading(
        line_id="Line-A",
        gauge_id="G1",
        value_mpa=value,
        confidence=0.95,
        raw_text="ok",
        need_retry=False,
        risk_flags=[],
    )


def test_pressure_below_target_maps_to_clockwise_increase_prompt():
    cfg = load_config("configs/lunar_pressure.yaml")
    planner = LocalPressurePlanner(cfg)

    plan = planner.plan(reading(0.0), "from_observation_pose")

    assert plan.plan_kind == "increase_pressure"
    assert plan.direction == "clockwise"
    assert plan.canonical_prompt == cfg.canonical_prompts.from_observation_pose.increase_pressure


def test_pressure_above_target_maps_to_counterclockwise_decrease_prompt():
    cfg = load_config("configs/lunar_pressure.yaml")
    planner = LocalPressurePlanner(cfg)

    plan = planner.plan(reading(0.2), "already_grasping_valve")

    assert plan.plan_kind == "decrease_pressure"
    assert plan.direction == "counterclockwise"
    assert plan.canonical_prompt == cfg.canonical_prompts.already_grasping_valve.decrease_pressure


def test_pressure_inside_tolerance_stops_without_vla_prompt():
    planner = LocalPressurePlanner(load_config("configs/lunar_pressure.yaml"))

    plan = planner.plan(reading(0.12), "already_grasping_valve")

    assert plan.plan_kind == "stop"
    assert plan.canonical_prompt is None

