from lunar_pressure.config import load_config


def test_default_config_matches_closed_architecture():
    cfg = load_config("configs/lunar_pressure.yaml")

    assert cfg.line_id == "Line-A"
    assert cfg.gauge_id == "G1"
    assert cfg.valve_id == "V1"
    assert cfg.target_pressure_mpa == 0.1
    assert cfg.tolerance_mpa == 0.05
    assert cfg.visual_hard_min_mpa == 0.0
    assert cfg.visual_hard_max_mpa == 0.2
    assert cfg.increase_pressure_direction == "clockwise"
    assert cfg.decrease_pressure_direction == "counterclockwise"
    assert cfg.use_tiny_action is False
    assert cfg.stop_protocol.stop_response_mode == "hold_send_action"
    assert cfg.forward_to_openpi == "保持原始 observation，仅替换 prompt"

