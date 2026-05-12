import numpy as np
import pytest

from lunar_pressure.observation_contract import (
    ObservationContractError,
    build_hold_response,
    extract_action_payload,
    get_image,
    get_state_vector,
    key_aliases,
    replace_prompt,
    resolve_observation_key,
)


def test_dot_slash_image_fallback():
    image = np.ones((2, 2, 3), dtype=np.uint8)
    obs = {"observation/scene_image": image}

    key, value = get_image(obs, "observation.scene_image")

    # get_image returns the RESOLVED alias (the key actually present in obs)
    assert key == "observation/scene_image"
    assert value is image


def test_state_from_split_joint_fields():
    obs = {
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([9], dtype=np.float32),
    }

    state = get_state_vector(obs)

    np.testing.assert_allclose(state, np.array([0, 1, 2, 3, 4, 5, 6, 9], dtype=np.float32))


def test_replace_prompt_keeps_original_observation():
    obs = {"prompt": "high level", "observation.state": np.ones(8)}
    forwarded = replace_prompt(obs, "canonical")

    assert forwarded["prompt"] == "canonical"
    assert obs["prompt"] == "high level"
    assert forwarded["observation.state"] is obs["observation.state"]


def test_hold_response_uses_state_and_rejects_all_zero():
    obs = {"observation.state": np.arange(1, 9, dtype=np.float32)}

    response = build_hold_response(obs, reason="target_reached")

    np.testing.assert_allclose(response["action"], obs["observation.state"])
    assert response["lunar_control"]["mode"] == "hold_send_action"

    with pytest.raises(ObservationContractError):
        build_hold_response({"observation.state": np.zeros(8, dtype=np.float32)}, reason="unsafe_zero")


def test_extract_action_payload_accepts_action_and_actions():
    one = extract_action_payload({"action": np.arange(8)})
    many = extract_action_payload({"actions": np.ones((2, 8))})
    nested = extract_action_payload({"output": {"actions": np.ones((3, 8))}})

    assert one.actions.shape == (1, 8)
    assert many.actions.shape == (2, 8)
    assert nested.actions.shape == (3, 8)


def test_key_aliases_no_malformed_images_form():
    """key_aliases must emit observation.images.X and observation/images/X but NOT observation/images.X."""
    aliases = key_aliases("observation.scene_image")
    assert "observation.images.scene_image" in aliases
    assert "observation/images/scene_image" in aliases
    # The old malformed form must NOT be present
    assert "observation/images.scene_image" not in aliases


def test_resolve_observation_key_returns_matched_alias():
    """resolve_observation_key returns the alias actually present in the observation."""
    obs = {"observation/scene_image": "img_data"}
    matched = resolve_observation_key(obs, "observation.scene_image")
    # The match is the slash form because that's what the dict contains
    assert matched == "observation/scene_image"

