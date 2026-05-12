from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class ObservationContractError(ValueError):
    """Raised when a required RoboCOIN observation field is unavailable."""


def slash_key(key: str) -> str:
    if key.startswith("observation."):
        return key.replace("observation.", "observation/", 1)
    return key


def dot_key(key: str) -> str:
    if key.startswith("observation/"):
        return key.replace("observation/", "observation.", 1)
    return key


def key_aliases(key: str) -> tuple[str, ...]:
    aliases = [key, slash_key(key), dot_key(key)]
    if key.startswith("observation.") and not key.startswith("observation.images."):
        suffix = key.removeprefix("observation.")
        aliases.append(f"observation.images.{suffix}")
        aliases.append(f"observation/images/{suffix}")
    if key.startswith("observation/") and not key.startswith("observation/images/"):
        suffix = key.removeprefix("observation/")
        aliases.append(f"observation.images.{suffix}")
        aliases.append(f"observation/images/{suffix}")
    seen: list[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.append(alias)
    return tuple(seen)


def resolve_observation_key(observation: Mapping[str, Any], key: str) -> str | None:
    for alias in key_aliases(key):
        if alias in observation:
            return alias
    return None


def get_observation_value(observation: Mapping[str, Any], key: str, *, required: bool = True) -> Any:
    alias = resolve_observation_key(observation, key)
    if alias is not None:
        return observation[alias]
    if required:
        raise ObservationContractError(f"Missing observation key {key!r}; tried {key_aliases(key)}")
    return None


def get_image(
    observation: Mapping[str, Any],
    primary_key: str,
    fallback_key: str | None = None,
) -> tuple[str, Any]:
    alias = resolve_observation_key(observation, primary_key)
    if alias is not None:
        return alias, observation[alias]
    if fallback_key:
        alias = resolve_observation_key(observation, fallback_key)
        if alias is not None:
            return alias, observation[alias]
    raise ObservationContractError(f"Missing image keys: primary={primary_key!r}, fallback={fallback_key!r}")


def replace_prompt(observation: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    forwarded = dict(observation)
    forwarded["prompt"] = prompt
    return forwarded


def get_high_level_task(observation: Mapping[str, Any]) -> str | None:
    value = observation.get("prompt")
    return str(value) if value is not None else None


def get_state_vector(observation: Mapping[str, Any]) -> np.ndarray:
    state = get_observation_value(observation, "observation.state", required=False)
    if state is not None:
        return np.asarray(state, dtype=np.float32).reshape(-1)

    joint_position = get_observation_value(observation, "observation.joint_position", required=False)
    gripper_position = get_observation_value(observation, "observation.gripper_position", required=False)
    if joint_position is None or gripper_position is None:
        raise ObservationContractError(
            "Missing hold-state fields; expected observation.state or observation.joint_position + "
            "observation.gripper_position"
        )
    joint = np.asarray(joint_position, dtype=np.float32).reshape(-1)
    gripper = np.asarray(gripper_position, dtype=np.float32).reshape(-1)
    return np.concatenate([joint, gripper])


def build_hold_action(observation: Mapping[str, Any]) -> np.ndarray:
    state = get_state_vector(observation)
    if state.size == 0:
        raise ObservationContractError("Cannot build hold action from an empty state vector")
    return state.astype(np.float32, copy=True)


def build_hold_response(observation: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    action = build_hold_action(observation)
    if np.allclose(action, 0.0):
        raise ObservationContractError(
            "Refusing to return an all-zero hold action. Confirm action semantics or provide real robot state."
        )
    return {
        "action": action,
        "lunar_control": {
            "stop": False,
            "reason": reason,
            "mode": "hold_send_action",
        },
    }


def observation_metadata(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value_metadata(value) for key, value in observation.items()}


def value_metadata(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (bytes, bytearray)):
        return {"type": type(value).__name__, "bytes": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": list(value.keys())}
    return {"type": type(value).__name__}


@dataclass(frozen=True)
class ActionPayload:
    actions: np.ndarray
    source_field: str


def extract_action_payload(response: Any) -> ActionPayload:
    if isinstance(response, Mapping):
        if "action" in response:
            raw = response["action"]
            source = "action"
        elif "actions" in response:
            raw = response["actions"]
            source = "actions"
        elif isinstance(response.get("output"), Mapping):
            output = response["output"]
            if "action" in output:
                raw = output["action"]
                source = "output.action"
            elif "actions" in output:
                raw = output["actions"]
                source = "output.actions"
            else:
                raise ObservationContractError(f"Missing action/actions fields in output: {list(output.keys())}")
        else:
            raise ObservationContractError(f"Missing action/actions fields in response: {list(response.keys())}")
    else:
        raw = response
        source = "raw"

    actions = np.asarray(raw, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ObservationContractError(f"Expected action array rank 1 or 2, got shape {actions.shape}")
    return ActionPayload(actions=actions, source_field=source)

