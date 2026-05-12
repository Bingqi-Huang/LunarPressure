from __future__ import annotations

from .schemas import CanonicalPromptSet, ContactState, PlanKind


class CanonicalPromptCompiler:
    """Selects fixed dataset-aligned VLA prompts.

    This class intentionally has no language-model logic. It only selects from
    the configured prompt set.
    """

    def __init__(self, prompt_set: CanonicalPromptSet):
        self._prompt_set = prompt_set

    def compile(self, plan_kind: PlanKind, contact_state: ContactState) -> str:
        if plan_kind == "stop":
            raise ValueError(
                "stop_data_label is a dataset label, not a runtime OpenPI prompt. "
                "Stop must be handled via the hold-action protocol, not as a VLA prompt."
            )
        return self._prompt_set.select(contact_state, plan_kind)

