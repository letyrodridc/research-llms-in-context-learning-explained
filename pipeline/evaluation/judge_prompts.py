from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..experiments.config import JUDGEABLE_PROMPT_TYPES
from ..utils.prompt_assets import build_asset_snapshot, require_assignment_blocks


@dataclass(frozen=True)
class JudgePromptSpec:
    prompt_type: str
    system_prompt: str
    condition_description: str
    max_tokens: int


JUDGE_PROMPT_ASSET_FILENAME = "judge_prompts.txt"
JUDGE_PROMPT_ASSET_KEYS = (
    "JUDGE_PROMPT",
    "NLE_JUDGE_CONDITION_DESCRIPTION",
    "FEATURES_JUDGE_CONDITION_DESCRIPTION",
    "LOGIC_RULES_JUDGE_CONDITION_DESCRIPTION",
    "DL_AXIOMS_JUDGE_CONDITION_DESCRIPTION",
    "JUDGE_EXPLAIN_ADDON",
)
JUDGE_PROMPT_ASSETS = require_assignment_blocks(
    JUDGE_PROMPT_ASSET_FILENAME,
    JUDGE_PROMPT_ASSET_KEYS,
)

JUDGE_CONDITION_DESCRIPTIONS = {
    "nle": JUDGE_PROMPT_ASSETS["NLE_JUDGE_CONDITION_DESCRIPTION"],
    "features": JUDGE_PROMPT_ASSETS["FEATURES_JUDGE_CONDITION_DESCRIPTION"],
    "rulebased": JUDGE_PROMPT_ASSETS["LOGIC_RULES_JUDGE_CONDITION_DESCRIPTION"],
    "axioms_ontology_v2": JUDGE_PROMPT_ASSETS["DL_AXIOMS_JUDGE_CONDITION_DESCRIPTION"],
}


def build_judge_prompt_specs(explain_scores: bool = False) -> Dict[str, JudgePromptSpec]:
    max_tokens = 8192 if explain_scores else 4096
    addon = JUDGE_PROMPT_ASSETS["JUDGE_EXPLAIN_ADDON"] if explain_scores else ""
    return {
        prompt_type: JudgePromptSpec(
            prompt_type=prompt_type,
            condition_description=JUDGE_CONDITION_DESCRIPTIONS[prompt_type],
            system_prompt=JUDGE_PROMPT_ASSETS["JUDGE_PROMPT"].format(
                CONDITION_DESCRIPTION=JUDGE_CONDITION_DESCRIPTIONS[prompt_type]
            ) + addon,
            max_tokens=max_tokens,
        )
        for prompt_type in JUDGEABLE_PROMPT_TYPES
    }


JUDGE_PROMPT_SPECS = build_judge_prompt_specs(explain_scores=False)


def export_judge_prompt_library_snapshot(explain_scores: bool = False) -> Dict[str, Any]:
    specs = build_judge_prompt_specs(explain_scores=explain_scores)
    return {
        "source_asset": build_asset_snapshot(
            JUDGE_PROMPT_ASSET_FILENAME,
            JUDGE_PROMPT_ASSET_KEYS,
        ),
        "explain_scores": explain_scores,
        "prompt_types": {
            prompt_type: {
                "system_prompt": spec.system_prompt,
                "condition_description": spec.condition_description,
                "max_tokens": spec.max_tokens,
            }
            for prompt_type, spec in specs.items()
        },
    }
