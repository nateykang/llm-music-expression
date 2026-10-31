"""Shared Part 2 run configuration, read by every judging leg.

The three base batches are the v3 corpus the original run judged. Extension
runs add their generation batches through the environment, so the same legs
(batch_anthropic / batch_openai / batch_openrouter / run_rt / run) judge the
new pieces with the old panel AND the old pieces with the new arms: every leg
derives its panel from the models present in BATCHES and skips any key already
in the content-keyed checkpoint, so only the missing cells are sent.

    PART2_EXTRA_BATCHES   comma list of docs/data batch dirs to add
    PART2_SKIP_JUDGES     comma list of arms to leave off the judge panel
    PART2_TAG             suffix for batch-leg manifests, so an extension run
                          never resumes the original run's (completed) manifests
"""
import os

BASE_BATCHES = [
    "20260819_201530__models_40_prompts_3",   # ABC arm (40 models)
    "20260819_203512__models_2_prompts_3",    # ABC top-up (grok-4.6 pair)
    "20260820_061907__models_42_prompts_3",   # codegen arm (42 models)
]


def _csv(var: str) -> list[str]:
    return [x.strip() for x in os.environ.get(var, "").split(",") if x.strip()]


BATCHES = BASE_BATCHES + [b for b in _csv("PART2_EXTRA_BATCHES") if b not in BASE_BATCHES]
SKIP_JUDGES = set(_csv("PART2_SKIP_JUDGES"))
TAG = os.environ.get("PART2_TAG", "")


def manifest_name(stem: str) -> str:
    """batch_anthropic_manifest -> batch_anthropic_manifest[_<tag>].json"""
    return f"{stem}_{TAG}.json" if TAG else f"{stem}.json"


def judges_from(panel) -> list[str]:
    return sorted(n for n in panel if n not in SKIP_JUDGES)
