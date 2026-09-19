from __future__ import annotations

from collections import Counter
from typing import Callable, TypeVar

from industrial_hazard_analysis.models import ExtractionCandidate, SceneSplitCandidate

T = TypeVar("T")


def object_key(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in values))


def vote_extraction(
    candidates: list[ExtractionCandidate],
    arbitrator: Callable[[list[ExtractionCandidate]], list[ExtractionCandidate]],
) -> tuple[list[ExtractionCandidate], str, bool]:
    if not candidates:
        raise ValueError("vote_extraction requires at least one candidate")
    candidate_keys = {
        (candidate.finding, object_key(candidate.object), candidate.scene)
        for candidate in candidates
    }
    key_counts = Counter(
        {
            key: len(
                {
                    candidate.source_model
                    for candidate in candidates
                    if (candidate.finding, object_key(candidate.object), candidate.scene) == key
                }
            )
            for key in candidate_keys
        }
    )
    winner, count = key_counts.most_common(1)[0]
    if count >= 2:
        selected = next(
            candidate
            for candidate in candidates
            if (candidate.finding, object_key(candidate.object), candidate.scene) == winner
        )
        return [selected], "majority_vote", False
    arbitrated = list(arbitrator(candidates))
    if not arbitrated:
        return [], "arbitration_empty", True
    return arbitrated, "arbitration", True


def vote_scene_split(
    candidates: list[SceneSplitCandidate],
    arbitrator: Callable[[list[SceneSplitCandidate]], SceneSplitCandidate],
) -> tuple[SceneSplitCandidate, str, bool]:
    if not candidates:
        raise ValueError("vote_scene_split requires at least one candidate")
    candidate_keys = {(candidate.loc_detail, candidate.risk_scene) for candidate in candidates}
    key_counts = Counter(
        {
            key: len(
                {
                    candidate.source_model
                    for candidate in candidates
                    if (candidate.loc_detail, candidate.risk_scene) == key
                }
            )
            for key in candidate_keys
        }
    )
    winner, count = key_counts.most_common(1)[0]
    if count >= 2:
        selected = next(
            candidate
            for candidate in candidates
            if (candidate.loc_detail, candidate.risk_scene) == winner
        )
        return selected, "majority_vote", False
    return arbitrator(candidates), "arbitration", True
