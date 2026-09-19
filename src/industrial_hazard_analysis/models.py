from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


NOISE_CLUSTER_ID = -1
MISSING_CHANNEL_CLUSTER_ID = -2


@dataclass
class RawHazardText:
    record_id: str
    raw_text: str


@dataclass
class ExtractionCandidate:
    source_model: str
    raw_record_id: str
    finding: str
    object: list[str]
    scene: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredRecord:
    record_id: str
    raw_record_id: str
    raw_text: str
    finding: str
    object: list[str]
    scene: str
    accepted_by: str
    arbitration_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneSplitCandidate:
    source_model: str
    structured_record_id: str
    loc_detail: str
    risk_scene: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneSplitRecord:
    record_id: str
    raw_record_id: str
    raw_text: str
    finding: str
    object: list[str]
    scene: str
    loc_detail: str
    risk_scene: str
    accepted_by: str
    arbitration_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelConfig:
    experiment_id: str
    fields: tuple[str, ...]
    name: str | None = None


@dataclass
class ChannelInput:
    experiment_id: str
    record_id: str
    input_text: str


@dataclass
class ClusterAssignment:
    experiment_id: str
    record_id: str
    cluster_id: int
    is_noise: bool
    distance_to_centroid: float | None = None
    representative_rank: int | None = None
    primary_cluster_id: int | None = None
    primary_is_noise: bool | None = None
    noise_recluster_id: int | None = None
    final_cluster_id: int | None = None
    cluster_level: str = "primary"


@dataclass
class ClusterName:
    experiment_id: str
    cluster_id: int
    cluster_name: str
    explanation: str
    sample_count: int


@dataclass
class ValidationIssue:
    stage: str
    record_id: str
    raw_record_id: str
    field: str
    value: str
    source_text: str
    is_valid: bool
    reason: str
