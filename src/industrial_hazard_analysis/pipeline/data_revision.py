from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
from hashlib import sha1
from typing import Iterable, Sequence

from industrial_hazard_analysis.models import RawHazardText, SceneSplitRecord, StructuredRecord


@dataclass(frozen=True)
class CanonicalRawTextGroup:
    analysis_record_id: str
    dedupe_group_id: str
    canonical_raw_record_id: str
    raw_text: str
    normalized_raw_text: str
    source_raw_record_ids: tuple[str, ...]

    @property
    def duplicate_count(self) -> int:
        return len(self.source_raw_record_ids)


@dataclass(frozen=True)
class RevisedUnits:
    kept_records: tuple[SceneSplitRecord, ...]
    audit_rows: tuple[dict, ...]


def normalize_text(value: object) -> str:
    """Normalize only whitespace; do not perform semantic or punctuation matching."""

    return " ".join(str(value or "").strip().split())


def build_canonical_raw_text_groups(
    raw_records: Sequence[RawHazardText],
) -> tuple[list[CanonicalRawTextGroup], list[dict]]:
    """Group source rows by normalized raw text and retain the first source row."""

    grouped: OrderedDict[str, list[RawHazardText]] = OrderedDict()
    for record in raw_records:
        normalized = normalize_text(record.raw_text)
        if not normalized:
            continue
        grouped.setdefault(normalized, []).append(record)

    groups: list[CanonicalRawTextGroup] = []
    audit_rows: list[dict] = []
    for index, (normalized, records) in enumerate(grouped.items(), start=1):
        group = CanonicalRawTextGroup(
            analysis_record_id=f"U{index:05d}",
            dedupe_group_id=f"RAW-D{index:05d}",
            canonical_raw_record_id=records[0].record_id,
            raw_text=records[0].raw_text,
            normalized_raw_text=normalized,
            source_raw_record_ids=tuple(record.record_id for record in records),
        )
        groups.append(group)
        key_hash = sha1(normalized.encode("utf-8")).hexdigest()
        for duplicate_index, record in enumerate(records, start=1):
            audit_rows.append(
                {
                    "dedupe_group_id": group.dedupe_group_id,
                    "analysis_record_id": group.analysis_record_id,
                    "canonical_raw_record_id": group.canonical_raw_record_id,
                    "original_raw_record_id": record.record_id,
                    "is_canonical": record.record_id == group.canonical_raw_record_id,
                    "duplicate_index": duplicate_index,
                    "duplicate_count": group.duplicate_count,
                    "normalized_raw_text_sha1": key_hash,
                    "raw_text": record.raw_text,
                }
            )
    return groups, audit_rows


def filter_and_deduplicate_units(
    groups: Sequence[CanonicalRawTextGroup],
    records: Sequence[SceneSplitRecord],
    *,
    source_label: str,
) -> RevisedUnits:
    """Keep canonical source rows, then remove exact raw_text+finding duplicates."""

    group_by_raw_id = {group.canonical_raw_record_id: group for group in groups}
    canonical_ids = set(group_by_raw_id)
    positions = {record.record_id: index for index, record in enumerate(records)}
    candidate_groups: OrderedDict[tuple[str, str], list[SceneSplitRecord]] = OrderedDict()
    empty_placeholder_ids: set[str] = set()
    for record in records:
        if record.raw_record_id not in canonical_ids:
            continue
        if not is_effective_unit(record):
            empty_placeholder_ids.add(record.record_id)
            continue
        key = (normalize_text(record.raw_text), normalize_text(record.finding))
        candidate_groups.setdefault(key, []).append(record)

    kept_by_original_id: dict[str, str] = {}
    kept_records: list[SceneSplitRecord] = []
    for unit_group_index, records_in_group in enumerate(candidate_groups.values(), start=1):
        kept = max(records_in_group, key=lambda item: _representative_score(item, positions))
        group = group_by_raw_id[kept.raw_record_id]
        unit_metadata = {
            "source_label": source_label,
            "unit_dedupe_group_id": f"UNIT-D{unit_group_index:05d}",
            "dedupe_key_fields": ["raw_text", "finding"],
            "duplicate_count": len(records_in_group),
            "kept_record_id": kept.record_id,
            "source_record_ids": [item.record_id for item in records_in_group],
            "analysis_record_id": group.analysis_record_id,
            "canonical_raw_record_id": group.canonical_raw_record_id,
        }
        metadata = dict(kept.metadata)
        metadata["data_revision"] = unit_metadata
        kept_records.append(replace(kept, metadata=metadata))
        for item in records_in_group:
            kept_by_original_id[item.record_id] = kept.record_id

    audit_rows: list[dict] = []
    for record in records:
        group = group_by_raw_id.get(record.raw_record_id)
        if group is None:
            audit_rows.append(
                {
                    "source_label": source_label,
                    "original_record_id": record.record_id,
                    "raw_record_id": record.raw_record_id,
                    "analysis_record_id": "",
                    "is_canonical_source": False,
                    "is_kept_unit": False,
                    "kept_record_id": "",
                    "exclusion_reason": "duplicate_source_raw_text",
                    "raw_text": record.raw_text,
                    "finding": record.finding,
                }
            )
            continue
        if record.record_id in empty_placeholder_ids:
            audit_rows.append(
                {
                    "source_label": source_label,
                    "original_record_id": record.record_id,
                    "raw_record_id": record.raw_record_id,
                    "analysis_record_id": group.analysis_record_id,
                    "is_canonical_source": True,
                    "is_kept_unit": False,
                    "kept_record_id": "",
                    "exclusion_reason": "empty_structured_placeholder",
                    "raw_text": record.raw_text,
                    "finding": record.finding,
                }
            )
            continue
        kept_record_id = kept_by_original_id[record.record_id]
        is_kept = record.record_id == kept_record_id
        audit_rows.append(
            {
                "source_label": source_label,
                "original_record_id": record.record_id,
                "raw_record_id": record.raw_record_id,
                "analysis_record_id": group.analysis_record_id,
                "is_canonical_source": True,
                "is_kept_unit": is_kept,
                "kept_record_id": kept_record_id,
                "exclusion_reason": "" if is_kept else "duplicate_raw_text_finding_within_canonical",
                "raw_text": record.raw_text,
                "finding": record.finding,
            }
        )

    return RevisedUnits(tuple(kept_records), tuple(audit_rows))


def consolidate_units_by_raw_text(
    groups: Sequence[CanonicalRawTextGroup],
    units: Sequence[SceneSplitRecord],
    *,
    source_label: str,
) -> list[SceneSplitRecord]:
    """Create one downstream analysis row per canonical raw description."""

    units_by_raw_id: dict[str, list[SceneSplitRecord]] = defaultdict(list)
    for unit in units:
        units_by_raw_id[unit.raw_record_id].append(unit)

    consolidated: list[SceneSplitRecord] = []
    for group in groups:
        source_units = units_by_raw_id.get(group.canonical_raw_record_id, [])
        findings = _ordered_unique(unit.finding for unit in source_units)
        objects = _ordered_unique(item for unit in source_units for item in unit.object)
        scenes = _ordered_unique(unit.scene for unit in source_units)
        loc_details = _ordered_unique(unit.loc_detail for unit in source_units)
        risk_scenes = _ordered_unique(unit.risk_scene for unit in source_units)
        consolidated.append(
            SceneSplitRecord(
                record_id=group.analysis_record_id,
                raw_record_id=group.canonical_raw_record_id,
                raw_text=group.raw_text,
                finding="；".join(findings),
                object=objects,
                scene="；".join(scenes),
                loc_detail="；".join(loc_details),
                risk_scene="；".join(risk_scenes),
                accepted_by=f"data_revision:{source_label}",
                arbitration_required=any(unit.arbitration_required for unit in source_units),
                metadata={
                    "data_revision": {
                        "source_label": source_label,
                        "analysis_grain": "unique_normalized_raw_text",
                        "analysis_record_id": group.analysis_record_id,
                        "canonical_raw_record_id": group.canonical_raw_record_id,
                        "source_raw_record_ids": list(group.source_raw_record_ids),
                        "source_duplicate_count": group.duplicate_count,
                        "source_unit_count": len(source_units),
                        "source_unit_record_ids": [unit.record_id for unit in source_units],
                        "aggregation_rule": "ordered_unique_values_joined_by_semicolon",
                    }
                },
            )
        )
    return consolidated


def to_structured_records(records: Iterable[SceneSplitRecord]) -> list[StructuredRecord]:
    return [
        StructuredRecord(
            record_id=record.record_id,
            raw_record_id=record.raw_record_id,
            raw_text=record.raw_text,
            finding=record.finding,
            object=record.object,
            scene=record.scene,
            accepted_by=record.accepted_by,
            arbitration_required=record.arbitration_required,
            metadata=record.metadata,
        )
        for record in records
    ]


def canonical_group_rows(groups: Sequence[CanonicalRawTextGroup]) -> list[dict]:
    return [
        {
            "analysis_record_id": group.analysis_record_id,
            "dedupe_group_id": group.dedupe_group_id,
            "canonical_raw_record_id": group.canonical_raw_record_id,
            "raw_text": group.raw_text,
            "normalized_raw_text_sha1": sha1(group.normalized_raw_text.encode("utf-8")).hexdigest(),
            "duplicate_count": group.duplicate_count,
            "source_raw_record_ids": list(group.source_raw_record_ids),
        }
        for group in groups
    ]


def _ordered_unique(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def is_effective_unit(record: SceneSplitRecord) -> bool:
    """Return whether a structured row contains any usable semantic content."""

    return bool(
        normalize_text(record.finding)
        or record.object
        or normalize_text(record.scene)
        or normalize_text(record.loc_detail)
        or normalize_text(record.risk_scene)
    )


def _representative_score(record: SceneSplitRecord, positions: dict[str, int]) -> tuple:
    semantic_field_count = sum(
        (
            bool(record.object),
            bool(normalize_text(record.scene)),
            bool(normalize_text(record.loc_detail)),
            bool(normalize_text(record.risk_scene)),
        )
    )
    return (
        semantic_field_count,
        bool(normalize_text(record.risk_scene)),
        bool(normalize_text(record.scene)),
        bool(record.object),
        bool(normalize_text(record.loc_detail)),
        len(record.object),
        record.accepted_by != "no_extraction_candidates",
        -positions.get(record.record_id, 10**9),
    )
