from __future__ import annotations

from industrial_hazard_analysis.models import SceneSplitRecord, StructuredRecord, ValidationIssue


def validate_structured_records(records: list[StructuredRecord]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for record in records:
        issues.extend(
            [
                _issue(
                    "structured_extraction",
                    record.record_id,
                    record.raw_record_id,
                    "finding",
                    record.finding,
                    record.raw_text,
                ),
                _issue(
                    "structured_extraction",
                    record.record_id,
                    record.raw_record_id,
                    "scene",
                    record.scene,
                    record.raw_text,
                ),
            ]
        )
        for obj in record.object:
            issues.append(
                _issue(
                    "structured_extraction",
                    record.record_id,
                    record.raw_record_id,
                    "object",
                    obj,
                    record.finding,
                    fallback_text=record.raw_text,
                    expected_scope="finding",
                )
            )
    return issues


def validate_scene_split_records(records: list[SceneSplitRecord]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for record in records:
        if record.loc_detail:
            issues.append(
                _issue(
                    "scene_secondary_split",
                    record.record_id,
                    record.raw_record_id,
                    "loc_detail",
                    record.loc_detail,
                    record.scene,
                    fallback_text=record.raw_text,
                    expected_scope="scene",
                )
            )
        if record.risk_scene:
            issues.append(
                _issue(
                    "scene_secondary_split",
                    record.record_id,
                    record.raw_record_id,
                    "risk_scene",
                    record.risk_scene,
                    record.scene,
                    fallback_text=record.raw_text,
                    expected_scope="scene",
                )
            )
    return issues


def validation_summary(issues: list[ValidationIssue]) -> dict:
    total = len(issues)
    invalid = sum(not issue.is_valid for issue in issues)
    return {
        "checked_fields": total,
        "invalid_fields": invalid,
        "valid_fields": total - invalid,
        "invalid_ratio": invalid / total if total else 0.0,
    }


def _issue(
    stage: str,
    record_id: str,
    raw_record_id: str,
    field: str,
    value: str,
    source_text: str,
    *,
    fallback_text: str | None = None,
    expected_scope: str = "raw_text",
) -> ValidationIssue:
    value = str(value or "").strip()
    source_text = str(source_text or "")
    if not value:
        return ValidationIssue(
            stage,
            record_id,
            raw_record_id,
            field,
            value,
            source_text,
            False,
            "empty_value",
        )
    if value in source_text:
        return ValidationIssue(
            stage,
            record_id,
            raw_record_id,
            field,
            value,
            source_text,
            True,
            f"continuous_substring_of_{expected_scope}",
        )
    if fallback_text and value in fallback_text:
        return ValidationIssue(
            stage,
            record_id,
            raw_record_id,
            field,
            value,
            source_text,
            False,
            f"not_in_{expected_scope}_but_found_in_raw_text",
        )
    return ValidationIssue(
        stage,
        record_id,
        raw_record_id,
        field,
        value,
        source_text,
        False,
        f"not_continuous_substring_of_{expected_scope}",
    )
