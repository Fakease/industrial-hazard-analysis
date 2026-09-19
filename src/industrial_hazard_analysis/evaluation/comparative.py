from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import (
    MISSING_CHANNEL_CLUSTER_ID,
    NOISE_CLUSTER_ID,
    char_f1,
    evaluate_extraction,
    is_effective_extraction_record,
    match_extraction_units,
    normalize_extraction_record,
    object_set_f1,
)


def compare_extraction_systems(
    systems: Mapping[str, Sequence[Any]],
    gold_records: Sequence[Any],
) -> dict[str, Any]:
    """Compare extraction systems with one matching rule and one gold set.

    The returned common-match table is restricted to gold units matched by
    every system. No raw text or row-level match content is returned.
    """

    if not systems:
        raise ValueError("At least one extraction system is required")
    gold = [normalize_extraction_record(row) for row in gold_records]
    _validate_gold_unit_keys(gold)
    gold_raw_ids = {row["raw_record_id"] for row in gold}

    summary_rows: list[dict[str, Any]] = []
    matches_by_system: dict[str, list[dict[str, Any]]] = {}
    matched_keys_by_system: dict[str, set[tuple[str, str]]] = {}

    for system_name, rows in systems.items():
        predictions = [
            normalized
            for row in rows
            if is_effective_extraction_record(
                normalized := normalize_extraction_record(row)
            )
        ]
        predictions_in_scope = [
            row for row in predictions if row.get("raw_record_id") in gold_raw_ids
        ]
        metrics = evaluate_extraction(predictions_in_scope, gold)
        matches, _, _ = match_extraction_units(predictions_in_scope, gold)
        matches_by_system[system_name] = matches
        matched_keys_by_system[system_name] = {
            _gold_unit_key(match["gold"]) for match in matches
        }
        summary_rows.append(
            {
                "系统": system_name,
                "预测单元": metrics["prediction_unit_count_in_scope"],
                "完整命中": metrics["complete_hit_count"],
                "Precision": metrics["unit_precision"],
                "Recall": metrics["unit_recall"],
                "Unit F1": metrics["unit_f1"],
                "Finding char-F1": metrics["finding_char_f1"],
                "Object set-F1": metrics["object_set_f1"],
                "Scene char-F1": metrics["scene_char_f1"],
            }
        )

    common_keys = set.intersection(*matched_keys_by_system.values())
    common_rows: list[dict[str, Any]] = []
    for system_name in systems:
        common_matches = [
            match
            for match in matches_by_system[system_name]
            if _gold_unit_key(match["gold"]) in common_keys
        ]
        fields = summarize_matched_fields(common_matches)
        common_rows.append(
            {
                "系统": system_name,
                "共同对应人工单元": len(common_matches),
                "Finding char-F1": fields["finding_char_f1"],
                "Object set-F1": fields["object_set_f1"],
                "Scene char-F1": fields["scene_char_f1"],
            }
        )

    return {
        "system_summary": summary_rows,
        "common_matched_summary": common_rows,
        "common_matched_gold_unit_count": len(common_keys),
        "_matches_by_system": matches_by_system,
    }


def summarize_matched_fields(matches: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    finding_scores: list[float] = []
    object_scores: list[float] = []
    scene_scores: list[float] = []
    for match in matches:
        prediction = match["prediction"]
        gold = match["gold"]
        predicted_finding = str(prediction.get("finding") or "").strip()
        gold_finding = str(gold.get("finding") or "").strip()
        if predicted_finding or gold_finding:
            finding_scores.append(char_f1(predicted_finding, gold_finding))
        predicted_objects = set(prediction.get("object") or [])
        gold_objects = set(gold.get("object") or [])
        if predicted_objects or gold_objects:
            object_scores.append(object_set_f1(predicted_objects, gold_objects))
        predicted_scene = str(prediction.get("scene") or "").strip()
        gold_scene = str(gold.get("scene") or "").strip()
        if predicted_scene or gold_scene:
            scene_scores.append(char_f1(predicted_scene, gold_scene))
    return {
        "finding_char_f1": float(np.mean(finding_scores)) if finding_scores else 0.0,
        "object_set_f1": float(np.mean(object_scores)) if object_scores else 0.0,
        "scene_char_f1": float(np.mean(scene_scores)) if scene_scores else 0.0,
    }


def summarize_prediction_sources(
    predictions: Sequence[Mapping[str, Any]],
    gold_records: Sequence[Any],
    *,
    source_column: str,
    source_labels: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate each Multi-LLM result source against the same gold standard."""

    source_labels = source_labels or {}
    gold_raw_ids = {
        normalize_extraction_record(row)["raw_record_id"] for row in gold_records
    }
    source_order: list[str] = []
    predictions_by_source: dict[str, list[dict[str, Any]]] = {}

    for source in predictions:
        normalized = normalize_extraction_record(source)
        if not is_effective_extraction_record(normalized):
            continue
        if normalized["raw_record_id"] not in gold_raw_ids:
            continue
        raw_source_value = source.get(source_column)
        source_value = (
            ""
            if raw_source_value is None or pd.isna(raw_source_value)
            else str(raw_source_value).strip()
        )
        if not source_value:
            raise ValueError(f"Missing {source_column!r} for an in-scope prediction")
        _prediction_unit_key(normalized)
        if source_value not in predictions_by_source:
            source_order.append(source_value)
            predictions_by_source[source_value] = []
        predictions_by_source[source_value].append(normalized)

    rows = []
    for source_value in source_order:
        metrics = evaluate_extraction(predictions_by_source[source_value], gold_records)
        rows.append(
            {
                "结果来源": source_labels.get(source_value, source_value),
                "预测单元": metrics["prediction_unit_count_in_scope"],
                "完整命中": metrics["complete_hit_count"],
                "Precision": metrics["unit_precision"],
                "Recall": metrics["unit_recall"],
                "Unit F1": metrics["unit_f1"],
                "Finding char-F1": metrics["finding_char_f1"],
                "Object set-F1": metrics["object_set_f1"],
                "Scene char-F1": metrics["scene_char_f1"],
            }
        )
    return rows


def cross_channel_correspondence(
    baseline: pd.DataFrame,
    target: pd.DataFrame,
    *,
    id_column: str = "raw_record_id",
    baseline_cluster_column: str = "cluster_id",
    target_cluster_column: str = "cluster_id",
    noise_cluster_id: int = NOISE_CLUSTER_ID,
    missing_cluster_id: int = MISSING_CHANNEL_CLUSTER_ID,
) -> dict[str, Any]:
    """Compare independent cluster assignments on common non-noise records."""

    _validate_assignment_frame(baseline, id_column, baseline_cluster_column, "baseline")
    _validate_assignment_frame(target, id_column, target_cluster_column, "target")
    left = baseline[[id_column, baseline_cluster_column]].rename(
        columns={baseline_cluster_column: "baseline_cluster"}
    )
    right = target[[id_column, target_cluster_column]].rename(
        columns={target_cluster_column: "target_cluster"}
    )
    merged = left.merge(right, on=id_column, how="inner", validate="one_to_one")
    clean = _common_non_noise_assignments(
        merged,
        noise_cluster_id=noise_cluster_id,
        missing_cluster_id=missing_cluster_id,
    )

    return _correspondence_metrics(clean)


def parent_inherited_cluster_correspondence(
    baseline: pd.DataFrame,
    target_units: pd.DataFrame,
    *,
    parent_id_column: str = "raw_record_id",
    target_unit_id_column: str = "record_id",
    baseline_cluster_column: str = "cluster_id",
    target_cluster_column: str = "cluster_id",
    noise_cluster_id: int = NOISE_CLUSTER_ID,
    missing_cluster_id: int = MISSING_CHANNEL_CLUSTER_ID,
) -> dict[str, Any]:
    """Compare record-level baseline labels with unit-level target labels.

    Each target unit inherits the cluster label of its parent baseline record.
    AMI and related statistics are therefore calculated over hazard units, while
    the baseline itself remains clustered at the original record grain.
    """

    _validate_assignment_frame(
        baseline,
        parent_id_column,
        baseline_cluster_column,
        "baseline parent",
    )
    _validate_assignment_frame(
        target_units,
        target_unit_id_column,
        target_cluster_column,
        "target unit",
    )
    if parent_id_column not in target_units:
        raise ValueError(
            f"target unit assignments are missing columns: [{parent_id_column!r}]"
        )
    if (
        target_units[parent_id_column].isna().any()
        or target_units[parent_id_column].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"target unit assignments contain blank {parent_id_column}")

    left = baseline[[parent_id_column, baseline_cluster_column]].rename(
        columns={baseline_cluster_column: "baseline_cluster"}
    )
    right = target_units[
        [target_unit_id_column, parent_id_column, target_cluster_column]
    ].rename(columns={target_cluster_column: "target_cluster"})
    merged = right.merge(
        left,
        on=parent_id_column,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmapped = merged[merged["_merge"] != "both"]
    if not unmapped.empty:
        raise ValueError(
            f"{len(unmapped)} target unit(s) have no parent baseline assignment"
        )
    merged = merged.drop(columns="_merge")
    clean = _common_non_noise_assignments(
        merged,
        noise_cluster_id=noise_cluster_id,
        missing_cluster_id=missing_cluster_id,
    )
    metrics = _correspondence_metrics(clean)
    metrics["target_unit_count"] = int(len(target_units))
    metrics["common_non_noise_units"] = metrics.pop("common_non_noise_records")
    return metrics


def _common_non_noise_assignments(
    merged: pd.DataFrame,
    *,
    noise_cluster_id: int,
    missing_cluster_id: int,
) -> pd.DataFrame:
    for column in ("baseline_cluster", "target_cluster"):
        merged[column] = pd.to_numeric(merged[column], errors="raise").astype(int)
    invalid_labels = {noise_cluster_id, missing_cluster_id}
    clean = merged[
        ~merged["baseline_cluster"].isin(invalid_labels)
        & ~merged["target_cluster"].isin(invalid_labels)
    ].copy()
    if clean.empty:
        raise ValueError("No common non-noise assignments remain for comparison")
    return clean


def _correspondence_metrics(clean: pd.DataFrame) -> dict[str, Any]:

    try:
        from sklearn.metrics import adjusted_mutual_info_score, v_measure_score
    except ImportError as exc:
        raise RuntimeError(
            'Cross-channel metrics require scikit-learn; install with pip install -e ".[clustering]"'
        ) from exc

    contingency = pd.crosstab(clean["baseline_cluster"], clean["target_cluster"])
    dominant_count = int(contingency.max(axis=1).sum())
    record_count = int(len(clean))
    tie_aware = _tie_aware_correspondence_summary(contingency)
    return {
        "common_non_noise_records": record_count,
        "ami": float(
            adjusted_mutual_info_score(
                clean["baseline_cluster"], clean["target_cluster"]
            )
        ),
        "v_measure": float(
            v_measure_score(clean["baseline_cluster"], clean["target_cluster"])
        ),
        "dominant_correspondence_ratio": dominant_count / record_count,
        "baseline_cluster_count": int(clean["baseline_cluster"].nunique()),
        "target_cluster_count": int(clean["target_cluster"].nunique()),
        **tie_aware,
    }


def cross_channel_contingency(
    baseline: pd.DataFrame,
    target: pd.DataFrame,
    *,
    id_column: str = "raw_record_id",
    baseline_cluster_column: str = "cluster_id",
    target_cluster_column: str = "cluster_id",
    noise_cluster_id: int = NOISE_CLUSTER_ID,
    missing_cluster_id: int = MISSING_CHANNEL_CLUSTER_ID,
) -> pd.DataFrame:
    """Return an aggregate non-zero contingency table with tie-aware flags.

    The output contains cluster identifiers and counts only; no record identifiers
    or source texts are retained.
    """

    _validate_assignment_frame(baseline, id_column, baseline_cluster_column, "baseline")
    _validate_assignment_frame(target, id_column, target_cluster_column, "target")
    merged = baseline[[id_column, baseline_cluster_column]].rename(
        columns={baseline_cluster_column: "baseline_cluster"}
    ).merge(
        target[[id_column, target_cluster_column]].rename(
            columns={target_cluster_column: "target_cluster"}
        ),
        on=id_column,
        how="inner",
        validate="one_to_one",
    )
    for column in ("baseline_cluster", "target_cluster"):
        merged[column] = pd.to_numeric(merged[column], errors="raise").astype(int)
    invalid_labels = {noise_cluster_id, missing_cluster_id}
    clean = merged[
        ~merged["baseline_cluster"].isin(invalid_labels)
        & ~merged["target_cluster"].isin(invalid_labels)
    ].copy()
    if clean.empty:
        raise ValueError("No common non-noise records remain for comparison")

    contingency = pd.crosstab(clean["baseline_cluster"], clean["target_cluster"])
    row_max = contingency.max(axis=1)
    column_max = contingency.max(axis=0)
    row_ties = contingency.eq(row_max, axis=0).sum(axis=1)
    column_ties = contingency.eq(column_max, axis=1).sum(axis=0)
    rows: list[dict[str, Any]] = []
    for baseline_cluster in contingency.index:
        baseline_total = int(contingency.loc[baseline_cluster].sum())
        for target_cluster in contingency.columns:
            count = int(contingency.loc[baseline_cluster, target_cluster])
            if count <= 0:
                continue
            target_total = int(contingency[target_cluster].sum())
            baseline_dominant = count == int(row_max.loc[baseline_cluster])
            target_dominant = count == int(column_max.loc[target_cluster])
            rows.append(
                {
                    "baseline_cluster": int(baseline_cluster),
                    "target_cluster": int(target_cluster),
                    "common_records": count,
                    "baseline_cluster_records": baseline_total,
                    "target_cluster_records": target_total,
                    "baseline_share": count / baseline_total,
                    "target_share": count / target_total,
                    "baseline_dominant": baseline_dominant,
                    "target_dominant": target_dominant,
                    "bidirectional_dominant_tie_aware": (
                        baseline_dominant and target_dominant
                    ),
                    "baseline_dominant_target_count": int(
                        row_ties.loc[baseline_cluster]
                    ),
                    "target_dominant_source_count": int(
                        column_ties.loc[target_cluster]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _tie_aware_correspondence_summary(contingency: pd.DataFrame) -> dict[str, Any]:
    row_max = contingency.max(axis=1)
    column_max = contingency.max(axis=0)
    row_dominant = contingency.eq(row_max, axis=0)
    column_dominant = contingency.eq(column_max, axis=1)
    mutual = row_dominant & column_dominant

    tie_aware_pairs = int(mutual.to_numpy().sum())
    tie_aware_records = int(contingency.where(mutual, 0).to_numpy().sum())
    record_count = int(contingency.to_numpy().sum())

    row_choice = {
        row: min(contingency.columns[row_dominant.loc[row]].tolist())
        for row in contingency.index
    }
    column_choice = {
        column: min(contingency.index[column_dominant[column]].tolist())
        for column in contingency.columns
    }
    single_pairs = [
        (row, column)
        for row, column in row_choice.items()
        if column_choice.get(column) == row
    ]
    single_records = int(
        sum(int(contingency.loc[row, column]) for row, column in single_pairs)
    )
    return {
        "baseline_clusters_with_dominant_ties": int(
            (row_dominant.sum(axis=1) > 1).sum()
        ),
        "target_clusters_with_dominant_ties": int(
            (column_dominant.sum(axis=0) > 1).sum()
        ),
        "tie_aware_bidirectional_pair_count": tie_aware_pairs,
        "tie_aware_bidirectional_record_count": tie_aware_records,
        "tie_aware_bidirectional_record_ratio": tie_aware_records / record_count,
        "single_tie_break_bidirectional_pair_count": len(single_pairs),
        "single_tie_break_bidirectional_record_count": single_records,
        "single_tie_break_bidirectional_record_ratio": single_records / record_count,
    }


def cross_channel_cluster_profiles(
    baseline: pd.DataFrame,
    target: pd.DataFrame,
    *,
    id_column: str = "raw_record_id",
    baseline_cluster_column: str = "cluster_id",
    target_cluster_column: str = "cluster_id",
    noise_cluster_id: int = NOISE_CLUSTER_ID,
    missing_cluster_id: int = MISSING_CHANNEL_CLUSTER_ID,
) -> pd.DataFrame:
    """Describe target-cluster concentration within each baseline cluster.

    The function returns aggregate cluster-level statistics only. Normalized
    entropy is zero when all records map to one target cluster and approaches
    one as records are distributed evenly across the observed target clusters.
    """

    _validate_assignment_frame(baseline, id_column, baseline_cluster_column, "baseline")
    _validate_assignment_frame(target, id_column, target_cluster_column, "target")
    merged = baseline[[id_column, baseline_cluster_column]].rename(
        columns={baseline_cluster_column: "baseline_cluster"}
    ).merge(
        target[[id_column, target_cluster_column]].rename(
            columns={target_cluster_column: "target_cluster"}
        ),
        on=id_column,
        how="inner",
        validate="one_to_one",
    )
    for column in ("baseline_cluster", "target_cluster"):
        merged[column] = pd.to_numeric(merged[column], errors="raise").astype(int)
    invalid_labels = {noise_cluster_id, missing_cluster_id}
    clean = merged[
        ~merged["baseline_cluster"].isin(invalid_labels)
        & ~merged["target_cluster"].isin(invalid_labels)
    ].copy()
    if clean.empty:
        raise ValueError("No common non-noise records remain for comparison")

    rows: list[dict[str, Any]] = []
    for baseline_cluster, group in clean.groupby("baseline_cluster", sort=True):
        counts = group["target_cluster"].value_counts().sort_index()
        total = int(counts.sum())
        probabilities = counts.to_numpy(dtype=float) / total
        observed_target_count = int(len(counts))
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        normalized_entropy = (
            entropy / float(np.log(observed_target_count))
            if observed_target_count > 1
            else 0.0
        )
        dominant_target = int(counts.idxmax())
        dominant_count = int(counts.max())
        rows.append(
            {
                "baseline_cluster": int(baseline_cluster),
                "common_records": total,
                "observed_target_clusters": observed_target_count,
                "dominant_target_cluster": dominant_target,
                "dominant_target_records": dominant_count,
                "dominant_target_share": dominant_count / total,
                "normalized_entropy": normalized_entropy,
            }
        )
    return pd.DataFrame(rows)


def complete_field_subset(
    assignments: pd.DataFrame,
    fields: pd.DataFrame,
    required_fields: Sequence[str],
    *,
    id_column: str = "raw_record_id",
) -> pd.DataFrame:
    """Return assignment rows whose required record-level fields are non-empty."""

    missing = [column for column in [id_column, *required_fields] if column not in fields]
    if missing:
        raise ValueError(f"Field data is missing columns: {missing}")
    if fields[id_column].duplicated().any():
        raise ValueError(f"Field data must contain one row per {id_column}")
    flags = fields[[id_column, *required_fields]].copy()
    complete = pd.Series(True, index=flags.index)
    for column in required_fields:
        complete &= flags[column].map(_is_nonempty_field)
    eligible_ids = set(flags.loc[complete, id_column].astype(str))
    return assignments[assignments[id_column].astype(str).isin(eligible_ids)].copy()


def _validate_gold_unit_keys(gold: Sequence[Mapping[str, Any]]) -> None:
    keys = [_gold_unit_key(row) for row in gold]
    if any(not raw_id or not unit_id for raw_id, unit_id in keys):
        raise ValueError(
            "Every gold unit needs raw_record_id and a stable record_id/gold_record_id"
        )
    if len(keys) != len(set(keys)):
        raise ValueError("Gold unit identifiers must be unique within raw records")


def _gold_unit_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("raw_record_id", "")), str(
        row.get("gold_record_id") or row.get("record_id") or ""
    )


def _prediction_unit_key(row: Mapping[str, Any]) -> tuple[str, str]:
    raw_id = str(row.get("raw_record_id", ""))
    record_id = str(row.get("record_id", ""))
    if not raw_id or not record_id:
        raise ValueError(
            "Result-source analysis requires raw_record_id and a stable prediction record_id"
        )
    return raw_id, record_id


def _validate_assignment_frame(
    frame: pd.DataFrame,
    id_column: str,
    cluster_column: str,
    label: str,
) -> None:
    missing = [column for column in (id_column, cluster_column) if column not in frame]
    if missing:
        raise ValueError(f"{label} assignments are missing columns: {missing}")
    if frame[id_column].isna().any() or frame[id_column].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} assignments contain blank {id_column}")
    if frame[id_column].duplicated().any():
        raise ValueError(f"{label} assignments must contain one row per {id_column}")


def _is_nonempty_field(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    text = str(value).strip()
    return text.lower() not in {"", "nan", "none", "null", "[]", "{}"}
