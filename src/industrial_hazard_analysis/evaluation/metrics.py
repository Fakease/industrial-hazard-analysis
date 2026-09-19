from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


NOISE_CLUSTER_ID = -1
MISSING_CHANNEL_CLUSTER_ID = -2
TEXT_FIELDS = ("finding", "scene", "loc_detail", "risk_scene")
CORE_COMPLETE_FIELDS = ("finding", "object", "scene")


def evaluate_extraction(
    predictions: list[Any],
    gold_records: list[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate structured extraction against the human gold standard.

    Predictions are evaluated only for raw_record_id values that appear in the
    gold standard. A unit is a complete hit only when finding, object, and scene
    all match exactly after normalization. Remaining units are paired within
    each raw record by a maximum-similarity assignment solely for field-level
    diagnostic metrics.
    """

    if not gold_records:
        return {"status": "skipped", "reason": "gold_records_missing"}

    gold = [_normalize_record(row) for row in gold_records]
    pred_rows = [_normalize_record(row) for row in predictions]
    pred_all = [row for row in pred_rows if is_effective_extraction_record(row)]
    gold_raw_ids = {row["raw_record_id"] for row in gold if row.get("raw_record_id")}
    predictions_in_scope = [
        row for row in pred_all if not gold_raw_ids or row.get("raw_record_id") in gold_raw_ids
    ]
    matches, unpaired_predictions, unpaired_gold = match_extraction_units(
        predictions_in_scope, gold
    )

    complete_hits = sum(bool(match["complete_hit"]) for match in matches)
    unit_precision, unit_recall, unit_f1 = _prf(
        complete_hits,
        len(predictions_in_scope) - complete_hits,
        len(gold) - complete_hits,
    )
    text_field_metrics = {
        field: _text_field_metrics(matches, field)
        for field in TEXT_FIELDS
    }
    object_metrics = _object_field_metrics(matches)

    return {
        "status": "ok",
        "gold_unit_count": len(gold),
        "prediction_unit_count": len(pred_all),
        "prediction_unit_count_in_scope": len(predictions_in_scope),
        "complete_hit_count": complete_hits,
        "unit_false_positive_count": len(predictions_in_scope) - complete_hits,
        "unit_false_negative_count": len(gold) - complete_hits,
        "diagnostic_pair_count": len(matches),
        "unpaired_prediction_count": len(unpaired_predictions),
        "unpaired_gold_count": len(unpaired_gold),
        "unit_precision": unit_precision,
        "unit_recall": unit_recall,
        "unit_f1": unit_f1,
        "finding_char_f1": text_field_metrics["finding"]["char_f1"],
        "scene_char_f1": text_field_metrics["scene"]["char_f1"],
        "loc_detail_char_f1": text_field_metrics["loc_detail"]["char_f1"],
        "risk_scene_char_f1": text_field_metrics["risk_scene"]["char_f1"],
        "object_set_f1": object_metrics["set_f1"],
        "text_fields": text_field_metrics,
        "object": object_metrics,
        "omission_overextraction": _omission_overextraction(matches),
        "pairs": [_match_summary(match) for match in matches],
        "unpaired_predictions": unpaired_predictions,
        "unpaired_gold": unpaired_gold,
    }


def evaluate_scene_split(
    predictions: list[Any],
    gold_records: list[Any] | None = None,
) -> dict[str, Any]:
    if not gold_records:
        return {"status": "skipped", "reason": "gold_records_missing"}
    metrics = evaluate_extraction(predictions, gold_records)
    return {
        "status": metrics["status"],
        "diagnostic_pair_count": metrics.get("diagnostic_pair_count", 0),
        "loc_detail_char_f1": metrics["text_fields"]["loc_detail"]["char_f1"],
        "risk_scene_char_f1": metrics["text_fields"]["risk_scene"]["char_f1"],
        "scene_char_f1": metrics["text_fields"]["scene"]["char_f1"],
    }


def evaluate_clustering(
    assignments: list[Any],
    gold_labels: list[Any] | None = None,
    *,
    embeddings: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    labels = [_cluster_id(row) for row in assignments]
    structure = cluster_structure_metrics(labels)
    result: dict[str, Any] = {
        "status": "ok",
        **structure,
    }

    if embeddings is not None:
        result["silhouette_cosine"] = silhouette_cosine(embeddings, labels)
        result.update(topic_similarity_metrics(embeddings, labels))
    else:
        result["silhouette_cosine"] = None
        result["topic_similarity_status"] = "skipped_embeddings_missing"

    if gold_labels:
        result.update(_external_label_metrics(labels, gold_labels))
    else:
        result["external_metrics_status"] = "skipped_gold_labels_missing"
    return result


def match_extraction_units(
    predictions: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair extraction units within each source record.

    Exact core-field matches are locked first. The remaining units are paired
    with the Hungarian algorithm using the equal-weight mean of valid finding,
    object, and scene similarities. No similarity threshold is applied. The
    pairing score is diagnostic only and never enters unit precision/recall/F1.
    """

    predictions_by_raw = _group_by_raw_id(predictions)
    gold_by_raw = _group_by_raw_id(gold_records)
    matches: list[dict[str, Any]] = []
    unpaired_predictions: list[dict[str, Any]] = []
    unpaired_gold: list[dict[str, Any]] = []

    for raw_id in sorted(set(predictions_by_raw) | set(gold_by_raw)):
        pred_group = predictions_by_raw.get(raw_id, [])
        gold_group = gold_by_raw.get(raw_id, [])
        used_pred: set[int] = set()
        used_gold: set[int] = set()

        gold_indices_by_key: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for gold_idx, gold in enumerate(gold_group):
            gold_indices_by_key[_complete_unit_key(gold)].append(gold_idx)

        for pred_idx, pred in enumerate(pred_group):
            candidates = gold_indices_by_key.get(_complete_unit_key(pred), [])
            gold_idx = next((idx for idx in candidates if idx not in used_gold), None)
            if gold_idx is None:
                continue
            gold = gold_group[gold_idx]
            matches.append(
                {
                    "raw_record_id": raw_id,
                    "prediction": pred,
                    "gold": gold,
                    "complete_hit": True,
                    "pairing_score": 1.0,
                }
            )
            used_pred.add(pred_idx)
            used_gold.add(gold_idx)

        remaining_pred_indices = [
            idx for idx in range(len(pred_group)) if idx not in used_pred
        ]
        remaining_gold_indices = [
            idx for idx in range(len(gold_group)) if idx not in used_gold
        ]
        if remaining_pred_indices and remaining_gold_indices:
            similarity = np.asarray(
                [
                    [
                        _unit_pairing_similarity(
                            pred_group[pred_idx], gold_group[gold_idx]
                        )
                        for gold_idx in remaining_gold_indices
                    ]
                    for pred_idx in remaining_pred_indices
                ],
                dtype=float,
            )
            row_indices, column_indices = linear_sum_assignment(-similarity)
            for row_idx, column_idx in zip(row_indices, column_indices):
                pred_idx = remaining_pred_indices[int(row_idx)]
                gold_idx = remaining_gold_indices[int(column_idx)]
                matches.append(
                    {
                        "raw_record_id": raw_id,
                        "prediction": pred_group[pred_idx],
                        "gold": gold_group[gold_idx],
                        "complete_hit": False,
                        "pairing_score": float(similarity[row_idx, column_idx]),
                    }
                )
                used_pred.add(pred_idx)
                used_gold.add(gold_idx)

        unpaired_predictions.extend(
            _minimal_record(row)
            for idx, row in enumerate(pred_group)
            if idx not in used_pred
        )
        unpaired_gold.extend(
            _minimal_record(row)
            for idx, row in enumerate(gold_group)
            if idx not in used_gold
        )

    return matches, unpaired_predictions, unpaired_gold


def normalize_extraction_record(row: Any) -> dict[str, Any]:
    return _normalize_record(row)


def is_effective_extraction_record(row: Any) -> bool:
    """Return whether a row contains at least one extracted semantic field.

    Empty arbitration/no-candidate rows are retained by the pipeline as audit
    placeholders. They are not predicted hazard units and therefore must not
    enter unit counts or extraction precision denominators.
    """

    normalized = row if _is_normalized_extraction_record(row) else _normalize_record(row)
    return bool(
        normalized.get("finding")
        or normalized.get("object")
        or normalized.get("scene")
        or normalized.get("loc_detail")
        or normalized.get("risk_scene")
    )


def cluster_structure_metrics(labels: Sequence[Any]) -> dict[str, Any]:
    normalized = [_safe_int(label) for label in labels]
    total_count = len(normalized)
    missing_count = sum(label == MISSING_CHANNEL_CLUSTER_ID for label in normalized)
    clustered_record_count = total_count - missing_count
    noise_count = sum(label == NOISE_CLUSTER_ID for label in normalized)
    topic_labels = [
        label
        for label in normalized
        if label not in {NOISE_CLUSTER_ID, MISSING_CHANNEL_CLUSTER_ID}
    ]
    sizes = sorted(Counter(topic_labels).values())
    largest_topic_size = max(sizes, default=0)
    return {
        "record_count": total_count,
        "clustered_record_count": clustered_record_count,
        "missing_channel_count": missing_count,
        "missing_channel_ratio": missing_count / total_count if total_count else 0.0,
        "cluster_count": len(sizes),
        "noise_count": noise_count,
        "noise_ratio": noise_count / total_count if total_count else 0.0,
        "outlier_ratio": noise_count / total_count if total_count else 0.0,
        "median_topic_size": float(median(sizes)) if sizes else 0.0,
        "largest_topic_size": largest_topic_size,
        "largest_topic_ratio": (
            largest_topic_size / clustered_record_count if clustered_record_count else 0.0
        ),
        "label_distribution": dict(Counter(normalized)),
    }


def silhouette_cosine(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[Any],
) -> float | None:
    x, y = _topic_matrix_and_labels(embeddings, labels)
    if x is None or y is None:
        return None
    unique_labels = sorted(set(int(label) for label in y))
    if len(unique_labels) < 2:
        return None

    distance = 1.0 - np.clip(x @ x.T, -1.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    mean_distances = np.zeros((len(y), len(unique_labels)), dtype=np.float32)
    label_to_column = {label: idx for idx, label in enumerate(unique_labels)}
    for label in unique_labels:
        idx = np.where(y == label)[0]
        mean_distances[:, label_to_column[label]] = distance[:, idx].mean(axis=1)

    silhouettes = np.zeros(len(y), dtype=np.float32)
    for row_idx, label in enumerate(y):
        own_column = label_to_column[int(label)]
        same_count = int(np.sum(y == label))
        if same_count <= 1:
            continue
        a_value = (mean_distances[row_idx, own_column] * same_count) / (same_count - 1)
        other_values = np.delete(mean_distances[row_idx], own_column)
        b_value = float(np.min(other_values)) if len(other_values) else 0.0
        denom = max(float(a_value), b_value)
        silhouettes[row_idx] = (b_value - float(a_value)) / denom if denom else 0.0
    return float(np.mean(silhouettes))


def topic_similarity_metrics(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[Any],
) -> dict[str, Any]:
    x, y = _topic_matrix_and_labels(embeddings, labels)
    if x is None or y is None:
        return {
            "topic_similarity_status": "skipped_insufficient_embeddings",
            "mean_topic_topic_similarity": None,
            "max_topic_topic_similarity": None,
        }
    unique_labels = sorted(set(int(label) for label in y))
    if len(unique_labels) < 2:
        return {
            "topic_similarity_status": "skipped_less_than_two_topics",
            "mean_topic_topic_similarity": None,
            "max_topic_topic_similarity": None,
        }

    centroids = []
    for label in unique_labels:
        centroid = x[y == label].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids.append(centroid / norm if norm else centroid)
    centroid_matrix = np.vstack(centroids).astype(np.float32)
    sim = np.clip(centroid_matrix @ centroid_matrix.T, -1.0, 1.0)
    mask = ~np.eye(len(unique_labels), dtype=bool)
    off_diag = sim[mask]
    mean_similarity = float(np.mean(off_diag))
    max_similarity = float(np.max(off_diag))
    return {
        "topic_similarity_status": "ok",
        "mean_topic_topic_similarity": mean_similarity,
        "max_topic_topic_similarity": max_similarity,
    }


def char_f1(predicted: Any, gold: Any) -> float:
    pred = _normalize_text(predicted)
    ref = _normalize_text(gold)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / sum(pred_counts.values()) if pred_counts else 0.0
    recall = overlap / sum(ref_counts.values()) if ref_counts else 0.0
    return _f1(precision, recall)


def object_set_f1(predicted: Any, gold: Any) -> float:
    pred_set = set(_parse_objects(predicted))
    gold_set = set(_parse_objects(gold))
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return _f1(precision, recall)


def _topic_matrix_and_labels(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must have the same length")
    clean_labels = np.asarray([_safe_int(label) for label in labels], dtype=int)
    mask = np.asarray(
        [
            label not in {NOISE_CLUSTER_ID, MISSING_CHANNEL_CLUSTER_ID}
            for label in clean_labels
        ],
        dtype=bool,
    )
    if int(mask.sum()) < 2:
        return None, None
    x = np.asarray(embeddings, dtype=np.float32)[mask]
    y = clean_labels[mask]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms, y


def _normalize_record(row: Any) -> dict[str, Any]:
    data = _asdict(row)
    normalized: dict[str, Any] = {}
    normalized["record_id"] = _first_text(data, ["record_id", "gold_record_id", "id"])
    normalized["gold_record_id"] = _first_text(data, ["gold_record_id", "record_id", "id"])
    normalized["raw_record_id"] = _first_text(data, ["raw_record_id", "raw_id", "source_id"])
    normalized["raw_text"] = _first_text(data, ["raw_text", "text", "隐患描述"])
    normalized["finding"] = _first_text(data, ["finding", "phenomenon_label"])
    normalized["scene"] = _first_text(data, ["scene", "scene_label"])
    normalized["loc_detail"] = _first_text(data, ["loc_detail", "location_label"])
    normalized["risk_scene"] = _first_text(data, ["risk_scene"])
    normalized["object"] = _parse_objects(
        data.get("object", data.get("object_json", data.get("object_label", [])))
    )
    return normalized


def _is_normalized_extraction_record(row: Any) -> bool:
    return isinstance(row, dict) and isinstance(row.get("object"), list)


def _asdict(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    if hasattr(row, "to_dict"):
        return dict(row.to_dict())
    return {
        key: getattr(row, key)
        for key in dir(row)
        if not key.startswith("_") and not callable(getattr(row, key))
    }


def _first_text(data: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _parse_objects(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = re.split(r"[;；,，、|/]+", text)
    flattened = _flatten_objects(parsed)
    return sorted({_normalize_object_text(item) for item in flattened if _normalize_object_text(item)})


def _flatten_objects(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_objects(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten_objects(item))
        return out
    return [str(value)]


def _normalize_object_text(value: Any) -> str:
    text = _normalize_text(value)
    return text.strip("[]{}()（）\"'")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def _complete_unit_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _normalize_text(row.get("finding", "")),
        tuple(sorted(set(_parse_objects(row.get("object", []))))),
        _normalize_text(row.get("scene", "")),
    )


def _unit_pairing_similarity(
    prediction: dict[str, Any], gold: dict[str, Any]
) -> float:
    field_scores: list[float] = []
    for field in ("finding", "scene"):
        predicted = _normalize_text(prediction.get(field, ""))
        reference = _normalize_text(gold.get(field, ""))
        if predicted or reference:
            field_scores.append(char_f1(predicted, reference))

    predicted_objects = _parse_objects(prediction.get("object", []))
    gold_objects = _parse_objects(gold.get("object", []))
    if predicted_objects or gold_objects:
        field_scores.append(object_set_f1(predicted_objects, gold_objects))
    return float(np.mean(field_scores)) if field_scores else 0.0


def _text_field_metrics(matches: list[dict[str, Any]], field: str) -> dict[str, Any]:
    scores: list[float] = []
    exact: list[bool] = []
    for match in matches:
        predicted = _normalize_text(match["prediction"].get(field, ""))
        reference = _normalize_text(match["gold"].get(field, ""))
        if not predicted and not reference:
            continue
        scores.append(char_f1(predicted, reference))
        exact.append(predicted == reference)
    return {
        "char_f1": float(np.mean(scores)) if scores else 0.0,
        "exact_match_rate": sum(exact) / len(exact) if exact else 0.0,
        "evaluated_count": len(scores),
    }


def _object_field_metrics(matches: list[dict[str, Any]]) -> dict[str, Any]:
    pair_scores: list[float] = []
    exact_matches: list[bool] = []
    for match in matches:
        pred_set = set(_parse_objects(match["prediction"].get("object", [])))
        gold_set = set(_parse_objects(match["gold"].get("object", [])))
        if not pred_set and not gold_set:
            continue
        pair_scores.append(object_set_f1(pred_set, gold_set))
        exact_matches.append(pred_set == gold_set)
    mean_pair_f1 = float(np.mean(pair_scores)) if pair_scores else 0.0
    return {
        "set_f1": mean_pair_f1,
        "average_pair_f1": mean_pair_f1,
        "exact_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "evaluated_count": len(pair_scores),
    }


def _omission_overextraction(matches: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for field in TEXT_FIELDS:
        counts = {"omission": 0, "over_extraction": 0, "both_empty": 0, "both_present": 0}
        for match in matches:
            pred = _normalize_text(match["prediction"].get(field, ""))
            gold = _normalize_text(match["gold"].get(field, ""))
            if gold and not pred:
                counts["omission"] += 1
            elif pred and not gold:
                counts["over_extraction"] += 1
            elif not pred and not gold:
                counts["both_empty"] += 1
            else:
                counts["both_present"] += 1
        out[field] = counts
    return out


def _match_summary(match: dict[str, Any]) -> dict[str, Any]:
    pred = match["prediction"]
    gold = match["gold"]
    return {
        "raw_record_id": match["raw_record_id"],
        "prediction_record_id": pred.get("record_id", ""),
        "gold_record_id": gold.get("gold_record_id", ""),
        "complete_hit": bool(match["complete_hit"]),
        "pairing_score": match["pairing_score"],
        "finding_char_f1": char_f1(pred.get("finding", ""), gold.get("finding", "")),
        "object_set_f1": object_set_f1(pred.get("object", []), gold.get("object", [])),
        "scene_char_f1": char_f1(pred.get("scene", ""), gold.get("scene", "")),
        "loc_detail_char_f1": char_f1(
            pred.get("loc_detail", ""), gold.get("loc_detail", "")
        ),
        "risk_scene_char_f1": char_f1(pred.get("risk_scene", ""), gold.get("risk_scene", "")),
        "prediction_finding": pred.get("finding", ""),
        "gold_finding": gold.get("finding", ""),
    }


def _minimal_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_record_id": row.get("raw_record_id", ""),
        "record_id": row.get("record_id", ""),
        "gold_record_id": row.get("gold_record_id", ""),
        "finding": row.get("finding", ""),
        "object": row.get("object", []),
        "scene": row.get("scene", ""),
        "loc_detail": row.get("loc_detail", ""),
        "risk_scene": row.get("risk_scene", ""),
    }


def _group_by_raw_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("raw_record_id", "")].append(row)
    return grouped


def _cluster_id(row: Any) -> int:
    data = _asdict(row)
    return _safe_int(data.get("cluster_id", data.get("final_cluster_id", 0)))


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        if isinstance(value, str) and not value.strip():
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _external_label_metrics(labels: Sequence[Any], gold_labels: Sequence[Any]) -> dict[str, Any]:
    if len(labels) != len(gold_labels):
        return {
            "external_metrics_status": "skipped_length_mismatch",
            "nmi": None,
            "ari": None,
        }
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except Exception:
        return {
            "external_metrics_status": "skipped_sklearn_missing",
            "nmi": None,
            "ari": None,
        }
    return {
        "external_metrics_status": "ok",
        "nmi": float(normalized_mutual_info_score(gold_labels, labels)),
        "ari": float(adjusted_rand_score(gold_labels, labels)),
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, _f1(precision, recall)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
