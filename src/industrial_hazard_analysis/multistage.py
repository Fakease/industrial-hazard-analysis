from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from industrial_hazard_analysis.evaluation import (
    is_effective_extraction_record,
    silhouette_cosine,
)


@dataclass(frozen=True)
class MultistagePath:
    path_id: str
    parent_channel: str
    target_channel: str
    parent_label: str
    target_label: str
    analysis_question: str


@dataclass(frozen=True)
class LocalClusteringParameters:
    candidate_id: str
    umap: dict[str, Any]
    hdbscan: dict[str, Any]
    random_seed: int = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_cache_key(
    provider: str,
    model: str,
    dimension: int,
    text: str,
) -> str:
    raw = "\n".join((provider, model, "embedding", str(dimension), text))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_embedding_cache(
    path: Path,
    needed_keys: set[str],
    *,
    dimension: int,
) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item.get("cache_key", ""))
            if key not in needed_keys:
                continue
            value = item.get("value")
            if not isinstance(value, list) or len(value) != dimension:
                raise ValueError(
                    f"Invalid {dimension}-dimensional embedding at cache line {line_number}"
                )
            found[key] = np.asarray(value, dtype=np.float32)
    missing = needed_keys - set(found)
    if missing:
        raise RuntimeError(
            f"Embedding cache is missing {len(missing)} required text vectors; "
            "API calls are disabled for multistage analysis"
        )
    return found


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains blank {column} values")
    duplicated = frame[column].astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, column].astype(str).head(5).tolist()
        raise ValueError(f"{label} contains duplicate {column} values: {examples}")


def prepare_path_frame(
    units: pd.DataFrame,
    parent_assignments: pd.DataFrame,
    target_inputs: pd.DataFrame,
    path: MultistagePath,
) -> pd.DataFrame:
    require_columns(units, ["record_id", "raw_record_id"], "structured units")
    effective_mask = units.apply(
        lambda row: is_effective_extraction_record(row.to_dict()),
        axis=1,
    )
    effective_units = units.loc[effective_mask].copy()
    if effective_units.empty:
        raise ValueError("structured units contain no effective hazard units")
    require_columns(
        parent_assignments,
        ["record_id", "final_cluster_id"],
        f"{path.parent_channel} assignments",
    )
    require_columns(
        target_inputs,
        ["record_id", "input_text"],
        f"{path.target_channel} inputs",
    )
    for frame, label in (
        (units, "structured units"),
        (parent_assignments, f"{path.parent_channel} assignments"),
        (target_inputs, f"{path.target_channel} inputs"),
    ):
        require_unique(frame, "record_id", label)

    unit_ids = set(effective_units["record_id"].astype(str))
    assignment_ids = set(parent_assignments["record_id"].astype(str))
    target_ids = set(target_inputs["record_id"].astype(str))
    if assignment_ids != unit_ids:
        raise ValueError(
            f"{path.parent_channel} assignment IDs do not match structured hazard units"
        )
    if target_ids != unit_ids:
        raise ValueError(
            f"{path.target_channel} input IDs do not match structured hazard units"
        )

    unit_copy = effective_units.copy()
    unit_copy["record_id"] = unit_copy["record_id"].astype(str)
    assignment_copy = parent_assignments[["record_id", "final_cluster_id"]].copy()
    assignment_copy["record_id"] = assignment_copy["record_id"].astype(str)
    assignment_copy["parent_topic_id"] = pd.to_numeric(
        assignment_copy.pop("final_cluster_id"), errors="raise"
    ).astype(int)
    target_copy = target_inputs[["record_id", "input_text"]].copy()
    target_copy["record_id"] = target_copy["record_id"].astype(str)
    target_copy["target_text"] = target_copy.pop("input_text").fillna("").astype(str).str.strip()

    merged = unit_copy.merge(
        assignment_copy,
        on="record_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        target_copy,
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(effective_units):
        raise ValueError(f"{path.path_id} join changed the structured-unit row count")
    merged.insert(0, "path_id", path.path_id)
    merged.insert(1, "parent_channel", path.parent_channel)
    merged.insert(2, "target_channel", path.target_channel)
    merged["target_valid"] = merged["target_text"].ne("")
    return merged


def parent_profiles(path_frame: pd.DataFrame) -> pd.DataFrame:
    non_noise = path_frame[path_frame["parent_topic_id"] >= 0].copy()
    rows: list[dict[str, Any]] = []
    for parent_topic_id, group in non_noise.groupby("parent_topic_id", sort=True):
        parent_units = int(len(group))
        target_valid_units = int(group["target_valid"].sum())
        rows.append(
            {
                "path_id": str(group["path_id"].iloc[0]),
                "parent_channel": str(group["parent_channel"].iloc[0]),
                "target_channel": str(group["target_channel"].iloc[0]),
                "parent_topic_id": int(parent_topic_id),
                "parent_units": parent_units,
                "target_valid_units": target_valid_units,
                "target_coverage": (
                    target_valid_units / parent_units if parent_units else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_summaries(
    profiles: pd.DataFrame,
    *,
    parent_min_values: Iterable[int],
    target_min_values: Iterable[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    path_id = str(profiles["path_id"].iloc[0]) if len(profiles) else ""
    for parent_min in parent_min_values:
        for target_min in target_min_values:
            eligible = profiles[
                (profiles["parent_units"] >= int(parent_min))
                & (profiles["target_valid_units"] >= int(target_min))
            ]
            parent_units = int(eligible["parent_units"].sum()) if len(eligible) else 0
            target_units = int(eligible["target_valid_units"].sum()) if len(eligible) else 0
            rows.append(
                {
                    "path_id": path_id,
                    "parent_min_units": int(parent_min),
                    "target_min_units": int(target_min),
                    "eligible_parent_topics": int(len(eligible)),
                    "eligible_parent_units": parent_units,
                    "target_valid_units": target_units,
                    "target_coverage": target_units / parent_units if parent_units else math.nan,
                }
            )
    return pd.DataFrame(rows)


def eligible_parent_ids(
    profiles: pd.DataFrame,
    *,
    parent_min_units: int,
    target_min_units: int,
) -> list[int]:
    eligible = profiles[
        (profiles["parent_units"] >= int(parent_min_units))
        & (profiles["target_valid_units"] >= int(target_min_units))
    ]
    return sorted(eligible["parent_topic_id"].astype(int).tolist())


def cluster_embeddings(
    embeddings: np.ndarray,
    parameters: LocalClusteringParameters,
    *,
    numba_cache_dir: str | Path | None = None,
) -> np.ndarray:
    if len(embeddings) < 4:
        return np.zeros(len(embeddings), dtype=int)
    n_components = int(parameters.umap.get("n_components", 5))
    if len(embeddings) <= n_components + 1:
        raise ValueError(
            f"Local sample size {len(embeddings)} is too small for "
            f"UMAP n_components={n_components}"
        )
    min_cluster_size = int(parameters.hdbscan.get("min_cluster_size", 3))
    min_samples = int(parameters.hdbscan.get("min_samples", 1))
    if min_cluster_size < 2 or min_cluster_size > len(embeddings):
        raise ValueError("Invalid local HDBSCAN min_cluster_size")
    if min_samples < 1 or min_samples > len(embeddings):
        raise ValueError("Invalid local HDBSCAN min_samples")
    if numba_cache_dir:
        cache_path = Path(numba_cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache_path)

    import hdbscan
    import umap

    n_neighbors = min(
        int(parameters.umap.get("n_neighbors", 8)),
        max(2, len(embeddings) - 1),
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=float(parameters.umap.get("min_dist", 0.05)),
        metric=str(parameters.umap.get("metric", "cosine")),
        random_state=int(parameters.random_seed),
    )
    reduced = reducer.fit_transform(np.asarray(embeddings, dtype=np.float32))
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=str(parameters.hdbscan.get("metric", "euclidean")),
        cluster_selection_method=str(
            parameters.hdbscan.get("cluster_selection_method", "eom")
        ),
        cluster_selection_epsilon=float(
            parameters.hdbscan.get("cluster_selection_epsilon", 0.0)
        ),
    )
    return np.asarray(model.fit_predict(reduced), dtype=int)


def centroid_distances_and_ranks(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(len(labels), np.nan, dtype=float)
    ranks = np.zeros(len(labels), dtype=int)
    for label in sorted({int(value) for value in labels if int(value) >= 0}):
        indices = np.where(labels == label)[0]
        centroid = embeddings[indices].mean(axis=0)
        cluster_distances = np.linalg.norm(embeddings[indices] - centroid, axis=1)
        order = np.argsort(cluster_distances, kind="stable")
        for rank, position in enumerate(order, start=1):
            source_index = indices[position]
            distances[source_index] = float(cluster_distances[position])
            ranks[source_index] = rank
    return distances, ranks


def run_local_analysis(
    path_frame: pd.DataFrame,
    profiles: pd.DataFrame,
    vectors_by_key: dict[str, np.ndarray],
    *,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimension: int,
    parent_min_units: int,
    target_min_units: int,
    parameters: LocalClusteringParameters,
    cluster_function: Callable[[np.ndarray, LocalClusteringParameters], np.ndarray] | None = None,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
    numba_cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_function = cluster_function or (
        lambda vectors, params: cluster_embeddings(
            vectors,
            params,
            numba_cache_dir=numba_cache_dir,
        )
    )
    profile_by_id = profiles.set_index("parent_topic_id")
    parent_ids = eligible_parent_ids(
        profiles,
        parent_min_units=parent_min_units,
        target_min_units=target_min_units,
    )
    assignment_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    for parent_index, parent_topic_id in enumerate(parent_ids, start=1):
        selected = path_frame[
            (path_frame["parent_topic_id"] == parent_topic_id)
            & path_frame["target_valid"]
        ].copy()
        if progress_callback is not None:
            progress_callback(
                parent_index,
                len(parent_ids),
                int(parent_topic_id),
                len(selected),
            )
        keys = [
            embedding_cache_key(
                embedding_provider,
                embedding_model,
                embedding_dimension,
                text,
            )
            for text in selected["target_text"].astype(str)
        ]
        embeddings = np.vstack([vectors_by_key[key] for key in keys]).astype(np.float32)
        labels = np.asarray(cluster_function(embeddings, parameters), dtype=int)
        if len(labels) != len(selected):
            raise ValueError("Local clustering returned an unexpected number of labels")
        distances, ranks = centroid_distances_and_ranks(embeddings, labels)
        child_topic_count = len({int(value) for value in labels if int(value) >= 0})
        noise_count = int(np.sum(labels == -1))
        non_noise_count = int(np.sum(labels >= 0))
        silhouette = silhouette_cosine(embeddings, labels)
        profile = profile_by_id.loc[parent_topic_id]
        parent_rows.append(
            {
                "path_id": str(selected["path_id"].iloc[0]),
                "candidate_id": parameters.candidate_id,
                "parent_topic_id": int(parent_topic_id),
                "parent_units": int(profile["parent_units"]),
                "target_valid_units": int(len(selected)),
                "child_topic_count": int(child_topic_count),
                "noise_count": noise_count,
                "noise_rate": noise_count / len(selected) if len(selected) else math.nan,
                "non_noise_count": non_noise_count,
                "silhouette_cosine": silhouette,
            }
        )
        for row_index, (_, row) in enumerate(selected.iterrows()):
            item = row.to_dict()
            item.update(
                {
                    "candidate_id": parameters.candidate_id,
                    "child_topic_id": int(labels[row_index]),
                    "is_noise": bool(labels[row_index] == -1),
                    "distance_to_centroid": (
                        None if math.isnan(float(distances[row_index])) else float(distances[row_index])
                    ),
                    "representative_rank": int(ranks[row_index]) or None,
                }
            )
            assignment_rows.append(item)
    return pd.DataFrame(parent_rows), pd.DataFrame(assignment_rows)


def summarize_local_results(parent_results: pd.DataFrame) -> dict[str, Any]:
    if parent_results.empty:
        return {
            "eligible_parent_topics": 0,
            "eligible_parent_units": 0,
            "target_valid_units": 0,
            "target_coverage": None,
            "multi_child_parent_topics": 0,
            "multi_child_parent_share": None,
            "child_topics_median": None,
            "child_topics_q1": None,
            "child_topics_q3": None,
            "noise_rate": None,
            "silhouette_parent_topics": 0,
            "silhouette_sample_share": None,
            "weighted_silhouette_cosine": None,
        }
    parent_units = int(parent_results["parent_units"].sum())
    target_units = int(parent_results["target_valid_units"].sum())
    noise_count = int(parent_results["noise_count"].sum())
    multi_count = int((parent_results["child_topic_count"] >= 2).sum())
    valid_silhouette = parent_results[parent_results["silhouette_cosine"].notna()].copy()
    silhouette_non_noise = int(valid_silhouette["non_noise_count"].sum())
    all_non_noise = int(parent_results["non_noise_count"].sum())
    if silhouette_non_noise:
        weighted_silhouette = float(
            np.average(
                valid_silhouette["silhouette_cosine"].astype(float),
                weights=valid_silhouette["non_noise_count"].astype(float),
            )
        )
    else:
        weighted_silhouette = None
    counts = parent_results["child_topic_count"].astype(float)
    return {
        "eligible_parent_topics": int(len(parent_results)),
        "eligible_parent_units": parent_units,
        "target_valid_units": target_units,
        "target_coverage": target_units / parent_units if parent_units else None,
        "multi_child_parent_topics": multi_count,
        "multi_child_parent_share": multi_count / len(parent_results),
        "child_topics_median": float(counts.median()),
        "child_topics_q1": float(counts.quantile(0.25)),
        "child_topics_q3": float(counts.quantile(0.75)),
        "noise_rate": noise_count / target_units if target_units else None,
        "silhouette_parent_topics": int(len(valid_silhouette)),
        "silhouette_sample_share": (
            silhouette_non_noise / all_non_noise if all_non_noise else None
        ),
        "weighted_silhouette_cosine": weighted_silhouette,
    }


def select_representative_parent(parent_results: pd.DataFrame) -> int | None:
    branching = parent_results[parent_results["child_topic_count"] >= 2].copy()
    if branching.empty:
        return None
    selected = branching.sort_values(
        ["parent_units", "parent_topic_id"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    return int(selected["parent_topic_id"])


def naming_request_rows(assignments: pd.DataFrame, *, representative_count: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if assignments.empty:
        return rows
    non_noise = assignments[assignments["child_topic_id"] >= 0].copy()
    for (path_id, parent_topic_id, child_topic_id), group in non_noise.groupby(
        ["path_id", "parent_topic_id", "child_topic_id"],
        sort=True,
    ):
        representatives = group.sort_values(
            ["representative_rank", "record_id"], kind="stable"
        ).head(representative_count)
        target_channels = group["target_channel"].astype(str).unique().tolist()
        if len(target_channels) != 1:
            raise ValueError(
                "Each multistage child topic must have exactly one target channel"
            )
        target_channel = target_channels[0]
        target_field_by_channel = {"T1": "finding", "T2": "object"}
        try:
            target_field = target_field_by_channel[target_channel]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported multistage naming target channel: {target_channel}"
            ) from exc
        rows.append(
            {
                "request_id": f"{path_id}-P{int(parent_topic_id)}-C{int(child_topic_id)}",
                "path_id": str(path_id),
                "parent_topic_id": int(parent_topic_id),
                "child_topic_id": int(child_topic_id),
                "target_channel": target_channel,
                "channel_fields": [target_field],
                "sample_count": int(len(group)),
                "representative_samples": [
                    {
                        "record_id": str(row.record_id),
                        "channel_text": str(row.target_text),
                        "distance_to_centroid": float(row.distance_to_centroid),
                        "representative_rank": int(row.representative_rank),
                    }
                    for row in representatives.itertuples(index=False)
                ],
                "field_values": {
                    target_field: group["target_text"].astype(str).tolist(),
                },
            }
        )
    return rows
