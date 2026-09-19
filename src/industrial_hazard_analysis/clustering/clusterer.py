from __future__ import annotations

from collections import Counter
from copy import deepcopy
import os
from typing import Any

import numpy as np

from industrial_hazard_analysis.models import ChannelInput, ClusterAssignment
from industrial_hazard_analysis.providers.base import EmbeddingProvider


SECONDARY_NOISE_ONLY_DEFAULTS = {
    "umap": {
        "n_neighbors": 8,
        "n_components": 5,
        "min_dist": 0.05,
        "metric": "cosine",
    },
    "hdbscan": {
        "min_cluster_size": 3,
        "min_samples": 1,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "cluster_selection_epsilon": 0.0,
    },
}


class Clusterer:
    def __init__(self, embedding_provider: EmbeddingProvider, config: dict[str, Any]):
        self.embedding_provider = embedding_provider
        self.config = config

    def cluster(self, experiment_id: str, inputs: list[ChannelInput]) -> tuple[list[ClusterAssignment], dict]:
        texts = [item.input_text for item in inputs]
        print(f"Clusterer {experiment_id}: embedding {len(texts)} text(s)", flush=True)
        embeddings = np.asarray(self.embedding_provider.embed_texts(texts), dtype=float)
        print(f"Clusterer {experiment_id}: running UMAP-HDBSCAN", flush=True)
        primary_labels, backend = self._umap_hdbscan(embeddings, self.config)
        primary_noise_count = int(np.sum(primary_labels == -1))
        final_labels, assignment_levels, recluster_metadata = self._apply_noise_recluster(
            experiment_id,
            embeddings,
            primary_labels,
        )
        print(f"Clusterer {experiment_id}: assigning representative ranks", flush=True)
        distances, ranks = self._centroid_distances_and_ranks(embeddings, final_labels)
        assignments = [
            ClusterAssignment(
                experiment_id=experiment_id,
                record_id=item.record_id,
                cluster_id=int(final_labels[index]),
                is_noise=int(final_labels[index]) == -1,
                distance_to_centroid=float(distances[index]),
                representative_rank=int(ranks[index]),
                primary_cluster_id=int(primary_labels[index]),
                primary_is_noise=int(primary_labels[index]) == -1,
                noise_recluster_id=assignment_levels[index]["noise_recluster_id"],
                final_cluster_id=int(final_labels[index]),
                cluster_level=assignment_levels[index]["cluster_level"],
            )
            for index, item in enumerate(inputs)
        ]
        noise_count = sum(assignment.is_noise for assignment in assignments)
        metadata = {
            "backend": backend,
            "record_count": len(assignments),
            "cluster_count": len({a.cluster_id for a in assignments if not a.is_noise}),
            "noise_count": noise_count,
            "noise_ratio": noise_count / len(assignments) if assignments else 0.0,
            "primary_cluster_count": len({int(label) for label in primary_labels if int(label) != -1}),
            "primary_noise_count": primary_noise_count,
            "primary_noise_ratio": primary_noise_count / len(assignments) if assignments else 0.0,
            "primary_label_distribution": dict(Counter(int(label) for label in primary_labels)),
            "label_distribution": dict(Counter(int(label) for label in final_labels)),
            "noise_recluster": recluster_metadata,
        }
        print(
            f"Clusterer {experiment_id}: backend={backend}, "
            f"clusters={metadata['cluster_count']}, residual_noise={metadata['noise_count']}, "
            f"primary_noise={metadata['primary_noise_count']}",
            flush=True,
        )
        return assignments, metadata

    def _umap_hdbscan(self, embeddings: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, str]:
        if len(embeddings) == 0:
            return np.asarray([], dtype=int), "empty"
        if len(embeddings) < 4:
            return np.zeros(len(embeddings), dtype=int), "small_sample_single_cluster"
        if not bool(config.get("use_real_backend", False)):
            return self._fallback_labels(embeddings), "fallback_hash_clusterer:disabled_by_config"

        try:
            self._ensure_numba_cache_dir(config)

            import hdbscan
            import umap

            umap_cfg = config.get("umap", {})
            hdbscan_cfg = config.get("hdbscan", {})
            n_neighbors = min(int(umap_cfg.get("n_neighbors", 15)), max(2, len(embeddings) - 1))
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=int(umap_cfg.get("n_components", 10)),
                min_dist=float(umap_cfg.get("min_dist", 0.05)),
                metric=umap_cfg.get("metric", "cosine"),
                random_state=int(config.get("random_seed", 42)),
            )
            reduced = reducer.fit_transform(embeddings)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min(int(hdbscan_cfg.get("min_cluster_size", 10)), len(embeddings)),
                min_samples=min(int(hdbscan_cfg.get("min_samples", 3)), len(embeddings)),
                metric=hdbscan_cfg.get("metric", "euclidean"),
                cluster_selection_method=hdbscan_cfg.get("cluster_selection_method", "eom"),
                cluster_selection_epsilon=float(hdbscan_cfg.get("cluster_selection_epsilon", 0.0)),
            )
            return clusterer.fit_predict(reduced), "umap_hdbscan"
        except Exception as exc:
            if not bool(config.get("allow_backend_fallback", False)):
                raise RuntimeError(
                    "UMAP-HDBSCAN backend failed while clustering. "
                    "Install/enable umap-learn, hdbscan and scikit-learn, or set "
                    "clustering.allow_backend_fallback=true for mock/debug runs."
                ) from exc
            labels = self._fallback_labels(embeddings)
            return labels, f"fallback_hash_clusterer:{exc.__class__.__name__}"

    @staticmethod
    def _ensure_numba_cache_dir(config: dict[str, Any]) -> None:
        cache_dir = config.get("numba_cache_dir", ".numba_cache")
        if not cache_dir or os.environ.get("NUMBA_CACHE_DIR"):
            return
        path = os.path.abspath(os.fspath(cache_dir))
        os.makedirs(path, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = path

    def _apply_noise_recluster(
        self,
        experiment_id: str,
        embeddings: np.ndarray,
        primary_labels: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
        final_labels = np.asarray(primary_labels, dtype=int).copy()
        assignment_levels = [
            {
                "cluster_level": "primary" if int(label) != -1 else "primary_noise",
                "noise_recluster_id": None,
            }
            for label in primary_labels
        ]
        noise_indices = np.where(primary_labels == -1)[0]
        noise_count = int(len(noise_indices))
        record_count = int(len(primary_labels))
        noise_ratio = noise_count / record_count if record_count else 0.0
        cfg = self.config.get("noise_recluster", {})
        enabled = bool(cfg.get("enabled", True))
        min_noise_count = int(cfg.get("min_noise_count", 100))
        min_noise_ratio = float(cfg.get("min_noise_ratio", 0.10))
        metadata: dict[str, Any] = {
            "enabled": enabled,
            "triggered": False,
            "min_noise_count": min_noise_count,
            "min_noise_ratio": min_noise_ratio,
            "primary_noise_count": noise_count,
            "primary_noise_ratio": noise_ratio,
            "input_count": 0,
            "cluster_count": 0,
            "residual_noise_count": noise_count,
            "residual_noise_ratio": noise_ratio,
            "backend": "",
            "strategy": str(cfg.get("strategy", "secondary_noise_only")),
            "reason": "",
        }
        if not enabled:
            metadata["reason"] = "disabled"
            return final_labels, assignment_levels, metadata
        if noise_count < min_noise_count:
            metadata["reason"] = "below_min_noise_count"
            return final_labels, assignment_levels, metadata
        if noise_ratio < min_noise_ratio:
            metadata["reason"] = "below_min_noise_ratio"
            return final_labels, assignment_levels, metadata

        recluster_config = self._noise_recluster_config(cfg)
        print(
            f"Clusterer {experiment_id}: noise-recluster triggered for "
            f"{noise_count} primary noise row(s) ({noise_ratio:.2%})",
            flush=True,
        )
        noise_labels, backend = self._umap_hdbscan(embeddings[noise_indices], recluster_config)
        primary_max = max((int(label) for label in primary_labels if int(label) != -1), default=-1)
        label_offset = primary_max + 1
        for local_index, original_index in enumerate(noise_indices):
            noise_label = int(noise_labels[local_index])
            assignment_levels[original_index]["noise_recluster_id"] = noise_label
            if noise_label == -1:
                final_labels[original_index] = -1
                assignment_levels[original_index]["cluster_level"] = "residual_noise"
            else:
                final_labels[original_index] = label_offset + noise_label
                assignment_levels[original_index]["cluster_level"] = "noise_recluster"

        residual_noise_count = int(np.sum(final_labels == -1))
        metadata.update(
            {
                "triggered": True,
                "reason": "threshold_met",
                "input_count": noise_count,
                "cluster_count": len({int(label) for label in noise_labels if int(label) != -1}),
                "residual_noise_count": residual_noise_count,
                "residual_noise_ratio": residual_noise_count / record_count if record_count else 0.0,
                "backend": backend,
                "label_offset": label_offset,
                "label_distribution": dict(Counter(int(label) for label in noise_labels)),
                "config": {
                    "umap": dict(recluster_config.get("umap", {})),
                    "hdbscan": dict(recluster_config.get("hdbscan", {})),
                },
            }
        )
        return final_labels, assignment_levels, metadata

    def _noise_recluster_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        recluster_config = deepcopy(self.config)
        recluster_config.setdefault("umap", {}).update(SECONDARY_NOISE_ONLY_DEFAULTS["umap"])
        recluster_config.setdefault("hdbscan", {}).update(SECONDARY_NOISE_ONLY_DEFAULTS["hdbscan"])
        recluster_config["umap"].update(cfg.get("umap", {}))
        recluster_config["hdbscan"].update(cfg.get("hdbscan", {}))
        return recluster_config

    @staticmethod
    def _fallback_labels(embeddings: np.ndarray) -> np.ndarray:
        labels = []
        for vector in embeddings:
            strongest = int(np.argmax(np.abs(vector)))
            labels.append(strongest % 3)
        return np.asarray(labels, dtype=int)

    @staticmethod
    def _centroid_distances_and_ranks(
        embeddings: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.zeros(len(labels), dtype=float)
        ranks = np.zeros(len(labels), dtype=int)
        for label in sorted(set(int(value) for value in labels)):
            indices = np.where(labels == label)[0]
            if len(indices) == 0:
                continue
            centroid = embeddings[indices].mean(axis=0)
            cluster_distances = np.linalg.norm(embeddings[indices] - centroid, axis=1)
            ordered_positions = np.argsort(cluster_distances, kind="stable")
            for rank, position in enumerate(ordered_positions, start=1):
                original_index = indices[position]
                distances[original_index] = cluster_distances[position]
                ranks[original_index] = rank
        return distances, ranks
