from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from industrial_hazard_analysis.models import (
    ChannelInput,
    ClusterAssignment,
    ClusterName,
    MISSING_CHANNEL_CLUSTER_ID,
    NOISE_CLUSTER_ID,
    SceneSplitRecord,
)
from industrial_hazard_analysis.providers.base import LLMProvider


FAILED_CLUSTER_NAME_PREFIX = "命名失败待复核-"


def is_failed_cluster_name(item: ClusterName) -> bool:
    return item.cluster_name.startswith(FAILED_CLUSTER_NAME_PREFIX)


class ClusterNamingService:
    def __init__(self, llm_provider: LLMProvider, *, max_workers: int = 1):
        self.llm_provider = llm_provider
        self.max_workers = max(1, int(max_workers))

    def name_clusters(
        self,
        experiment_id: str,
        assignments: list[ClusterAssignment],
        records_by_id: dict[str, SceneSplitRecord],
        channel_inputs: list[ChannelInput],
        channel_fields: tuple[str, ...],
        max_samples: int = 5,
    ) -> list[ClusterName]:
        grouped: dict[int, list[ClusterAssignment]] = defaultdict(list)
        for assignment in assignments:
            grouped[assignment.cluster_id].append(assignment)

        inputs_by_id = {item.record_id: item for item in channel_inputs}
        sorted_groups = sorted(grouped.items(), key=lambda item: item[0])
        names_by_cluster: dict[int, ClusterName] = {}
        pending_groups: list[tuple[int, list[ClusterAssignment]]] = []
        for index, (cluster_id, cluster_assignments) in enumerate(sorted_groups, start=1):
            print(
                f"Naming {experiment_id}: preparing cluster {index}/{len(sorted_groups)} "
                f"id={cluster_id}, rows={len(cluster_assignments)}",
                flush=True,
            )
            if cluster_id == MISSING_CHANNEL_CLUSTER_ID:
                names_by_cluster[cluster_id] = ClusterName(
                    experiment_id=experiment_id,
                    cluster_id=cluster_id,
                    cluster_name="字段缺失类",
                    explanation="当前语义通道字段为空或缺失，未参与 UMAP-HDBSCAN 聚类，统一归入字段缺失类。",
                    sample_count=len(cluster_assignments),
                )
                continue
            if cluster_id == NOISE_CLUSTER_ID:
                names_by_cluster[cluster_id] = ClusterName(
                    experiment_id=experiment_id,
                    cluster_id=cluster_id,
                    cluster_name="噪声类",
                    explanation="HDBSCAN 输出的噪声点整体归入噪声类，不进行语义命名。",
                    sample_count=len(cluster_assignments),
                )
                continue
            pending_groups.append((cluster_id, cluster_assignments))

        if self.max_workers == 1:
            for index, (cluster_id, cluster_assignments) in enumerate(pending_groups, start=1):
                names_by_cluster[cluster_id] = self._name_one_cluster(
                    experiment_id,
                    cluster_id,
                    cluster_assignments,
                    records_by_id,
                    inputs_by_id,
                    channel_fields,
                    max_samples,
                )
                print(
                    f"Naming {experiment_id}: completed {index}/{len(pending_groups)} "
                    f"normal cluster(s), id={cluster_id}",
                    flush=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._name_one_cluster,
                        experiment_id,
                        cluster_id,
                        cluster_assignments,
                        records_by_id,
                        inputs_by_id,
                        channel_fields,
                        max_samples,
                    ): cluster_id
                    for cluster_id, cluster_assignments in pending_groups
                }
                completed = 0
                for future in as_completed(futures):
                    cluster_id = futures[future]
                    names_by_cluster[cluster_id] = future.result()
                    completed += 1
                    print(
                        f"Naming {experiment_id}: completed {completed}/{len(pending_groups)} "
                        f"normal cluster(s), id={cluster_id}, workers={self.max_workers}",
                        flush=True,
                    )

        return [names_by_cluster[cluster_id] for cluster_id, _ in sorted_groups]

    def _name_one_cluster(
        self,
        experiment_id: str,
        cluster_id: int,
        cluster_assignments: list[ClusterAssignment],
        records_by_id: dict[str, SceneSplitRecord],
        inputs_by_id: dict[str, ChannelInput],
        channel_fields: tuple[str, ...],
        max_samples: int,
    ) -> ClusterName:

        ranked_assignments = sorted(
            cluster_assignments,
            key=lambda item: (
                item.representative_rank if item.representative_rank is not None else 10**9,
                item.record_id,
            ),
        )
        representative_assignments = ranked_assignments[:max_samples]
        cluster_records = [
            records_by_id[assignment.record_id]
            for assignment in cluster_assignments
            if assignment.record_id in records_by_id
        ]
        payload = {
            "experiment_id": experiment_id,
            "cluster_id": cluster_id,
            "channel_fields": list(channel_fields),
            "representative_samples": [
                {
                    "record_id": assignment.record_id,
                    "channel_text": inputs_by_id[assignment.record_id].input_text,
                    "distance_to_centroid": assignment.distance_to_centroid,
                    "representative_rank": assignment.representative_rank,
                }
                for assignment in representative_assignments
                if assignment.record_id in inputs_by_id
            ],
            "field_values": _collect_field_values(cluster_records, channel_fields),
        }
        try:
            response = self.llm_provider.name_cluster(payload)
            cluster_name = str(response["cluster_name"]).strip()
            explanation = str(response["explanation"]).strip()
        except Exception as exc:
            cluster_name = f"{FAILED_CLUSTER_NAME_PREFIX}{cluster_id}"
            explanation = (
                "LLM 未返回可解析的聚类命名 JSON，程序已保留该簇并继续后续命名；"
                f"请在人工审阅时复核。错误：{type(exc).__name__}: {exc}"
            )
        return ClusterName(
            experiment_id=experiment_id,
            cluster_id=cluster_id,
            cluster_name=cluster_name or f"待复核-{cluster_id}",
            explanation=explanation or "LLM 返回内容缺少解释字段，请人工复核。",
            sample_count=len(cluster_assignments),
        )


def _collect_field_values(
    records: list[SceneSplitRecord], channel_fields: tuple[str, ...]
) -> dict[str, list[str]]:
    field_values: dict[str, list[str]] = {}
    for field in channel_fields:
        values: list[str] = []
        for record in records:
            value = getattr(record, field)
            if isinstance(value, list):
                values.extend(str(item) for item in value if item)
            elif value:
                values.append(str(value))
        field_values[field] = values
    return field_values
