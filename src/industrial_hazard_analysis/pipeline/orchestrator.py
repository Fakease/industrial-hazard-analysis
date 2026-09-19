from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from industrial_hazard_analysis.clustering import ClusterNamingService, Clusterer
from industrial_hazard_analysis.models import RawHazardText, SceneSplitRecord, StructuredRecord
from industrial_hazard_analysis.pipeline.consistency import vote_extraction, vote_scene_split
from industrial_hazard_analysis.pipeline.llm_tasks import LLMTaskPolicy, build_llm_task_policies
from industrial_hazard_analysis.pipeline.validation import validation_summary
from industrial_hazard_analysis.providers.base import EmbeddingProvider, LLMProvider


class HazardAnalysisPipeline:
    def __init__(
        self,
        extractor_providers: list[LLMProvider],
        arbitrator_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        cluster_namer_provider: LLMProvider,
        config: dict[str, Any],
        extraction_providers: list[LLMProvider] | None = None,
        scene_split_providers: list[LLMProvider] | None = None,
        extraction_arbitrator_provider: LLMProvider | None = None,
        scene_split_arbitrator_provider: LLMProvider | None = None,
        llm_tasks: dict[str, LLMTaskPolicy] | None = None,
    ):
        self.extractor_providers = extractor_providers
        self.arbitrator_provider = arbitrator_provider
        self.embedding_provider = embedding_provider
        self.cluster_namer_provider = cluster_namer_provider
        self.config = config
        self.llm_tasks = llm_tasks or build_llm_task_policies(
            config.get("llm", {}),
            providers_by_name={provider.name: provider for provider in extractor_providers},
            default_worker_providers=extractor_providers,
            default_arbitrator_provider=arbitrator_provider,
            task_defaults={"structured_extraction", "scene_secondary_split"},
        )
        self.extraction_providers = extraction_providers or self.llm_task("structured_extraction").worker_providers
        self.scene_split_providers = scene_split_providers or self.llm_task("scene_secondary_split").worker_providers
        self.extraction_arbitrator_provider = (
            extraction_arbitrator_provider
            or self.llm_task("structured_extraction").arbitrator_provider
            or arbitrator_provider
        )
        self.scene_split_arbitrator_provider = (
            scene_split_arbitrator_provider
            or self.llm_task("scene_secondary_split").arbitrator_provider
            or arbitrator_provider
        )

    def run(
        self,
        raw_records: list[RawHazardText],
        output_dir: str | Path,
        source_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "HazardAnalysisPipeline.run() is a deprecated legacy full-run entry point. "
            "Use ExperimentStageRunner.run('all') or the scripts in scripts/ so outputs follow "
            "the 01_structured/02_clustering/03_cluster_naming/04_ablation layout."
        )

    def structure_records(self, raw_records: list[RawHazardText]) -> list[StructuredRecord]:
        structured: list[StructuredRecord] = []
        record_index = 1
        task = self.llm_task("structured_extraction")
        batch_size = task.batch_size
        batches = list(_batches(raw_records, batch_size))
        for batch_index, raw_batch in enumerate(batches, start=1):
            print(
                f"Appendix A extraction batch {batch_index}/{len(batches)}: "
                f"{len(raw_batch)} raw row(s)",
                flush=True,
            )
            candidates_by_provider = []
            candidates_by_provider = _run_provider_calls(
                self.extraction_providers,
                action_label="extracting",
                done_label="done",
                call=lambda provider: provider.extract_hazards(raw_batch),
            )
            for raw in raw_batch:
                candidates = [
                    candidate
                    for candidates_by_raw_id in candidates_by_provider
                    for candidate in candidates_by_raw_id.get(raw.record_id, [])
                ]
                candidate_groups = self._candidate_groups(candidates)
                if not candidate_groups:
                    print(
                        f"  raw_record_id {raw.record_id}: no extraction candidates, writing empty placeholder",
                        flush=True,
                    )
                    structured.append(
                        StructuredRecord(
                            record_id=f"S{record_index:05d}",
                            raw_record_id=raw.record_id,
                            raw_text=raw.raw_text,
                            finding="",
                            object=[],
                            scene="",
                            accepted_by="no_extraction_candidates",
                            arbitration_required=False,
                            metadata={
                                "candidate_count": 0,
                                "raw_candidate_count": 0,
                                "raw_record_output_count": 0,
                                "llm_batch_size": batch_size,
                                "warning": "no_extraction_candidates",
                            },
                        )
                    )
                    record_index += 1
                    continue
                if task.is_lightweight:
                    for candidate in candidates:
                        structured.append(
                            StructuredRecord(
                                record_id=f"S{record_index:05d}",
                                raw_record_id=raw.record_id,
                                raw_text=raw.raw_text,
                                finding=candidate.finding,
                                object=candidate.object,
                                scene=candidate.scene,
                                accepted_by=f"lightweight:{candidate.source_model}",
                                arbitration_required=False,
                                metadata={
                                    "candidate_count": 1,
                                    "raw_candidate_count": len(candidates),
                                    "raw_record_output_count": len(candidates),
                                    "llm_batch_size": batch_size,
                                    "llm_mode": task.mode,
                                    "llm_task": task.task_key,
                                },
                            )
                        )
                        record_index += 1
                    continue
                for group_index, candidate_group in enumerate(candidate_groups, start=1):
                    selected_candidates, accepted_by, arbitration_required = vote_extraction(
                        candidate_group,
                        lambda values, raw=raw: self.extraction_arbitrator_provider.arbitrate_extraction(raw, values),
                    )
                    if not selected_candidates:
                        print(
                            f"  raw_record_id {raw.record_id}: arbitration returned no valid records; "
                            "writing empty placeholder",
                            flush=True,
                        )
                        structured.append(
                            StructuredRecord(
                                record_id=f"S{record_index:05d}",
                                raw_record_id=raw.record_id,
                                raw_text=raw.raw_text,
                                finding="",
                                object=[],
                                scene="",
                                accepted_by=accepted_by,
                                arbitration_required=arbitration_required,
                                metadata={
                                    "candidate_count": len(candidate_group),
                                    "raw_candidate_count": len(candidates),
                                    "candidate_group_index": group_index,
                                    "candidate_group_count": len(candidate_groups),
                                    "arbitration_output_count": 0,
                                    "llm_batch_size": batch_size,
                                    "llm_mode": task.mode,
                                    "llm_task": task.task_key,
                                    "warning": "arbitration_returned_no_valid_records",
                                },
                            )
                        )
                        record_index += 1
                        continue
                    for arbitration_output_index, selected in enumerate(
                        selected_candidates, start=1
                    ):
                        structured.append(
                            StructuredRecord(
                                record_id=f"S{record_index:05d}",
                                raw_record_id=raw.record_id,
                                raw_text=raw.raw_text,
                                finding=selected.finding,
                                object=selected.object,
                                scene=selected.scene,
                                accepted_by=accepted_by,
                                arbitration_required=arbitration_required,
                                metadata={
                                    "candidate_count": len(candidate_group),
                                    "raw_candidate_count": len(candidates),
                                    "candidate_group_index": group_index,
                                    "candidate_group_count": len(candidate_groups),
                                    "arbitration_output_index": arbitration_output_index,
                                    "arbitration_output_count": len(selected_candidates),
                                    "llm_batch_size": batch_size,
                                    "llm_mode": task.mode,
                                    "llm_task": task.task_key,
                                },
                            )
                        )
                        record_index += 1
        return structured

    def split_scenes(self, records: list[StructuredRecord]) -> list[SceneSplitRecord]:
        split_records: list[SceneSplitRecord] = []
        task = self.llm_task("scene_secondary_split")
        batch_size = task.batch_size
        batches = list(_batches(records, batch_size))
        for batch_index, record_batch in enumerate(batches, start=1):
            print(
                f"Appendix C scene split batch {batch_index}/{len(batches)}: "
                f"{len(record_batch)} structured row(s)",
                flush=True,
            )
            active_batch = [
                record for record in record_batch if _structured_record_is_effective(record)
            ]
            candidates_by_provider = (
                _run_provider_calls(
                    self.scene_split_providers,
                    action_label="splitting scenes",
                    done_label="done",
                    call=lambda provider: provider.split_scenes(active_batch),
                )
                if active_batch
                else []
            )
            for record in record_batch:
                if not _structured_record_is_effective(record):
                    split_records.append(
                        SceneSplitRecord(
                            record_id=record.record_id,
                            raw_record_id=record.raw_record_id,
                            raw_text=record.raw_text,
                            finding=record.finding,
                            object=record.object,
                            scene=record.scene,
                            loc_detail="",
                            risk_scene="",
                            accepted_by=record.accepted_by,
                            arbitration_required=record.arbitration_required,
                            metadata=dict(record.metadata),
                        )
                    )
                    continue
                candidates = [
                    candidate
                    for candidates_by_record_id in candidates_by_provider
                    if (candidate := candidates_by_record_id.get(record.record_id)) is not None
                ]
                if not candidates:
                    print(
                        f"  record_id {record.record_id}: no scene split candidates, writing empty split",
                        flush=True,
                    )
                    split_records.append(
                        SceneSplitRecord(
                            record_id=record.record_id,
                            raw_record_id=record.raw_record_id,
                            raw_text=record.raw_text,
                            finding=record.finding,
                            object=record.object,
                            scene=record.scene,
                            loc_detail="",
                            risk_scene="",
                            accepted_by="no_scene_split_candidates",
                            arbitration_required=False,
                            metadata={
                                "candidate_count": 0,
                                "llm_batch_size": batch_size,
                                "warning": "no_scene_split_candidates",
                            },
                        )
                    )
                    continue
                if task.is_lightweight:
                    selected = candidates[0]
                    split_records.append(
                        SceneSplitRecord(
                            record_id=record.record_id,
                            raw_record_id=record.raw_record_id,
                            raw_text=record.raw_text,
                            finding=record.finding,
                            object=record.object,
                            scene=record.scene,
                            loc_detail=selected.loc_detail,
                            risk_scene=selected.risk_scene,
                            accepted_by=f"lightweight:{selected.source_model}",
                            arbitration_required=False,
                            metadata={
                                "candidate_count": len(candidates),
                                "llm_batch_size": batch_size,
                                "llm_mode": task.mode,
                                "llm_task": task.task_key,
                            },
                        )
                    )
                    continue
                selected, accepted_by, arbitration_required = vote_scene_split(
                    candidates,
                    lambda values, record=record: self.scene_split_arbitrator_provider.arbitrate_scene_split(record, values),
                )
                split_records.append(
                    SceneSplitRecord(
                        record_id=record.record_id,
                        raw_record_id=record.raw_record_id,
                        raw_text=record.raw_text,
                        finding=record.finding,
                        object=record.object,
                        scene=record.scene,
                        loc_detail=selected.loc_detail,
                        risk_scene=selected.risk_scene,
                        accepted_by=accepted_by,
                        arbitration_required=arbitration_required,
                        metadata={
                            "candidate_count": len(candidates),
                            "llm_batch_size": batch_size,
                            "llm_mode": task.mode,
                            "llm_task": task.task_key,
                        },
                    )
                )
        return split_records

    def run_ablation_experiments(
        self,
        raw_records: list[RawHazardText],
        full_records: list[SceneSplitRecord],
        output_dir: Path,
        clusterer: Clusterer,
        namer: ClusterNamingService,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "HazardAnalysisPipeline.run_ablation_experiments() is deprecated. "
            "Use ExperimentStageRunner.run('ablation-extract'), "
            "run('ablation-cluster'), and run('ablation-name') so outputs follow "
            "04_ablation/ablation_results.xlsx plus 04_ablation/_machine/."
        )

    def _extractor_provider_by_name(self, name: str) -> LLMProvider:
        for provider in self.extractor_providers:
            if provider.name == name:
                return provider
        available = [provider.name for provider in self.extractor_providers]
        raise ValueError(f"Unknown extractor provider {name!r}. Available: {available}")

    def _llm_batch_size(self, key: str) -> int:
        task_key_by_legacy_key = {
            "extraction_batch_size": "structured_extraction",
            "scene_split_batch_size": "scene_secondary_split",
        }
        task_key = task_key_by_legacy_key.get(key)
        value = self.llm_task(task_key).batch_size if task_key else int(self.config.get("llm", {}).get(key, 1))
        if value < 1:
            raise ValueError(f"llm.{key} must be >= 1")
        return value

    def _llm_task_mode(self, task_key: str) -> str:
        return self.llm_task(task_key).mode

    def llm_task(self, task_key: str | None) -> LLMTaskPolicy:
        if not task_key:
            raise ValueError("task_key is required")
        if task_key in self.llm_tasks:
            return self.llm_tasks[task_key]
        self.llm_tasks[task_key] = build_llm_task_policies(
            self.config.get("llm", {}),
            providers_by_name={provider.name: provider for provider in self.extractor_providers},
            default_worker_providers=self.extractor_providers,
            default_arbitrator_provider=self.arbitrator_provider,
            task_defaults={task_key},
        )[task_key]
        return self.llm_tasks[task_key]

    @staticmethod
    def _candidate_groups(candidates):
        if not candidates:
            return []
        provider_count = len({candidate.source_model for candidate in candidates})
        if len(candidates) <= provider_count:
            return [candidates]

        groups_by_finding: dict[str, list] = {}
        for candidate in candidates:
            key = candidate.finding.strip()
            groups_by_finding.setdefault(key, []).append(candidate)
        return list(groups_by_finding.values())

    def write_run_manifest(
        self,
        output_dir: Path,
        raw_records: list[RawHazardText],
        structured_records: list[StructuredRecord],
        scene_split_records: list[SceneSplitRecord],
        clustering_metadata: dict[str, Any],
        ablation_result: dict[str, Any],
        validation_issues,
        source_info: dict[str, Any] | None,
    ) -> None:
        from datetime import datetime

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_info": source_info or {},
            "counts": {
                "raw_records": len(raw_records),
                "structured_records": len(structured_records),
                "scene_split_records": len(scene_split_records),
            },
            "validation": validation_summary(validation_issues),
            "main_experiments": sorted(clustering_metadata),
            "ablation_experiments": sorted(ablation_result.get("metadata", {})),
            "config": self.config,
        }
        with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def _batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _structured_record_is_effective(record: StructuredRecord) -> bool:
    return bool(
        record.finding.strip()
        or record.object
        or record.scene.strip()
    )


def _run_provider_calls(
    providers: list[LLMProvider],
    *,
    action_label: str,
    done_label: str,
    call,
) -> list[Any]:
    if len(providers) <= 1:
        results = []
        for provider in providers:
            print(f"  provider {provider.name}: {action_label}...", flush=True)
            results.append(call(provider))
            print(f"  provider {provider.name}: {done_label}", flush=True)
        return results

    print(
        "  provider parallel: "
        + ", ".join(provider.name for provider in providers)
        + f" ({action_label})",
        flush=True,
    )
    results_by_index: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        future_to_provider = {
            executor.submit(call, provider): (index, provider)
            for index, provider in enumerate(providers)
        }
        for future in as_completed(future_to_provider):
            index, provider = future_to_provider[future]
            results_by_index[index] = future.result()
            print(f"  provider {provider.name}: {done_label}", flush=True)
    return [results_by_index[index] for index in range(len(providers))]


# Backward-compatible alias for older project notes and scripts.
MockPipeline = HazardAnalysisPipeline
