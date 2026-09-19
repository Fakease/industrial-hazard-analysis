from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_hazard_analysis.clustering import (
    ClusterNamingService,
    Clusterer,
    is_failed_cluster_name,
)
from industrial_hazard_analysis.io import read_raw_texts, write_dataclass_csv
from industrial_hazard_analysis.models import (
    ChannelConfig,
    ChannelInput,
    ClusterAssignment,
    ClusterName,
    MISSING_CHANNEL_CLUSTER_ID,
    RawHazardText,
    StructuredRecord,
    SceneSplitRecord,
)
from industrial_hazard_analysis.pipeline.ablation import (
    generate_ablation_inputs,
    generate_no_consistency_records_by_provider,
    prepare_no_consistency_record_level,
)
from industrial_hazard_analysis.pipeline.channeling import (
    generate_baseline_inputs,
    generate_channel_inputs,
    load_channel_configs,
)
from industrial_hazard_analysis.pipeline.data_revision import (
    build_canonical_raw_text_groups,
    canonical_group_rows,
    consolidate_units_by_raw_text,
    filter_and_deduplicate_units,
    is_effective_unit,
    to_structured_records,
)
from industrial_hazard_analysis.pipeline.orchestrator import HazardAnalysisPipeline
from industrial_hazard_analysis.pipeline.validation import (
    validate_scene_split_records,
    validate_structured_records,
)


MAIN_STAGES = {"llm-extract", "unit-dedupe", "cluster", "name"}
ABLATION_STAGES = {"ablation-extract", "ablation-cluster", "ablation-name"}
REVIEW_STAGES = {"review"}
STAGES = {"all", *MAIN_STAGES, *ABLATION_STAGES, *REVIEW_STAGES}


class ExperimentStageRunner:
    def __init__(
        self,
        pipeline: HazardAnalysisPipeline,
        output_dir: str | Path,
        *,
        source_output_dir: str | Path | None = None,
        input_path: str | Path | None = None,
        input_options: dict[str, Any] | None = None,
        limit: int | None = None,
        experiment_ids: list[str] | None = None,
        ablation_common_records_path: str | Path | None = None,
        deduplicate_raw_text: bool = True,
    ):
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.source_output_dir = Path(source_output_dir) if source_output_dir is not None else self.output_dir
        self.input_path = Path(input_path) if input_path is not None else None
        self.input_options = input_options or {}
        self.limit = limit
        self.experiment_ids = experiment_ids
        self.deduplicate_raw_text = deduplicate_raw_text
        self.ablation_common_records_path = (
            Path(ablation_common_records_path) if ablation_common_records_path is not None else None
        )

    def run(self, stage: str) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Available: {sorted(STAGES)}")
        if stage == "all":
            return self.run_all()
        if stage == "llm-extract":
            return self.run_main_extract()
        if stage == "unit-dedupe":
            return self.run_unit_dedupe()
        if stage == "cluster":
            return self.run_main_cluster()
        if stage == "name":
            return self.run_main_name()
        if stage == "review":
            return self.run_review_outputs()
        if stage == "ablation-extract":
            return self.run_ablation_extract()
        if stage == "ablation-cluster":
            return self.run_ablation_cluster()
        if stage == "ablation-name":
            return self.run_ablation_name()
        raise AssertionError(stage)

    def run_all(self) -> dict[str, Any]:
        extract_result = self.run_main_extract()
        unit_dedupe_result = self.run_unit_dedupe()
        cluster_result = self.run_main_cluster()
        name_result = self.run_main_name()
        return {
            "stage": "all",
            "llm_extract": extract_result,
            "unit_dedupe": unit_dedupe_result,
            "cluster": cluster_result,
            "name": name_result,
        }

    def run_main_extract(self) -> dict[str, Any]:
        raw_records = self.read_raw_records()
        self.structured_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Stage llm-extract: loaded {len(raw_records)} unique raw row(s) "
            "after pre-extraction exact-text deduplication",
            flush=True,
        )
        structured_records = self.pipeline.structure_records(raw_records)
        print(f"Stage llm-extract: produced {len(structured_records)} structured row(s)", flush=True)
        scene_split_records = self.pipeline.split_scenes(structured_records)
        print(f"Stage llm-extract: produced {len(scene_split_records)} scene split row(s)", flush=True)
        validation_issues = validate_structured_records(structured_records)
        validation_issues.extend(validate_scene_split_records(scene_split_records))
        write_dataclass_csv(structured_records, self.structured_machine_dir / "structured_records.csv")
        write_dataclass_csv(scene_split_records, self.structured_machine_dir / "scene_split_records.csv")
        write_dataclass_csv(validation_issues, self.structured_machine_dir / "validation_report.csv")
        write_excel_workbook(
            self.structured_workbook_path,
            {
                "structured_records": structured_records,
                "scene_split_records": scene_split_records,
                "validation_report": validation_issues,
            },
        )
        return {
            "stage": "llm-extract",
            "raw_records": len(raw_records),
            "structured_records": len(structured_records),
            "scene_split_records": len(scene_split_records),
        }

    def run_unit_dedupe(self) -> dict[str, Any]:
        structured_records = self.read_structured_records(original=True)
        scene_split_records = self.read_scene_split_records(original=True)
        source_raw_records = self.read_source_raw_records()
        raw_groups, _ = build_canonical_raw_text_groups(source_raw_records)
        revised_units = filter_and_deduplicate_units(
            raw_groups,
            scene_split_records,
            source_label="A2_consensus",
        )
        unit_scene_records = list(revised_units.kept_records)
        unit_structured_records = to_structured_records(unit_scene_records)
        record_level_scene_records = consolidate_units_by_raw_text(
            raw_groups,
            unit_scene_records,
            source_label="A2_consensus",
        )
        record_level_structured_records = to_structured_records(record_level_scene_records)
        write_dataclass_csv(
            unit_structured_records,
            self.structured_machine_dir / "structured_units_dedup.csv",
        )
        write_dataclass_csv(
            unit_scene_records,
            self.structured_machine_dir / "scene_split_units_dedup.csv",
        )
        write_dataclass_csv(
            record_level_structured_records,
            self.structured_machine_dir / "structured_records_dedup.csv",
        )
        write_dataclass_csv(
            record_level_scene_records,
            self.structured_machine_dir / "scene_split_records_dedup.csv",
        )
        write_dataclass_csv(
            revised_units.audit_rows,
            self.structured_machine_dir / "deduplication_map.csv",
        )
        workbook_sheets = {
            "structured_records": structured_records,
            "scene_split_records": scene_split_records,
            "structured_units_dedup": unit_structured_records,
            "scene_split_units_dedup": unit_scene_records,
            "structured_records_dedup": record_level_structured_records,
            "scene_split_records_dedup": record_level_scene_records,
            "deduplication_map": revised_units.audit_rows,
        }
        validation_path = self.first_existing_structured_path("validation_report.csv")
        if validation_path.exists():
            workbook_sheets["validation_report"] = read_table(validation_path).to_dict(orient="records")
        write_excel_workbook(self.structured_workbook_path, workbook_sheets)
        self.write_dedupe_stale_markers()
        return {
            "stage": "unit-dedupe",
            "dedupe_key": "whitespace-normalized raw_text + finding exact match",
            "input_unit_count": len(scene_split_records),
            "kept_unit_count": len(unit_scene_records),
            "dropped_unit_count": len(scene_split_records) - len(unit_scene_records),
            "unit_group_count": len(unit_scene_records),
            "unique_raw_text_count": len(record_level_scene_records),
        }

    def run_main_cluster(self) -> dict[str, Any]:
        baseline_records = self.read_scene_split_records()
        unit_records = self.read_scene_split_units()
        channel_inputs, _ = self.build_main_channel_inputs(baseline_records, unit_records)
        all_channel_inputs, _ = self.build_main_channel_inputs(
            baseline_records,
            unit_records,
            apply_experiment_filter=False,
        )
        records_by_id = merge_records_by_id(baseline_records, unit_records)
        channel_dir = self.clustering_machine_dir / "channel_inputs"
        for experiment_id, inputs in channel_inputs.items():
            write_dataclass_csv(inputs, channel_dir / f"{experiment_id}.csv")

        clusterer = self.clusterer()
        clustering_metadata = self.read_clustering_metadata()
        assignment_sheets: dict[str, list[Any]] = {}
        review_sheets: dict[str, list[dict[str, Any]]] = {}
        for index, (experiment_id, inputs) in enumerate(channel_inputs.items(), start=1):
            print(
                f"Stage cluster: experiment {index}/{len(channel_inputs)} {experiment_id}, "
                f"{len(inputs)} input row(s)",
                flush=True,
            )
            assignments, metadata = self.cluster_nonempty_channel_inputs(
                clusterer,
                experiment_id,
                inputs,
            )
            clustering_metadata[experiment_id] = metadata
            write_dataclass_csv(
                assignments,
                self.clustering_machine_dir / "clustering_results" / f"{experiment_id}_assignments.csv",
            )
            review_rows = self.write_assignment_review(experiment_id, assignments, records_by_id, inputs)
            assignment_sheets[f"{experiment_id}_assignments"] = assignments
            review_sheets[f"{experiment_id}_review"] = review_rows
            print(
                f"Stage cluster: {experiment_id} done, "
                f"{metadata.get('cluster_count', 0)} cluster(s), {metadata.get('noise_count', 0)} noise row(s)",
                flush=True,
            )

        self.clustering_dir.mkdir(parents=True, exist_ok=True)
        with (self.clustering_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(clustering_metadata, f, ensure_ascii=False, indent=2)
        self.write_clustering_workbook_from_existing(
            clustering_metadata,
            all_channel_inputs,
            records_by_id,
            assignment_overrides=assignment_sheets,
            review_overrides=review_sheets,
        )
        return {"stage": "cluster", "experiments": len(channel_inputs)}

    def run_main_name(self) -> dict[str, Any]:
        baseline_records = self.read_scene_split_records()
        unit_records = self.read_scene_split_units()
        channel_inputs, experiment_configs = self.build_main_channel_inputs(
            baseline_records,
            unit_records,
        )
        all_channel_inputs, _ = self.build_main_channel_inputs(
            baseline_records,
            unit_records,
            apply_experiment_filter=False,
        )
        records_by_id = merge_records_by_id(baseline_records, unit_records)
        namer = self.namer()
        failed_cluster_names = 0

        for index, (experiment_id, inputs) in enumerate(channel_inputs.items(), start=1):
            print(f"Stage name: experiment {index}/{len(channel_inputs)} {experiment_id}", flush=True)
            assignments = self.read_cluster_assignments(self.cluster_assignments_path(experiment_id))
            ensure_assignments_match_records(experiment_id, assignments, records_by_id)
            experiment_cluster_names = namer.name_clusters(
                experiment_id,
                assignments,
                records_by_id,
                inputs,
                experiment_configs[experiment_id].fields,
            )
            write_dataclass_csv(
                experiment_cluster_names,
                self.naming_machine_dir / f"{experiment_id}_cluster_names.csv",
            )
            self.write_cluster_name_review(
                experiment_id,
                experiment_cluster_names,
                assignments,
                records_by_id,
                inputs,
            )
            experiment_failures = sum(
                is_failed_cluster_name(item) for item in experiment_cluster_names
            )
            failed_cluster_names += experiment_failures
            if experiment_failures:
                print(
                    f"Stage name: {experiment_id} partial; "
                    f"{experiment_failures} cluster name(s) require retry or review",
                    flush=True,
                )
            else:
                print(f"Stage name: {experiment_id} done", flush=True)
        all_cluster_names, all_review_rows = self.write_naming_workbook_from_existing(
            all_channel_inputs,
            records_by_id,
        )
        return {
            "stage": "name",
            "status": "partial" if failed_cluster_names else "completed",
            "cluster_names": len(all_cluster_names),
            "failed_cluster_names": failed_cluster_names,
        }

    def run_review_outputs(self) -> dict[str, Any]:
        structured_workbook = self.write_structured_workbook_from_existing()
        baseline_records = self.read_scene_split_records()
        unit_records = self.read_scene_split_units()
        channel_inputs, _ = self.build_main_channel_inputs(baseline_records, unit_records)
        records_by_id = merge_records_by_id(baseline_records, unit_records)

        assignment_reviews = 0
        name_reviews = 0
        all_name_review_rows = []
        assignment_sheets: dict[str, list[Any]] = {}
        review_sheets: dict[str, list[dict[str, Any]]] = {}
        for experiment_id, inputs in channel_inputs.items():
            assignments_path = self.cluster_assignments_path(experiment_id)
            if not assignments_path.exists():
                continue
            assignments = self.read_cluster_assignments(assignments_path)
            ensure_assignments_match_records(experiment_id, assignments, records_by_id)
            assignment_review_rows_for_experiment = self.write_assignment_review(
                experiment_id,
                assignments,
                records_by_id,
                inputs,
            )
            assignment_sheets[f"{experiment_id}_assignments"] = assignments
            review_sheets[f"{experiment_id}_review"] = assignment_review_rows_for_experiment
            assignment_reviews += 1

            cluster_names_path = self.cluster_names_path(experiment_id)
            if not cluster_names_path.exists():
                continue
            cluster_names = read_cluster_names(cluster_names_path)
            review_rows = self.write_cluster_name_review(
                experiment_id,
                cluster_names,
                assignments,
                records_by_id,
                inputs,
            )
            all_name_review_rows.extend(review_rows)
            name_reviews += 1

        if all_name_review_rows:
            write_dataclass_csv(all_name_review_rows, self.naming_machine_dir / "cluster_name_review.csv")
            all_cluster_names = []
            for experiment_id in channel_inputs:
                cluster_names_path = self.cluster_names_path(experiment_id)
                if cluster_names_path.exists():
                    all_cluster_names.extend(read_cluster_names(cluster_names_path))
            write_excel_workbook(
                self.naming_dir / "cluster_names.xlsx",
                {
                    "cluster_names": all_cluster_names,
                    "review": all_name_review_rows,
                },
            )
        if assignment_sheets:
            metadata_path = self.clustering_dir / "run_metadata.json"
            clustering_metadata = {}
            if metadata_path.exists():
                clustering_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.write_clustering_workbook(clustering_metadata, assignment_sheets, review_sheets)
        return {
            "stage": "review",
            "structured_workbook": structured_workbook,
            "assignment_reviews": assignment_reviews,
            "name_reviews": name_reviews,
        }

    def run_ablation_extract(self) -> dict[str, Any]:
        raw_records = self.read_raw_records()
        provider_names = self.ablation_provider_names()
        providers = [self.pipeline._extractor_provider_by_name(name) for name in provider_names]
        print(
            f"Stage ablation-extract: loaded {len(raw_records)} raw row(s), "
            f"{len(providers)} provider(s)",
            flush=True,
        )
        records_by_provider = generate_no_consistency_records_by_provider(
            raw_records,
            providers,
            batch_size=self.pipeline._llm_batch_size("extraction_batch_size"),
        )
        unit_counts: dict[str, int] = {}
        record_counts: dict[str, int] = {}
        for provider_name, records in records_by_provider.items():
            unit_records, record_level_records = prepare_no_consistency_record_level(
                raw_records,
                records,
                provider_name=provider_name,
            )
            write_dataclass_csv(
                unit_records,
                self.ablation_machine_dir
                / "ablation_records"
                / f"no_consistency_units_{provider_name}.csv",
            )
            write_dataclass_csv(
                record_level_records,
                self.ablation_machine_dir / "ablation_records" / f"no_consistency_records_{provider_name}.csv",
            )
            unit_counts[provider_name] = len(unit_records)
            record_counts[provider_name] = len(record_level_records)
        self.write_ablation_workbook_from_existing()
        return {
            "stage": "ablation-extract",
            "providers": len(records_by_provider),
            "unit_records": unit_counts,
            "record_level_records": record_counts,
        }

    def run_ablation_cluster(self) -> dict[str, Any]:
        full_units = self.read_scene_split_units()
        no_consistency_units = self.read_no_consistency_units_by_provider()
        full_units, no_consistency_units, ablation_parent_ids = self.filter_ablation_record_universe(
            full_units,
            no_consistency_units,
        )
        inputs_by_experiment, configs_by_experiment, records_by_experiment = generate_ablation_inputs(
            self.pipeline.config,
            full_units,
            no_consistency_units,
        )
        inputs_by_experiment, configs_by_experiment, records_by_experiment = self.filter_ablation_experiments(
            inputs_by_experiment,
            configs_by_experiment,
            records_by_experiment,
        )
        input_dir = self.ablation_machine_dir / "ablation_channel_inputs"
        for experiment_id, inputs in inputs_by_experiment.items():
            write_dataclass_csv(inputs, input_dir / f"{experiment_id}.csv")

        clusterer = self.clusterer()
        metadata_by_experiment = self.read_ablation_metadata()
        clustering_dir = self.ablation_machine_dir / "ablation_clustering_results"
        assignment_sheets: dict[str, list[Any]] = {}
        review_sheets: dict[str, list[dict[str, Any]]] = {}
        for index, (experiment_id, inputs) in enumerate(inputs_by_experiment.items(), start=1):
            print(
                f"Stage ablation-cluster: experiment {index}/{len(inputs_by_experiment)} "
                f"{experiment_id}, {len(inputs)} input row(s)",
                flush=True,
            )
            records_by_id = {
                record.record_id: record
                for record in records_by_experiment[experiment_id]
            }
            assignments, metadata = self.cluster_nonempty_channel_inputs(
                clusterer,
                experiment_id,
                inputs,
            )
            observed_parent_ids = {
                record.raw_record_id
                for record in records_by_experiment[experiment_id]
            }
            unexpected_parent_ids = observed_parent_ids - ablation_parent_ids
            if unexpected_parent_ids:
                raise ValueError(
                    f"{experiment_id} contains {len(unexpected_parent_ids)} parent record(s) "
                    "outside the ablation parent universe"
                )
            metadata = add_parent_coverage_metadata(
                metadata,
                parent_record_count=len(ablation_parent_ids),
                covered_parent_record_count=len(observed_parent_ids),
            )
            metadata_by_experiment[experiment_id] = metadata
            write_dataclass_csv(assignments, clustering_dir / f"{experiment_id}_assignments.csv")
            review_rows = assignment_review_rows(experiment_id, assignments, records_by_id, inputs)
            write_dataclass_csv(review_rows, clustering_dir / f"{experiment_id}_assignments_review.csv")
            assignment_sheets[f"{experiment_id}_assignments"] = assignments
            review_sheets[f"{experiment_id}_review"] = review_rows
            print(f"Stage ablation-cluster: {experiment_id} done", flush=True)

        self.ablation_dir.mkdir(parents=True, exist_ok=True)
        with (self.ablation_machine_dir / "ablation_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata_by_experiment, f, ensure_ascii=False, indent=2)
        self.write_ablation_workbook_from_existing(
            metadata_by_experiment,
            inputs_by_experiment,
            records_by_experiment,
            assignment_overrides=assignment_sheets,
            review_overrides=review_sheets,
        )
        return {
            "stage": "ablation-cluster",
            "experiments": len(inputs_by_experiment),
            "parent_record_count": len(ablation_parent_ids),
        }

    def run_ablation_name(self) -> dict[str, Any]:
        full_units = self.read_scene_split_units()
        no_consistency_units = self.read_no_consistency_units_by_provider()
        full_units, no_consistency_units, ablation_parent_ids = self.filter_ablation_record_universe(
            full_units,
            no_consistency_units,
        )
        inputs_by_experiment, configs_by_experiment, records_by_experiment = generate_ablation_inputs(
            self.pipeline.config,
            full_units,
            no_consistency_units,
        )
        inputs_by_experiment, configs_by_experiment, records_by_experiment = self.filter_ablation_experiments(
            inputs_by_experiment,
            configs_by_experiment,
            records_by_experiment,
        )

        namer = self.namer()
        all_cluster_names = []
        for index, (experiment_id, inputs) in enumerate(inputs_by_experiment.items(), start=1):
            print(
                f"Stage ablation-name: experiment {index}/{len(inputs_by_experiment)} {experiment_id}",
                flush=True,
            )
            assignments = self.read_cluster_assignments(self.ablation_assignments_path(experiment_id))
            records_by_id = {
                record.record_id: record
                for record in records_by_experiment[experiment_id]
            }
            experiment_cluster_names = namer.name_clusters(
                experiment_id,
                assignments,
                records_by_id,
                inputs,
                configs_by_experiment[experiment_id].fields,
            )
            all_cluster_names.extend(experiment_cluster_names)
            write_dataclass_csv(
                experiment_cluster_names,
                self.ablation_machine_dir / f"{experiment_id}_cluster_names.csv",
            )
            write_dataclass_csv(
                cluster_name_review_rows(
                    experiment_id,
                    experiment_cluster_names,
                    assignments,
                    records_by_id,
                    inputs,
                ),
                self.ablation_machine_dir / f"{experiment_id}_cluster_name_review.csv",
            )
            print(f"Stage ablation-name: {experiment_id} done", flush=True)
        all_cluster_names, _ = self.write_ablation_workbook_from_existing(
            inputs_by_experiment=inputs_by_experiment,
            records_by_experiment=records_by_experiment,
        )
        return {
            "stage": "ablation-name",
            "cluster_names": len(all_cluster_names),
            "parent_record_count": len(ablation_parent_ids),
        }

    def filter_ablation_record_universe(
        self,
        full_records: list[SceneSplitRecord],
        no_consistency_records: dict[str, list[SceneSplitRecord]],
    ) -> tuple[list[SceneSplitRecord], dict[str, list[SceneSplitRecord]], set[str]]:
        parent_universe_ids = {
            record.raw_record_id
            for record in self.read_scene_split_records()
        }
        if not parent_universe_ids:
            raise SystemExit("Unit-level ablation has no parent records")

        full_parent_ids = {record.raw_record_id for record in full_records}
        provider_parent_ids = {
            provider: {record.raw_record_id for record in records}
            for provider, records in no_consistency_records.items()
        }

        selected_by_file = set(parent_universe_ids)
        analysis_mode = "full_parent_universe"
        if self.ablation_common_records_path is not None:
            if not self.ablation_common_records_path.exists():
                raise SystemExit(
                    f"Missing ablation common-record file: {self.ablation_common_records_path}"
                )
            frame = read_table(self.ablation_common_records_path)
            required_columns = {"canonical_raw_record_id", "included_in_common_ablation"}
            missing_columns = required_columns - set(frame.columns)
            if missing_columns:
                raise SystemExit(
                    "Ablation common-record file is missing column(s): "
                    + ", ".join(sorted(missing_columns))
                )
            file_ids = {
                str(value)
                for value in frame["canonical_raw_record_id"].tolist()
            }
            unexpected_file_ids = file_ids - parent_universe_ids
            if unexpected_file_ids:
                raise SystemExit(
                    "Ablation common-record file contains unknown parent id(s): "
                    + ", ".join(sorted(unexpected_file_ids)[:10])
                )
            selected_by_file = {
                str(row["canonical_raw_record_id"])
                for row in frame.to_dict(orient="records")
                if _bool(row["included_in_common_ablation"])
            }
            if not selected_by_file:
                raise SystemExit("Ablation common-record file selects zero records")
            analysis_mode = "paired_complete_case"

        if analysis_mode == "paired_complete_case":
            ablation_parent_ids = set(selected_by_file)
            ablation_parent_ids.intersection_update(full_parent_ids)
            for parent_ids in provider_parent_ids.values():
                ablation_parent_ids.intersection_update(parent_ids)
            if not ablation_parent_ids:
                raise SystemExit("Unit-level ablation has no common parent records")
        else:
            # Retain the complete deduplicated parent-record universe. A branch that
            # produces no effective unit is represented in the coverage audit and
            # denominator instead of being removed before the ablation comparison.
            ablation_parent_ids = set(parent_universe_ids)

        filtered_full = [
            record for record in full_records if record.raw_record_id in ablation_parent_ids
        ]
        filtered_by_provider = {
            provider: [
                record
                for record in records
                if record.raw_record_id in ablation_parent_ids
            ]
            for provider, records in no_consistency_records.items()
        }

        audit_rows = []
        for raw_id in sorted(parent_universe_ids):
            missing_branches = []
            if raw_id not in full_parent_ids:
                missing_branches.append("multi")
            for provider, parent_ids in provider_parent_ids.items():
                if raw_id not in parent_ids:
                    missing_branches.append(provider)
            included = raw_id in ablation_parent_ids
            if included:
                exclusion_reason = ""
            elif raw_id not in selected_by_file:
                exclusion_reason = "not_selected_by_paired_parent_file"
            else:
                exclusion_reason = "no_effective_unit_in_at_least_one_branch"
            row = {
                "canonical_raw_record_id": raw_id,
                "analysis_mode": analysis_mode,
                "included_in_unit_level_ablation": included,
                "multi_has_effective_unit": raw_id in full_parent_ids,
                "missing_effective_branches": ";".join(missing_branches),
                "exclusion_reason": exclusion_reason,
            }
            for provider, parent_ids in provider_parent_ids.items():
                row[f"{provider}_has_effective_unit"] = raw_id in parent_ids
            audit_rows.append(row)

        audit_path = self.ablation_machine_dir / "unit_level_ablation_parents.csv"
        write_dataclass_csv(audit_rows, audit_path)
        if analysis_mode == "paired_complete_case":
            write_dataclass_csv(
                audit_rows,
                self.ablation_machine_dir / "unit_level_common_ablation_parents.csv",
            )
        return filtered_full, filtered_by_provider, ablation_parent_ids

    def filter_ablation_experiments(
        self,
        inputs_by_experiment: dict[str, list[ChannelInput]],
        configs_by_experiment: dict[str, ChannelConfig],
        records_by_experiment: dict[str, list[SceneSplitRecord]],
    ) -> tuple[
        dict[str, list[ChannelInput]],
        dict[str, ChannelConfig],
        dict[str, list[SceneSplitRecord]],
    ]:
        if not self.experiment_ids:
            return inputs_by_experiment, configs_by_experiment, records_by_experiment
        missing = [
            experiment_id
            for experiment_id in self.experiment_ids
            if experiment_id not in inputs_by_experiment
        ]
        if missing:
            raise SystemExit(
                "Unknown ablation experiment id(s): "
                + ", ".join(missing)
                + ". Available: "
                + ", ".join(inputs_by_experiment)
            )
        selected = list(self.experiment_ids)
        return (
            {experiment_id: inputs_by_experiment[experiment_id] for experiment_id in selected},
            {experiment_id: configs_by_experiment[experiment_id] for experiment_id in selected},
            {experiment_id: records_by_experiment[experiment_id] for experiment_id in selected},
        )

    def read_raw_records(self) -> list[RawHazardText]:
        records = self.read_source_raw_records()
        if not self.deduplicate_raw_text:
            return records

        groups, audit_rows = build_canonical_raw_text_groups(records)
        records_by_id = {record.record_id: record for record in records}
        canonical_records = [records_by_id[group.canonical_raw_record_id] for group in groups]
        self.data_revision_dir.mkdir(parents=True, exist_ok=True)
        write_dataclass_csv(canonical_group_rows(groups), self.data_revision_dir / "canonical_raw_records.csv")
        write_dataclass_csv(audit_rows, self.data_revision_dir / "raw_text_dedup_map.csv")
        (self.data_revision_dir / "source_preprocessing_summary.json").write_text(
            json.dumps(
                {
                    "source_record_count": len(records),
                    "unique_raw_text_count": len(canonical_records),
                    "duplicate_source_row_count": len(records) - len(canonical_records),
                    "dedupe_key": "whitespace-normalized raw_text exact match",
                    "canonical_rule": "first source row in original order",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return canonical_records

    def read_source_raw_records(self) -> list[RawHazardText]:
        if self.input_path is None:
            raise SystemExit("Input path is required for this stage.")
        records = read_raw_texts(self.input_path, **self.input_options)
        if self.limit is not None:
            records = records[: self.limit]
        return records

    def read_structured_records(self, *, original: bool = False) -> list[StructuredRecord]:
        path = self.structured_records_path(original=original)
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run the prerequisite extraction stage first.")
        sheet_name = None
        if path.suffix.lower() in {".xlsx", ".xls"}:
            preferred = "structured_records" if original else "structured_records_dedup"
            sheet_name = preferred_excel_sheet(path, preferred, fallback="structured_records")
        df = read_table(path, sheet_name=sheet_name)
        records: list[StructuredRecord] = []
        for row in df.to_dict(orient="records"):
            records.append(
                StructuredRecord(
                    record_id=str(row["record_id"]),
                    raw_record_id=str(row["raw_record_id"]),
                    raw_text=_clean(row.get("raw_text")),
                    finding=_clean(row.get("finding")),
                    object=_json_list(row.get("object")),
                    scene=_clean(row.get("scene")),
                    accepted_by=_clean(row.get("accepted_by")),
                    arbitration_required=_bool(row.get("arbitration_required", False)),
                    metadata=_json_dict(row.get("metadata")),
                )
            )
        return records

    def source_info(self) -> dict[str, Any]:
        return {"input_path": str(self.input_path), "limit": self.limit, **self.input_options}

    def clusterer(self) -> Clusterer:
        return Clusterer(self.pipeline.embedding_provider, self.pipeline.config.get("clustering", {}))

    def namer(self) -> ClusterNamingService:
        naming_config = (
            self.pipeline.config.get("llm", {}).get("cluster_namer", {})
        )
        return ClusterNamingService(
            self.pipeline.cluster_namer_provider,
            max_workers=int(naming_config.get("max_workers", 1)),
        )

    def cluster_nonempty_channel_inputs(
        self,
        clusterer: Clusterer,
        experiment_id: str,
        inputs: list[ChannelInput],
    ) -> tuple[list[ClusterAssignment], dict[str, Any]]:
        clusterable_inputs = [item for item in inputs if item.input_text.strip()]
        missing_inputs = [item for item in inputs if not item.input_text.strip()]
        if missing_inputs:
            print(
                f"Stage cluster: {experiment_id} excludes {len(missing_inputs)} empty channel row(s) "
                "as missing_channel",
                flush=True,
            )
        if clusterable_inputs:
            assignments, metadata = clusterer.cluster(experiment_id, clusterable_inputs)
        else:
            assignments = []
            metadata = self.empty_channel_metadata()

        missing_assignments = [
            ClusterAssignment(
                experiment_id=experiment_id,
                record_id=item.record_id,
                cluster_id=MISSING_CHANNEL_CLUSTER_ID,
                is_noise=False,
                distance_to_centroid=None,
                representative_rank=None,
                primary_cluster_id=None,
                primary_is_noise=None,
                noise_recluster_id=None,
                final_cluster_id=MISSING_CHANNEL_CLUSTER_ID,
                cluster_level="missing_channel",
            )
            for item in missing_inputs
        ]
        assignments_by_record_id = {
            assignment.record_id: assignment
            for assignment in [*assignments, *missing_assignments]
        }
        ordered_assignments = [
            assignments_by_record_id[item.record_id]
            for item in inputs
        ]
        return ordered_assignments, add_missing_channel_metadata(
            metadata,
            total_count=len(inputs),
            clusterable_count=len(clusterable_inputs),
            missing_count=len(missing_inputs),
        )

    def empty_channel_metadata(self) -> dict[str, Any]:
        cfg = self.pipeline.config.get("clustering", {}).get("noise_recluster", {})
        return {
            "backend": "empty_channel",
            "record_count": 0,
            "cluster_count": 0,
            "noise_count": 0,
            "noise_ratio": 0.0,
            "primary_cluster_count": 0,
            "primary_noise_count": 0,
            "primary_noise_ratio": 0.0,
            "primary_label_distribution": {},
            "label_distribution": {},
            "noise_recluster": {
                "enabled": bool(cfg.get("enabled", True)),
                "triggered": False,
                "min_noise_count": int(cfg.get("min_noise_count", 100)),
                "min_noise_ratio": float(cfg.get("min_noise_ratio", 0.10)),
                "primary_noise_count": 0,
                "primary_noise_ratio": 0.0,
                "input_count": 0,
                "cluster_count": 0,
                "residual_noise_count": 0,
                "residual_noise_ratio": 0.0,
                "backend": "",
                "strategy": str(cfg.get("strategy", "secondary_noise_only")),
                "reason": "no_clusterable_records",
            },
        }

    def build_main_channel_inputs(
        self,
        baseline_records: list[SceneSplitRecord],
        unit_records: list[SceneSplitRecord] | None = None,
        *,
        apply_experiment_filter: bool = True,
    ) -> tuple[dict[str, list[ChannelInput]], dict[str, ChannelConfig]]:
        unit_records = baseline_records if unit_records is None else unit_records
        channel_configs = load_channel_configs(self.pipeline.config)
        experiment_configs = {
            "B1": ChannelConfig(
                experiment_id="B1",
                fields=("raw_text",),
                name="full_text_text_embedding_v4_umap_hdbscan",
            ),
        }
        experiment_configs.update({config.experiment_id: config for config in channel_configs})
        channel_inputs = generate_baseline_inputs(baseline_records)
        channel_inputs.update(generate_channel_inputs(unit_records, channel_configs))
        if apply_experiment_filter and self.experiment_ids:
            missing = [experiment_id for experiment_id in self.experiment_ids if experiment_id not in channel_inputs]
            if missing:
                raise SystemExit(
                    "Unknown experiment id(s): "
                    + ", ".join(missing)
                    + ". Available: "
                    + ", ".join(channel_inputs)
                )
            channel_inputs = {
                experiment_id: channel_inputs[experiment_id]
                for experiment_id in self.experiment_ids
            }
            experiment_configs = {
                experiment_id: experiment_configs[experiment_id]
                for experiment_id in self.experiment_ids
            }
        return channel_inputs, experiment_configs

    def ablation_provider_names(self) -> list[str]:
        return self.pipeline.config.get("ablation", {}).get(
            "no_consistency_providers",
            ["qwen"],
        )

    @property
    def data_revision_dir(self) -> Path:
        return self.output_dir / "00_data_revision"

    @property
    def structured_dir(self) -> Path:
        return self.output_dir / "01_structured"

    @property
    def structured_workbook_path(self) -> Path:
        return self.structured_dir / "structured_data.xlsx"

    @property
    def structured_machine_dir(self) -> Path:
        return self.structured_dir / "_machine"

    @property
    def clustering_dir(self) -> Path:
        return self.output_dir / "02_clustering"

    @property
    def clustering_machine_dir(self) -> Path:
        return self.clustering_dir / "_machine"

    @property
    def naming_dir(self) -> Path:
        return self.output_dir / "03_cluster_naming"

    @property
    def naming_machine_dir(self) -> Path:
        return self.naming_dir / "_machine"

    @property
    def ablation_dir(self) -> Path:
        return self.output_dir / "04_ablation"

    @property
    def ablation_machine_dir(self) -> Path:
        return self.ablation_dir / "_machine"

    @property
    def ablation_workbook_path(self) -> Path:
        return self.ablation_dir / "ablation_results.xlsx"

    def read_scene_split_records(self, *, original: bool = False) -> list[SceneSplitRecord]:
        path = self.scene_split_records_path(original=original)
        sheet_name = None
        if path.suffix.lower() in {".xlsx", ".xls"}:
            preferred = "scene_split_records" if original else "scene_split_records_dedup"
            sheet_name = preferred_excel_sheet(path, preferred, fallback="scene_split_records")
        return read_scene_split_records(path, sheet_name=sheet_name)

    def read_scene_split_units(self) -> list[SceneSplitRecord]:
        """Read effective deduplicated hazard units for T1-T8 analysis."""

        path = self.scene_split_units_path()
        sheet_name = None
        if path.suffix.lower() in {".xlsx", ".xls"}:
            sheet_name = preferred_excel_sheet(
                path,
                "scene_split_units_dedup",
                fallback="scene_split_records",
            )
        records = read_scene_split_records(path, sheet_name=sheet_name)
        effective_records = [record for record in records if is_effective_unit(record)]
        if not effective_records:
            raise SystemExit(f"No effective structured hazard units found in {path}")
        return effective_records

    def read_no_consistency_records_by_provider(self) -> dict[str, list[SceneSplitRecord]]:
        records_by_provider = {}
        for provider_name in self.ablation_provider_names():
            path = self.no_consistency_records_path(provider_name)
            records_by_provider[provider_name] = read_scene_split_records(path)
        return records_by_provider

    def read_no_consistency_units_by_provider(self) -> dict[str, list[SceneSplitRecord]]:
        """Read effective single-model hazard units for unit-level ablations."""

        records_by_provider: dict[str, list[SceneSplitRecord]] = {}
        for provider_name in self.ablation_provider_names():
            path = self.no_consistency_units_path(provider_name)
            records = read_scene_split_records(path)
            effective_records = [record for record in records if is_effective_unit(record)]
            if not effective_records:
                raise SystemExit(
                    f"No effective single-model hazard units found for {provider_name}: {path}"
                )
            records_by_provider[provider_name] = effective_records
        return records_by_provider

    def write_structured_workbook_from_existing(self) -> bool:
        required_paths = {
            "structured_records": self.first_existing_structured_path("structured_records.csv"),
            "scene_split_records": self.first_existing_structured_path("scene_split_records.csv"),
        }
        if not all(path.exists() for path in required_paths.values()):
            return False

        optional_paths = {
            "structured_records_dedup": self.first_existing_structured_path("structured_records_dedup.csv"),
            "scene_split_records_dedup": self.first_existing_structured_path("scene_split_records_dedup.csv"),
            "deduplication_map": self.first_existing_structured_path("deduplication_map.csv"),
            "validation_report": self.first_existing_structured_path("validation_report.csv"),
        }
        sheets = {}
        for sheet_name, path in {**required_paths, **optional_paths}.items():
            if path.exists():
                sheets[sheet_name] = read_table(path).to_dict(orient="records")
        write_excel_workbook(self.structured_workbook_path, sheets)
        return True

    def write_assignment_review(
        self,
        experiment_id: str,
        assignments: list[ClusterAssignment],
        records_by_id: dict[str, SceneSplitRecord],
        channel_inputs: list[ChannelInput],
    ) -> list[dict[str, Any]]:
        rows = assignment_review_rows(experiment_id, assignments, records_by_id, channel_inputs)
        write_dataclass_csv(
            rows,
            self.clustering_machine_dir / "clustering_results" / f"{experiment_id}_assignments_review.csv",
        )
        return rows

    def write_clustering_workbook(
        self,
        clustering_metadata: dict[str, Any],
        assignment_sheets: dict[str, list[Any]],
        review_sheets: dict[str, list[dict[str, Any]]],
    ) -> None:
        sheets: dict[str, list[Any]] = {
            "summary": clustering_summary_rows(clustering_metadata),
        }
        sheets.update(assignment_sheets)
        sheets.update(review_sheets)
        write_excel_workbook(
            self.clustering_dir / "clustering_results.xlsx",
            sheets,
        )

    def write_clustering_workbook_from_existing(
        self,
        clustering_metadata: dict[str, Any],
        channel_inputs: dict[str, list[ChannelInput]],
        records_by_id: dict[str, SceneSplitRecord],
        *,
        assignment_overrides: dict[str, list[Any]] | None = None,
        review_overrides: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        assignment_overrides = assignment_overrides or {}
        review_overrides = review_overrides or {}
        assignment_sheets: dict[str, list[Any]] = {}
        review_sheets: dict[str, list[dict[str, Any]]] = {}
        for experiment_id, inputs in channel_inputs.items():
            assignment_sheet_name = f"{experiment_id}_assignments"
            review_sheet_name = f"{experiment_id}_review"
            if assignment_sheet_name in assignment_overrides:
                assignments = assignment_overrides[assignment_sheet_name]
            else:
                assignments_path = self.cluster_assignments_path(experiment_id)
                if not assignments_path.exists():
                    continue
                assignments = self.read_cluster_assignments(assignments_path)
            assignment_sheets[assignment_sheet_name] = assignments
            if review_sheet_name in review_overrides:
                review_sheets[review_sheet_name] = review_overrides[review_sheet_name]
            else:
                review_sheets[review_sheet_name] = self.write_assignment_review(
                    experiment_id,
                    assignments,
                    records_by_id,
                    inputs,
                )
        if assignment_sheets:
            self.write_clustering_workbook(clustering_metadata, assignment_sheets, review_sheets)

    def read_clustering_metadata(self) -> dict[str, Any]:
        metadata_path = self.clustering_dir / "run_metadata.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def write_cluster_name_review(
        self,
        experiment_id: str,
        cluster_names: list[ClusterName],
        assignments: list[ClusterAssignment],
        records_by_id: dict[str, SceneSplitRecord],
        channel_inputs: list[ChannelInput],
    ) -> list[dict[str, Any]]:
        rows = cluster_name_review_rows(
            experiment_id,
            cluster_names,
            assignments,
            records_by_id,
            channel_inputs,
        )
        write_dataclass_csv(rows, self.naming_machine_dir / f"{experiment_id}_cluster_name_review.csv")
        return rows

    def write_naming_workbook_from_existing(
        self,
        channel_inputs: dict[str, list[ChannelInput]],
        records_by_id: dict[str, SceneSplitRecord],
    ) -> tuple[list[ClusterName], list[dict[str, Any]]]:
        all_cluster_names: list[ClusterName] = []
        all_review_rows: list[dict[str, Any]] = []
        for experiment_id, inputs in channel_inputs.items():
            cluster_names_path = self.cluster_names_path(experiment_id)
            assignments_path = self.cluster_assignments_path(experiment_id)
            if not cluster_names_path.exists() or not assignments_path.exists():
                continue
            cluster_names = read_cluster_names(cluster_names_path)
            assignments = self.read_cluster_assignments(assignments_path)
            ensure_assignments_match_records(experiment_id, assignments, records_by_id)
            review_rows = self.write_cluster_name_review(
                experiment_id,
                cluster_names,
                assignments,
                records_by_id,
                inputs,
            )
            all_cluster_names.extend(cluster_names)
            all_review_rows.extend(review_rows)
        if all_cluster_names:
            write_dataclass_csv(all_cluster_names, self.naming_machine_dir / "cluster_names.csv")
            write_dataclass_csv(all_review_rows, self.naming_machine_dir / "cluster_name_review.csv")
            write_excel_workbook(
                self.naming_dir / "cluster_names.xlsx",
                {
                    "cluster_names": all_cluster_names,
                    "review": all_review_rows,
                },
            )
        return all_cluster_names, all_review_rows

    def write_ablation_workbook_from_existing(
        self,
        metadata_by_experiment: dict[str, Any] | None = None,
        inputs_by_experiment: dict[str, list[ChannelInput]] | None = None,
        records_by_experiment: dict[str, list[SceneSplitRecord]] | None = None,
        *,
        assignment_overrides: dict[str, list[Any]] | None = None,
        review_overrides: dict[str, list[dict[str, Any]]] | None = None,
    ) -> tuple[list[ClusterName], list[dict[str, Any]]]:
        metadata_by_experiment = metadata_by_experiment if metadata_by_experiment is not None else self.read_ablation_metadata()
        inputs_by_experiment = inputs_by_experiment or {}
        records_by_experiment = records_by_experiment or {}
        assignment_overrides = assignment_overrides or {}
        review_overrides = review_overrides or {}

        sheets: dict[str, list[Any]] = {}
        for provider_name in self.ablation_provider_names():
            unit_path = self.no_consistency_units_path(provider_name)
            if unit_path.exists():
                sheets[f"{provider_name}_units"] = read_table(unit_path).to_dict(orient="records")
            record_path = self.no_consistency_records_path(provider_name)
            if record_path.exists():
                sheets[f"{provider_name}_records"] = read_table(record_path).to_dict(orient="records")

        if metadata_by_experiment:
            sheets["summary"] = clustering_summary_rows(metadata_by_experiment)

        experiment_ids = self.ablation_experiment_ids(metadata_by_experiment)
        all_cluster_names: list[ClusterName] = []
        all_name_review_rows: list[dict[str, Any]] = []
        for experiment_id in experiment_ids:
            assignment_sheet_name = f"{experiment_id}_assignments"
            review_sheet_name = f"{experiment_id}_review"
            if assignment_sheet_name in assignment_overrides:
                assignments = assignment_overrides[assignment_sheet_name]
            else:
                assignments_path = self.ablation_assignments_path(experiment_id)
                assignments = self.read_cluster_assignments(assignments_path) if assignments_path.exists() else []
            if assignments:
                sheets[assignment_sheet_name] = assignments
                if review_sheet_name in review_overrides:
                    sheets[review_sheet_name] = review_overrides[review_sheet_name]
                else:
                    review_path = self.ablation_assignment_review_path(experiment_id)
                    if review_path.exists():
                        sheets[review_sheet_name] = read_table(review_path).to_dict(orient="records")
                    elif experiment_id in inputs_by_experiment and experiment_id in records_by_experiment:
                        records_by_id = {
                            record.record_id: record
                            for record in records_by_experiment[experiment_id]
                        }
                        sheets[review_sheet_name] = assignment_review_rows(
                            experiment_id,
                            assignments,
                            records_by_id,
                            inputs_by_experiment[experiment_id],
                        )

            cluster_names_path = self.ablation_cluster_names_path(experiment_id)
            if not cluster_names_path.exists():
                continue
            cluster_names = read_cluster_names(cluster_names_path)
            all_cluster_names.extend(cluster_names)

            name_review_path = self.ablation_cluster_name_review_path(experiment_id)
            if name_review_path.exists():
                all_name_review_rows.extend(read_table(name_review_path).to_dict(orient="records"))
            elif assignments and experiment_id in inputs_by_experiment and experiment_id in records_by_experiment:
                records_by_id = {
                    record.record_id: record
                    for record in records_by_experiment[experiment_id]
                }
                all_name_review_rows.extend(
                    cluster_name_review_rows(
                        experiment_id,
                        cluster_names,
                        assignments,
                        records_by_id,
                        inputs_by_experiment[experiment_id],
                    )
                )

        if all_cluster_names:
            sheets["cluster_names"] = all_cluster_names
            write_dataclass_csv(all_cluster_names, self.ablation_machine_dir / "ablation_cluster_names.csv")
        if all_name_review_rows:
            sheets["name_review"] = all_name_review_rows
            write_dataclass_csv(all_name_review_rows, self.ablation_machine_dir / "ablation_cluster_name_review.csv")
        if sheets:
            write_excel_workbook(self.ablation_workbook_path, sheets)
        return all_cluster_names, all_name_review_rows

    def read_ablation_metadata(self) -> dict[str, Any]:
        candidates = [
            self.ablation_machine_dir / "ablation_metadata.json",
            self.source_ablation_dir / "_machine" / "ablation_metadata.json",
            self.ablation_dir / "ablation_metadata.json",
            self.source_ablation_dir / "ablation_metadata.json",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def ablation_experiment_ids(self, metadata_by_experiment: dict[str, Any]) -> list[str]:
        ids = list(metadata_by_experiment)
        for path in sorted((self.ablation_machine_dir / "ablation_clustering_results").glob("*_assignments.csv")):
            experiment_id = remove_suffix(path.name, "_assignments.csv")
            if experiment_id not in ids:
                ids.append(experiment_id)
        for path in sorted(self.ablation_machine_dir.glob("*_cluster_names.csv")):
            if path.name == "ablation_cluster_names.csv":
                continue
            experiment_id = remove_suffix(path.name, "_cluster_names.csv")
            if experiment_id not in ids:
                ids.append(experiment_id)
        return ids

    def no_consistency_records_path(self, provider_name: str) -> Path:
        filename = f"no_consistency_records_{provider_name}.csv"
        candidates = [
            self.ablation_machine_dir / "ablation_records" / filename,
            self.source_ablation_dir / "_machine" / "ablation_records" / filename,
            self.ablation_dir / "ablation_records" / filename,
            self.source_ablation_dir / "ablation_records" / filename,
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def no_consistency_units_path(self, provider_name: str) -> Path:
        filename = f"no_consistency_units_{provider_name}.csv"
        candidates = [
            self.ablation_machine_dir / "ablation_records" / filename,
            self.source_ablation_dir / "_machine" / "ablation_records" / filename,
            self.ablation_dir / "ablation_records" / filename,
            self.source_ablation_dir / "ablation_records" / filename,
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def ablation_assignments_path(self, experiment_id: str) -> Path:
        candidates = [
            self.ablation_machine_dir / "ablation_clustering_results" / f"{experiment_id}_assignments.csv",
            self.ablation_dir / "ablation_clustering_results" / f"{experiment_id}_assignments.csv",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def ablation_assignment_review_path(self, experiment_id: str) -> Path:
        candidates = [
            self.ablation_machine_dir / "ablation_clustering_results" / f"{experiment_id}_assignments_review.csv",
            self.ablation_dir / "ablation_clustering_results" / f"{experiment_id}_assignments_review.csv",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def ablation_cluster_names_path(self, experiment_id: str) -> Path:
        candidates = [
            self.ablation_machine_dir / f"{experiment_id}_cluster_names.csv",
            self.ablation_dir / f"{experiment_id}_cluster_names.csv",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def ablation_cluster_name_review_path(self, experiment_id: str) -> Path:
        candidates = [
            self.ablation_machine_dir / f"{experiment_id}_cluster_name_review.csv",
            self.ablation_dir / f"{experiment_id}_cluster_name_review.csv",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    @staticmethod
    def read_cluster_assignments(path: Path) -> list[ClusterAssignment]:
        return read_cluster_assignments(path)

    def first_existing_structured_path(self, filename: str) -> Path:
        candidates = [
            self.structured_machine_dir / filename,
            self.structured_dir / filename,
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def structured_records_path(self, *, original: bool = False) -> Path:
        if not original:
            dedup_candidates = [
                self.source_structured_dir / "_machine" / "structured_records_dedup.csv",
                self.source_structured_dir / "structured_data.xlsx",
                self.source_structured_dir / "deduplicated_outputs.xlsx",
            ]
            for path in dedup_candidates:
                if path.exists():
                    return path
        candidates = [
            self.source_structured_dir / "_machine" / "structured_records.csv",
            self.source_structured_dir / "structured_records.csv",
            self.source_structured_dir / "structured_data.xlsx",
            self.source_structured_dir / "structured_outputs.xlsx",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def scene_split_records_path(self, *, original: bool = False) -> Path:
        if not original:
            dedup_candidates = [
                self.source_structured_dir / "_machine" / "scene_split_records_dedup.csv",
                self.source_structured_dir / "structured_data.xlsx",
                self.source_structured_dir / "deduplicated_outputs.xlsx",
            ]
            for path in dedup_candidates:
                if path.exists():
                    return path
        candidates = [
            self.source_structured_dir / "_machine" / "scene_split_records.csv",
            self.source_structured_dir / "scene_split_records.csv",
            self.source_structured_dir / "structured_data.xlsx",
            self.source_structured_dir / "structured_outputs.xlsx",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def scene_split_units_path(self) -> Path:
        candidates = [
            self.source_structured_dir / "_machine" / "scene_split_units_dedup.csv",
            self.source_structured_dir / "scene_split_units_dedup.csv",
            self.source_structured_dir / "structured_data.xlsx",
            self.source_structured_dir / "deduplicated_outputs.xlsx",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def write_dedupe_stale_markers(self) -> None:
        message = (
            "# Stale After Unit Check And Aggregation\n\n"
            "The structured units for this run have been checked for exact duplicates and "
            "aggregated to the record level. "
            "Outputs in this directory may have been generated from earlier structured records. "
            "Rerun the dependent stage before using these results for analysis.\n"
        )
        for path in [self.clustering_dir, self.naming_dir, self.ablation_dir]:
            if path.exists():
                (path / "_STALE_AFTER_DEDUPE.md").write_text(message, encoding="utf-8")

    def cluster_assignments_path(self, experiment_id: str) -> Path:
        candidates = [
            self.clustering_machine_dir / "clustering_results" / f"{experiment_id}_assignments.csv",
            self.clustering_dir / "clustering_results" / f"{experiment_id}_assignments.csv",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def source_cluster_assignments_path(self, experiment_id: str) -> Path:
        candidates = [
            self.source_output_dir / "02_clustering" / "_machine" / "clustering_results" / f"{experiment_id}_assignments.csv",
            self.source_output_dir / "02_clustering" / "clustering_results" / f"{experiment_id}_assignments.csv",
            self.cluster_assignments_path(experiment_id),
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def cluster_names_path(self, experiment_id: str) -> Path:
        candidates = [
            self.naming_machine_dir / f"{experiment_id}_cluster_names.csv",
            self.naming_dir / f"{experiment_id}_cluster_names.csv",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def source_cluster_names_path(self, experiment_id: str) -> Path:
        candidates = [
            self.source_output_dir / "03_cluster_naming" / "_machine" / f"{experiment_id}_cluster_names.csv",
            self.source_output_dir / "03_cluster_naming" / f"{experiment_id}_cluster_names.csv",
            self.cluster_names_path(experiment_id),
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    @property
    def source_structured_dir(self) -> Path:
        return self.source_output_dir / "01_structured"

    @property
    def source_ablation_dir(self) -> Path:
        return self.source_output_dir / "04_ablation"


def read_table(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    return pd.read_csv(path)


def preferred_excel_sheet(path: Path, preferred: str, *, fallback: str) -> str:
    sheet_names = pd.ExcelFile(path).sheet_names
    if preferred in sheet_names:
        return preferred
    if fallback in sheet_names:
        return fallback
    return preferred


def remove_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if value.endswith(suffix) else value


def add_missing_channel_metadata(
    metadata: dict[str, Any],
    *,
    total_count: int,
    clusterable_count: int,
    missing_count: int,
) -> dict[str, Any]:
    metadata = dict(metadata)
    metadata["record_count"] = total_count
    metadata["clustered_record_count"] = clusterable_count
    metadata["missing_channel_count"] = missing_count
    metadata["missing_channel_ratio"] = missing_count / total_count if total_count else 0.0

    noise_count = int(metadata.get("noise_count", 0))
    metadata["noise_ratio"] = noise_count / total_count if total_count else 0.0
    primary_noise_count = int(metadata.get("primary_noise_count", 0))
    metadata["primary_noise_ratio"] = primary_noise_count / total_count if total_count else 0.0

    label_distribution = dict(metadata.get("label_distribution", {}))
    if missing_count:
        label_distribution[MISSING_CHANNEL_CLUSTER_ID] = missing_count
    metadata["label_distribution"] = label_distribution

    noise_recluster = dict(metadata.get("noise_recluster", {}))
    residual_noise_count = int(noise_recluster.get("residual_noise_count", noise_count))
    noise_recluster["residual_noise_ratio"] = residual_noise_count / total_count if total_count else 0.0
    metadata["noise_recluster"] = noise_recluster
    return metadata


def add_parent_coverage_metadata(
    metadata: dict[str, Any],
    *,
    parent_record_count: int,
    covered_parent_record_count: int,
) -> dict[str, Any]:
    if parent_record_count < 1:
        raise ValueError("parent_record_count must be positive")
    if not 0 <= covered_parent_record_count <= parent_record_count:
        raise ValueError(
            "covered_parent_record_count must be between zero and parent_record_count"
        )
    metadata = dict(metadata)
    no_output_parent_record_count = parent_record_count - covered_parent_record_count
    metadata["parent_record_count"] = parent_record_count
    metadata["covered_parent_record_count"] = covered_parent_record_count
    metadata["no_output_parent_record_count"] = no_output_parent_record_count
    metadata["parent_record_coverage"] = covered_parent_record_count / parent_record_count
    metadata["no_output_parent_record_rate"] = no_output_parent_record_count / parent_record_count
    return metadata


def read_scene_split_records(path: Path, sheet_name: str | None = None) -> list[SceneSplitRecord]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the prerequisite stage first.")
    df = read_table(path, sheet_name=sheet_name)
    records: list[SceneSplitRecord] = []
    for row in df.to_dict(orient="records"):
        records.append(
            SceneSplitRecord(
                record_id=str(row["record_id"]),
                raw_record_id=str(row["raw_record_id"]),
                raw_text=_clean(row.get("raw_text")),
                finding=_clean(row.get("finding")),
                object=_json_list(row.get("object")),
                scene=_clean(row.get("scene")),
                loc_detail=_clean(row.get("loc_detail")),
                risk_scene=_clean(row.get("risk_scene")),
                accepted_by=_clean(row.get("accepted_by")),
                arbitration_required=_bool(row.get("arbitration_required", False)),
                metadata=_json_dict(row.get("metadata")),
            )
        )
    return records


def read_cluster_assignments(path: Path) -> list[ClusterAssignment]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the prerequisite clustering stage first.")
    df = read_table(path)
    assignments: list[ClusterAssignment] = []
    for row in df.to_dict(orient="records"):
        assignments.append(
            ClusterAssignment(
                experiment_id=str(row["experiment_id"]),
                record_id=str(row["record_id"]),
                cluster_id=int(row["cluster_id"]),
                is_noise=_bool(row["is_noise"]),
                distance_to_centroid=_optional_float(row.get("distance_to_centroid")),
                representative_rank=_optional_int(row.get("representative_rank")),
                primary_cluster_id=_optional_int(row.get("primary_cluster_id")),
                primary_is_noise=_optional_bool(row.get("primary_is_noise")),
                noise_recluster_id=_optional_int(row.get("noise_recluster_id")),
                final_cluster_id=_optional_int(row.get("final_cluster_id"))
                if _optional_int(row.get("final_cluster_id")) is not None
                else int(row["cluster_id"]),
                cluster_level=_clean(row.get("cluster_level")) or ("primary_noise" if _bool(row["is_noise"]) else "primary"),
            )
        )
    return assignments


def read_cluster_names(path: Path) -> list[ClusterName]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the prerequisite naming stage first.")
    df = read_table(path)
    names: list[ClusterName] = []
    for row in df.to_dict(orient="records"):
        names.append(
            ClusterName(
                experiment_id=str(row["experiment_id"]),
                cluster_id=int(row["cluster_id"]),
                cluster_name=_clean(row.get("cluster_name")),
                explanation=_clean(row.get("explanation")),
                sample_count=int(row.get("sample_count", 0)),
            )
        )
    return names


def ensure_assignments_match_records(
    experiment_id: str,
    assignments: list[ClusterAssignment],
    records_by_id: dict[str, SceneSplitRecord],
) -> None:
    missing_record_ids = sorted(
        {assignment.record_id for assignment in assignments if assignment.record_id not in records_by_id}
    )
    if missing_record_ids:
        preview = ", ".join(missing_record_ids[:10])
        raise SystemExit(
            f"Clustering assignments for {experiment_id} do not match the current structured records. "
            f"{len(missing_record_ids)} assignment record(s) are missing after dedupe, for example: {preview}. "
            "Rerun --stage cluster before running review or name."
        )


def merge_records_by_id(
    *record_groups: list[SceneSplitRecord],
) -> dict[str, SceneSplitRecord]:
    """Merge record- and unit-grain rows while rejecting identifier collisions."""

    merged: dict[str, SceneSplitRecord] = {}
    for records in record_groups:
        for record in records:
            existing = merged.get(record.record_id)
            if existing is not None and existing != record:
                raise SystemExit(
                    "Record- and unit-grain inputs contain a conflicting record_id: "
                    f"{record.record_id}"
                )
            merged[record.record_id] = record
    return merged


def assignment_review_rows(
    experiment_id: str,
    assignments: list[ClusterAssignment],
    records_by_id: dict[str, SceneSplitRecord],
    channel_inputs: list[ChannelInput],
) -> list[dict[str, Any]]:
    inputs_by_id = {item.record_id: item for item in channel_inputs}
    cluster_sizes = Counter(assignment.cluster_id for assignment in assignments)
    rows = []
    for assignment in sorted(assignments, key=_assignment_sort_key):
        record = records_by_id.get(assignment.record_id)
        channel_input = inputs_by_id.get(assignment.record_id)
        rows.append(
            {
                "experiment_id": experiment_id,
                "cluster_id": assignment.cluster_id,
                "cluster_size": cluster_sizes[assignment.cluster_id],
                "is_noise": assignment.is_noise,
                "primary_cluster_id": assignment.primary_cluster_id,
                "primary_is_noise": assignment.primary_is_noise,
                "noise_recluster_id": assignment.noise_recluster_id,
                "final_cluster_id": assignment.final_cluster_id,
                "cluster_level": assignment.cluster_level,
                "representative_rank": assignment.representative_rank,
                "distance_to_centroid": assignment.distance_to_centroid,
                "record_id": assignment.record_id,
                "raw_record_id": record.raw_record_id if record else "",
                "channel_text": channel_input.input_text if channel_input else "",
                "raw_text": record.raw_text if record else "",
                "finding": record.finding if record else "",
                "object": record.object if record else [],
                "scene": record.scene if record else "",
                "loc_detail": record.loc_detail if record else "",
                "risk_scene": record.risk_scene if record else "",
                "accepted_by": record.accepted_by if record else "",
                "arbitration_required": record.arbitration_required if record else "",
            }
        )
    return rows


def cluster_name_review_rows(
    experiment_id: str,
    cluster_names: list[ClusterName],
    assignments: list[ClusterAssignment],
    records_by_id: dict[str, SceneSplitRecord],
    channel_inputs: list[ChannelInput],
) -> list[dict[str, Any]]:
    names_by_cluster_id = {name.cluster_id: name for name in cluster_names}
    rows = []
    for row in assignment_review_rows(experiment_id, assignments, records_by_id, channel_inputs):
        cluster_name = names_by_cluster_id.get(int(row["cluster_id"]))
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "cluster_id": row["cluster_id"],
                "cluster_name": cluster_name.cluster_name if cluster_name else "",
                "cluster_explanation": cluster_name.explanation if cluster_name else "",
                "cluster_size": row["cluster_size"],
                "name_sample_count": cluster_name.sample_count if cluster_name else "",
                "is_noise": row["is_noise"],
                "primary_cluster_id": row["primary_cluster_id"],
                "primary_is_noise": row["primary_is_noise"],
                "noise_recluster_id": row["noise_recluster_id"],
                "final_cluster_id": row["final_cluster_id"],
                "cluster_level": row["cluster_level"],
                "representative_rank": row["representative_rank"],
                "distance_to_centroid": row["distance_to_centroid"],
                "record_id": row["record_id"],
                "raw_record_id": row["raw_record_id"],
                "channel_text": row["channel_text"],
                "raw_text": row["raw_text"],
                "finding": row["finding"],
                "object": row["object"],
                "scene": row["scene"],
                "loc_detail": row["loc_detail"],
                "risk_scene": row["risk_scene"],
                "accepted_by": row["accepted_by"],
                "arbitration_required": row["arbitration_required"],
            }
        )
    return rows


def clustering_summary_rows(clustering_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for experiment_id, metadata in clustering_metadata.items():
        rows.append(
            {
                "experiment_id": experiment_id,
                "backend": metadata.get("backend", ""),
                "record_count": metadata.get("record_count", 0),
                "clustered_record_count": metadata.get("clustered_record_count", metadata.get("record_count", 0)),
                "missing_channel_count": metadata.get("missing_channel_count", 0),
                "missing_channel_ratio": metadata.get("missing_channel_ratio", 0),
                "cluster_count": metadata.get("cluster_count", 0),
                "noise_count": metadata.get("noise_count", 0),
                "noise_ratio": metadata.get("noise_ratio", 0),
                "primary_cluster_count": metadata.get("primary_cluster_count", 0),
                "primary_noise_count": metadata.get("primary_noise_count", 0),
                "primary_noise_ratio": metadata.get("primary_noise_ratio", 0),
                "noise_recluster_enabled": metadata.get("noise_recluster", {}).get("enabled", False),
                "noise_recluster_triggered": metadata.get("noise_recluster", {}).get("triggered", False),
                "noise_recluster_reason": metadata.get("noise_recluster", {}).get("reason", ""),
                "noise_recluster_input_count": metadata.get("noise_recluster", {}).get("input_count", 0),
                "noise_recluster_cluster_count": metadata.get("noise_recluster", {}).get("cluster_count", 0),
                "residual_noise_count": metadata.get("noise_recluster", {}).get(
                    "residual_noise_count",
                    metadata.get("noise_count", 0),
                ),
                "residual_noise_ratio": metadata.get("noise_recluster", {}).get(
                    "residual_noise_ratio",
                    metadata.get("noise_ratio", 0),
                ),
            }
        )
    return rows


def write_excel_workbook(path: Path, sheets: dict[str, list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, rows in sheets.items():
            _rows_to_dataframe(rows).to_excel(writer, sheet_name=sheet_name[:31], index=False)


def _rows_to_dataframe(rows: list[Any]) -> pd.DataFrame:
    dict_rows = []
    for row in rows:
        data = asdict(row) if is_dataclass(row) else dict(row)
        dict_rows.append({key: _normalize_table_value(value) for key, value in data.items()})
    return pd.DataFrame(dict_rows)


def _normalize_table_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _assignment_sort_key(assignment: ClusterAssignment) -> tuple:
    representative_rank = (
        assignment.representative_rank
        if assignment.representative_rank is not None
        else 10**9
    )
    if assignment.cluster_level == "missing_channel":
        group_order = 2
    elif assignment.is_noise:
        group_order = 3
    else:
        group_order = 0
    return (group_order, assignment.cluster_id, representative_rank, assignment.record_id)


def _json_list(value) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)] if str(value) else []
    return [str(item) for item in parsed if str(item)] if isinstance(parsed, list) else []


def _json_dict(value) -> dict:
    if pd.isna(value):
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _optional_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _optional_bool(value) -> bool | None:
    if pd.isna(value):
        return None
    return _bool(value)
