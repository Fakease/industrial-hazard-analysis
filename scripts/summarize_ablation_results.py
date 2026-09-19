from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EXPERIMENTS = (
    "A1-F-qwen",
    "A1-F-deepseek",
    "A1-F-doubao",
    "A2-F",
    "A1-O-qwen",
    "A1-O-deepseek",
    "A1-O-doubao",
    "A2-O",
    "A2-R",
    "A3-R",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the paper's full-universe ablation clustering outputs."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--assignments-dir", type=Path, required=True)
    parser.add_argument("--channel-inputs-dir", type=Path, required=True)
    parser.add_argument(
        "--parent-records",
        "--common-records",
        dest="parent_records",
        type=Path,
        required=True,
        help=(
            "Parent-record audit table. The current paper uses all deduplicated parent "
            "records and retains branch-level no-output cases as coverage outcomes."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--experiments",
        default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated experiment IDs in output order.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe_range(value: float, references: list[float], label: str) -> str:
    lower = min(references)
    upper = max(references)
    if lower <= value <= upper:
        return f"{label}位于三个单模型结果范围内"
    if value < lower:
        return f"{label}低于三个单模型结果范围下限"
    return f"{label}高于三个单模型结果范围上限"


def main() -> None:
    args = parse_args()
    experiments = tuple(item.strip() for item in args.experiments.split(",") if item.strip())
    if not experiments or len(experiments) != len(set(experiments)):
        raise SystemExit("Experiment IDs must be non-empty and unique")

    metadata: dict[str, dict[str, Any]] = json.loads(args.metadata.read_text(encoding="utf-8"))
    if set(metadata) != set(experiments):
        raise ValueError(
            f"Metadata experiments differ from requested experiments: "
            f"metadata={sorted(metadata)}, requested={sorted(experiments)}"
        )

    parent_frame = pd.read_csv(args.parent_records, dtype=str).fillna("")
    inclusion_column = next(
        (
            column
            for column in (
                "included_in_unit_level_ablation",
                "included_in_common_ablation",
            )
            if column in parent_frame.columns
        ),
        None,
    )
    if inclusion_column:
        included = parent_frame[inclusion_column].str.strip().str.lower()
        parent_frame = parent_frame[included.isin({"true", "1", "yes"})].copy()
    parent_key = (
        "record_id" if "record_id" in parent_frame.columns else "canonical_raw_record_id"
    )
    if parent_key not in parent_frame.columns:
        raise ValueError("Parent-record file must contain record_id or canonical_raw_record_id")
    if parent_frame[parent_key].duplicated().any():
        raise ValueError(f"Parent-record file contains duplicate {parent_key} values")
    parent_record_ids = set(parent_frame[parent_key].astype(str))
    if not parent_record_ids:
        raise ValueError("Parent-record file selects zero records")

    rows: list[dict[str, Any]] = []
    assignment_hashes: dict[str, str] = {}
    for experiment in experiments:
        assignment_path = args.assignments_dir / f"{experiment}_assignments.csv"
        input_path = args.channel_inputs_dir / f"{experiment}.csv"
        if not assignment_path.is_file():
            raise FileNotFoundError(assignment_path)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        assignments = pd.read_csv(assignment_path, dtype=str).fillna("")
        inputs = pd.read_csv(input_path, dtype=str).fillna("")
        review_path = args.assignments_dir / f"{experiment}_assignments_review.csv"
        if not review_path.is_file():
            raise FileNotFoundError(review_path)
        reviews = pd.read_csv(review_path, dtype=str).fillna("")
        required = {"experiment_id", "record_id", "final_cluster_id"}
        if not required.issubset(assignments.columns):
            raise ValueError(
                f"{experiment} assignments missing {sorted(required - set(assignments.columns))}"
            )
        if set(assignments["experiment_id"]) != {experiment}:
            raise ValueError(f"{experiment} assignments contain another experiment ID")
        if assignments["record_id"].duplicated().any():
            raise ValueError(f"{experiment} contains duplicate record_id values")
        if "record_id" not in inputs.columns or inputs["record_id"].duplicated().any():
            raise ValueError(f"{experiment} channel input lacks a unique record_id")
        assignment_ids = set(assignments["record_id"])
        input_ids = set(inputs["record_id"])
        if assignment_ids != input_ids:
            raise ValueError(
                f"{experiment} assignment coverage differs from its channel input: "
                f"missing={len(input_ids - assignment_ids)}, extra={len(assignment_ids - input_ids)}"
            )
        review_required = {"record_id", "raw_record_id"}
        if not review_required.issubset(reviews.columns):
            raise ValueError(
                f"{experiment} assignment review is missing "
                f"{sorted(review_required - set(reviews.columns))}"
            )
        if reviews["record_id"].duplicated().any():
            raise ValueError(f"{experiment} assignment review contains duplicate unit IDs")
        if set(reviews["record_id"]) != assignment_ids:
            raise ValueError(f"{experiment} assignment review differs from assignments")
        observed_parent_ids = set(reviews["raw_record_id"].astype(str))
        unexpected_parent_ids = observed_parent_ids - parent_record_ids
        if unexpected_parent_ids:
            raise ValueError(
                f"{experiment} contains {len(unexpected_parent_ids)} parent record(s) "
                "outside the declared parent universe"
            )
        covered_parent_record_count = len(observed_parent_ids)
        no_output_parent_record_count = len(parent_record_ids) - covered_parent_record_count

        labels = pd.to_numeric(assignments["final_cluster_id"], errors="raise").astype(int)
        observed = {
            "record_count": int(len(assignments)),
            "cluster_count": int(labels[labels >= 0].nunique()),
            "noise_count": int((labels == -1).sum()),
            "missing_channel_count": int((labels == -2).sum()),
            "clustered_record_count": int((labels != -2).sum()),
        }
        item = metadata[experiment]
        for key, value in observed.items():
            if int(item[key]) != value:
                raise ValueError(
                    f"{experiment} metadata mismatch for {key}: metadata={item[key]}, observed={value}"
                )
        parent_observed = {
            "parent_record_count": len(parent_record_ids),
            "covered_parent_record_count": covered_parent_record_count,
            "no_output_parent_record_count": no_output_parent_record_count,
        }
        for key, value in parent_observed.items():
            if key in item and int(item[key]) != value:
                raise ValueError(
                    f"{experiment} parent metadata mismatch for {key}: "
                    f"metadata={item[key]}, observed={value}"
                )
        if sum(observed[key] for key in ("noise_count", "missing_channel_count")) + int(
            (labels >= 0).sum()
        ) != observed["record_count"]:
            raise ValueError(f"{experiment} final labels do not partition all records")

        denominator = observed["record_count"]
        rows.append(
            {
                "experiment": experiment,
                "parent_record_count": len(parent_record_ids),
                "covered_parent_record_count": covered_parent_record_count,
                "no_output_parent_record_count": no_output_parent_record_count,
                "parent_record_coverage": round(
                    covered_parent_record_count / len(parent_record_ids), 4
                ),
                "no_output_parent_record_rate": round(
                    no_output_parent_record_count / len(parent_record_ids), 4
                ),
                "analysis_unit_count": denominator,
                "clustered_unit_count": observed["clustered_record_count"],
                "missing_unit_count": observed["missing_channel_count"],
                "missing_unit_rate": round(observed["missing_channel_count"] / denominator, 4),
                "cluster_count": observed["cluster_count"],
                "residual_noise_unit_count": observed["noise_count"],
                "residual_noise_unit_rate": round(observed["noise_count"] / denominator, 4),
                "primary_cluster_count": int(item["primary_cluster_count"]),
                "primary_noise_unit_count": int(item["primary_noise_count"]),
                "primary_noise_unit_rate": round(int(item["primary_noise_count"]) / denominator, 4),
                "noise_recluster_triggered": bool(item["noise_recluster"]["triggered"]),
            }
        )
        assignment_hashes[experiment] = sha256(assignment_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    csv_path = args.out_dir / "table_3_6_ablation_clustering.csv"
    md_path = args.out_dir / "table_3_6_ablation_clustering.md"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_markdown(table, md_path)

    indexed = table.set_index("experiment")
    finding_models = ["A1-F-qwen", "A1-F-deepseek", "A1-F-doubao"]
    object_models = ["A1-O-qwen", "A1-O-deepseek", "A1-O-doubao"]
    finding_feature = "；".join(
        (
            describe_range(
                float(indexed.loc["A2-F", column]),
                [float(indexed.loc[name, column]) for name in finding_models],
                label,
            )
        )
        for column, label in (
            ("analysis_unit_count", "分析单元数"),
            ("cluster_count", "主题数"),
            ("residual_noise_unit_rate", "总体未归类占比"),
        )
    ) + "。"
    object_feature = "；".join(
        (
            describe_range(
                float(indexed.loc["A2-O", column]),
                [float(indexed.loc[name, column]) for name in object_models],
                label,
            )
        )
        for column, label in (
            ("analysis_unit_count", "分析单元数"),
            ("cluster_count", "主题数"),
            ("residual_noise_unit_rate", "总体未归类占比"),
        )
    ) + "。"
    summary = pd.DataFrame(
        [
            {
                "对比": "finding",
                "单模型或参照结果": "；".join(
                    f"{name}："
                    f"{int(indexed.loc[name, 'analysis_unit_count'])}个单元、"
                    f"{int(indexed.loc[name, 'cluster_count'])}个主题、"
                    f"总体未归类占比{indexed.loc[name, 'residual_noise_unit_rate'] * 100:.2f}%"
                    for name in finding_models
                ),
                "一致性—仲裁或组合结果": (
                    "A2-F："
                    f"{int(indexed.loc['A2-F', 'analysis_unit_count'])}个单元、"
                    f"{int(indexed.loc['A2-F', 'cluster_count'])}个主题、"
                    f"总体未归类占比{indexed.loc['A2-F', 'residual_noise_unit_rate'] * 100:.2f}%"
                ),
                "结果特征": finding_feature,
            },
            {
                "对比": "object",
                "单模型或参照结果": "；".join(
                    f"{name}："
                    f"{int(indexed.loc[name, 'analysis_unit_count'])}个单元、"
                    f"{int(indexed.loc[name, 'cluster_count'])}个主题、"
                    f"总体未归类占比{indexed.loc[name, 'residual_noise_unit_rate'] * 100:.2f}%"
                    for name in object_models
                ),
                "一致性—仲裁或组合结果": (
                    "A2-O："
                    f"{int(indexed.loc['A2-O', 'analysis_unit_count'])}个单元、"
                    f"{int(indexed.loc['A2-O', 'cluster_count'])}个主题、"
                    f"总体未归类占比{indexed.loc['A2-O', 'residual_noise_unit_rate'] * 100:.2f}%"
                ),
                "结果特征": object_feature,
            },
            {
                "对比": "场景输入",
                "单模型或参照结果": (
                    "A2-R："
                    f"{int(indexed.loc['A2-R', 'analysis_unit_count'])}个单元、"
                    f"缺失率{indexed.loc['A2-R', 'missing_unit_rate'] * 100:.2f}%、"
                    f"{int(indexed.loc['A2-R', 'cluster_count'])}个主题、"
                    f"总体未归类占比{indexed.loc['A2-R', 'residual_noise_unit_rate'] * 100:.2f}%"
                ),
                "一致性—仲裁或组合结果": (
                    "A3-R："
                    f"{int(indexed.loc['A3-R', 'analysis_unit_count'])}个单元、"
                    f"缺失率{indexed.loc['A3-R', 'missing_unit_rate'] * 100:.2f}%、"
                    f"{int(indexed.loc['A3-R', 'cluster_count'])}个主题、"
                    f"总体未归类占比{indexed.loc['A3-R', 'residual_noise_unit_rate'] * 100:.2f}%"
                ),
                "结果特征": (
                    "A3-R消融场景拆分步骤并将位置细节与风险场景重新合并，比较场景拆分对"
                    "可用单元、主题数和总体未归类占比的影响。"
                ),
            },
        ]
    )
    summary.to_csv(
        args.out_dir / "table_3_6_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_markdown(summary, args.out_dir / "table_3_6_ablation_summary.md")

    validation = {
        "status": "passed",
        "experiment_count": len(experiments),
        "parent_record_count": len(parent_record_ids),
        "complete_case_filter_applied": False,
        "no_output_parent_records_retained_in_denominator": True,
        "parent_record_coverage_identical_across_experiments": (
            table["covered_parent_record_count"].nunique() == 1
        ),
        "parent_record_coverage_by_experiment": dict(
            zip(table["experiment"], table["parent_record_coverage"])
        ),
        "analysis_unit_counts_may_differ_across_structuring_branches": True,
        "duplicate_unit_ids": 0,
        "metadata_reconciled_to_assignments": True,
        "rate_denominators": {
            "parent_record_coverage": "all deduplicated parent records",
            "missing_channel_and_noise": "per-experiment hazard-unit input count",
        },
        "metadata_sha256": sha256(args.metadata),
        "parent_records_sha256": sha256(args.parent_records),
        "assignment_sha256": assignment_hashes,
    }
    (args.out_dir / "ablation_clustering_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated and summarized {len(table)} ablation experiments")


if __name__ == "__main__":
    main()
