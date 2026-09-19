from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_hazard_analysis.evaluation import evaluate_extraction  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "outputs" / "evaluation"


CORE_ROWS = [
    ("Prediction units", "prediction_unit_count_in_scope"),
    ("Gold units", "gold_unit_count"),
    ("Complete hits", "complete_hit_count"),
    ("Unit precision", "unit_precision"),
    ("Unit recall", "unit_recall"),
    ("Unit F1", "unit_f1"),
    ("Finding char-F1", "finding_char_f1"),
    ("Object set-F1", "object_set_f1"),
    ("Scene char-F1", "scene_char_f1"),
    ("Loc detail char-F1", "loc_detail_char_f1"),
    ("Risk scene char-F1", "risk_scene_char_f1"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-LLM structured extraction against the human gold standard."
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--raw-dedup-map",
        type=Path,
        default=None,
        help="Optional raw_text_dedup_map.csv used to remap gold raw IDs to canonical IDs.",
    )
    parser.add_argument(
        "--write-pairs",
        action="store_true",
        help="Write row-level matched pairs; disabled by default for privacy.",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        sheet_name: str | int = (
            "Gold_Records" if "Gold_Records" in workbook.sheet_names else 0
        )
        df = pd.read_excel(workbook, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path)
    if "object" not in df.columns and "object_json" in df.columns:
        df = df.rename(columns={"object_json": "object"})
    return df.where(pd.notna(df), None).to_dict(orient="records")


def remap_gold_raw_ids(
    gold_records: list[dict[str, Any]],
    dedup_map_path: Path,
) -> tuple[list[dict[str, Any]], int, int, int]:
    dedup_rows = pd.read_csv(dedup_map_path).where(lambda frame: pd.notna(frame), None)
    raw_id_map = {
        str(row["original_raw_record_id"]): str(row["canonical_raw_record_id"])
        for row in dedup_rows.to_dict(orient="records")
    }
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    unmapped = 0
    for source in gold_records:
        row = dict(source)
        raw_id = str(row.get("raw_record_id") or "")
        canonical_id = raw_id_map.get(raw_id)
        if canonical_id is None:
            unmapped += 1
            continue
        row["raw_record_id"] = canonical_id
        grouped.setdefault(canonical_id, {}).setdefault(raw_id, []).append(row)

    remapped: list[dict[str, Any]] = []
    changed = 0
    duplicate_units = 0
    for canonical_id, source_groups in grouped.items():
        chosen_source = (
            canonical_id if canonical_id in source_groups else next(iter(source_groups))
        )
        chosen_rows = source_groups[chosen_source]
        remapped.extend(chosen_rows)
        if chosen_source != canonical_id:
            changed += len(chosen_rows)
        duplicate_units += sum(
            len(rows)
            for source_id, rows in source_groups.items()
            if source_id != chosen_source
        )
    return remapped, changed, unmapped, duplicate_units


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["指标", "数值", "说明"]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers)
                + " |\n"
            )


def flatten_summary(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    notes = {
        "Prediction units": "金标准范围内的结构化隐患单元数",
        "Gold units": "人工金标准隐患单元数",
        "Complete hits": "问题表现、风险对象和原始场景均严格一致的单元数",
        "Unit precision": "完整命中数与结构化隐患单元数之比",
        "Unit recall": "完整命中数与人工金标准隐患单元数之比",
        "Unit F1": "严格完整命中的 precision/recall 调和均值",
        "Finding char-F1": "对应单元中问题表现的平均字符级 F1",
        "Object set-F1": "对应单元中风险对象的平均集合 F1",
        "Scene char-F1": "对应单元中原始场景的平均字符级 F1",
        "Loc detail char-F1": "对应单元中位置细节的平均字符级 F1",
        "Risk scene char-F1": "对应单元中风险场景的平均字符级 F1",
    }
    rows = []
    for label, key in CORE_ROWS:
        value = metrics.get(key)
        if label in {"Prediction units", "Gold units", "Complete hits"}:
            formatted = str(int(value))
        else:
            formatted = f"{float(value):.3f}" if isinstance(value, (int, float)) else value
        rows.append(
            {
                "指标": label,
                "数值": formatted,
                "说明": notes[label],
            }
        )
    return rows


def field_summary(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field, values in metrics.get("text_fields", {}).items():
        rows.append(
            {
                "field": field,
                "char_f1": values["char_f1"],
                "exact_match_rate": values["exact_match_rate"],
                "evaluated_count": values["evaluated_count"],
            }
        )
    rows.append(
        {
            "field": "object",
            "char_f1": metrics["object"]["set_f1"],
            "exact_match_rate": metrics["object"]["exact_match_rate"],
            "evaluated_count": metrics["object"]["evaluated_count"],
        }
    )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gold_records = read_records(args.gold)
    predictions = read_records(args.predictions)
    remapped_gold_count = 0
    unmapped_gold_count = 0
    duplicate_gold_unit_count = 0
    if args.raw_dedup_map is not None:
        (
            gold_records,
            remapped_gold_count,
            unmapped_gold_count,
            duplicate_gold_unit_count,
        ) = remap_gold_raw_ids(
            gold_records,
            args.raw_dedup_map,
        )
    metrics = evaluate_extraction(predictions, gold_records)
    metrics["evaluation_context"] = {
        "predictions_file": args.predictions.name,
        "predictions_sha256": sha256(args.predictions),
        "gold_file": args.gold.name,
        "gold_sha256": sha256(args.gold),
        "raw_dedup_map_file": args.raw_dedup_map.name if args.raw_dedup_map else None,
        "raw_dedup_map_sha256": sha256(args.raw_dedup_map) if args.raw_dedup_map else None,
        "remapped_gold_unit_count": remapped_gold_count,
        "unmapped_gold_unit_count": unmapped_gold_count,
        "duplicate_gold_unit_count_removed_after_raw_text_dedup": (
            duplicate_gold_unit_count
        ),
        "row_level_outputs_written": bool(args.write_pairs),
    }

    summary_rows = flatten_summary(metrics)
    field_rows = field_summary(metrics)
    pairs = metrics.pop("pairs", [])
    metrics.pop("unpaired_predictions", None)
    metrics.pop("unpaired_gold", None)

    (args.out_dir / "extraction_reliability_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "extraction_reliability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_markdown_table(args.out_dir / "extraction_reliability_summary.md", summary_rows)
    pd.DataFrame(summary_rows).rename(columns={"说明": "结果含义"}).to_csv(
        args.out_dir / "table_3_2_extraction_reliability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_markdown_table(
        args.out_dir / "table_3_2_extraction_reliability.md", summary_rows
    )
    pd.DataFrame(field_rows).to_csv(
        args.out_dir / "extraction_reliability_field_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if args.write_pairs:
        pd.DataFrame(pairs).to_csv(
            args.out_dir / "extraction_reliability_pairs.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"Wrote extraction reliability evaluation to {args.out_dir}")
    print(
        "Core metrics: "
        + ", ".join(f"{row['指标']}={row['数值']}" for row in summary_rows)
    )


if __name__ == "__main__":
    main()
