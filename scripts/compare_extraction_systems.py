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

from industrial_hazard_analysis.evaluation import (  # noqa: E402
    compare_extraction_systems,
    summarize_prediction_sources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple extraction systems with one gold set and one matching rule. "
            "Only aggregate outputs are written."
        )
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for each system, for example --system '多LLM=multi.csv'.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--raw-dedup-map", type=Path, default=None)
    parser.add_argument(
        "--source-system",
        default=None,
        help="Optional system name for majority/arbitration result-source summary.",
    )
    parser.add_argument(
        "--source-column",
        default=None,
        help="Prediction column containing result source; requires --source-system.",
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        default=None,
        help="Optional unit-level file supplying the source column for the source system.",
    )
    parser.add_argument("--source-map-key", default="record_id")
    parser.add_argument(
        "--source-label",
        action="append",
        default=[],
        metavar="RAW=DISPLAY",
        help="Optional display mapping for result-source values.",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        sheet_name: str | int = (
            "Gold_Records" if "Gold_Records" in workbook.sheet_names else 0
        )
        frame = pd.read_excel(workbook, sheet_name=sheet_name, dtype=str)
    else:
        frame = pd.read_csv(path, dtype=str)
    if "object" not in frame.columns and "object_json" in frame.columns:
        frame = frame.rename(columns={"object_json": "object"})
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def parse_named_paths(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Expected NAME=PATH, received: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not name or name in parsed:
            raise SystemExit(f"System names must be non-empty and unique: {name!r}")
        path = Path(raw_path.strip())
        if not path.exists():
            raise SystemExit(f"System prediction file not found: {path}")
        parsed[name] = path
    return parsed


def parse_value_map(specs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Expected RAW=DISPLAY, received: {spec}")
        raw, display = spec.split("=", 1)
        mapping[raw.strip()] = display.strip()
    return mapping


def remap_gold_raw_ids(
    gold_records: list[dict[str, Any]],
    dedup_map_path: Path,
) -> tuple[list[dict[str, Any]], int, int, int]:
    mapping_frame = pd.read_csv(dedup_map_path).where(lambda frame: pd.notna(frame), None)
    required = {"original_raw_record_id", "canonical_raw_record_id"}
    if not required.issubset(mapping_frame.columns):
        raise SystemExit(f"Dedup map must contain columns: {sorted(required)}")
    raw_id_map = {
        str(row["original_raw_record_id"]): str(row["canonical_raw_record_id"])
        for row in mapping_frame.to_dict(orient="records")
    }
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    unmapped = 0
    for source in gold_records:
        row = dict(source)
        raw_id = str(row.get("raw_record_id") or "")
        if raw_id not in raw_id_map:
            unmapped += 1
            continue
        canonical_id = raw_id_map[raw_id]
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(frame: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rounded_frame(rows: list[dict[str, Any]], digits: int = 3) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include="number").columns
    for column in numeric:
        if not pd.api.types.is_integer_dtype(frame[column]):
            frame[column] = frame[column].round(digits)
    return frame


def main() -> None:
    args = parse_args()
    system_paths = parse_named_paths(args.system)
    gold_records = read_records(args.gold)
    remapped_gold_count = 0
    unmapped_gold_count = 0
    duplicate_gold_unit_count = 0
    if args.raw_dedup_map is not None:
        (
            gold_records,
            remapped_gold_count,
            unmapped_gold_count,
            duplicate_gold_unit_count,
        ) = remap_gold_raw_ids(gold_records, args.raw_dedup_map)
    system_records = {name: read_records(path) for name, path in system_paths.items()}
    if args.source_map is not None:
        if not args.source_system or not args.source_column:
            raise SystemExit("--source-map requires --source-system and --source-column")
        source_frame = pd.read_csv(args.source_map, dtype=str).fillna("")
        required = {args.source_map_key, args.source_column}
        if not required.issubset(source_frame.columns):
            raise SystemExit(f"Source map must contain columns: {sorted(required)}")
        if source_frame[args.source_map_key].duplicated().any():
            raise SystemExit("Source map keys must be unique")
        source_by_id = dict(
            zip(source_frame[args.source_map_key], source_frame[args.source_column])
        )
        for record in system_records.get(args.source_system, []):
            record_id = str(record.get(args.source_map_key) or "")
            if record_id not in source_by_id:
                raise SystemExit(f"Source map is missing prediction unit: {record_id}")
            record[args.source_column] = source_by_id[record_id]
    comparison = compare_extraction_systems(system_records, gold_records)
    comparison.pop("_matches_by_system")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_table = rounded_frame(comparison["system_summary"])
    write_table(
        main_table,
        args.out_dir / "table_3_2a_single_multi_extraction.csv",
        args.out_dir / "table_3_2a_single_multi_extraction.md",
    )
    common_table = rounded_frame(comparison["common_matched_summary"])
    write_table(
        common_table,
        args.out_dir / "table_s2_common_matched_field_comparison.csv",
        args.out_dir / "table_s2_common_matched_field_comparison.md",
    )

    if bool(args.source_system) != bool(args.source_column):
        raise SystemExit("--source-system and --source-column must be supplied together")
    if args.source_system:
        if args.source_system not in system_records:
            raise SystemExit(f"Unknown source system: {args.source_system}")
        source_rows = summarize_prediction_sources(
            system_records[args.source_system],
            gold_records,
            source_column=args.source_column,
            source_labels=parse_value_map(args.source_label),
        )
        source_table = rounded_frame(source_rows)
        write_table(
            source_table,
            args.out_dir / "table_3_2b_multi_llm_result_source.csv",
            args.out_dir / "table_3_2b_multi_llm_result_source.md",
        )

    metadata = {
        "gold_file": args.gold.name,
        "gold_sha256": sha256(args.gold),
        "gold_unit_count": len(gold_records),
        "remapped_gold_unit_count": remapped_gold_count,
        "unmapped_gold_unit_count": unmapped_gold_count,
        "duplicate_gold_unit_count_removed_after_raw_text_dedup": duplicate_gold_unit_count,
        "system_files": {
            name: {"file": path.name, "sha256": sha256(path)}
            for name, path in system_paths.items()
        },
        "effective_unit_rule": (
            "at least one of finding, object, scene, loc_detail, or risk_scene is non-empty"
        ),
        "source_map": (
            {"file": args.source_map.name, "sha256": sha256(args.source_map)}
            if args.source_map is not None
            else None
        ),
        "unit_complete_hit_rule": (
            "normalized finding, object set, and scene all exactly equal within raw_record_id"
        ),
        "diagnostic_pairing_rule": (
            "exact hits locked first; remaining units paired globally with the Hungarian "
            "algorithm using equal-weight valid-field similarity and no threshold"
        ),
        "field_f1_aggregation": (
            "mean of per-pair field F1 over pairs where at least one side is non-empty"
        ),
        "common_matched_gold_unit_count": comparison[
            "common_matched_gold_unit_count"
        ],
        "row_level_outputs_written": False,
    }
    (args.out_dir / "single_multi_extraction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Compared {len(system_records)} systems; common matched gold units: "
        f"{comparison['common_matched_gold_unit_count']}"
    )


if __name__ == "__main__":
    main()
