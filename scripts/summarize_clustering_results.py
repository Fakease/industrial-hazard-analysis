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

from industrial_hazard_analysis.evaluation import silhouette_cosine  # noqa: E402


CHANNEL_LABELS = {
    "B1": "原始全文",
    "T1": "问题表现",
    "T2": "对象",
    "T3": "风险场景",
    "T4": "位置细节",
    "T5": "风险对象与问题表现",
    "T6": "风险场景与问题表现",
    "T7": "风险场景与对象组合",
    "T8": "风险场景、风险对象与问题表现",
}
COMBINATION_FIELDS = {
    "T5": ("object", "finding"),
    "T6": ("risk_scene", "finding"),
    "T7": ("risk_scene", "object"),
    "T8": ("risk_scene", "object", "finding"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the aggregate B1–T8 clustering table from final assignments and cached embeddings."
    )
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--channel-input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--assignment", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--id-column", default="record_id")
    parser.add_argument("--text-column", default="input_text")
    parser.add_argument("--cluster-column", default="final_cluster_id")
    parser.add_argument("--embedding-provider", default="qwen")
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument(
        "--field-data",
        type=Path,
        default=None,
        help="Optional record-level structured fields used to summarize T5–T8 composition.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def parse_named_paths(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Expected NAME=PATH, received: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name or name in result:
            raise SystemExit(f"Channel names must be non-empty and unique: {name!r}")
        if not path.is_file():
            raise SystemExit(f"Input file not found: {path}")
        result[name] = path
    return result


def cache_key(provider: str, model: str, dimension: int, text: str) -> str:
    raw = "\n".join((provider, model, "embedding", str(dimension), text))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cache(path: Path, needed: set[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item.get("cache_key", ""))
            if key not in needed:
                continue
            value = item.get("value")
            if not isinstance(value, list):
                raise ValueError(f"Invalid embedding at cache line {line_number}")
            found[key] = value
    return found


def write_table(frame: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def field_present(value: object) -> bool:
    text = "" if pd.isna(value) else str(value).strip()
    return text.lower() not in {"", "nan", "none", "null", "[]", "{}"}


def summarize_channel_composition(
    fields: pd.DataFrame,
    clustering_rows: dict[str, dict[str, Any]],
    *,
    id_column: str,
) -> pd.DataFrame:
    required = {id_column, *{field for values in COMBINATION_FIELDS.values() for field in values}}
    missing = sorted(required - set(fields.columns))
    if missing:
        raise ValueError(f"Field data is missing columns: {missing}")
    if fields[id_column].duplicated().any():
        raise ValueError("Field data must contain one row per record identifier")

    rows: list[dict[str, Any]] = []
    for channel, channel_fields in COMBINATION_FIELDS.items():
        presence = pd.DataFrame(
            {
                field: fields[field].map(field_present)
                for field in channel_fields
            },
            index=fields.index,
        )
        usable = presence.any(axis=1)
        complete = presence.all(axis=1)
        usable_count = int(usable.sum())
        expected = int(clustering_rows[channel]["进入聚类"])
        if usable_count != expected:
            raise ValueError(
                f"{channel} field composition has {usable_count} usable records; "
                f"clustering table reports {expected}"
            )
        partial_counts: dict[str, int] = {}
        for _, row in presence[usable & ~complete].iterrows():
            present_fields = tuple(field for field in channel_fields if bool(row[field]))
            label = " + ".join(present_fields)
            partial_counts[label] = partial_counts.get(label, 0) + 1
        partial_description = "；".join(
            f"{count}条仅含 {label}"
            for label, count in sorted(partial_counts.items(), key=lambda item: (-item[1], item[0]))
        )
        rows.append(
            {
                "通道": channel,
                "进入聚类的隐患单元": usable_count,
                "字段完整单元": int(complete.sum()),
                "完整字段比例": round(float(complete.sum() / usable_count), 3),
                "部分字段单元": int((usable & ~complete).sum()),
                "部分字段构成": partial_description,
                "全部指定字段缺失单元": int((~usable).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    channel_paths = parse_named_paths(args.channel_input)
    assignment_paths = parse_named_paths(args.assignment)
    if set(channel_paths) != set(assignment_paths):
        raise SystemExit("Channel inputs and assignments must have the same channel names")

    prepared: dict[str, pd.DataFrame] = {}
    keys_by_channel: dict[str, list[str]] = {}
    needed: set[str] = set()
    for channel, input_path in channel_paths.items():
        inputs = pd.read_csv(input_path, dtype=str).fillna("")
        assignments = pd.read_csv(assignment_paths[channel], dtype=str).fillna("")
        required_input = {args.id_column, args.text_column}
        required_assignment = {args.id_column, args.cluster_column}
        if not required_input.issubset(inputs.columns):
            raise ValueError(f"{channel} input is missing {sorted(required_input - set(inputs.columns))}")
        if not required_assignment.issubset(assignments.columns):
            raise ValueError(
                f"{channel} assignments are missing {sorted(required_assignment - set(assignments.columns))}"
            )
        merged = inputs[[args.id_column, args.text_column]].merge(
            assignments[[args.id_column, args.cluster_column]],
            on=args.id_column,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(assignments):
            raise ValueError(f"{channel} channel inputs and assignments do not have identical record coverage")
        merged[args.cluster_column] = pd.to_numeric(
            merged[args.cluster_column], errors="raise"
        ).astype(int)
        non_noise = merged[merged[args.cluster_column] >= 0].copy()
        keys = [
            cache_key(
                args.embedding_provider,
                args.embedding_model,
                args.embedding_dimension,
                str(text),
            )
            for text in non_noise[args.text_column]
        ]
        prepared[channel] = merged
        keys_by_channel[channel] = keys
        needed.update(keys)

    cached = read_cache(args.embedding_cache, needed)
    missing = sorted(needed - set(cached))
    if missing:
        raise RuntimeError(
            f"Embedding cache is missing {len(missing)} required text vectors; API calls are disabled"
        )

    rows: list[dict[str, Any]] = []
    for channel in channel_paths:
        merged = prepared[channel]
        labels = merged[args.cluster_column]
        clusterable = merged[labels != -2]
        non_noise = merged[labels >= 0]
        sizes = non_noise[args.cluster_column].value_counts()
        vectors = [cached[key] for key in keys_by_channel[channel]]
        if any(len(vector) != args.embedding_dimension for vector in vectors):
            raise ValueError(f"{channel} contains an unexpected embedding dimension")
        silhouette = silhouette_cosine(vectors, non_noise[args.cluster_column].tolist())
        analysis_unit = "隐患记录" if channel == "B1" else "隐患单元"
        total_rows = int(len(merged))
        clusterable_rows = int(len(clusterable))
        noise_rows = int((labels == -1).sum())
        rows.append(
            {
                "通道": channel,
                "输入表示": CHANNEL_LABELS.get(channel, channel),
                "分析单位": analysis_unit,
                "分析总体": total_rows,
                "进入聚类": clusterable_rows,
                "覆盖率": round(float(clusterable_rows / total_rows), 3),
                "最终主题数": int(len(sizes)),
                "未归类": noise_rows,
                "总体未归类占比": round(float(noise_rows / total_rows), 3),
                "Silhouette-cosine": round(float(silhouette), 3),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    write_table(
        table,
        args.out_dir / "table_3_3_main_clustering.csv",
        args.out_dir / "table_3_3_main_clustering.md",
    )
    composition_written = False
    if args.field_data is not None:
        fields = pd.read_csv(args.field_data, dtype=str).fillna("")
        unit_channel = next(
            (channel for channel in channel_paths if channel != "B1"),
            None,
        )
        if unit_channel is None:
            raise ValueError("Field composition requires at least one T channel")
        unit_ids = set(prepared[unit_channel][args.id_column].astype(str))
        fields = fields[fields[args.id_column].astype(str).isin(unit_ids)].copy()
        if len(fields) != len(unit_ids):
            raise ValueError(
                "Field data does not contain exactly one row for every unit-level channel input"
            )
        composition = summarize_channel_composition(
            fields,
            {str(row["通道"]): row for row in rows},
            id_column=args.id_column,
        )
        write_table(
            composition,
            args.out_dir / "table_3_3a_channel_composition.csv",
            args.out_dir / "table_3_3a_channel_composition.md",
        )
        composition_written = True
    metadata = {
        "embedding_cache_file": args.embedding_cache.name,
        "embedding_cache_sha256": sha256(args.embedding_cache),
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "required_cache_keys": len(needed),
        "missing_cache_keys": 0,
        "analysis_grain_by_channel": {
            name: "deduplicated full-text record" if name == "B1" else "hazard unit"
            for name in channel_paths
        },
        "noise_rate_denominator": "all analysis rows in each channel",
        "maximum_cluster_share_denominator": "clusterable analysis rows in each channel",
        "silhouette_scope": (
            "final topic-assigned analysis rows only; labels -1 (unclassified) and "
            "-2 (missing channel) excluded"
        ),
        "silhouette_embedding_space": "original cached 1024-dimensional embeddings",
        "silhouette_distance": "cosine",
        "field_data": (
            {"file": args.field_data.name, "sha256": sha256(args.field_data)}
            if args.field_data is not None
            else None
        ),
        "channel_composition_written": composition_written,
        "channels": {
            name: {
                "channel_input_sha256": sha256(channel_paths[name]),
                "assignment_sha256": sha256(assignment_paths[name]),
            }
            for name in channel_paths
        },
        "row_level_outputs_written": False,
    }
    (args.out_dir / "main_clustering_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Summarized clustering results for {len(table)} channels")


if __name__ == "__main__":
    main()
