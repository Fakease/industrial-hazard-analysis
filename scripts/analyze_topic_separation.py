from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_hazard_analysis.evaluation import topic_similarity_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate topic-centroid similarity for multiple channels from an "
            "existing embedding cache. Only aggregate outputs are written."
        )
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
            if key in needed:
                value = item.get("value")
                if not isinstance(value, list):
                    raise ValueError(f"Invalid embedding at cache line {line_number}")
                found[key] = value
    return found


def write_table(rows: list[dict[str, Any]], csv_path: Path, md_path: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    channel_paths = parse_named_paths(args.channel_input)
    assignment_paths = parse_named_paths(args.assignment)
    if set(channel_paths) != set(assignment_paths):
        raise SystemExit("Channel inputs and assignments must have the same channel names")

    prepared: dict[str, pd.DataFrame] = {}
    key_by_channel: dict[str, list[str]] = {}
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
        merged[args.cluster_column] = pd.to_numeric(
            merged[args.cluster_column], errors="raise"
        ).astype(int)
        clean = merged[merged[args.cluster_column] >= 0].copy()
        keys = [
            cache_key(
                args.embedding_provider,
                args.embedding_model,
                args.embedding_dimension,
                str(text),
            )
            for text in clean[args.text_column]
        ]
        prepared[channel] = clean
        key_by_channel[channel] = keys
        needed.update(keys)

    cached = read_cache(args.embedding_cache, needed)
    missing = sorted(needed - set(cached))
    if missing:
        raise RuntimeError(
            f"Embedding cache is missing {len(missing)} required text vectors; API calls are disabled"
        )

    rows: list[dict[str, Any]] = []
    for channel in channel_paths:
        frame = prepared[channel]
        vectors = [cached[key] for key in key_by_channel[channel]]
        if any(len(vector) != args.embedding_dimension for vector in vectors):
            raise ValueError(f"{channel} contains an unexpected embedding dimension")
        metrics = topic_similarity_metrics(vectors, frame[args.cluster_column].tolist())
        if metrics["topic_similarity_status"] != "ok":
            raise RuntimeError(f"Topic similarity failed for {channel}: {metrics}")
        rows.append(
            {
                "通道": channel,
                "非噪声分析行": int(len(frame)),
                "最终主题数": int(frame[args.cluster_column].nunique()),
                "平均主题相似度": round(float(metrics["mean_topic_topic_similarity"]), 3),
                "最高主题相似度": round(float(metrics["max_topic_topic_similarity"]), 3),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(
        rows,
        args.out_dir / "table_3_3b_topic_separation.csv",
        args.out_dir / "table_3_3b_topic_separation.md",
    )
    metadata = {
        "embedding_cache_file": args.embedding_cache.name,
        "embedding_cache_sha256": sha256(args.embedding_cache),
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "required_cache_keys": len(needed),
        "missing_cache_keys": 0,
        "reported_metrics": ["mean_topic_topic_similarity", "max_topic_topic_similarity"],
        "excluded_redundant_metric": (
            "semantic_topic_diversity was defined as 1 - mean_topic_topic_similarity "
            "and is therefore not reported as an independent metric"
        ),
        "channels": {
            name: {
                "channel_input_file": channel_paths[name].name,
                "channel_input_sha256": sha256(channel_paths[name]),
                "assignment_file": assignment_paths[name].name,
                "assignment_sha256": sha256(assignment_paths[name]),
            }
            for name in channel_paths
        },
        "row_level_outputs_written": False,
    }
    (args.out_dir / "topic_separation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Calculated topic separation for {len(rows)} channels from cached embeddings")


if __name__ == "__main__":
    main()
