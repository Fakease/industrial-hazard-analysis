from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_hazard_analysis.evaluation import (  # noqa: E402
    cross_channel_correspondence,
    parent_inherited_cluster_correspondence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate AMI after inheriting each record-level B1 label to its "
            "unit-level T1-T8 children. Only aggregate outputs are written."
        )
    )
    parser.add_argument("--baseline-review", type=Path, required=True)
    parser.add_argument(
        "--target-review",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for each unit-level target channel.",
    )
    parser.add_argument("--parent-id-column", default="raw_record_id")
    parser.add_argument("--unit-id-column", default="record_id")
    parser.add_argument("--cluster-column", default="final_cluster_id")
    parser.add_argument("--noise-cluster-id", type=int, default=-1)
    parser.add_argument("--missing-cluster-id", type=int, default=-2)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def parse_named_paths(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Expected NAME=PATH, received: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name:
            raise SystemExit("Target channel name cannot be empty")
        if name in parsed:
            raise SystemExit(f"Duplicate target channel: {name}")
        if not path.exists():
            raise SystemExit(f"Target review file not found: {path}")
        parsed[name] = path
    return parsed


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Review file not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in columns if column not in header]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    frame = pd.read_csv(path, usecols=columns, dtype=str)
    for column in columns:
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    targets = parse_named_paths(args.target_review)
    baseline = read_columns(
        args.baseline_review,
        [args.parent_id_column, args.cluster_column],
    )
    target_frames = {
        name: read_columns(
            path,
            [args.unit_id_column, args.parent_id_column, args.cluster_column],
        )
        for name, path in targets.items()
    }

    baseline_rows: list[dict[str, object]] = []
    for name, target in target_frames.items():
        metrics = parent_inherited_cluster_correspondence(
            baseline,
            target,
            parent_id_column=args.parent_id_column,
            target_unit_id_column=args.unit_id_column,
            baseline_cluster_column=args.cluster_column,
            target_cluster_column=args.cluster_column,
            noise_cluster_id=args.noise_cluster_id,
            missing_cluster_id=args.missing_cluster_id,
        )
        baseline_rows.append(
            {
                "baseline": "B1",
                "target": name,
                "target_units": metrics["target_unit_count"],
                "common_non_noise_units": metrics["common_non_noise_units"],
                "AMI": round(float(metrics["ami"]), 6),
            }
        )

    pairwise_rows: list[dict[str, object]] = []
    names = list(target_frames)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            metrics = cross_channel_correspondence(
                target_frames[left_name],
                target_frames[right_name],
                id_column=args.unit_id_column,
                baseline_cluster_column=args.cluster_column,
                target_cluster_column=args.cluster_column,
                noise_cluster_id=args.noise_cluster_id,
                missing_cluster_id=args.missing_cluster_id,
            )
            pairwise_rows.append(
                {
                    "channel_a": left_name,
                    "channel_b": right_name,
                    "common_non_noise_units": metrics["common_non_noise_records"],
                    "AMI": round(float(metrics["ami"]), 6),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(baseline_rows).to_csv(
        args.out_dir / "unit_level_ami_b1_to_channels.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(pairwise_rows).to_csv(
        args.out_dir / "unit_level_ami_between_channels.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "analysis_grain": "hazard_unit",
        "baseline_grain": "deduplicated_full_text_record",
        "inheritance_rule": (
            "Each hazard unit inherits the final B1 cluster label of its parent raw record"
        ),
        "baseline_review": {
            "file": args.baseline_review.name,
            "sha256": sha256(args.baseline_review),
        },
        "target_reviews": {
            name: {"file": path.name, "sha256": sha256(path)}
            for name, path in targets.items()
        },
        "parent_id_column": args.parent_id_column,
        "unit_id_column": args.unit_id_column,
        "cluster_column": args.cluster_column,
        "noise_cluster_id": args.noise_cluster_id,
        "missing_cluster_id": args.missing_cluster_id,
        "row_level_outputs_written": False,
    }
    (args.out_dir / "unit_level_ami_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(baseline_rows)} B1-to-channel and "
        f"{len(pairwise_rows)} channel-pair AMI rows to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
