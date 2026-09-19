from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paired record universe for A1/A2 clustering. A record is included "
            "when every supplied system has a non-empty finding."
        )
    )
    parser.add_argument("--system", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--id-column", default="raw_record_id")
    parser.add_argument("--finding-column", default="finding")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_systems(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Expected NAME=PATH, received: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name or name in result:
            raise SystemExit(f"System names must be non-empty and unique: {name!r}")
        if not path.is_file():
            raise SystemExit(f"System file not found: {path}")
        result[name] = path
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    systems = parse_systems(args.system)
    availability: dict[str, set[str]] = {}
    all_ids: set[str] = set()
    metadata: dict[str, object] = {"systems": {}}
    for name, path in systems.items():
        frame = pd.read_csv(path, dtype=str).fillna("")
        required = {args.id_column, args.finding_column}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} is missing {sorted(required - set(frame.columns))}")
        ids = frame[args.id_column].astype(str).str.strip()
        if ids.eq("").any() or ids.duplicated().any():
            raise ValueError(f"{name} must contain one non-empty row per {args.id_column}")
        available = set(ids[frame[args.finding_column].astype(str).str.strip().ne("")])
        availability[name] = available
        all_ids.update(ids)
        metadata["systems"][name] = {
            "file": path.name,
            "sha256": sha256(path),
            "records": int(len(frame)),
            "nonempty_finding_records": int(len(available)),
        }

    common_ids = set.intersection(*availability.values())
    rows = pd.DataFrame(
        {
            "canonical_raw_record_id": sorted(all_ids),
        }
    )
    rows["included_in_common_ablation"] = rows["canonical_raw_record_id"].isin(common_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False, encoding="utf-8-sig")
    metadata.update(
        {
            "rule": "non-empty finding in every supplied system",
            "total_record_ids": int(len(rows)),
            "included_record_ids": int(len(common_ids)),
            "output_file": args.output.name,
            "output_sha256": sha256(args.output),
        }
    )
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Built common ablation universe: {len(common_ids)}/{len(rows)} records")


if __name__ == "__main__":
    main()
