"""Execute caller-prepared K1 request batches with schema and identity checks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_hazard_analysis.topic_control_synthesis import (
    TopicControlSynthesisClient, load_topic_control_config,
)
from run_experiment_pipeline import load_local_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/k1.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline-cache-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Use an output directory outside the source repository")
    load_local_env(ROOT / ".env.local")
    config = load_topic_control_config(args.config)
    output.mkdir(parents=True, exist_ok=True)
    client = TopicControlSynthesisClient(
        config=config, cache_path=output / "cache.jsonl",
        offline_cache_only=args.offline_cache_only,
    )
    with args.requests.open(encoding="utf-8-sig") as source:
        requests = [json.loads(line) for line in source if line.strip()]
    ids = [str(request["request_id"]) for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("Request IDs must be unique within a batch")
    with (output / "responses.jsonl").open("x", encoding="utf-8") as target:
        for request in requests:
            target.write(json.dumps(asdict(client.run(request)), ensure_ascii=False) + "\n")
            target.flush()


if __name__ == "__main__":
    main()
