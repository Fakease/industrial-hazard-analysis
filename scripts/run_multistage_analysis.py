"""Run P1/P2 local clustering on caller-supplied pipeline outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_hazard_analysis.multistage import (
    LocalClusteringParameters, MultistagePath, embedding_cache_key,
    naming_request_rows, parent_profiles, prepare_path_frame,
    read_embedding_cache, run_local_analysis, summarize_local_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Use an output directory outside the source repository")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings, embedding = config["multistage"], config["embedding"]
    parameters = LocalClusteringParameters(
        candidate_id=str(settings["candidate_id"]), umap=settings["umap"],
        hdbscan=settings["hdbscan"], random_seed=int(settings["random_seed"]),
    )
    def read_csv(relative):
        return pd.read_csv(args.run_dir / relative, dtype=str, keep_default_na=False)

    units = read_csv("01_structured/_machine/scene_split_units_dedup.csv")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path_id, definition in settings["paths"].items():
        parent, target = definition["parent_channel"], definition["target_channel"]
        path = MultistagePath(path_id, parent, target, parent, target, "")
        assignments = read_csv(f"02_clustering/_machine/clustering_results/{parent}_assignments.csv")
        inputs = read_csv(f"02_clustering/_machine/channel_inputs/{target}.csv")
        frame = prepare_path_frame(units, assignments, inputs, path)
        profiles = parent_profiles(frame)
        keys = {embedding_cache_key(embedding["provider"], embedding["model"],
                int(embedding["dimension"]), text)
                for text in frame.loc[frame["target_valid"], "target_text"].astype(str)}
        vectors = read_embedding_cache(args.embedding_cache, keys, dimension=int(embedding["dimension"]))
        parents, mapping = run_local_analysis(
            frame, profiles, vectors,
            embedding_provider=embedding["provider"], embedding_model=embedding["model"],
            embedding_dimension=int(embedding["dimension"]),
            parent_min_units=int(settings["parent_min_units"]),
            target_min_units=int(settings["target_min_units"]), parameters=parameters,
            numba_cache_dir=output / "cache",
        )
        parents.to_csv(output / f"{path_id}_parents.csv", index=False, encoding="utf-8-sig")
        mapping.to_csv(output / f"{path_id}_units.csv", index=False, encoding="utf-8-sig")
        with (output / f"{path_id}_naming_inputs.jsonl").open("w", encoding="utf-8") as stream:
            for request in naming_request_rows(mapping):
                stream.write(json.dumps(request, ensure_ascii=False) + "\n")
        summaries.append({"path_id": path_id, **summarize_local_results(parents)})
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
