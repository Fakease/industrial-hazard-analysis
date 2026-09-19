from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_hazard_analysis.config import load_config, project_path
from industrial_hazard_analysis.pipeline import HazardAnalysisPipeline
from industrial_hazard_analysis.pipeline.llm_tasks import apply_llm_task_override, build_llm_task_policies
from industrial_hazard_analysis.pipeline.run_artifacts import (
    experiment_default_output_dir,
    experiment_output_category,
    resolve_output_dir,
    write_run_info,
)


from industrial_hazard_analysis.pipeline.stages import STAGES, ExperimentStageRunner
from industrial_hazard_analysis.providers import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
)


DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
}


def build_pipeline(config: dict, output_dir: Path, *, cache_dir: Path | None = None) -> HazardAnalysisPipeline:
    llm_config = config["llm"]
    offline_cache_only = bool(config.get("runtime", {}).get("offline_cache_only", False))
    cache_dir = cache_dir or output_dir / "cache"
    extractors = [
        OpenAICompatibleLLMProvider(
            name=item["provider"],
            model=item["model"],
            api_key_env=item["api_key_env"],
            base_url=item.get("base_url") or DEFAULT_BASE_URLS[item["provider"]],
            prompt_paths=_worker_prompt_paths(llm_config),
            cache_dir=cache_dir,
            offline_cache_only=offline_cache_only,
        )
        for item in llm_config["extractors"]
    ]
    extractors_by_name = {provider.name: provider for provider in extractors}
    arbitrator_config = llm_config["arbitrator"]
    arbitrator = OpenAICompatibleLLMProvider(
        name=arbitrator_config["provider"],
        model=arbitrator_config["model"],
        api_key_env=arbitrator_config["api_key_env"],
        base_url=arbitrator_config.get("base_url") or DEFAULT_BASE_URLS[arbitrator_config["provider"]],
        prompt_paths=_arbitrator_prompt_paths(llm_config),
        cache_dir=cache_dir,
        offline_cache_only=offline_cache_only,
    )
    namer_config = llm_config["cluster_namer"]
    cluster_namer = OpenAICompatibleLLMProvider(
        name=namer_config["provider"],
        model=namer_config["model"],
        api_key_env=namer_config["api_key_env"],
        base_url=namer_config.get("base_url") or DEFAULT_BASE_URLS[namer_config["provider"]],
        prompt_paths={"cluster_naming": str(project_path(ROOT, namer_config["prompt_path"]))},
        cache_dir=cache_dir,
        offline_cache_only=offline_cache_only,
    )
    embedding_config = config["embedding"]
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        name=embedding_config["provider"],
        model=embedding_config["model"],
        api_key_env=embedding_config["api_key_env"],
        base_url=embedding_config.get("base_url") or DEFAULT_BASE_URLS[embedding_config["provider"]],
        dimension=int(embedding_config.get("dimension", 1024)),
        batch_size=int(embedding_config.get("batch_size", 10)),
        cache_dir=cache_dir,
        offline_cache_only=offline_cache_only,
    )
    experiment_config = dict(config)
    experiment_config["clustering"] = dict(config.get("clustering", {}))
    experiment_config["clustering"]["use_real_backend"] = True
    llm_tasks = build_llm_task_policies(
        llm_config,
        providers_by_name=extractors_by_name,
        default_worker_providers=extractors,
        default_arbitrator_provider=arbitrator,
        task_defaults={"structured_extraction", "scene_secondary_split"},
    )
    return HazardAnalysisPipeline(
        extractor_providers=extractors,
        arbitrator_provider=arbitrator,
        embedding_provider=embedding_provider,
        cluster_namer_provider=cluster_namer,
        config=experiment_config,
        llm_tasks=llm_tasks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real industrial hazard experiment pipeline.")
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    parser.add_argument("--input", default=None, help="Override input CSV/XLSX path.")
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name when input is .xlsx.")
    parser.add_argument("--id-column", default=None, help="Optional record id column.")
    parser.add_argument("--text-column", default=None, help="Source hazard text column.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Parent directory used with --run-name.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Named run directory under --output-root. Use 'auto' for a timestamped descriptive name.",
    )
    parser.add_argument(
        "--source-run-name",
        default=None,
        help=(
            "Read prerequisite artifacts from another run under --output-root/main. "
            "Useful for clustering parameter variants that reuse an existing llm-extract run."
        ),
    )
    parser.add_argument(
        "--source-output-dir",
        default=None,
        help="Read prerequisite artifacts from this explicit output directory.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Read and write provider caches in this explicit directory. "
            "By default, reuse the prerequisite run cache."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Read only the first N input rows for pilot runs.")
    parser.add_argument("--umap-n-neighbors", type=int, default=None, help="Override UMAP n_neighbors.")
    parser.add_argument("--umap-n-components", type=int, default=None, help="Override UMAP n_components.")
    parser.add_argument("--umap-min-dist", type=float, default=None, help="Override UMAP min_dist.")
    parser.add_argument("--umap-metric", default=None, help="Override UMAP metric, for example cosine.")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=None, help="Override HDBSCAN min_cluster_size.")
    parser.add_argument("--hdbscan-min-samples", type=int, default=None, help="Override HDBSCAN min_samples.")
    parser.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=None,
        help="Override HDBSCAN cluster_selection_epsilon; larger values merge nearby clusters.",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-method",
        choices=["eom", "leaf"],
        default=None,
        help="Override HDBSCAN cluster_selection_method.",
    )
    parser.add_argument(
        "--disable-noise-recluster",
        action="store_true",
        help="Disable automatic second-stage clustering for primary -1 noise rows.",
    )
    parser.add_argument(
        "--noise-recluster-min-ratio",
        type=float,
        default=None,
        help="Trigger noise-recluster only when primary noise ratio is at least this value.",
    )
    parser.add_argument(
        "--noise-recluster-min-count",
        type=int,
        default=None,
        help="Trigger noise-recluster only when primary noise count is at least this value.",
    )
    parser.add_argument(
        "--extraction-batch-size",
        type=int,
        default=None,
        help="Appendix A extraction LLM batch size. Use 1 for single-record prompts.",
    )
    parser.add_argument(
        "--scene-batch-size",
        type=int,
        default=None,
        help="Appendix C scene split LLM batch size. Use 1 for single-record prompts.",
    )
    parser.add_argument(
        "--extraction-mode",
        choices=["arbitration", "lightweight"],
        default=None,
        help="Override structured extraction mode for this run.",
    )
    parser.add_argument(
        "--extraction-workers",
        default=None,
        help="Comma-separated structured extraction worker providers, for example qwen,deepseek,doubao.",
    )
    parser.add_argument(
        "--extraction-arbitrator",
        default=None,
        help="Structured extraction arbitrator provider when --extraction-mode arbitration.",
    )
    parser.add_argument(
        "--scene-mode",
        choices=["arbitration", "lightweight"],
        default=None,
        help="Override scene split mode for this run.",
    )
    parser.add_argument(
        "--scene-workers",
        default=None,
        help="Comma-separated scene split worker providers, for example deepseek or qwen,deepseek,doubao.",
    )
    parser.add_argument(
        "--scene-arbitrator",
        default=None,
        help="Scene split arbitrator provider when --scene-mode arbitration.",
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGES),
        default="all",
        help="Run all stages or one debuggable stage at a time.",
    )
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated main experiment ids to run, for example B1,T1. Omit to run B1 and T1-T8.",
    )
    parser.add_argument(
        "--ablation-common-records",
        default=None,
        help=(
            "CSV/XLSX audit table containing canonical_raw_record_id and "
            "included_in_common_ablation. Restricts A1/A2 to a paired record universe."
        ),
    )
    parser.add_argument(
        "--naming-workers",
        type=int,
        default=None,
        help="Concurrent cluster-naming API requests. Default is 1; use a small value such as 4-6.",
    )
    parser.add_argument(
        "--offline-cache-only",
        action="store_true",
        help="Forbid all LLM and embedding API requests; stop immediately if any required cache item is missing.",
    )
    args = parser.parse_args()

    load_local_env(ROOT / ".env.local")
    config = load_config(args.config)
    if args.offline_cache_only:
        config.setdefault("runtime", {})["offline_cache_only"] = True
    if args.naming_workers is not None:
        if args.naming_workers < 1:
            raise SystemExit("--naming-workers must be at least 1")
        config.setdefault("llm", {}).setdefault("cluster_namer", {})[
            "max_workers"
        ] = args.naming_workers
    apply_task_overrides(
        config,
        extraction_batch_size=args.extraction_batch_size,
        scene_batch_size=args.scene_batch_size,
        extraction_mode=args.extraction_mode,
        extraction_workers=args.extraction_workers,
        extraction_arbitrator=args.extraction_arbitrator,
        scene_mode=args.scene_mode,
        scene_workers=args.scene_workers,
        scene_arbitrator=args.scene_arbitrator,
    )
    apply_clustering_overrides(
        config,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_n_components=args.umap_n_components,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
        disable_noise_recluster=args.disable_noise_recluster,
        noise_recluster_min_ratio=args.noise_recluster_min_ratio,
        noise_recluster_min_count=args.noise_recluster_min_count,
    )
    preflight_api_keys(config, args.stage)
    configured_source = config.get("paths", {}).get("source_data")
    if args.input:
        input_path = Path(args.input)
    elif configured_source:
        input_path = project_path(ROOT, configured_source)
    else:
        raise SystemExit(
            "No research dataset is bundled in the publication package. "
            "Pass its CSV/XLSX path with --input."
        )
    output_root = project_path(ROOT, args.output_root)
    output_dir_arg = project_path(ROOT, args.output_dir) if args.output_dir else None
    source_output_dir_arg = project_path(ROOT, args.source_output_dir) if args.source_output_dir else None
    output_dir = resolve_output_dir(
        output_dir=output_dir_arg,
        output_root=output_root,
        output_category=experiment_output_category(args.stage),
        run_name=args.run_name,
        default_output_dir=project_path(ROOT, experiment_default_output_dir(args.stage)),
        stage=args.stage,
        config=config,
        limit=args.limit,
    )
    source_output_dir = resolve_source_output_dir(
        output_dir=output_dir,
        output_root=output_root,
        output_category=experiment_output_category(args.stage),
        source_run_name=args.source_run_name,
        source_output_dir=source_output_dir_arg,
    )
    cache_dir = (
        project_path(ROOT, args.cache_dir)
        if args.cache_dir
        else source_output_dir / "cache"
        if source_output_dir != output_dir
        else output_dir / "cache"
    )
    input_config = config.get("input", {})
    input_options = {
        "sheet_name": args.sheet_name or input_config.get("sheet_name"),
        "id_column": args.id_column or input_config.get("id_column"),
        "text_column": args.text_column or input_config.get("text_column"),
    }
    runner = ExperimentStageRunner(
        build_pipeline(config, output_dir, cache_dir=cache_dir),
        output_dir,
        source_output_dir=source_output_dir,
        input_path=input_path,
        input_options=input_options,
        limit=args.limit,
        experiment_ids=parse_experiment_ids(args.experiments),
        ablation_common_records_path=(
            project_path(ROOT, args.ablation_common_records)
            if args.ablation_common_records
            else None
        ),
        deduplicate_raw_text=bool(input_config.get("deduplicate_raw_text", True)),
    )
    result = runner.run(args.stage)
    write_run_info(
        output_dir,
        index_root=run_index_root(output_dir),
        script_name=Path(__file__).name,
        stage=args.stage,
        input_path=input_path,
        input_options=input_options,
        limit=args.limit,
        config=config,
        result=result,
        project_root=ROOT,
    )
    if result.get("status") == "partial":
        print(f"Experiment stage partially completed: {args.stage}")
    else:
        print(f"Experiment stage completed: {args.stage}")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"Outputs written to: {output_dir}")
    if source_output_dir != output_dir:
        print(f"Prerequisite artifacts read from: {source_output_dir}")


def run_index_root(output_dir: Path) -> Path:
    """Keep the run index beside external/private runs instead of publication/."""

    try:
        output_dir.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return output_dir.parent
    return ROOT / "outputs"


def preflight_api_keys(config: dict, stage: str) -> None:
    if bool(config.get("runtime", {}).get("offline_cache_only", False)):
        return
    required = required_api_key_envs(config, stage)
    missing = sorted(env for env in required if not os.environ.get(env))
    if missing:
        raise SystemExit("Missing API key environment variable(s): " + ", ".join(missing))


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"Invalid local env line {line_number} in {path}: missing '='.")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            raise SystemExit(f"Invalid local env line {line_number} in {path}: empty key.")
        os.environ.setdefault(key, value)


def parse_experiment_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_source_output_dir(
    *,
    output_dir: Path,
    output_root: str | Path,
    output_category: str,
    source_run_name: str | None,
    source_output_dir: str | None,
) -> Path:
    if source_output_dir:
        return Path(source_output_dir)
    if source_run_name:
        return Path(output_root) / output_category / source_run_name
    return output_dir


def apply_clustering_overrides(
    config: dict,
    *,
    umap_n_neighbors: int | None,
    umap_n_components: int | None,
    umap_min_dist: float | None,
    umap_metric: str | None,
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
    hdbscan_cluster_selection_epsilon: float | None,
    hdbscan_cluster_selection_method: str | None,
    disable_noise_recluster: bool = False,
    noise_recluster_min_ratio: float | None = None,
    noise_recluster_min_count: int | None = None,
) -> None:
    clustering_config = config.setdefault("clustering", {})
    umap_config = clustering_config.setdefault("umap", {})
    hdbscan_config = clustering_config.setdefault("hdbscan", {})
    noise_recluster_config = clustering_config.setdefault("noise_recluster", {})
    if umap_n_neighbors is not None:
        umap_config["n_neighbors"] = umap_n_neighbors
    if umap_n_components is not None:
        umap_config["n_components"] = umap_n_components
    if umap_min_dist is not None:
        umap_config["min_dist"] = umap_min_dist
    if umap_metric is not None:
        umap_config["metric"] = umap_metric
    if hdbscan_min_cluster_size is not None:
        hdbscan_config["min_cluster_size"] = hdbscan_min_cluster_size
    if hdbscan_min_samples is not None:
        hdbscan_config["min_samples"] = hdbscan_min_samples
    if hdbscan_cluster_selection_epsilon is not None:
        hdbscan_config["cluster_selection_epsilon"] = hdbscan_cluster_selection_epsilon
    if hdbscan_cluster_selection_method is not None:
        hdbscan_config["cluster_selection_method"] = hdbscan_cluster_selection_method
    if disable_noise_recluster:
        noise_recluster_config["enabled"] = False
    if noise_recluster_min_ratio is not None:
        noise_recluster_config["min_noise_ratio"] = noise_recluster_min_ratio
    if noise_recluster_min_count is not None:
        noise_recluster_config["min_noise_count"] = noise_recluster_min_count


def apply_task_overrides(
    config: dict,
    *,
    extraction_batch_size: int | None,
    scene_batch_size: int | None,
    extraction_mode: str | None = None,
    extraction_workers: str | None = None,
    extraction_arbitrator: str | None = None,
    scene_mode: str | None = None,
    scene_workers: str | None = None,
    scene_arbitrator: str | None = None,
) -> None:
    llm_config = config.setdefault("llm", {})
    try:
        apply_llm_task_override(
            llm_config,
            "structured_extraction",
            mode=extraction_mode,
            worker_providers=extraction_workers,
            arbitrator_provider=extraction_arbitrator,
            batch_size=extraction_batch_size,
        )
        apply_llm_task_override(
            llm_config,
            "scene_secondary_split",
            mode=scene_mode,
            worker_providers=scene_workers,
            arbitrator_provider=scene_arbitrator,
            batch_size=scene_batch_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def required_api_key_envs(config: dict, stage: str) -> set[str]:
    llm_config = config["llm"]
    extractor_configs_by_name = {item["provider"]: item for item in llm_config["extractors"]}
    extraction_envs = _task_api_key_envs(llm_config, "structured_extraction", extractor_configs_by_name)
    scene_envs = _task_api_key_envs(llm_config, "scene_secondary_split", extractor_configs_by_name)
    arbitrator_env = {llm_config["arbitrator"]["api_key_env"]}
    namer_env = {llm_config["cluster_namer"]["api_key_env"]}
    embedding_env = {config["embedding"]["api_key_env"]}

    if stage == "all":
        return extraction_envs | scene_envs | _task_arbitrator_envs(llm_config) | namer_env | embedding_env
    if stage in {"llm-extract", "ablation-extract"}:
        return extraction_envs | scene_envs | _task_arbitrator_envs(llm_config)
    if stage in {"cluster", "ablation-cluster"}:
        return embedding_env
    if stage in {"name", "ablation-name"}:
        return namer_env
    return set()


def _worker_prompt_paths(llm_config: dict) -> dict[str, str]:
    tasks = llm_config.get("tasks", {})
    structured_task = tasks["structured_extraction"]
    scene_task = tasks["scene_secondary_split"]
    return {
        "structured_extraction": str(project_path(ROOT, structured_task["prompt_path"])),
        "structured_extraction_batch": str(project_path(ROOT, structured_task["batch_prompt_path"])),
        "scene_secondary_split": str(project_path(ROOT, scene_task["prompt_path"])),
        "scene_secondary_split_batch": str(project_path(ROOT, scene_task["batch_prompt_path"])),
    }


def _arbitrator_prompt_paths(llm_config: dict) -> dict[str, str]:
    tasks = llm_config.get("tasks", {})
    structured_task = tasks["structured_extraction"]
    scene_task = tasks["scene_secondary_split"]
    prompt_paths = {}
    if structured_task.get("arbitration_prompt_path"):
        prompt_paths["structured_extraction_arbitration"] = str(
            project_path(ROOT, structured_task["arbitration_prompt_path"])
        )
    if scene_task.get("arbitration_prompt_path"):
        prompt_paths["scene_split_arbitration"] = str(
            project_path(ROOT, scene_task["arbitration_prompt_path"])
        )
    return prompt_paths


def _task_api_key_envs(
    llm_config: dict,
    task_key: str,
    extractor_configs_by_name: dict[str, dict],
) -> set[str]:
    task_config = llm_config.get("tasks", {}).get(task_key, {})
    names = task_config.get("worker_providers") or [item["provider"] for item in llm_config["extractors"]]
    return {extractor_configs_by_name[name]["api_key_env"] for name in names}


def _task_arbitrator_envs(llm_config: dict) -> set[str]:
    arbitrator_env = {llm_config["arbitrator"]["api_key_env"]}
    tasks = llm_config.get("tasks", {})
    if not tasks:
        return arbitrator_env
    return {
        env
        for task_config in tasks.values()
        if str(task_config.get("mode", "arbitration")).lower() == "arbitration"
        for env in arbitrator_env
    }


if __name__ == "__main__":
    main()
