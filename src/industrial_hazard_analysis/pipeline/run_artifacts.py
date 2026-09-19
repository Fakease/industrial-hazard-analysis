from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def resolve_output_dir(
    *,
    output_dir: str | None,
    output_root: str | Path,
    output_category: str,
    run_name: str | None,
    default_output_dir: str | Path,
    stage: str,
    config: dict[str, Any],
    limit: int | None,
) -> Path:
    if output_dir:
        return Path(output_dir)
    if run_name:
        name = auto_run_name(stage, config, limit) if run_name == "auto" else _safe_name(run_name)
        return Path(output_root) / output_category / name
    return Path(default_output_dir)


def experiment_output_category(stage: str) -> str:
    return "main"


def experiment_default_output_dir(stage: str) -> Path:
    return Path("outputs") / experiment_output_category(stage) / "latest"


def write_run_info(
    output_dir: str | Path,
    *,
    index_root: str | Path,
    script_name: str,
    stage: str,
    input_path: str | Path | None,
    input_options: dict[str, Any],
    limit: int | None,
    config: dict[str, Any],
    result: dict[str, Any],
    project_root: str | Path | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().isoformat(timespec="seconds")
    llm_tasks = _llm_task_summaries(config)
    clustering = _clustering_summary(config)
    prompt_sha256 = _prompt_hashes(config, Path(project_root) if project_root else Path.cwd())
    payload = {
        "created_at": created_at,
        "script": script_name,
        "stage": stage,
        "output_dir": str(output_path),
        "input_path": str(input_path) if input_path is not None else "",
        "input_options": input_options,
        "limit": limit,
        "llm_tasks": llm_tasks,
        "clustering": clustering,
        "prompt_sha256": prompt_sha256,
        "result": result,
    }
    (output_path / "_run_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "_RUN_INFO.md").write_text(_run_info_markdown(payload), encoding="utf-8")
    try:
        _append_run_index(Path(index_root), payload)
    except PermissionError as exc:
        print(
            f"Warning: run index not updated because {Path(index_root) / '_runs_index.csv'} is not writable: {exc}",
            flush=True,
        )


def auto_run_name(stage: str, config: dict[str, Any], limit: int | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, stage]
    if limit is not None:
        parts.append(f"limit{limit}")
    for task_key, prefix in [
        ("structured_extraction", "ext"),
        ("scene_secondary_split", "scene"),
    ]:
        task = config.get("llm", {}).get("tasks", {}).get(task_key, {})
        mode = str(task.get("mode", "mode")).strip()
        batch_size = task.get("batch_size")
        bits = [prefix, mode]
        if batch_size:
            bits.append(f"b{batch_size}")
        parts.append("-".join(bits))
    return _safe_name("_".join(parts))


def _llm_task_summaries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for task_key, task in config.get("llm", {}).get("tasks", {}).items():
        summaries[task_key] = {
            "mode": task.get("mode"),
            "worker_providers": task.get("worker_providers", []),
            "arbitrator_provider": task.get("arbitrator_provider", ""),
            "batch_size": task.get("batch_size"),
        }
    return summaries


def _clustering_summary(config: dict[str, Any]) -> dict[str, Any]:
    clustering = config.get("clustering", {})
    return {
        "umap": dict(clustering.get("umap", {})),
        "hdbscan": dict(clustering.get("hdbscan", {})),
        "noise_recluster": dict(clustering.get("noise_recluster", {})),
    }


def _prompt_hashes(config: dict[str, Any], project_root: Path) -> dict[str, str]:
    configured_paths: dict[str, object] = {}
    for task_key, task in config.get("llm", {}).get("tasks", {}).items():
        for field in ("prompt_path", "batch_prompt_path", "arbitration_prompt_path"):
            if task.get(field):
                configured_paths[f"{task_key}.{field}"] = task[field]
    namer = config.get("llm", {}).get("cluster_namer", {})
    if namer.get("prompt_path"):
        configured_paths["cluster_namer.prompt_path"] = namer["prompt_path"]

    result: dict[str, str] = {}
    for label, configured_path in configured_paths.items():
        path = Path(str(configured_path))
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"Configured prompt does not exist: {path}")
        result[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _run_info_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Run Info",
        "",
        f"- Created at: {payload['created_at']}",
        f"- Script: `{payload['script']}`",
        f"- Stage: `{payload['stage']}`",
        f"- Output: `{payload['output_dir']}`",
        f"- Input: `{payload['input_path']}`",
        f"- Limit: `{payload['limit']}`",
        "",
        "## LLM Tasks",
    ]
    for task_key, task in payload["llm_tasks"].items():
        workers = ",".join(str(item) for item in task.get("worker_providers", []))
        lines.append(
            f"- `{task_key}`: mode=`{task.get('mode')}`, workers=`{workers}`, "
            f"arbitrator=`{task.get('arbitrator_provider')}`, batch=`{task.get('batch_size')}`"
        )
    clustering = payload.get("clustering", {})
    lines.extend(["", "## Clustering"])
    for section in ["umap", "hdbscan", "noise_recluster"]:
        values = clustering.get(section, {})
        if not values:
            continue
        rendered = ", ".join(f"{key}={value}" for key, value in values.items())
        lines.append(f"- `{section}`: {rendered}")

    lines.extend(["", "## Prompt SHA-256"])
    for label, digest in payload.get("prompt_sha256", {}).items():
        lines.append(f"- `{label}`: `{digest}`")

    lines.extend(["", "## Result", "", "```json"])
    lines.append(json.dumps(payload["result"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _append_run_index(index_root: Path, payload: dict[str, Any]) -> None:
    index_root.mkdir(parents=True, exist_ok=True)
    index_path = index_root / "_runs_index.csv"
    fields = [
        "created_at",
        "script",
        "stage",
        "output_dir",
        "input_path",
        "limit",
        "structured_extraction",
        "scene_secondary_split",
        "clustering",
        "result",
    ]
    row = {
        "created_at": payload["created_at"],
        "script": payload["script"],
        "stage": payload["stage"],
        "output_dir": payload["output_dir"],
        "input_path": payload["input_path"],
        "limit": payload["limit"],
        "structured_extraction": _task_index_value(payload["llm_tasks"].get("structured_extraction", {})),
        "scene_secondary_split": _task_index_value(payload["llm_tasks"].get("scene_secondary_split", {})),
        "clustering": _clustering_index_value(payload.get("clustering", {})),
        "result": json.dumps(payload["result"], ensure_ascii=False),
    }
    if index_path.exists():
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fields != fields:
            with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({field: existing_row.get(field, "") for field in fields})
    write_header = not index_path.exists()
    with index_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _task_index_value(task: dict[str, Any]) -> str:
    workers = ",".join(str(item) for item in task.get("worker_providers", []))
    return f"{task.get('mode')} workers={workers} batch={task.get('batch_size')}"


def _clustering_index_value(clustering: dict[str, Any]) -> str:
    umap = clustering.get("umap", {})
    hdbscan = clustering.get("hdbscan", {})
    noise_recluster = clustering.get("noise_recluster", {})
    return (
        "umap "
        f"n_neighbors={umap.get('n_neighbors')} "
        f"n_components={umap.get('n_components')} "
        f"min_dist={umap.get('min_dist')} "
        "hdbscan "
        f"min_cluster_size={hdbscan.get('min_cluster_size')} "
        f"min_samples={hdbscan.get('min_samples')} "
        f"epsilon={hdbscan.get('cluster_selection_epsilon')} "
        "noise_recluster "
        f"enabled={noise_recluster.get('enabled')} "
        f"min_ratio={noise_recluster.get('min_noise_ratio')} "
        f"min_count={noise_recluster.get('min_noise_count')} "
        f"strategy={noise_recluster.get('strategy')}"
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "run"
