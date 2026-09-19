from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from industrial_hazard_analysis.providers.base import LLMProvider


LLM_TASK_MODES = {"arbitration", "lightweight"}


@dataclass(frozen=True)
class LLMTaskPolicy:
    task_key: str
    mode: str
    worker_providers: list[LLMProvider]
    arbitrator_provider: LLMProvider | None
    batch_size: int
    prompt_path: str
    batch_prompt_path: str
    arbitration_prompt_path: str | None

    @property
    def is_lightweight(self) -> bool:
        return self.mode == "lightweight"

    @property
    def is_arbitration(self) -> bool:
        return self.mode == "arbitration"


def build_llm_task_policies(
    llm_config: dict[str, Any],
    *,
    providers_by_name: dict[str, LLMProvider],
    default_worker_providers: list[LLMProvider],
    default_arbitrator_provider: LLMProvider,
    task_defaults: set[str],
) -> dict[str, LLMTaskPolicy]:
    tasks_config = llm_config.get("tasks", {})
    task_keys = sorted(task_defaults | set(tasks_config))
    return {
        task_key: build_llm_task_policy(
            llm_config,
            task_key,
            providers_by_name=providers_by_name,
            default_worker_providers=default_worker_providers,
            default_arbitrator_provider=default_arbitrator_provider,
        )
        for task_key in task_keys
    }


def build_llm_task_policy(
    llm_config: dict[str, Any],
    task_key: str,
    *,
    providers_by_name: dict[str, LLMProvider],
    default_worker_providers: list[LLMProvider],
    default_arbitrator_provider: LLMProvider,
) -> LLMTaskPolicy:
    task_config = llm_config.get("tasks", {}).get(task_key, {})
    mode = _task_mode(task_key, task_config)
    worker_providers = _task_workers(
        task_key,
        mode,
        task_config,
        providers_by_name=providers_by_name,
        default_worker_providers=default_worker_providers,
    )
    arbitrator_provider = _task_arbitrator(
        task_key,
        mode,
        task_config,
        default_arbitrator_provider=default_arbitrator_provider,
    )
    batch_size = _task_batch_size(task_key, task_config)
    prompt_path = _required_task_path(task_key, task_config, "prompt_path")
    batch_prompt_path = _required_task_path(task_key, task_config, "batch_prompt_path")
    arbitration_prompt_path = (
        _required_task_path(task_key, task_config, "arbitration_prompt_path")
        if mode == "arbitration"
        else task_config.get("arbitration_prompt_path")
    )
    return LLMTaskPolicy(
        task_key=task_key,
        mode=mode,
        worker_providers=worker_providers,
        arbitrator_provider=arbitrator_provider,
        batch_size=batch_size,
        prompt_path=prompt_path,
        batch_prompt_path=batch_prompt_path,
        arbitration_prompt_path=arbitration_prompt_path,
    )


def apply_llm_task_override(
    llm_config: dict[str, Any],
    task_key: str,
    *,
    mode: str | None = None,
    worker_providers: str | list[str] | None = None,
    arbitrator_provider: str | None = None,
    batch_size: int | None = None,
) -> None:
    task_config = llm_config.setdefault("tasks", {}).setdefault(task_key, {})
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"llm.tasks.{task_key}.batch_size must be >= 1")
        task_config["batch_size"] = batch_size

    if worker_providers is not None:
        task_config["worker_providers"] = _provider_names(worker_providers)

    if arbitrator_provider is not None:
        task_config["arbitrator_provider"] = arbitrator_provider

    if mode is None:
        return

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in LLM_TASK_MODES:
        raise ValueError(f"llm.tasks.{task_key}.mode must be 'arbitration' or 'lightweight'")

    task_config["mode"] = normalized_mode
    if worker_providers is None:
        task_config["worker_providers"] = _default_worker_names(llm_config, task_config, normalized_mode)
    if normalized_mode == "arbitration":
        task_config.setdefault("arbitrator_provider", llm_config["arbitrator"]["provider"])


def _task_mode(task_key: str, task_config: dict[str, Any]) -> str:
    mode = str(task_config.get("mode", "arbitration")).strip().lower()
    if mode not in LLM_TASK_MODES:
        raise ValueError(f"llm.tasks.{task_key}.mode must be 'arbitration' or 'lightweight'")
    return mode


def _task_workers(
    task_key: str,
    mode: str,
    task_config: dict[str, Any],
    *,
    providers_by_name: dict[str, LLMProvider],
    default_worker_providers: list[LLMProvider],
) -> list[LLMProvider]:
    names = task_config.get("worker_providers")
    if not names:
        workers = list(default_worker_providers)
    else:
        workers = []
        for name in names:
            if name not in providers_by_name:
                raise ValueError(f"Unknown worker provider {name!r} in llm.tasks.{task_key}.worker_providers")
            workers.append(providers_by_name[name])

    if mode == "lightweight" and len(workers) != 1:
        raise ValueError(f"llm.tasks.{task_key}.worker_providers must contain exactly one provider in lightweight mode")
    if mode == "arbitration" and len(workers) < 2:
        raise ValueError(f"llm.tasks.{task_key}.worker_providers must contain at least two providers in arbitration mode")
    return workers


def _task_arbitrator(
    task_key: str,
    mode: str,
    task_config: dict[str, Any],
    *,
    default_arbitrator_provider: LLMProvider,
) -> LLMProvider | None:
    if mode == "lightweight":
        return None

    name = task_config.get("arbitrator_provider")
    if name and name != default_arbitrator_provider.name:
        raise ValueError(
            f"llm.tasks.{task_key}.arbitrator_provider={name!r} does not match "
            f"configured llm.arbitrator provider {default_arbitrator_provider.name!r}"
        )
    return default_arbitrator_provider


def _task_batch_size(
    task_key: str,
    task_config: dict[str, Any],
) -> int:
    if "batch_size" not in task_config:
        raise ValueError(f"llm.tasks.{task_key}.batch_size is required")
    batch_size = int(task_config["batch_size"])
    if batch_size < 1:
        raise ValueError(f"llm.tasks.{task_key}.batch_size must be >= 1")
    return batch_size


def _required_task_path(task_key: str, task_config: dict[str, Any], field: str) -> str:
    value = str(task_config.get(field, "")).strip()
    if not value:
        raise ValueError(f"llm.tasks.{task_key}.{field} is required")
    return value


def _provider_names(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        names = [name.strip() for name in value.split(",")]
    else:
        names = [str(name).strip() for name in value]
    return [name for name in names if name]


def _default_worker_names(
    llm_config: dict[str, Any],
    task_config: dict[str, Any],
    mode: str,
) -> list[str]:
    current = _provider_names(task_config.get("worker_providers", []))
    if mode == "arbitration":
        if len(current) >= 2:
            return current
        return [item["provider"] for item in llm_config.get("extractors", [])]

    if len(current) == 1:
        return current
    return [task_config.get("arbitrator_provider") or llm_config["arbitrator"]["provider"]]
