from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config_with_prompts(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = load_config(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    prompts_path = config.pop("prompts_path", None)
    if prompts_path is None:
        return config
    prompt_file = config_path.parent / prompts_path
    prompts = load_config(prompt_file)
    if not isinstance(prompts, dict):
        raise ValueError(f"Prompts must be a mapping: {prompt_file}")
    overlapping_keys = config.keys() & prompts.keys()
    if overlapping_keys:
        raise ValueError(f"Duplicate configuration and prompt keys: {sorted(overlapping_keys)}")
    return {**config, **prompts}


def project_path(root: str | Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(root) / path
