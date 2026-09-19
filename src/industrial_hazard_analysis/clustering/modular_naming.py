from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industrial_hazard_analysis.config import load_config_with_prompts
from industrial_hazard_analysis.providers.json_schema import (
    get_schema_definition,
    validate_schema_payload,
)
from industrial_hazard_analysis.providers.openai_compatible import (
    JsonlCache,
    OpenAICompatibleChatClient,
    _parse_json_object,
)


SUPPORTED_FIELDS = {"full_text", "finding", "object", "risk_scene", "loc_detail"}
MODULE_ORDER = (
    "BASE_TASK",
    "FIELD_DEFINITION",
    "CHANNEL_POLICY",
    "LEVEL_CONTEXT",
    "NAMING_CONSTRAINTS",
    "OUTPUT_SCHEMA",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_naming_config(path: str | Path) -> dict[str, Any]:
    return load_config_with_prompts(path)


@dataclass(frozen=True)
class PromptBuild:
    text: str
    prompt_version: str
    builder_hash: str
    prompt_hash: str
    injected_fields: tuple[str, ...]
    channel_policy_key: str
    topic_level: str


@dataclass(frozen=True)
class NamingCallResult:
    response: dict[str, Any]
    attempt_count: int
    retry_count: int
    from_cache: bool
    cache_key: str


class ModularNamingPromptBuilder:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.prompt_version = str(config["prompt_version"])
        self.field_definitions = dict(config["field_definitions"])
        self.channel_policies = dict(config["channel_policies"])
        self.modules = dict(config["modules"])
        self.level_contexts = dict(config["level_contexts"])
        self.input_json_label = str(config["input_json_label"])
        missing_modules = [name for name in MODULE_ORDER if name not in self.modules]
        if missing_modules:
            raise ValueError(f"Naming modules are missing: {missing_modules}")
        unknown_fields = set(self.field_definitions) - SUPPORTED_FIELDS
        if unknown_fields:
            raise ValueError(f"Unsupported field definitions: {sorted(unknown_fields)}")
        self.builder_hash = sha256_text(
            canonical_json(
                {
                    "prompt_version": self.prompt_version,
                    "field_definitions": self.field_definitions,
                    "channel_policies": self.channel_policies,
                    "modules": {name: self.modules[name] for name in MODULE_ORDER},
                    "level_contexts": self.level_contexts,
                    "input_json_label": self.input_json_label,
                }
            )
        )

    def build(self, payload: dict[str, Any]) -> PromptBuild:
        fields = tuple(str(item) for item in payload.get("channel_fields", []))
        if not fields:
            raise ValueError("channel_fields cannot be empty")
        unsupported = set(fields) - SUPPORTED_FIELDS
        if unsupported:
            raise ValueError(f"Unsupported channel fields: {sorted(unsupported)}")
        policy_key = "+".join(fields)
        try:
            policy = self.channel_policies[policy_key]
        except KeyError as exc:
            raise ValueError(f"No channel policy for {policy_key!r}") from exc

        topic_level = str(payload.get("topic_level", ""))
        if topic_level == "global":
            if payload.get("parent_context") is not None:
                raise ValueError("Global topics must use parent_context=null")
            level_context = str(self.level_contexts["global"]).format(
                ANALYSIS_PATH=canonical_json(payload.get("analysis_path", list(fields)))
            )
        elif topic_level == "level_2":
            parent = payload.get("parent_context")
            if not isinstance(parent, dict):
                raise ValueError("Level-2 topics require a parent_context object")
            required_parent = {"cluster_id", "cluster_name", "channel_fields"}
            missing_parent = sorted(required_parent - set(parent))
            if missing_parent:
                raise ValueError(f"parent_context is missing: {missing_parent}")
            level_context = str(self.level_contexts["level_2"]).format(
                ANALYSIS_PATH=canonical_json(payload.get("analysis_path", [])),
                PARENT_CONTEXT=canonical_json(parent),
            )
        else:
            raise ValueError(f"Unsupported topic_level: {topic_level!r}")

        definitions = "\n".join(
            f"- {field}: {self.field_definitions[field]}" for field in fields
        )
        rendered = {
            "BASE_TASK": str(self.modules["BASE_TASK"]),
            "FIELD_DEFINITION": str(self.modules["FIELD_DEFINITION"]).replace(
                "{FIELD_DEFINITIONS}", definitions
            ),
            "CHANNEL_POLICY": str(self.modules["CHANNEL_POLICY"])
            .replace("{CHANNEL_FIELDS}", canonical_json(list(fields)))
            .replace("{CHANNEL_POLICY}", str(policy)),
            "LEVEL_CONTEXT": str(self.modules["LEVEL_CONTEXT"]).replace(
                "{LEVEL_CONTEXT}", level_context
            ),
            "NAMING_CONSTRAINTS": str(self.modules["NAMING_CONSTRAINTS"]),
            "OUTPUT_SCHEMA": str(self.modules["OUTPUT_SCHEMA"]),
        }
        prompt_text = "\n\n".join(rendered[name].strip() for name in MODULE_ORDER)
        prompt_text += "\n\n" + self.input_json_label + "\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2
        )
        return PromptBuild(
            text=prompt_text,
            prompt_version=self.prompt_version,
            builder_hash=self.builder_hash,
            prompt_hash=sha256_text(prompt_text),
            injected_fields=fields,
            channel_policy_key=policy_key,
            topic_level=topic_level,
        )


class ModularClusterNamingClient:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        cache_path: str | Path | None = None,
        offline_cache_only: bool = False,
    ):
        self.config = config
        self.model_config = dict(config["model"])
        self.generation = dict(self.model_config["generation"])
        self.max_attempts = int(self.model_config["max_attempts"])
        self.backoff_seconds = [
            float(value) for value in self.model_config.get("retry_backoff_seconds", [])
        ]
        self.offline_cache_only = offline_cache_only
        self.builder = ModularNamingPromptBuilder(config)
        self.client = OpenAICompatibleChatClient(
            api_key_env=str(self.model_config["api_key_env"]),
            base_url=str(self.model_config["base_url"]),
            model=str(self.model_config["model"]),
            provider_name=str(self.model_config["provider"]),
            timeout=float(self.model_config["timeout_seconds"]),
            max_retries=1,
        )
        self.cache = JsonlCache(cache_path)
        schema = get_schema_definition("cluster_naming_v2_object")
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "cluster_naming_v2",
                "strict": True,
                "schema": schema,
            },
        }
        self.schema_hash = sha256_text(canonical_json(schema))

    def request_options(self) -> dict[str, Any]:
        return {
            "temperature": float(self.generation["temperature"]),
            "top_p": float(self.generation["top_p"]),
            "max_completion_tokens": int(self.generation["max_completion_tokens"]),
            "enable_thinking": bool(self.generation["enable_thinking"]),
            "response_format": self.response_format,
        }

    def name(self, payload: dict[str, Any]) -> tuple[NamingCallResult, PromptBuild]:
        prompt = self.builder.build(payload)
        key = sha256_text(
            canonical_json(
                {
                    "provider": self.model_config["provider"],
                    "model": self.model_config["model"],
                    "prompt_version": prompt.prompt_version,
                    "prompt_hash": prompt.prompt_hash,
                    "schema_hash": self.schema_hash,
                    "request_options": self.request_options(),
                }
            )
        )
        cached = self.cache.get(key)
        if cached is not None:
            if not isinstance(cached, dict) or "response" not in cached:
                raise ValueError("Modular naming cache entry has an unsupported format")
            response = validate_schema_payload(
                "cluster_naming_v2_object", cached["response"]
            )
            return (
                NamingCallResult(
                    response=response,
                    attempt_count=int(cached.get("attempt_count", 1)),
                    retry_count=int(cached.get("retry_count", 0)),
                    from_cache=True,
                    cache_key=key,
                ),
                prompt,
            )
        if self.offline_cache_only:
            raise RuntimeError(
                "Offline cache-only mode blocked an uncached modular naming request: "
                f"cluster_id={payload.get('cluster_id')!r}"
            )

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                content = self.client.chat(
                    prompt.text,
                    request_overrides=self.request_options(),
                )
                response = validate_schema_payload(
                    "cluster_naming_v2_object", _parse_json_object(content)
                )
                expected_id = str(payload["cluster_id"])
                if str(response["cluster_id"]).strip() != expected_id:
                    raise ValueError(
                        "Naming response cluster_id mismatch: "
                        f"expected={expected_id!r}, received={response['cluster_id']!r}"
                    )
                if not str(response["cluster_name"]).strip():
                    raise ValueError("Naming response contains an empty cluster_name")
                if not str(response["explanation"]).strip():
                    raise ValueError("Naming response contains an empty explanation")
                cache_value = {
                    "response": response,
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                }
                self.cache.set(key, cache_value)
                return (
                    NamingCallResult(
                        response=response,
                        attempt_count=attempt,
                        retry_count=attempt - 1,
                        from_cache=False,
                        cache_key=key,
                    ),
                    prompt,
                )
            except Exception as exc:  # live provider and schema failures share one rule
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                delay_index = min(attempt - 1, max(len(self.backoff_seconds) - 1, 0))
                delay = self.backoff_seconds[delay_index] if self.backoff_seconds else 0
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError(
            f"Modular naming failed after {self.max_attempts} attempt(s): {last_error}"
        ) from last_error
