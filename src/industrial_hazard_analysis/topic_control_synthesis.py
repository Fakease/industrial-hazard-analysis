# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from industrial_hazard_analysis.config import load_config_with_prompts
from industrial_hazard_analysis.providers.json_schema import (
    get_schema_definition,
    validate_schema_payload,
)
from industrial_hazard_analysis.providers.openai_compatible import (
    JsonlCache,
    NonRetryableChatError,
    OpenAICompatibleChatClient,
    _parse_json_object,
)


RELATION_LABELS = (
    "same_control_same_failure",
    "same_control_different_failure",
    "related_but_different_control",
    "not_same_or_insufficient",
)
POSITIVE_RELATIONS = {
    "same_control_same_failure",
    "same_control_different_failure",
}
TASK_SCHEMA = {
    "relation_batch": "topic_control_relation_batch_object",
    "group_validation": "topic_control_group_validation_object",
    "unit_validation_batch": "topic_control_unit_validation_batch_object",
    "knowledge_card_semantics": "topic_control_knowledge_card_semantics_object",
    "synthesis_group_validation": "safety_control_synthesis_validation_object",
    "synthesis_unit_evidence_batch": "safety_control_unit_evidence_batch_object",
    "synthesis_knowledge_card_semantics": (
        "safety_control_knowledge_card_semantics_object"
    ),
}
TASK_PROMPT_KEY = {
    "relation_batch": "pair_task",
    "group_validation": "group_task",
    "unit_validation_batch": "unit_task",
    "knowledge_card_semantics": "card_task",
    "synthesis_group_validation": "synthesis_group_task",
    "synthesis_unit_evidence_batch": "synthesis_unit_task",
    "synthesis_knowledge_card_semantics": "synthesis_card_task",
}
PROVIDER_EXCLUDED_SCHEMA_KEYWORDS = {
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
}


def provider_schema(schema_name: str) -> dict[str, Any]:
    """Return the self-contained DashScope transport schema.

    DashScope's strict-schema converter rejects a small set of standard array
    keywords. They are removed only from the transmitted schema; every response
    is still validated against the complete local Draft 7 schema.
    """
    return get_schema_definition(
        schema_name,
        inline_references=True,
        excluded_keywords=PROVIDER_EXCLUDED_SCHEMA_KEYWORDS,
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_topic_control_config(path: str | Path) -> dict[str, Any]:
    payload = load_config_with_prompts(path)
    required = {
        "prompt_version",
        "candidate_generation",
        "group_construction",
        "unit_validation",
        "model",
        "shared_definitions",
        "pair_task",
        "group_task",
        "unit_task",
        "card_task",
        "output_constraints",
        "input_json_label",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Topic-control configuration is missing: {missing}")
    return payload


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Centroids must be a two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Centroids contain zero-norm rows")
    return values / norms


def _pair_key(topic_a: str, topic_b: str) -> tuple[str, str]:
    if topic_a == topic_b:
        raise ValueError("A topic pair cannot contain the same topic twice")
    return tuple(sorted((str(topic_a), str(topic_b))))


def generate_candidate_pairs(
    topics: pd.DataFrame,
    centroids: np.ndarray,
    *,
    top_k: int = 5,
    high_similarity_quantile: float = 0.95,
    threshold_reference_scope: str = "cross_parent",
    selected_rule: str = "mutual_top_k_union_high_similarity",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate topic-pair candidates without consulting names or examples."""
    required = {"topic_id", "parent_context"}
    missing = sorted(required - set(topics.columns))
    if missing:
        raise ValueError(f"Topic table is missing columns: {missing}")
    if topics["topic_id"].astype(str).duplicated().any():
        raise ValueError("topic_id values must be unique")
    if not 0 < float(high_similarity_quantile) < 1:
        raise ValueError("high_similarity_quantile must lie in (0, 1)")
    if int(top_k) < 1 or int(top_k) >= len(topics):
        raise ValueError("top_k must be between 1 and topic_count - 1")

    centers = normalize_rows(centroids)
    if centers.shape[0] != len(topics):
        raise ValueError("Topic row count and centroid row count differ")
    similarities = np.clip(centers @ centers.T, -1.0, 1.0)
    topic_ids = topics["topic_id"].astype(str).tolist()
    parents = topics["parent_context"].astype(str).tolist()

    pair_rows: list[dict[str, Any]] = []
    for left in range(len(topic_ids)):
        for right in range(left + 1, len(topic_ids)):
            pair_rows.append(
                {
                    "topic_a": topic_ids[left],
                    "topic_b": topic_ids[right],
                    "parent_context_a": parents[left],
                    "parent_context_b": parents[right],
                    "same_parent_context": parents[left] == parents[right],
                    "cosine_similarity": float(similarities[left, right]),
                }
            )
    all_pairs = pd.DataFrame(pair_rows)
    reference = all_pairs
    if threshold_reference_scope == "cross_parent":
        reference = all_pairs[~all_pairs["same_parent_context"]]
    elif threshold_reference_scope != "all_pairs":
        raise ValueError(
            "threshold_reference_scope must be 'cross_parent' or 'all_pairs'"
        )
    threshold = float(
        reference["cosine_similarity"].quantile(float(high_similarity_quantile))
    )

    directed_top: set[tuple[str, str]] = set()
    topic_index = {topic_id: index for index, topic_id in enumerate(topic_ids)}
    for topic_id in topic_ids:
        index = topic_index[topic_id]
        neighbors = [
            (topic_ids[other], float(similarities[index, other]))
            for other in range(len(topic_ids))
            if other != index
        ]
        neighbors.sort(key=lambda item: (-item[1], item[0]))
        directed_top.update((topic_id, other) for other, _ in neighbors[: int(top_k)])

    top_edges = {_pair_key(left, right) for left, right in directed_top}
    mutual_edges = {
        _pair_key(left, right)
        for left, right in directed_top
        if (right, left) in directed_top
    }
    high_edges = {
        _pair_key(row.topic_a, row.topic_b)
        for row in all_pairs.itertuples(index=False)
        if float(row.cosine_similarity) >= threshold
    }
    rule_edges = {
        "top_k": top_edges,
        "high_similarity": high_edges,
        "mutual_top_k": mutual_edges,
        "top_k_union_high_similarity": top_edges | high_edges,
        "mutual_top_k_union_high_similarity": mutual_edges | high_edges,
    }
    try:
        selected_edges = rule_edges[selected_rule]
    except KeyError as exc:
        raise ValueError(f"Unsupported selected candidate rule: {selected_rule}") from exc

    all_pairs["pair_key"] = [
        "|".join(_pair_key(left, right))
        for left, right in zip(all_pairs["topic_a"], all_pairs["topic_b"])
    ]
    for rule_name, edges in rule_edges.items():
        all_pairs[f"rule_{rule_name}"] = [
            _pair_key(left, right) in edges
            for left, right in zip(all_pairs["topic_a"], all_pairs["topic_b"])
        ]

    selected = all_pairs[
        all_pairs[f"rule_{selected_rule}"]
    ].copy()
    selected = selected.sort_values(
        ["cosine_similarity", "topic_a", "topic_b"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected.insert(0, "pair_id", [f"PAIR-{index:04d}" for index in range(1, len(selected) + 1)])
    selected["candidate_rule"] = [
        "+".join(
            label
            for label, column in (
                ("top_k", "rule_top_k"),
                ("mutual_top_k", "rule_mutual_top_k"),
                ("high_similarity", "rule_high_similarity"),
            )
            if bool(row[column])
        )
        for _, row in selected.iterrows()
    ]

    comparison_rows = []
    theoretical = len(all_pairs)
    for rule_name in (
        "top_k",
        "high_similarity",
        "mutual_top_k",
        "top_k_union_high_similarity",
        "mutual_top_k_union_high_similarity",
    ):
        edge_count = len(rule_edges[rule_name])
        comparison_rows.append(
            {
                "candidate_rule": rule_name,
                "pair_count": edge_count,
                "theoretical_pair_count": theoretical,
                "candidate_share": edge_count / theoretical,
                "compression_ratio": 1.0 - edge_count / theoretical,
                "selected": rule_name == selected_rule,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    metadata = {
        "topic_count": len(topics),
        "theoretical_pair_count": theoretical,
        "top_k": int(top_k),
        "high_similarity_quantile": float(high_similarity_quantile),
        "threshold_reference_scope": threshold_reference_scope,
        "high_similarity_threshold": threshold,
        "selected_rule": selected_rule,
        "selected_pair_count": len(selected),
        "selected_cross_parent_pair_count": int(
            (~selected["same_parent_context"]).sum()
        ),
        "topic_names_used": False,
        "topic_explanations_used": False,
        "sanity_cases_used": False,
    }
    return selected, comparison, metadata


@dataclass(frozen=True)
class PromptBuild:
    task: str
    text: str
    prompt_version: str
    builder_hash: str
    prompt_hash: str
    schema_name: str
    schema_hash: str


class TopicControlPromptBuilder:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.prompt_version = str(config["prompt_version"])
        self.shared_definitions = str(config["shared_definitions"]).strip()
        self.output_constraints = str(config["output_constraints"]).strip()
        self.input_json_label = str(config["input_json_label"])
        self.tasks = {
            task: str(config[key]).strip()
            for task, key in TASK_PROMPT_KEY.items()
            if key in config
        }
        self.builder_hash = sha256_text(
            canonical_json(
                {
                    "prompt_version": self.prompt_version,
                    "shared_definitions": self.shared_definitions,
                    "tasks": self.tasks,
                    "output_constraints": self.output_constraints,
                    "input_json_label": self.input_json_label,
                    "schemas": {
                        task: provider_schema(schema_name)
                        for task, schema_name in TASK_SCHEMA.items()
                        if task in self.tasks
                    },
                }
            )
        )

    def build(self, task: str, payload: dict[str, Any]) -> PromptBuild:
        if task not in TASK_SCHEMA or task not in self.tasks:
            raise ValueError(f"Unsupported topic-control task: {task}")
        schema_name = TASK_SCHEMA[task]
        schema = provider_schema(schema_name)
        text = "\n\n".join(
            (
                self.shared_definitions,
                self.tasks[task],
                self.output_constraints,
                self.input_json_label + "\n"
                + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            )
        )
        return PromptBuild(
            task=task,
            text=text,
            prompt_version=self.prompt_version,
            builder_hash=self.builder_hash,
            prompt_hash=sha256_text(text),
            schema_name=schema_name,
            schema_hash=sha256_text(canonical_json(schema)),
        )


def build_relation_requests(
    candidate_pairs: pd.DataFrame,
    topic_objects: dict[str, dict[str, Any]],
    builder: TopicControlPromptBuilder,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    required = {"pair_id", "topic_a", "topic_b"}
    missing = sorted(required - set(candidate_pairs.columns))
    if missing:
        raise ValueError(f"Candidate table is missing: {missing}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = candidate_pairs.sort_values("pair_id", kind="stable").to_dict("records")
    requests: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        request_id = f"REL-{start // batch_size + 1:04d}"
        topic_ids = sorted(
            {str(row[side]) for row in batch for side in ("topic_a", "topic_b")}
        )
        payload = {
            "request_id": request_id,
            "topics": [topic_objects[topic_id] for topic_id in topic_ids],
            "pairs": [
                {
                    "pair_id": str(row["pair_id"]),
                    "topic_a": str(row["topic_a"]),
                    "topic_b": str(row["topic_b"]),
                }
                for row in batch
            ],
        }
        prompt = builder.build("relation_batch", payload)
        requests.append(request_record(prompt, payload))
    return requests


def request_record(prompt: PromptBuild, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(payload["request_id"]),
        "task": prompt.task,
        "prompt_version": prompt.prompt_version,
        "prompt_builder_hash": prompt.builder_hash,
        "prompt_hash": prompt.prompt_hash,
        "schema_name": prompt.schema_name,
        "schema_hash": prompt.schema_hash,
        "payload_sha256": sha256_text(canonical_json(payload)),
        "prompt_char_count": len(prompt.text),
        "payload": payload,
        "prompt": prompt.text,
    }


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def validate_external_payload_structure(payload: dict[str, Any]) -> None:
    forbidden_keys = {
        "record_id",
        "raw_record_id",
        "enterprise",
        "enterprise_id",
        "enterprise_name",
        "source_topic_key",
        "member_unit_ids",
    }
    observed = {key.lower() for key in _walk_keys(payload)}
    leaked = sorted(forbidden_keys & observed)
    if leaked:
        raise ValueError(f"External payload contains forbidden keys: {leaked}")


@dataclass(frozen=True)
class StructuredCallResult:
    request_id: str
    task: str
    response: dict[str, Any]
    attempt_count: int
    retry_count: int
    from_cache: bool
    cache_key: str
    latency_seconds: float
    usage: dict[str, Any]
    provider_response_id: str
    raw_response_sha256: str
    normalization_events: tuple[dict[str, Any], ...]


_META_REASONING_MARKERS = (
    "*修正思考*",
    "修正思考：",
    "再次修正：",
    "重新审视：",
    "让我们再看",
)


def _concise_contract_string(value: str, maximum: int) -> str:
    text = str(value).strip()
    marker_positions = [
        position
        for marker in _META_REASONING_MARKERS
        for position in [text.find(marker)]
        if position >= 0
    ]
    if marker_positions:
        first_marker = min(marker_positions)
        text = text[:first_marker].strip() if first_marker > 0 else ""
    if len(text) <= maximum:
        return text
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？；])|[\r\n]+", text)
        if item.strip()
    ]
    if sentences:
        text = sentences[0]
    return text[:maximum].strip()


def normalize_provider_response(
    task: str,
    response: dict[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Apply narrow, auditable transport-contract repairs before validation.

    Topic/unit identities, Boolean evidence judgments, and positive evidence
    membership are never invented here. For synthesis groups, a model-emitted
    control supported by fewer than two distinct topics is conservatively
    demoted to the exclusion list, which enforces the frozen minimum subgroup
    rule without inventing a replacement control. The full local JSON Schema
    and task identity checks remain authoritative after normalization.
    """
    schema_name = TASK_SCHEMA[task]
    schema = get_schema_definition(schema_name, inline_references=True)
    normalized = deepcopy(response)
    events: list[dict[str, Any]] = []

    def visit(value: Any, rule: dict[str, Any], path: str) -> Any:
        rule_type = rule.get("type")
        if isinstance(rule_type, list):
            rule_type = next(
                (item for item in rule_type if item != "null"), rule_type[0]
            )
        if rule_type == "object" and isinstance(value, dict):
            properties = rule.get("properties", {})
            return {
                key: visit(child, properties.get(key, {}), f"{path}.{key}")
                for key, child in value.items()
            }
        if rule_type == "array" and isinstance(value, list):
            items = [
                visit(child, rule.get("items", {}), f"{path}[{index}]")
                for index, child in enumerate(value)
            ]
            if bool(rule.get("uniqueItems")):
                deduplicated: list[Any] = []
                seen: set[str] = set()
                for item in items:
                    key = canonical_json(item)
                    if key not in seen:
                        seen.add(key)
                        deduplicated.append(item)
                if len(deduplicated) != len(items):
                    events.append(
                        {
                            "path": path,
                            "action": "deduplicate_array_preserving_order",
                            "before_count": len(items),
                            "after_count": len(deduplicated),
                        }
                    )
                items = deduplicated
            return items
        if rule_type == "string" and isinstance(value, str):
            maximum = rule.get("maxLength")
            markers_present = any(marker in value for marker in _META_REASONING_MARKERS)
            if maximum is not None and (len(value) > int(maximum) or markers_present):
                repaired = _concise_contract_string(value, int(maximum))
                if repaired != value:
                    events.append(
                        {
                            "path": path,
                            "action": "retain_first_conclusion_within_max_length",
                            "before_length": len(value),
                            "after_length": len(repaired),
                            "before_sha256": sha256_text(value),
                        }
                    )
                return repaired
        return value

    normalized = visit(normalized, schema, "$")
    if task == "relation_batch":
        for index, item in enumerate(normalized.get("judgments", [])):
            if (
                item.get("relation") not in POSITIVE_RELATIONS
                and str(item.get("shared_control_name", "")).strip()
            ):
                previous = str(item["shared_control_name"])
                item["shared_control_name"] = ""
                events.append(
                    {
                        "path": f"$.judgments[{index}].shared_control_name",
                        "action": "clear_control_name_for_nonpositive_relation",
                        "before_length": len(previous),
                        "after_length": 0,
                        "before_sha256": sha256_text(previous),
                    }
                )
    elif task == "synthesis_group_validation":
        controls = normalized.get("validated_controls", [])
        assignments = normalized.get("topic_assignments", [])
        excluded_topics = normalized.get("excluded_topics", [])
        if (
            isinstance(controls, list)
            and isinstance(assignments, list)
            and isinstance(excluded_topics, list)
        ):
            explicitly_excluded = {
                str(item.get("topic_id", ""))
                for item in excluded_topics
                if isinstance(item, dict)
            }
            overlapping_assignments = [
                item
                for item in assignments
                if isinstance(item, dict)
                and str(item.get("topic_id", "")) in explicitly_excluded
            ]
            if overlapping_assignments:
                assignments = [
                    item
                    for item in assignments
                    if not isinstance(item, dict)
                    or str(item.get("topic_id", "")) not in explicitly_excluded
                ]
                normalized["topic_assignments"] = assignments
                events.append(
                    {
                        "path": "$.topic_assignments",
                        "action": "prefer_explicit_exclusion_over_duplicate_assignment",
                        "removed_assignment_count": len(overlapping_assignments),
                        "affected_topic_count": len(
                            {
                                str(item.get("topic_id", ""))
                                for item in overlapping_assignments
                            }
                        ),
                    }
                )
            assigned_topics_by_control: dict[str, set[str]] = {}
            for assignment in assignments:
                if not isinstance(assignment, dict):
                    continue
                control_id = str(assignment.get("assigned_control_id", ""))
                topic_id = str(assignment.get("topic_id", ""))
                assigned_topics_by_control.setdefault(control_id, set()).add(topic_id)
            singleton_control_ids = {
                str(control.get("subgroup_id", ""))
                for control in controls
                if isinstance(control, dict)
                and len(
                    assigned_topics_by_control.get(
                        str(control.get("subgroup_id", "")), set()
                    )
                )
                < 2
            }
            if singleton_control_ids:
                previous_decision = str(normalized.get("decision", ""))
                demoted_topic_ids = sorted(
                    {
                        str(item.get("topic_id", ""))
                        for item in assignments
                        if isinstance(item, dict)
                        and str(item.get("assigned_control_id", ""))
                        in singleton_control_ids
                    }
                )
                normalized["validated_controls"] = [
                    item
                    for item in controls
                    if str(item.get("subgroup_id", ""))
                    not in singleton_control_ids
                ]
                normalized["topic_assignments"] = [
                    item
                    for item in assignments
                    if str(item.get("assigned_control_id", ""))
                    not in singleton_control_ids
                ]
                existing_excluded = {
                    str(item.get("topic_id", ""))
                    for item in excluded_topics
                    if isinstance(item, dict)
                }
                normalized["excluded_topics"] = list(excluded_topics) + [
                    {
                        "topic_id": topic_id,
                        "reason": "未与其他主题形成至少含两个主题的共同安全控制。",
                    }
                    for topic_id in demoted_topic_ids
                    if topic_id not in existing_excluded
                ]
                retained_control_count = len(normalized["validated_controls"])
                if retained_control_count == 0:
                    normalized["decision"] = "reject"
                elif retained_control_count == 1:
                    normalized["decision"] = (
                        "accept_after_excluding_topics"
                        if normalized["excluded_topics"]
                        else "accept"
                    )
                else:
                    normalized["decision"] = "split"
                events.append(
                    {
                        "path": "$.validated_controls",
                        "action": "demote_controls_with_fewer_than_two_topics",
                        "removed_control_count": len(singleton_control_ids),
                        "demoted_topic_count": len(demoted_topic_ids),
                        "decision_before": previous_decision,
                        "decision_after": str(normalized["decision"]),
                    }
                )
    return normalized, tuple(events)


class TopicControlSynthesisClient:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        cache_path: str | Path | None,
        offline_cache_only: bool = False,
    ):
        self.config = config
        self.model = dict(config["model"])
        self.builder = TopicControlPromptBuilder(config)
        self.cache = JsonlCache(cache_path)
        cache_file = Path(cache_path) if cache_path else None
        self.invalid_response_cache = JsonlCache(
            cache_file.with_name(f"{cache_file.stem}_invalid_responses.jsonl")
            if cache_file
            else None
        )
        self.offline_cache_only = bool(offline_cache_only)
        self.max_attempts = int(self.model["max_attempts"])
        self.backoff = [float(value) for value in self.model["retry_backoff_seconds"]]
        self.chat = OpenAICompatibleChatClient(
            api_key_env=str(self.model["api_key_env"]),
            base_url=str(self.model["base_url"]),
            model=str(self.model["model"]),
            provider_name=str(self.model["provider"]),
            timeout=float(self.model["timeout_seconds"]),
            max_retries=1,
        )

    def request_options(self, task: str, schema_name: str) -> dict[str, Any]:
        schema = provider_schema(schema_name)
        max_tokens = int(self.model["max_completion_tokens"][task])
        return {
            "temperature": float(self.model["temperature"]),
            "top_p": float(self.model["top_p"]),
            "max_completion_tokens": max_tokens,
            "enable_thinking": bool(self.model["enable_thinking"]),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }

    def run(self, record: dict[str, Any]) -> StructuredCallResult:
        task = str(record["task"])
        schema_name = str(record["schema_name"])
        if TASK_SCHEMA.get(task) != schema_name:
            raise ValueError("Request task/schema mismatch")
        payload = record["payload"]
        validate_external_payload_structure(payload)
        prompt = self.builder.build(task, payload)
        if prompt.prompt_hash != record["prompt_hash"]:
            raise ValueError("Request prompt hash changed after preparation")
        options = self.request_options(task, schema_name)
        cache_key = sha256_text(
            canonical_json(
                {
                    "provider": self.model["provider"],
                    "model": self.model["model"],
                    "task": task,
                    "prompt_version": prompt.prompt_version,
                    "prompt_hash": prompt.prompt_hash,
                    "schema_hash": prompt.schema_hash,
                    "request_options": options,
                }
            )
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            raw_cached = cached.get("provider_response_raw", cached["response"])
            response, normalization_events = normalize_provider_response(
                task, raw_cached
            )
            response = validate_schema_payload(schema_name, response)
            validate_response_identity(task, payload, response)
            return StructuredCallResult(
                request_id=str(payload["request_id"]),
                task=task,
                response=response,
                attempt_count=int(cached.get("attempt_count", 1)),
                retry_count=int(cached.get("retry_count", 0)),
                from_cache=True,
                cache_key=cache_key,
                latency_seconds=float(cached.get("latency_seconds", 0.0)),
                usage=dict(cached.get("usage", {})),
                provider_response_id=str(cached.get("provider_response_id", "")),
                raw_response_sha256=str(
                    cached.get(
                        "raw_response_sha256",
                        sha256_text(canonical_json(raw_cached)),
                    )
                ),
                normalization_events=normalization_events,
            )
        if self.offline_cache_only:
            raise RuntimeError(
                f"Offline cache-only mode blocked request {payload['request_id']}"
            )

        last_error: Exception | None = None
        attempts_used = 0
        for attempt in range(1, self.max_attempts + 1):
            attempts_used = attempt
            started = time.perf_counter()
            provider: dict[str, Any] | None = None
            raw_response: dict[str, Any] | None = None
            try:
                provider = self.chat.chat_response(
                    prompt.text,
                    request_overrides=options,
                )
                latency = time.perf_counter() - started
                content = provider["choices"][0]["message"]["content"]
                raw_response = _parse_json_object(content)
                response, normalization_events = normalize_provider_response(
                    task, raw_response
                )
                response = validate_schema_payload(schema_name, response)
                validate_response_identity(task, payload, response)
                cache_value = {
                    "response": response,
                    "provider_response_raw": raw_response,
                    "raw_response_sha256": sha256_text(canonical_json(raw_response)),
                    "normalization_events": list(normalization_events),
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "latency_seconds": latency,
                    "usage": provider.get("usage", {}),
                    "provider_response_id": provider.get("id", ""),
                }
                self.cache.set(cache_key, cache_value)
                return StructuredCallResult(
                    request_id=str(payload["request_id"]),
                    task=task,
                    response=response,
                    attempt_count=attempt,
                    retry_count=attempt - 1,
                    from_cache=False,
                    cache_key=cache_key,
                    latency_seconds=latency,
                    usage=dict(provider.get("usage", {})),
                    provider_response_id=str(provider.get("id", "")),
                    raw_response_sha256=cache_value["raw_response_sha256"],
                    normalization_events=normalization_events,
                )
            except NonRetryableChatError as exc:
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                if raw_response is not None:
                    raw_sha256 = sha256_text(canonical_json(raw_response))
                    failure_key = sha256_text(
                        canonical_json(
                            {
                                "cache_key": cache_key,
                                "attempt": attempt,
                                "raw_response_sha256": raw_sha256,
                            }
                        )
                    )
                    self.invalid_response_cache.set(
                        failure_key,
                        {
                            "request_id": str(payload["request_id"]),
                            "task": task,
                            "prompt_hash": prompt.prompt_hash,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "provider_response_id": str(
                                provider.get("id", "") if provider else ""
                            ),
                            "usage": dict(provider.get("usage", {}) if provider else {}),
                            "raw_response_sha256": raw_sha256,
                            "raw_response": raw_response,
                        },
                    )
                if attempt >= self.max_attempts:
                    break
                delay_index = min(attempt - 1, max(len(self.backoff) - 1, 0))
                delay = self.backoff[delay_index] if self.backoff else 0.0
                if delay:
                    time.sleep(delay)
        raise RuntimeError(
            f"Topic-control request {payload['request_id']} failed after "
            f"{attempts_used} attempt(s): {last_error}"
        ) from last_error


def validate_response_identity(
    task: str, payload: dict[str, Any], response: dict[str, Any]
) -> None:
    if str(response.get("request_id", "")) != str(payload["request_id"]):
        raise ValueError("Response request_id does not match request")
    if task == "relation_batch":
        expected = {
            str(item["pair_id"]): (str(item["topic_a"]), str(item["topic_b"]))
            for item in payload["pairs"]
        }
        observed = {str(item["pair_id"]): item for item in response["judgments"]}
        if set(observed) != set(expected):
            raise ValueError("Relation response pair IDs do not exactly match request")
        topic_lookup = {str(item["topic_id"]): item for item in payload["topics"]}
        for pair_id, item in observed.items():
            topic_a, topic_b = expected[pair_id]
            if (str(item["topic_a"]), str(item["topic_b"])) != (topic_a, topic_b):
                raise ValueError(f"Relation response topics changed for {pair_id}")
            if item["relation"] in POSITIVE_RELATIONS:
                if not str(item["shared_control_name"]).strip():
                    raise ValueError(f"Positive relation lacks control name: {pair_id}")
            elif str(item["shared_control_name"]).strip():
                raise ValueError(f"Negative relation supplied control name: {pair_id}")
            allowed_a = {
                str(sample["sample_id"])
                for sample in topic_lookup[topic_a]["representative_samples"]
            }
            allowed_b = {
                str(sample["sample_id"])
                for sample in topic_lookup[topic_b]["representative_samples"]
            }
            if not set(map(str, item["evidence_a"])).issubset(allowed_a):
                raise ValueError(f"Unknown evidence_a sample in {pair_id}")
            if not set(map(str, item["evidence_b"])).issubset(allowed_b):
                raise ValueError(f"Unknown evidence_b sample in {pair_id}")
    elif task == "group_validation":
        if str(response["group_id"]) != str(payload["group_id"]):
            raise ValueError("Group validation changed group_id")
        allowed = {str(item["topic_id"]) for item in payload["topics"]}
        assigned: list[str] = []
        subgroup_ids: list[str] = []
        for group in response["validated_groups"]:
            subgroup_ids.append(str(group["subgroup_id"]))
            members = list(map(str, group["member_topic_ids"]))
            if not set(members).issubset(allowed):
                raise ValueError("Group validation returned unknown topic IDs")
            assigned.extend(members)
        if len(subgroup_ids) != len(set(subgroup_ids)):
            raise ValueError("Validated subgroup IDs duplicate within one response")
        if len(assigned) != len(set(assigned)):
            raise ValueError("Validated subgroups overlap within one response")
        removed = set(map(str, response["removed_topic_ids"]))
        if not removed.issubset(allowed) or removed.intersection(assigned):
            raise ValueError("Invalid removed_topic_ids in group validation")
        if set(assigned).union(removed) != allowed:
            raise ValueError("Group validation did not account for every input topic")
        decision = str(response["decision"])
        if decision == "accept":
            if (
                len(response["validated_groups"]) != 1
                or set(assigned) != allowed
                or removed
            ):
                raise ValueError("Accepted group must preserve all input topics")
        elif decision == "remove_members":
            if len(response["validated_groups"]) != 1 or not removed:
                raise ValueError(
                    "remove_members requires one retained subgroup and removed topics"
                )
        elif decision == "split":
            if len(response["validated_groups"]) < 2:
                raise ValueError("Split decision requires at least two subgroups")
        elif decision == "reject":
            if response["validated_groups"] or removed != allowed:
                raise ValueError(
                    "Rejected group must remove all topics and return no subgroup"
                )
    elif task == "unit_validation_batch":
        if str(response["control_group_id"]) != str(payload["control_group_id"]):
            raise ValueError("Unit validation changed control_group_id")
        expected = {str(item["unit_id"]) for item in payload["units"]}
        observed = [str(item["unit_id"]) for item in response["judgments"]]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError("Unit response IDs do not exactly match request")
        for item in response["judgments"]:
            if bool(item["supports_control"]) and not str(item["failure_mode"]).strip():
                raise ValueError("Supporting unit lacks failure_mode")
            if not bool(item["supports_control"]) and not str(item["exception_reason"]).strip():
                raise ValueError("Non-supporting unit lacks exception_reason")
    elif task == "knowledge_card_semantics":
        if str(response["control_group_id"]) != str(payload["control_group_id"]):
            raise ValueError("Knowledge-card response changed control_group_id")
    elif task == "synthesis_group_validation":
        if str(response["candidate_group_id"]) != str(
            payload["candidate_group_id"]
        ):
            raise ValueError("Synthesis validation changed candidate_group_id")
        allowed = {str(item["topic_id"]) for item in payload["topics"]}
        controls = response["validated_controls"]
        subgroup_ids = [str(item["subgroup_id"]) for item in controls]
        if len(subgroup_ids) != len(set(subgroup_ids)):
            raise ValueError("Validated control IDs duplicate within one response")
        subgroup_id_set = set(subgroup_ids)

        assignments = response["topic_assignments"]
        assigned = [str(item["topic_id"]) for item in assignments]
        if len(assigned) != len(set(assigned)):
            raise ValueError("Topic assignments duplicate a topic")
        if not set(assigned).issubset(allowed):
            raise ValueError("Synthesis validation returned unknown topic IDs")
        if any(
            str(item["assigned_control_id"]) not in subgroup_id_set
            for item in assignments
        ):
            raise ValueError("Topic assignment references an unknown control ID")
        assignment_counts = {
            subgroup_id: sum(
                str(item["assigned_control_id"]) == subgroup_id
                for item in assignments
            )
            for subgroup_id in subgroup_ids
        }
        if any(count < 2 for count in assignment_counts.values()):
            raise ValueError(
                "Each validated control requires at least two assigned topics"
            )
        excluded_ids = [
            str(item["topic_id"]) for item in response["excluded_topics"]
        ]
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("Excluded topics duplicate within one response")
        excluded = set(excluded_ids)
        if not excluded.issubset(allowed) or excluded.intersection(assigned):
            raise ValueError("Invalid excluded topics in synthesis validation")
        if set(assigned).union(excluded) != allowed:
            raise ValueError(
                "Synthesis validation did not account for every input topic"
            )
        decision = str(response["decision"])
        if decision == "accept":
            if len(controls) != 1 or set(assigned) != allowed or excluded:
                raise ValueError("Accepted control must preserve all input topics")
        elif decision == "accept_after_excluding_topics":
            if len(controls) != 1 or not excluded:
                raise ValueError(
                    "accept_after_excluding_topics requires one control and exclusions"
                )
        elif decision == "split":
            if len(controls) < 2:
                raise ValueError("Split decision requires at least two controls")
        elif decision in {"reject", "unresolved"}:
            if controls or excluded != allowed:
                raise ValueError(
                    f"{decision} must return no control and account for all topics"
                )
            if decision == "unresolved" and not bool(
                response["needs_further_review"]
            ):
                raise ValueError("Unresolved decision requires further review")
    elif task == "synthesis_unit_evidence_batch":
        if str(response["control_id"]) != str(payload["control_id"]):
            raise ValueError("Unit-evidence validation changed control_id")
        expected = {str(item["unit_id"]) for item in payload["units"]}
        observed = [str(item["unit_id"]) for item in response["judgments"]]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError("Unit-evidence response IDs do not exactly match request")
        for item in response["judgments"]:
            supports = bool(item["supports_control"])
            failure_mode = str(item["failure_mode"]).strip()
            exception_reason = str(item["exception_reason"]).strip()
            if supports and (not failure_mode or exception_reason):
                raise ValueError(
                    "Supporting unit requires failure_mode and empty exception_reason"
                )
            if not supports and (failure_mode or not exception_reason):
                raise ValueError(
                    "Non-supporting unit requires empty failure_mode and exception_reason"
                )
    elif task == "synthesis_knowledge_card_semantics":
        if str(response["control_id"]) != str(payload["control_id"]):
            raise ValueError("Synthesis knowledge-card response changed control_id")
    else:
        raise ValueError(f"Unsupported task for identity validation: {task}")


def flatten_relation_results(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        response = record["response"]
        for item in response["judgments"]:
            values = dict(item)
            evidence_a = values.pop("evidence_a")
            evidence_b = values.pop("evidence_b")
            rows.append(
                {
                    "request_id": response["request_id"],
                    **values,
                    "evidence_a_json": canonical_json(evidence_a),
                    "evidence_b_json": canonical_json(evidence_b),
                }
            )
    return pd.DataFrame(rows)


def _maximal_cliques(adjacency: dict[str, set[str]]) -> list[tuple[str, ...]]:
    cliques: list[tuple[str, ...]] = []

    def visit(current: set[str], candidates: set[str], excluded: set[str]) -> None:
        if not candidates and not excluded:
            if len(current) >= 2:
                cliques.append(tuple(sorted(current)))
            return
        pivot_pool = candidates | excluded
        pivot = max(
            pivot_pool,
            key=lambda node: (len(candidates & adjacency[node]), node),
        ) if pivot_pool else None
        remaining = candidates - (adjacency[pivot] if pivot is not None else set())
        for node in sorted(remaining):
            visit(
                current | {node},
                candidates & adjacency[node],
                excluded & adjacency[node],
            )
            candidates.remove(node)
            excluded.add(node)

    visit(set(), set(adjacency), set())
    return sorted(set(cliques), key=lambda group: (-len(group), group))


def build_provisional_groups(
    relation_judgments: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"pair_id", "topic_a", "topic_b", "relation"}
    missing = sorted(required - set(relation_judgments.columns))
    if missing:
        raise ValueError(f"Relation judgments are missing: {missing}")
    pair_info = candidate_pairs.set_index("pair_id").to_dict("index")
    positive = relation_judgments[
        relation_judgments["relation"].isin(POSITIVE_RELATIONS)
    ].copy()
    nodes = sorted(set(positive["topic_a"]) | set(positive["topic_b"]))
    adjacency = {node: set() for node in nodes}
    edge_by_topics: dict[tuple[str, str], dict[str, Any]] = {}
    for row in positive.to_dict("records"):
        left, right = str(row["topic_a"]), str(row["topic_b"])
        adjacency[left].add(right)
        adjacency[right].add(left)
        edge_by_topics[_pair_key(left, right)] = row
    cliques = _maximal_cliques(adjacency) if adjacency else []

    group_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for index, members in enumerate(cliques, start=1):
        group_id = f"PG-{index:03d}"
        internal_edges = [
            edge_by_topics[_pair_key(members[left], members[right])]
            for left in range(len(members))
            for right in range(left + 1, len(members))
        ]
        similarities = [
            float(pair_info[str(edge["pair_id"])]["cosine_similarity"])
            for edge in internal_edges
        ]
        group_rows.append(
            {
                "provisional_group_id": group_id,
                "topic_count": len(members),
                "topic_ids_json": canonical_json(list(members)),
                "positive_edge_count": len(internal_edges),
                "same_failure_edge_count": sum(
                    edge["relation"] == "same_control_same_failure"
                    for edge in internal_edges
                ),
                "different_failure_edge_count": sum(
                    edge["relation"] == "same_control_different_failure"
                    for edge in internal_edges
                ),
                "internal_similarity_mean": float(np.mean(similarities)),
                "internal_similarity_min": float(np.min(similarities)),
                "construction_rule": "maximal_clique_of_explicit_positive_edges",
            }
        )
        membership_rows.extend(
            {
                "provisional_group_id": group_id,
                "topic_id": topic_id,
            }
            for topic_id in members
        )
    return pd.DataFrame(group_rows), pd.DataFrame(membership_rows)


def build_group_requests(
    group_summary: pd.DataFrame,
    group_membership: pd.DataFrame,
    relation_judgments: pd.DataFrame,
    topic_objects: dict[str, dict[str, Any]],
    builder: TopicControlPromptBuilder,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for index, group in enumerate(
        group_summary.sort_values("provisional_group_id").to_dict("records"), start=1
    ):
        group_id = str(group["provisional_group_id"])
        members = sorted(
            group_membership.loc[
                group_membership["provisional_group_id"].eq(group_id), "topic_id"
            ].astype(str)
        )
        member_set = set(members)
        edges = relation_judgments[
            relation_judgments["topic_a"].isin(member_set)
            & relation_judgments["topic_b"].isin(member_set)
            & relation_judgments["relation"].isin(POSITIVE_RELATIONS)
        ]
        request_id = f"GRP-{index:04d}"
        payload = {
            "request_id": request_id,
            "group_id": group_id,
            "topics": [topic_objects[topic_id] for topic_id in members],
            "positive_relations": [
                {
                    "pair_id": str(row.pair_id),
                    "topic_a": str(row.topic_a),
                    "topic_b": str(row.topic_b),
                    "relation": str(row.relation),
                    "shared_control_name": str(row.shared_control_name),
                    "important_difference": str(row.important_difference),
                }
                for row in edges.itertuples(index=False)
            ],
        }
        requests.append(request_record(builder.build("group_validation", payload), payload))
    return requests


def final_groups_from_validations(
    validations: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    next_id = 1
    for record in sorted(validations, key=lambda item: item["request_id"]):
        response = record["response"]
        for subgroup in response["validated_groups"]:
            control_id = f"CTRL-{next_id:03d}"
            next_id += 1
            topic_ids = list(map(str, subgroup["member_topic_ids"]))
            groups.append(
                {
                    "control_group_id": control_id,
                    "source_provisional_group_id": str(response["group_id"]),
                    "source_subgroup_id": str(subgroup["subgroup_id"]),
                    "group_decision": str(response["decision"]),
                    "control_name": str(subgroup["control_name"]),
                    "common_safety_basis": str(subgroup["common_safety_basis"]),
                    "important_boundaries_json": canonical_json(
                        subgroup["important_boundaries"]
                    ),
                    "topic_count": len(topic_ids),
                    "needs_review": bool(response["needs_review"]),
                    "validation_summary": str(response["validation_summary"]),
                }
            )
            members.extend(
                {
                    "control_group_id": control_id,
                    "topic_id": topic_id,
                }
                for topic_id in topic_ids
            )
    return pd.DataFrame(groups), pd.DataFrame(members)


def build_unit_requests(
    final_groups: pd.DataFrame,
    final_membership: pd.DataFrame,
    unit_objects: pd.DataFrame,
    builder: TopicControlPromptBuilder,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    required = {"unit_id", "topic_id", "finding", "objects", "risk_scene"}
    missing = sorted(required - set(unit_objects.columns))
    if missing:
        raise ValueError(f"Unit table is missing: {missing}")
    if unit_objects["unit_id"].astype(str).duplicated().any():
        raise ValueError("unit_id values must be unique")
    group_lookup = final_groups.set_index("control_group_id").to_dict("index")
    requests: list[dict[str, Any]] = []
    request_index = 1
    for group_id in sorted(final_membership["control_group_id"].astype(str).unique()):
        topics = set(
            final_membership.loc[
                final_membership["control_group_id"].eq(group_id), "topic_id"
            ].astype(str)
        )
        units = unit_objects[unit_objects["topic_id"].isin(topics)].sort_values(
            "unit_id", kind="stable"
        )
        control = group_lookup[group_id]
        for start in range(0, len(units), batch_size):
            batch = units.iloc[start : start + batch_size]
            request_id = f"UNT-{request_index:05d}"
            request_index += 1
            payload = {
                "request_id": request_id,
                "control_group_id": group_id,
                "control_name": str(control["control_name"]),
                "common_safety_basis": str(control["common_safety_basis"]),
                "important_boundaries": json.loads(
                    str(control["important_boundaries_json"])
                ),
                "units": [
                    {
                        "unit_id": str(row.unit_id),
                        "topic_id": str(row.topic_id),
                        "finding": str(row.finding),
                        "objects": list(row.objects),
                        "risk_scene": str(row.risk_scene),
                    }
                    for row in batch.itertuples(index=False)
                ],
            }
            requests.append(
                request_record(builder.build("unit_validation_batch", payload), payload)
            )
    return requests


def flatten_unit_results(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        response = record["response"]
        for item in response["judgments"]:
            rows.append(
                {
                    "request_id": response["request_id"],
                    "control_group_id": response["control_group_id"],
                    **item,
                }
            )
    return pd.DataFrame(rows)


def build_knowledge_card_requests(
    final_groups: pd.DataFrame,
    final_membership: pd.DataFrame,
    unit_validations: pd.DataFrame,
    unit_objects: pd.DataFrame,
    builder: TopicControlPromptBuilder,
) -> list[dict[str, Any]]:
    unit_lookup = unit_objects.set_index("unit_id").to_dict("index")
    requests: list[dict[str, Any]] = []
    for index, group in enumerate(
        final_groups.sort_values("control_group_id").to_dict("records"), start=1
    ):
        group_id = str(group["control_group_id"])
        evidence = unit_validations[
            unit_validations["control_group_id"].eq(group_id)
            & unit_validations["supports_control"].astype(bool)
        ].copy()
        failure_counts = (
            evidence["failure_mode"].astype(str).value_counts().sort_index().to_dict()
        )
        scenes: set[str] = set()
        objects: set[str] = set()
        evidence_examples: list[dict[str, Any]] = []
        for row in evidence.sort_values("unit_id").itertuples(index=False):
            source = unit_lookup[str(row.unit_id)]
            if str(source["risk_scene"]).strip():
                scenes.add(str(source["risk_scene"]).strip())
            objects.update(str(item).strip() for item in source["objects"] if str(item).strip())
            if len(evidence_examples) < 12:
                evidence_examples.append(
                    {
                        "unit_id": str(row.unit_id),
                        "finding": str(source["finding"]),
                        "objects": list(source["objects"]),
                        "risk_scene": str(source["risk_scene"]),
                        "validated_failure_mode": str(row.failure_mode),
                    }
                )
        request_id = f"CRD-{index:04d}"
        payload = {
            "request_id": request_id,
            "control_group_id": group_id,
            "provisional_control_name": str(group["control_name"]),
            "validated_support_unit_n": len(evidence),
            "failure_mode_counts": failure_counts,
            "risk_scenes": sorted(scenes),
            "objects": sorted(objects),
            "representative_validated_evidence": evidence_examples,
            "existing_boundaries": json.loads(str(group["important_boundaries_json"])),
        }
        requests.append(
            request_record(builder.build("knowledge_card_semantics", payload), payload)
        )
    return requests


def assemble_knowledge_cards(
    semantic_results: list[dict[str, Any]],
    final_membership: pd.DataFrame,
    unit_validations: pd.DataFrame,
    unit_objects: pd.DataFrame,
) -> list[dict[str, Any]]:
    unit_lookup = unit_objects.set_index("unit_id").to_dict("index")
    cards: list[dict[str, Any]] = []
    for record in sorted(semantic_results, key=lambda item: item["request_id"]):
        response = record["response"]
        group_id = str(response["control_group_id"])
        evidence = unit_validations[
            unit_validations["control_group_id"].eq(group_id)
            & unit_validations["supports_control"].astype(bool)
        ].copy()
        failure_modes = [
            {"mode": str(mode), "unit_n": int(count)}
            for mode, count in evidence["failure_mode"].astype(str).value_counts().items()
        ]
        source_topic_ids = sorted(
            final_membership.loc[
                final_membership["control_group_id"].eq(group_id), "topic_id"
            ].astype(str)
        )
        source_unit_ids = sorted(evidence["unit_id"].astype(str))
        scenes = sorted(
            {
                str(unit_lookup[unit_id]["risk_scene"])
                for unit_id in source_unit_ids
                if str(unit_lookup[unit_id]["risk_scene"]).strip()
            }
        )
        objects = sorted(
            {
                str(item)
                for unit_id in source_unit_ids
                for item in unit_lookup[unit_id]["objects"]
                if str(item).strip()
            }
        )
        cards.append(
            {
                "control_id": group_id,
                "control_name": response["control_name"],
                "safety_basis_summary": response["safety_basis_summary"],
                "supported_unit_n": len(source_unit_ids),
                "source_topic_n": len(source_topic_ids),
                "risk_scenes": scenes,
                "objects": objects,
                "failure_modes": failure_modes,
                "inspection_focus": response["inspection_focus"],
                "important_boundaries": response["important_boundaries"],
                "source_topic_ids": source_topic_ids,
                "source_unit_ids": source_unit_ids,
                "needs_review": bool(response["needs_review"]),
            }
        )
    return cards


def build_synthesis_group_requests(
    group_summary: pd.DataFrame,
    group_membership: pd.DataFrame,
    relation_judgments: pd.DataFrame,
    topic_objects: dict[str, dict[str, Any]],
    builder: TopicControlPromptBuilder,
) -> list[dict[str, Any]]:
    """Build synthesis requests from pairwise-positive topic groups."""
    requests: list[dict[str, Any]] = []
    for index, group in enumerate(
        group_summary.sort_values("provisional_group_id", kind="stable").to_dict(
            "records"
        ),
        start=1,
    ):
        group_id = str(group["provisional_group_id"])
        members = sorted(
            group_membership.loc[
                group_membership["provisional_group_id"].eq(group_id), "topic_id"
            ].astype(str)
        )
        member_set = set(members)
        edges = relation_judgments[
            relation_judgments["topic_a"].astype(str).isin(member_set)
            & relation_judgments["topic_b"].astype(str).isin(member_set)
            & relation_judgments["relation"].isin(POSITIVE_RELATIONS)
        ].sort_values("pair_id", kind="stable")
        payload = {
            "request_id": f"SCG-{index:04d}",
            "candidate_group_id": group_id,
            "candidate_group_construction": str(group["construction_rule"]),
            "topics": [topic_objects[topic_id] for topic_id in members],
            "positive_relations": [
                {
                    "pair_id": str(row.pair_id),
                    "topic_a": str(row.topic_a),
                    "topic_b": str(row.topic_b),
                    "relation": str(row.relation),
                    "shared_control_name": str(row.shared_control_name),
                    "failure_mode_a": str(row.failure_mode_a),
                    "failure_mode_b": str(row.failure_mode_b),
                    "common_safety_basis": str(row.common_safety_basis),
                    "important_difference": str(row.important_difference),
                }
                for row in edges.itertuples(index=False)
            ],
        }
        prompt = builder.build("synthesis_group_validation", payload)
        requests.append(request_record(prompt, payload))
    return requests


def migrate_synthesis_group_response_v20(
    response: dict[str, Any],
) -> dict[str, Any]:
    """Convert the non-overlapping v2.0 group response to canonical v2.1.

    The v2.0 transport contract encoded topic membership twice: once as
    ``member_topic_ids`` and again as nested ``topic_assessments``. Responses
    that passed the old identity checks are losslessly normalized by retaining
    one top-level ``topic_assignments`` list. Already-canonical responses are
    returned unchanged (apart from a defensive copy).
    """
    migrated = deepcopy(response)
    if "topic_assignments" in migrated:
        return migrated

    topic_assignments: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for control in migrated.get("validated_controls", []):
        subgroup_id = str(control["subgroup_id"])
        members = list(map(str, control["member_topic_ids"]))
        assessments = list(control["topic_assessments"])
        assessed_ids = [str(item["topic_id"]) for item in assessments]
        if len(members) != len(set(members)) or set(members) != set(assessed_ids):
            raise ValueError(
                "Cannot migrate v2.0 control with inconsistent topic membership"
            )
        if any(
            str(item["assigned_control_id"]) != subgroup_id
            for item in assessments
        ):
            raise ValueError(
                "Cannot migrate v2.0 control with inconsistent assignment target"
            )
        controls.append(
            {
                "subgroup_id": subgroup_id,
                "control_name": str(control["control_name"]),
                "control_definition": str(control["control_definition"]),
                "important_boundaries": deepcopy(control["important_boundaries"]),
            }
        )
        topic_assignments.extend(deepcopy(assessments))

    migrated["validated_controls"] = controls
    migrated["topic_assignments"] = sorted(
        topic_assignments,
        key=lambda item: (
            str(item["topic_id"]),
            str(item["assigned_control_id"]),
        ),
    )
    return migrated


def final_controls_from_synthesis(
    validations: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Materialize deterministic controls, topic membership, and exclusions."""
    controls: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    next_id = 1
    for record in sorted(validations, key=lambda item: str(item["request_id"])):
        response = record["response"]
        candidate_group_id = str(response["candidate_group_id"])
        assignments_by_control: dict[str, list[dict[str, Any]]] = {}
        for assignment in response["topic_assignments"]:
            assignments_by_control.setdefault(
                str(assignment["assigned_control_id"]), []
            ).append(assignment)
        ordered_controls = sorted(
            response["validated_controls"],
            key=lambda item: (
                tuple(
                    sorted(
                        str(assignment["topic_id"])
                        for assignment in assignments_by_control.get(
                            str(item["subgroup_id"]), []
                        )
                    )
                ),
                str(item["subgroup_id"]),
            ),
        )
        for control in ordered_controls:
            control_id = f"CTRL-{next_id:03d}"
            next_id += 1
            subgroup_id = str(control["subgroup_id"])
            assessments = {
                str(item["topic_id"]): item
                for item in assignments_by_control.get(subgroup_id, [])
            }
            topic_ids = sorted(assessments)
            controls.append(
                {
                    "control_id": control_id,
                    "source_candidate_group_id": candidate_group_id,
                    "source_subgroup_id": subgroup_id,
                    "group_decision": str(response["decision"]),
                    "control_name": str(control["control_name"]),
                    "control_definition": str(control["control_definition"]),
                    "important_boundaries_json": canonical_json(
                        control["important_boundaries"]
                    ),
                    "topic_count": len(topic_ids),
                    "needs_further_review": bool(
                        response["needs_further_review"]
                    ),
                    "validation_summary": str(response["validation_summary"]),
                }
            )
            memberships.extend(
                {
                    "control_id": control_id,
                    "topic_id": topic_id,
                    "topic_failure_mode": str(
                        assessments[topic_id]["failure_mode"]
                    ),
                    "topic_important_difference": str(
                        assessments[topic_id]["important_difference"]
                    ),
                }
                for topic_id in topic_ids
            )
        exclusions.extend(
            {
                "source_candidate_group_id": candidate_group_id,
                "group_decision": str(response["decision"]),
                "topic_id": str(item["topic_id"]),
                "exclusion_reason": str(item["reason"]),
                "needs_further_review": bool(
                    response["needs_further_review"]
                ),
            }
            for item in response["excluded_topics"]
        )
    return (
        pd.DataFrame(controls),
        pd.DataFrame(memberships),
        pd.DataFrame(exclusions),
    )


def build_synthesis_unit_requests(
    controls: pd.DataFrame,
    control_membership: pd.DataFrame,
    unit_objects: pd.DataFrame,
    builder: TopicControlPromptBuilder,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Build unit evidence requests while carrying frozen facts only as inputs."""
    required = {"unit_id", "topic_id", "finding", "objects", "risk_scene"}
    missing = sorted(required - set(unit_objects.columns))
    if missing:
        raise ValueError(f"Unit table is missing: {missing}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if unit_objects["unit_id"].astype(str).duplicated().any():
        raise ValueError("unit_id values must be unique")
    control_lookup = controls.set_index("control_id").to_dict("index")
    requests: list[dict[str, Any]] = []
    request_index = 1
    for control_id in sorted(control_membership["control_id"].astype(str).unique()):
        topic_ids = set(
            control_membership.loc[
                control_membership["control_id"].eq(control_id), "topic_id"
            ].astype(str)
        )
        units = (
            unit_objects[unit_objects["topic_id"].astype(str).isin(topic_ids)]
            .drop_duplicates(subset=["unit_id"], keep="first")
            .sort_values("unit_id", kind="stable")
        )
        control = control_lookup[control_id]
        for start in range(0, len(units), batch_size):
            batch = units.iloc[start : start + batch_size]
            payload = {
                "request_id": f"SUE-{request_index:05d}",
                "control_id": control_id,
                "control_name": str(control["control_name"]),
                "control_definition": str(control["control_definition"]),
                "important_boundaries": json.loads(
                    str(control["important_boundaries_json"])
                ),
                "units": [
                    {
                        "unit_id": str(row.unit_id),
                        "topic_id": str(row.topic_id),
                        "finding": str(row.finding),
                        "objects": list(row.objects),
                        "risk_scene": str(row.risk_scene),
                    }
                    for row in batch.itertuples(index=False)
                ],
            }
            request_index += 1
            prompt = builder.build("synthesis_unit_evidence_batch", payload)
            requests.append(request_record(prompt, payload))
    return requests


def flatten_synthesis_unit_results(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        response = record["response"]
        for item in response["judgments"]:
            rows.append(
                {
                    "request_id": str(response["request_id"]),
                    "control_id": str(response["control_id"]),
                    **item,
                }
            )
    return pd.DataFrame(rows)


def _diverse_evidence_sample(
    evidence: pd.DataFrame, maximum: int
) -> pd.DataFrame:
    """Select deterministic round-robin examples across observed failure modes."""
    if len(evidence) <= maximum:
        return evidence.sort_values(["failure_mode", "unit_id"], kind="stable")
    groups = [
        group.sort_values("unit_id", kind="stable").to_dict("records")
        for _, group in evidence.groupby("failure_mode", sort=True)
    ]
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < maximum:
        added = False
        for group in groups:
            if offset < len(group) and len(selected) < maximum:
                selected.append(group[offset])
                added = True
        if not added:
            break
        offset += 1
    return pd.DataFrame(selected)


def build_synthesis_knowledge_card_requests(
    controls: pd.DataFrame,
    unit_validations: pd.DataFrame,
    unit_objects: pd.DataFrame,
    builder: TopicControlPromptBuilder,
    *,
    max_representatives: int,
) -> list[dict[str, Any]]:
    """Build semantics-only knowledge-card requests from supporting units."""
    unit_lookup = unit_objects.set_index("unit_id").to_dict("index")
    requests: list[dict[str, Any]] = []
    request_index = 1
    for control in controls.sort_values("control_id", kind="stable").to_dict(
        "records"
    ):
        control_id = str(control["control_id"])
        evidence = unit_validations[
            unit_validations["control_id"].astype(str).eq(control_id)
            & unit_validations["supports_control"].astype(bool)
        ].drop_duplicates(subset=["control_id", "unit_id"], keep="first")
        if evidence.empty:
            continue
        failure_counts = (
            evidence["failure_mode"].astype(str).value_counts().sort_index().to_dict()
        )
        scenes: set[str] = set()
        objects: set[str] = set()
        for unit_id in evidence["unit_id"].astype(str):
            source = unit_lookup[unit_id]
            if str(source["risk_scene"]).strip():
                scenes.add(str(source["risk_scene"]).strip())
            objects.update(
                str(item).strip()
                for item in source["objects"]
                if str(item).strip()
            )
        examples = _diverse_evidence_sample(evidence, max_representatives)
        representative_units = []
        for row in examples.itertuples(index=False):
            source = unit_lookup[str(row.unit_id)]
            representative_units.append(
                {
                    "sample_id": str(row.unit_id),
                    "finding": str(source["finding"]),
                    "objects": list(source["objects"]),
                    "risk_scene": str(source["risk_scene"]),
                    "validated_failure_mode": str(row.failure_mode),
                }
            )
        payload = {
            "request_id": f"SKC-{request_index:04d}",
            "control_id": control_id,
            "provisional_control_name": str(control["control_name"]),
            "provisional_control_definition": str(control["control_definition"]),
            "existing_boundaries": json.loads(
                str(control["important_boundaries_json"])
            ),
            "validated_evidence_summary": {
                "supported_unit_n": int(len(evidence)),
                "failure_mode_counts": {
                    str(key): int(value) for key, value in failure_counts.items()
                },
                "risk_scenes": sorted(scenes),
                "objects": sorted(objects),
                "representative_supported_units": representative_units,
            },
        }
        request_index += 1
        prompt = builder.build("synthesis_knowledge_card_semantics", payload)
        requests.append(request_record(prompt, payload))
    return requests


def assemble_synthesis_knowledge_cards(
    semantic_results: list[dict[str, Any]],
    controls: pd.DataFrame,
    unit_validations: pd.DataFrame,
    unit_objects: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Add deterministic evidence fields to the LLM-produced card semantics."""
    control_lookup = controls.set_index("control_id").to_dict("index")
    unit_lookup = unit_objects.set_index("unit_id").to_dict("index")
    cards: list[dict[str, Any]] = []
    for record in sorted(semantic_results, key=lambda item: str(item["request_id"])):
        response = record["response"]
        control_id = str(response["control_id"])
        control = control_lookup[control_id]
        all_judgments = unit_validations[
            unit_validations["control_id"].astype(str).eq(control_id)
        ].drop_duplicates(subset=["control_id", "unit_id"], keep="first")
        evidence = all_judgments[all_judgments["supports_control"].astype(bool)]
        source_unit_ids = sorted(evidence["unit_id"].astype(str).unique())
        source_topic_ids = sorted(
            {str(unit_lookup[unit_id]["topic_id"]) for unit_id in source_unit_ids}
        )
        scenes = sorted(
            {
                str(unit_lookup[unit_id]["risk_scene"]).strip()
                for unit_id in source_unit_ids
                if str(unit_lookup[unit_id]["risk_scene"]).strip()
            }
        )
        objects = sorted(
            {
                str(item).strip()
                for unit_id in source_unit_ids
                for item in unit_lookup[unit_id]["objects"]
                if str(item).strip()
            }
        )
        failure_modes = [
            {"mode": str(mode), "unit_n": int(count)}
            for mode, count in evidence["failure_mode"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        ]
        boundaries = list(
            dict.fromkeys(
                [
                    *json.loads(str(control["important_boundaries_json"])),
                    *list(response["important_boundaries"]),
                ]
            )
        )
        supported_n = len(source_unit_ids)
        candidate_n = len(all_judgments)
        needs_review = bool(
            control["needs_further_review"]
            or response["needs_further_review"]
            or all_judgments["needs_further_review"].astype(bool).any()
            or len(source_topic_ids) < 2
        )
        cards.append(
            {
                "control_id": control_id,
                "control_name": str(response["control_name"]),
                "control_definition": str(response["control_definition"]),
                "supported_unit_n": supported_n,
                "candidate_unit_n": candidate_n,
                "evidence_support_ratio": (
                    float(supported_n / candidate_n) if candidate_n else 0.0
                ),
                "source_topic_n": len(source_topic_ids),
                "risk_scenes": scenes,
                "objects": objects,
                "failure_modes": failure_modes,
                "inspection_focus": list(response["inspection_focus"]),
                "important_boundaries": boundaries,
                "source_topic_ids": source_topic_ids,
                "source_unit_ids": source_unit_ids,
                "needs_further_review": needs_review,
            }
        )
    return cards
