from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


SCHEMA_PATH = Path(__file__).with_name("llm_output_schemas.json")

TASK_SCHEMA_NAMES = {
    "extract": "extraction_array",
    "extract_batch": "extraction_batch_array",
    "arbitrate_extraction": "extraction_array",
    "scene_split": "scene_split_array",
    "scene_split_batch": "scene_split_array",
    "arbitrate_scene": "scene_split_array",
    "cluster_name": "cluster_naming_object",
}


def _schema_document() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


_DOCUMENT = _schema_document()
_DEFINITIONS = _DOCUMENT["definitions"]
_VALIDATORS = {
    name: Draft7Validator(
        {
            "$schema": _DOCUMENT["$schema"],
            "definitions": _DEFINITIONS,
            "$ref": f"#/definitions/{name}",
        }
    )
    for name in (
        "extraction_array",
        "extraction_batch_array",
        "scene_split_array",
        "cluster_naming_object",
        "cluster_naming_v2_object",
        "topic_control_relation_batch_object",
        "topic_control_group_validation_object",
        "topic_control_unit_validation_batch_object",
        "topic_control_knowledge_card_semantics_object",
        "safety_control_synthesis_validation_object",
        "safety_control_unit_evidence_batch_object",
        "safety_control_knowledge_card_semantics_object",
    )
}


def validate_schema_payload(schema_name: str, payload: Any) -> Any:
    """Validate a parsed payload and return it unchanged."""
    try:
        validator = _VALIDATORS[schema_name]
    except KeyError as exc:
        raise KeyError(f"Unknown LLM JSON Schema {schema_name!r}") from exc

    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.path
        )
        raise ValueError(
            f"LLM response failed JSON Schema validation for {schema_name!r} "
            f"at {path}: {error.message}"
        )
    return payload


def validate_task_payload(task: str, payload: Any) -> Any:
    """Validate payloads for known LLM tasks; leave test/extension tasks unchanged."""
    schema_name = TASK_SCHEMA_NAMES.get(task)
    if schema_name is None:
        return payload
    return validate_schema_payload(schema_name, payload)


def get_schema_definition(
    schema_name: str,
    *,
    inline_references: bool = False,
    excluded_keywords: set[str] | None = None,
) -> dict[str, Any]:
    """Return a detached schema definition for provider-side structured output.

    Some OpenAI-compatible providers accept strict JSON Schema but do not resolve
    local ``#/definitions/...`` references unless the whole source document is
    supplied. ``inline_references=True`` recursively embeds those definitions so
    the transmitted schema is self-contained. ``excluded_keywords`` supports a
    provider-specific transport projection; the full local schema remains the
    authority for response validation.
    """
    try:
        schema = _DEFINITIONS[schema_name]
    except KeyError as exc:
        raise KeyError(f"Unknown LLM JSON Schema {schema_name!r}") from exc
    detached = json.loads(json.dumps(schema, ensure_ascii=False))
    excluded = set(excluded_keywords or ())
    if not inline_references and not excluded:
        return detached

    def inline(value: Any, stack: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [inline(item, stack) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if reference is not None:
            prefix = "#/definitions/"
            if not isinstance(reference, str) or not reference.startswith(prefix):
                raise ValueError(f"Unsupported JSON Schema reference: {reference!r}")
            referenced_name = reference[len(prefix) :]
            if referenced_name in stack:
                raise ValueError(
                    "Recursive JSON Schema references cannot be inlined: "
                    + " -> ".join((*stack, referenced_name))
                )
            try:
                referenced = _DEFINITIONS[referenced_name]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown referenced JSON Schema {referenced_name!r}"
                ) from exc
            resolved = inline(
                json.loads(json.dumps(referenced, ensure_ascii=False)),
                (*stack, referenced_name),
            )
            siblings = {key: child for key, child in value.items() if key != "$ref"}
            if siblings:
                if not isinstance(resolved, dict):
                    raise ValueError("Cannot merge JSON Schema reference siblings")
                resolved.update(inline(siblings, stack))
            return resolved
        return {
            key: inline(child, stack)
            for key, child in value.items()
            if key not in excluded
        }

    return inline(detached, (schema_name,))


def infer_schema_name(payload: Any) -> str | None:
    """Infer the schema used by a cached payload without exposing its content."""
    if isinstance(payload, dict):
        keys = set(payload)
        if {"cluster_id", "cluster_name", "explanation", "key_evidence"} <= keys:
            return "cluster_naming_object"
        return None
    if not isinstance(payload, list):
        return None
    if not payload:
        return "extraction_array"
    first = payload[0]
    if not isinstance(first, dict):
        return None
    keys = set(first)
    if {"raw_record_id", "records"} <= keys:
        return "extraction_batch_array"
    if {"finding", "object", "scene"} <= keys:
        return "extraction_array"
    if {"record_id", "loc_detail", "risk_scene"} <= keys:
        return "scene_split_array"
    return None
