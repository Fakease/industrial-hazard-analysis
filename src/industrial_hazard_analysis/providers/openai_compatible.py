from __future__ import annotations

import hashlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from industrial_hazard_analysis.models import (
    ExtractionCandidate,
    RawHazardText,
    SceneSplitCandidate,
    StructuredRecord,
)
from industrial_hazard_analysis.providers.base import EmbeddingProvider, LLMProvider
from industrial_hazard_analysis.providers.json_schema import validate_task_payload


def _verified_ssl_context() -> ssl.SSLContext:
    """Build a verified TLS context, preferring certifi when it is installed."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


class NonRetryableChatError(RuntimeError):
    """A deterministic provider rejection that should not be retried unchanged."""


class JsonlCache:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.items: dict[str, Any] = {}
        self._lock = threading.RLock()
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = item.get("cache_key")
                    if key:
                        self.items[str(key)] = item.get("value")

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self.items.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.items[key] = value
            if not self.path:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"cache_key": key, "value": value}, ensure_ascii=False) + "\n")


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        api_key_env: str,
        base_url: str,
        model: str,
        provider_name: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        sleep: float = 0.2,
    ):
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider_name = provider_name or model
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep = sleep
        self.ssl_context = _verified_ssl_context()

    @property
    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RuntimeError(f"Missing API key. Set {self.api_key_env}.")
        return value

    def chat(
        self,
        prompt: str,
        *,
        request_overrides: dict[str, Any] | None = None,
    ) -> str:
        data = self.chat_response(prompt, request_overrides=request_overrides)
        return data["choices"][0]["message"]["content"]

    def chat_response(
        self,
        prompt: str,
        *,
        request_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the complete provider response for audited usage metadata."""
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,
        }
        if request_overrides:
            payload.update(request_overrides)
        return self._post_chat_response(url, payload)

    def _post_chat(self, url: str, payload: dict[str, Any]) -> str:
        data = self._post_chat_response(url, payload)
        return data["choices"][0]["message"]["content"]

    def _post_chat_response(
        self, url: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error_detail = ""
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout,
                    context=self.ssl_context,
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Chat response must decode to a JSON object")
                data["choices"][0]["message"]["content"]
                return data
            except urllib.error.HTTPError as exc:
                last_error_detail = _http_error_detail(exc)
                message = (
                    "Chat request failed: "
                    f"provider={self.provider_name!r}, model={self.model!r}, url={url!r}, "
                    f"http_status={exc.code}, detail={last_error_detail}"
                )
                if 400 <= int(exc.code) < 500 and int(exc.code) not in {408, 409, 429}:
                    raise NonRetryableChatError(message) from exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "Chat request failed after "
                        f"{attempt} attempts: provider={self.provider_name!r}, model={self.model!r}, url={url!r}, "
                        f"http_status={exc.code}, detail={last_error_detail}"
                    ) from exc
                time.sleep(min(2**attempt, 20))
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "Chat request failed after "
                        f"{attempt} attempts: provider={self.provider_name!r}, model={self.model!r}, url={url!r}, "
                        f"error={exc}, detail={last_error_detail}"
                    ) from exc
                time.sleep(min(2**attempt, 20))
        raise RuntimeError("Unreachable retry state")


class OpenAICompatibleLLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key_env: str,
        base_url: str,
        prompt_paths: dict[str, str],
        cache_dir: str | Path | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        offline_cache_only: bool = False,
    ):
        self.name = name
        self.model = model
        self.prompt_paths = prompt_paths
        self.offline_cache_only = offline_cache_only
        self.client = OpenAICompatibleChatClient(
            api_key_env=api_key_env,
            base_url=base_url,
            model=model,
            provider_name=name,
            timeout=timeout,
            max_retries=max_retries,
        )
        cache_path = Path(cache_dir) / f"llm_{name}_{model}.jsonl" if cache_dir else None
        self.cache = JsonlCache(cache_path)

    def extract_hazard(self, raw: RawHazardText) -> list[ExtractionCandidate]:
        prompt = self._prompt("structured_extraction").replace("{TEXT}", raw.raw_text)
        records = self._cached_json_array("extract", raw.record_id, raw.raw_text, prompt)
        return [
            ExtractionCandidate(
                source_model=self.name,
                raw_record_id=raw.record_id,
                finding=str(item.get("finding", "")).strip(),
                object=_clean_objects(item.get("object")),
                scene=str(item.get("scene", "")).strip(),
                metadata={"provider": self.name, "model": self.model},
            )
            for item in records
            if isinstance(item, dict)
        ]

    def extract_hazards(
        self,
        raw_records: list[RawHazardText],
    ) -> dict[str, list[ExtractionCandidate]]:
        if len(raw_records) <= 1 or "structured_extraction_batch" not in self.prompt_paths:
            return super().extract_hazards(raw_records)

        payload = [
            {"raw_record_id": raw.record_id, "raw_text": raw.raw_text}
            for raw in raw_records
        ]
        payload_text = json.dumps(payload, ensure_ascii=False)
        prompt = self._prompt("structured_extraction_batch").replace("{RAW_RECORDS}", payload_text)
        records = self._cached_json_array(
            "extract_batch",
            _batch_id(raw.record_id for raw in raw_records),
            payload_text,
            prompt,
        )

        candidates_by_raw_id = {raw.record_id: [] for raw in raw_records}
        for item in records:
            if not isinstance(item, dict):
                continue
            raw_record_id = str(item.get("raw_record_id", "")).strip()
            if raw_record_id not in candidates_by_raw_id:
                continue
            extracted_records = item.get("records", [])
            if not isinstance(extracted_records, list):
                extracted_records = []
            candidates_by_raw_id[raw_record_id] = [
                ExtractionCandidate(
                    source_model=self.name,
                    raw_record_id=raw_record_id,
                    finding=str(record.get("finding", "")).strip(),
                    object=_clean_objects(record.get("object")),
                    scene=str(record.get("scene", "")).strip(),
                    metadata={"provider": self.name, "model": self.model, "batch": True},
                )
                for record in extracted_records
                if isinstance(record, dict)
            ]
        return candidates_by_raw_id

    def arbitrate_extraction(
        self, raw: RawHazardText, candidates: list[ExtractionCandidate]
    ) -> list[ExtractionCandidate]:
        prompt_template = self._prompt("structured_extraction_arbitration")
        candidate_payload = [
            {
                "source_model": item.source_model,
                "finding": item.finding,
                "object": item.object,
                "scene": item.scene,
            }
            for item in candidates
        ]
        candidate_text = json.dumps(candidate_payload, ensure_ascii=False)
        candidates_by_model: list[list[dict[str, Any]]] = []
        for source_model in dict.fromkeys(item.source_model for item in candidates):
            candidates_by_model.append(
                [
                    payload
                    for item, payload in zip(candidates, candidate_payload)
                    if item.source_model == source_model
                ]
            )
        while len(candidates_by_model) < 3:
            candidates_by_model.append([])
        model_candidate_texts = [
            json.dumps(items, ensure_ascii=False) for items in candidates_by_model[:3]
        ]
        prompt = _fill_first_available(
            prompt_template,
            {
                "{TEXT}": raw.raw_text,
                "{CONSENSUS_RESULT}": "[]",
                "{MODEL_A_CONFLICT_RESULT}": model_candidate_texts[0],
                "{MODEL_B_CONFLICT_RESULT}": model_candidate_texts[1],
                "{MODEL_C_CONFLICT_RESULT}": model_candidate_texts[2],
                "{CANDIDATES}": candidate_text,
            },
        )
        records = self._cached_json_array(
            "arbitrate_extraction",
            raw.record_id,
            candidate_text,
            prompt,
        )
        return [
            ExtractionCandidate(
                source_model=self.name,
                raw_record_id=raw.record_id,
                finding=str(content.get("finding", "")).strip(),
                object=_clean_objects(content.get("object")),
                scene=str(content.get("scene", "")).strip(),
                metadata={"arbitrated": True, "provider": self.name, "model": self.model},
            )
            for content in records
            if isinstance(content, dict)
        ]

    def split_scene(self, record: StructuredRecord) -> SceneSplitCandidate:
        payload = [
            {
                "record_id": record.record_id,
                "finding": record.finding,
                "object": record.object,
                "scene": record.scene,
            }
        ]
        prompt = self._prompt("scene_secondary_split").replace(
            "{STRUCTURED_RECORDS}",
            json.dumps(payload, ensure_ascii=False),
        )
        records = self._cached_json_array("scene_split", record.record_id, record.scene, prompt)
        item = records[0] if records and isinstance(records[0], dict) else {}
        return SceneSplitCandidate(
            source_model=self.name,
            structured_record_id=record.record_id,
            loc_detail=str(item.get("loc_detail", "")).strip(),
            risk_scene=str(item.get("risk_scene", "")).strip(),
            metadata={"provider": self.name, "model": self.model},
        )

    def split_scenes(
        self,
        records: list[StructuredRecord],
    ) -> dict[str, SceneSplitCandidate]:
        if len(records) <= 1 or "scene_secondary_split_batch" not in self.prompt_paths:
            return super().split_scenes(records)

        payload = [
            {
                "record_id": record.record_id,
                "finding": record.finding,
                "object": record.object,
                "scene": record.scene,
            }
            for record in records
        ]
        payload_text = json.dumps(payload, ensure_ascii=False)
        prompt = self._prompt("scene_secondary_split_batch").replace(
            "{STRUCTURED_RECORDS}",
            payload_text,
        )
        records_json = self._cached_json_array(
            "scene_split_batch",
            _batch_id(record.record_id for record in records),
            payload_text,
            prompt,
        )

        candidates_by_record_id = {
            record.record_id: SceneSplitCandidate(
                source_model=self.name,
                structured_record_id=record.record_id,
                loc_detail="",
                risk_scene="",
                metadata={"provider": self.name, "model": self.model, "batch": True, "missing": True},
            )
            for record in records
        }
        for item in records_json:
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("record_id", "")).strip()
            if record_id not in candidates_by_record_id:
                continue
            candidates_by_record_id[record_id] = SceneSplitCandidate(
                source_model=self.name,
                structured_record_id=record_id,
                loc_detail=str(item.get("loc_detail", "")).strip(),
                risk_scene=str(item.get("risk_scene", "")).strip(),
                metadata={"provider": self.name, "model": self.model, "batch": True},
            )
        return candidates_by_record_id

    def arbitrate_scene_split(
        self, record: StructuredRecord, candidates: list[SceneSplitCandidate]
    ) -> SceneSplitCandidate:
        prompt_template = self._prompt("scene_split_arbitration")
        payload = {
            "record_id": record.record_id,
            "scene": record.scene,
            "candidates": [
                {
                    "source_model": item.source_model,
                    "loc_detail": item.loc_detail,
                    "risk_scene": item.risk_scene,
                }
                for item in candidates
            ],
        }
        payload_text = json.dumps(payload, ensure_ascii=False)
        prompt = _fill_first_available(
            prompt_template,
            {
                "{TEXT}": record.raw_text,
                "{STRUCTURED_RECORD}": json.dumps(
                    {
                        "record_id": record.record_id,
                        "finding": record.finding,
                        "object": record.object,
                        "scene": record.scene,
                    },
                    ensure_ascii=False,
                ),
                "{SCENE}": record.scene,
                "{MODEL_A_SCENE_SPLIT}": payload_text,
                "{MODEL_B_SCENE_SPLIT}": payload_text,
                "{MODEL_C_SCENE_SPLIT}": payload_text,
                "{CANDIDATES}": payload_text,
                "{STRUCTURED_RECORDS}": payload_text,
            },
        )
        records = self._cached_json_array(
            "arbitrate_scene",
            record.record_id,
            payload_text,
            prompt,
        )
        content = records[0] if records and isinstance(records[0], dict) else {}
        return SceneSplitCandidate(
            source_model=self.name,
            structured_record_id=record.record_id,
            loc_detail=str(content.get("loc_detail", "")).strip(),
            risk_scene=str(content.get("risk_scene", "")).strip(),
            metadata={"arbitrated": True, "provider": self.name, "model": self.model},
        )

    def name_cluster(self, payload: dict) -> dict:
        prompt = self._prompt("cluster_naming").replace("{CLUSTER_INFO}", json.dumps(payload, ensure_ascii=False))
        return self._cached_json_object(
            "cluster_name",
            str(payload.get("experiment_id", "")),
            json.dumps(payload, ensure_ascii=False),
            prompt,
        )

    def _prompt(self, key: str) -> str:
        path = self.prompt_paths.get(key)
        if not path:
            raise KeyError(f"Missing prompt path for {key!r} in provider {self.name!r}")
        return Path(path).read_text(encoding="utf-8")

    def _cached_json_array(self, task: str, record_id: str, source: str, prompt: str) -> list[Any]:
        key = _cache_key(self.name, self.model, task, record_id, source, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            return validate_task_payload(task, cached)
        if self.offline_cache_only:
            raise RuntimeError(
                "Offline cache-only mode blocked an uncached LLM request: "
                f"provider={self.name!r}, model={self.model!r}, task={task!r}, record_id={record_id!r}"
            )
        parsed = self._request_parsed_json(
            task,
            record_id,
            prompt,
            lambda content: validate_task_payload(task, _parse_json_array(content)),
        )
        self.cache.set(key, parsed)
        return parsed

    def _cached_json_object(self, task: str, record_id: str, source: str, prompt: str) -> dict[str, Any]:
        key = _cache_key(self.name, self.model, task, record_id, source, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            return validate_task_payload(task, cached)
        if self.offline_cache_only:
            raise RuntimeError(
                "Offline cache-only mode blocked an uncached LLM request: "
                f"provider={self.name!r}, model={self.model!r}, task={task!r}, record_id={record_id!r}"
            )
        parsed = self._request_parsed_json(
            task,
            record_id,
            prompt,
            lambda content: validate_task_payload(task, _parse_json_object(content)),
        )
        self.cache.set(key, parsed)
        return parsed

    def _request_parsed_json(self, task: str, record_id: str, prompt: str, parser):
        last_content = ""
        last_error: Exception | None = None
        for attempt in range(1, self.client.max_retries + 1):
            content = self.client.chat(prompt)
            last_content = content
            try:
                return parser(content)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                print(
                    "LLM JSON parse failed; retrying "
                    f"provider={self.name}, model={self.model}, task={task}, "
                    f"record_id={record_id}, attempt={attempt}/{self.client.max_retries}",
                    flush=True,
                )
                if attempt < self.client.max_retries:
                    time.sleep(min(2**attempt, 20))
        preview = last_content[:500].replace("\n", "\\n")
        raise ValueError(
            "LLM response was not valid JSON after "
            f"{self.client.max_retries} attempt(s): provider={self.name!r}, "
            f"model={self.model!r}, task={task!r}, record_id={record_id!r}, "
            f"last_error={last_error}, content_preview={preview!r}"
        )


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key_env: str,
        base_url: str,
        dimension: int,
        batch_size: int = 10,
        cache_dir: str | Path | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        offline_cache_only: bool = False,
    ):
        self.name = name
        self.model = model
        self.dimension = dimension
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.offline_cache_only = offline_cache_only
        cache_path = Path(cache_dir) / f"embedding_{name}_{model}_{dimension}.jsonl" if cache_dir else None
        self.cache = JsonlCache(cache_path)

    @property
    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RuntimeError(f"Missing API key. Set {self.api_key_env}.")
        return value

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings_by_text: dict[str, list[float]] = {}
        uncached = []
        for text in dict.fromkeys(texts):
            key = _cache_key(self.name, self.model, "embedding", str(self.dimension), text)
            cached = self.cache.get(key)
            if cached is not None:
                embeddings_by_text[text] = cached
            else:
                uncached.append(text)

        if uncached and self.offline_cache_only:
            raise RuntimeError(
                "Offline cache-only mode blocked uncached embedding requests: "
                f"provider={self.name!r}, model={self.model!r}, missing_texts={len(uncached)}"
            )

        for start in range(0, len(uncached), self.batch_size):
            batch = uncached[start : start + self.batch_size]
            print(
                f"Embedding {self.name}/{self.model}: "
                f"{min(start + self.batch_size, len(uncached))}/{len(uncached)} uncached text(s)",
                flush=True,
            )
            vectors = self._embed_batch(batch)
            for text, vector in zip(batch, vectors):
                key = _cache_key(self.name, self.model, "embedding", str(self.dimension), text)
                self.cache.set(key, vector)
                embeddings_by_text[text] = vector
        return [embeddings_by_text[text] for text in texts]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = self.base_url + "/embeddings"
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimension,
            "encoding_format": "float",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error_detail = ""
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                ordered = sorted(data["data"], key=lambda item: item["index"])
                return [item["embedding"] for item in ordered]
            except urllib.error.HTTPError as exc:
                last_error_detail = _http_error_detail(exc)
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "Embedding request failed after "
                        f"{attempt} attempts: model={self.model!r}, url={url!r}, "
                        f"http_status={exc.code}, detail={last_error_detail}"
                    ) from exc
                time.sleep(min(2**attempt, 20))
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "Embedding request failed after "
                        f"{attempt} attempts: model={self.model!r}, url={url!r}, "
                        f"error={exc}, detail={last_error_detail}"
                    ) from exc
                time.sleep(min(2**attempt, 20))
        raise RuntimeError("Unreachable retry state")


def _parse_json_array(content: str) -> list[Any]:
    parsed = _parse_json(content)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    return parsed


def _parse_json_object(content: str) -> dict[str, Any]:
    parsed = _parse_json(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def _parse_json(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        starts = [idx for idx in [text.find("["), text.find("{")] if idx >= 0]
        ends = [idx for idx in [text.rfind("]"), text.rfind("}")] if idx >= 0]
        if not starts or not ends:
            raise
        return json.loads(text[min(starts) : max(ends) + 1])


def _cache_key(*parts: str) -> str:
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _batch_id(parts) -> str:
    return "|".join(str(part) for part in parts)


def _clean_objects(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _fill_first_available(template: str, replacements: dict[str, str]) -> str:
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception:
        return str(exc)
