from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

import numpy as np

from industrial_hazard_analysis.models import (
    ExtractionCandidate,
    RawHazardText,
    SceneSplitCandidate,
    StructuredRecord,
)
from industrial_hazard_analysis.providers.base import EmbeddingProvider, LLMProvider


RISK_SCENE_KEYWORDS = [
    "配电室",
    "受限空间",
    "动火作业区",
    "动火区",
    "高处作业面",
    "吊装区域",
    "危化品仓库",
    "楼梯间",
    "泵房",
    "仓库",
    "平台",
]

OBJECT_KEYWORDS = [
    "电缆桥架盖板",
    "灭火器",
    "通道",
    "安全带",
    "警戒线",
    "通风设备",
    "照明",
    "临时电源线",
    "地面",
    "油污",
]

FINDING_MARKERS = [
    "缺失",
    "遮挡",
    "湿滑",
    "未规范悬挂",
    "进入作业半径",
    "未开启",
    "损坏",
    "拖地",
    "未做防护",
    "视线不足",
]


class MockLLMProvider(LLMProvider):
    def __init__(self, name: str, model: str, prompt_paths: dict[str, str] | None = None):
        self.name = name
        self.model = model
        self.prompt_paths = prompt_paths or {}

    def extract_hazard(self, raw: RawHazardText) -> list[ExtractionCandidate]:
        finding = _mock_finding(raw.raw_text)
        scene = _mock_scene(raw.raw_text)
        objects = _mock_objects(finding or raw.raw_text)

        if self.name == "qwen" and "且" in finding:
            finding = finding.split("且", maxsplit=1)[0].strip("，, ")
        if self.name == "deepseek" and not scene:
            scene = _prefix_until_marker(raw.raw_text)

        return [
            ExtractionCandidate(
                source_model=self.name,
                raw_record_id=raw.record_id,
                finding=finding,
                object=objects,
                scene=scene,
                confidence=0.8,
                metadata={"mock": True},
            )
        ]

    def arbitrate_extraction(
        self, raw: RawHazardText, candidates: list[ExtractionCandidate]
    ) -> list[ExtractionCandidate]:
        selected = sorted(
            candidates,
            key=lambda c: (len(c.finding), len(c.scene), len(c.object)),
            reverse=True,
        )[0]
        return [
            ExtractionCandidate(
                source_model=self.model,
                raw_record_id=raw.record_id,
                finding=selected.finding,
                object=selected.object,
                scene=selected.scene,
                confidence=0.7,
                metadata={"mock_arbitrated": True},
            )
        ]

    def split_scene(self, record: StructuredRecord) -> SceneSplitCandidate:
        loc_detail, risk_scene = _mock_scene_split(record.scene)
        if self.name == "doubao" and not loc_detail:
            loc_detail = record.scene.replace(risk_scene, "").strip("，, ")
        return SceneSplitCandidate(
            source_model=self.name,
            structured_record_id=record.record_id,
            loc_detail=loc_detail,
            risk_scene=risk_scene,
            confidence=0.8,
            metadata={"mock": True},
        )

    def arbitrate_scene_split(
        self, record: StructuredRecord, candidates: list[SceneSplitCandidate]
    ) -> SceneSplitCandidate:
        selected = sorted(
            candidates,
            key=lambda c: (bool(c.risk_scene), len(c.loc_detail), len(c.risk_scene)),
            reverse=True,
        )[0]
        return SceneSplitCandidate(
            source_model=self.model,
            structured_record_id=record.record_id,
            loc_detail=selected.loc_detail,
            risk_scene=selected.risk_scene,
            confidence=0.7,
            metadata={"mock_arbitrated": True},
        )

    def name_cluster(self, payload: dict[str, Any]) -> dict[str, str]:
        field_values = payload.get("field_values", {})
        channel_fields = payload.get("channel_fields", [])
        parts = [_most_common_text(field_values.get(field, [])) for field in channel_fields]
        parts = [part for part in parts if part]
        field_label = "+".join(channel_fields) if channel_fields else "channel"
        return {
            "cluster_name": " / ".join(parts[:3]) if parts else "未命名通道簇",
            "explanation": f"mock 命名：仅基于 {field_label} 通道的高频值和中心最近样本生成。",
        }


class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, name: str, model: str, dimension: int = 64):
        self.name = name
        self.model = model
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embedding(text, self.dimension) for text in texts]


def _mock_finding(text: str) -> str:
    parts = re.split(r"[，,；;]", text)
    marker_index = next(
        (idx for idx, part in enumerate(parts) if any(marker in part for marker in FINDING_MARKERS)),
        max(0, len(parts) - 1),
    )
    return "，".join(part.strip() for part in parts[marker_index:] if part.strip())


def _mock_scene(text: str) -> str:
    for keyword in RISK_SCENE_KEYWORDS:
        idx = text.find(keyword)
        if idx >= 0:
            return _expand_scene_left(text, idx, keyword)
    return _prefix_until_marker(text)


def _expand_scene_left(text: str, idx: int, keyword: str) -> str:
    left = text[:idx]
    boundary = max(left.rfind("，"), left.rfind(","), left.rfind("；"), left.rfind(";"))
    prefix = text[boundary + 1 : idx].strip() if boundary >= 0 else left.strip()
    return f"{prefix}{keyword}".strip()


def _prefix_until_marker(text: str) -> str:
    marker_positions = [text.find(marker) for marker in FINDING_MARKERS if marker in text]
    if not marker_positions:
        return text
    return text[: min(marker_positions)].strip("，, ")


def _mock_objects(text: str) -> list[str]:
    found = [keyword for keyword in OBJECT_KEYWORDS if keyword in text]
    if found:
        return found
    before_marker = _prefix_until_marker(text)
    tokens = re.split(r"[，,；;\s]", before_marker)
    return [tokens[-1]] if tokens and tokens[-1] else []


def _mock_scene_split(scene: str) -> tuple[str, str]:
    for keyword in RISK_SCENE_KEYWORDS:
        if keyword in scene:
            loc_detail = scene.replace(keyword, "").strip("，, ")
            return loc_detail, keyword
    return scene, ""


def _hash_embedding(text: str, dimension: int) -> list[float]:
    vector = np.zeros(dimension, dtype=float)
    for token in _char_ngrams(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm
    return vector.tolist()


def _char_ngrams(text: str) -> list[str]:
    clean = text.strip()
    if len(clean) <= 2:
        return [clean] if clean else ["<empty>"]
    return [clean[i : i + 2] for i in range(len(clean) - 1)]


def _most_common_text(values: list[str]) -> str:
    flattened = [str(value) for value in values if value]
    return Counter(flattened).most_common(1)[0][0] if flattened else ""
