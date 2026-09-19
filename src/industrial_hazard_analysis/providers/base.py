from __future__ import annotations

from abc import ABC, abstractmethod

from industrial_hazard_analysis.models import (
    ExtractionCandidate,
    RawHazardText,
    SceneSplitCandidate,
    StructuredRecord,
)


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def extract_hazard(self, raw: RawHazardText) -> list[ExtractionCandidate]:
        raise NotImplementedError

    def extract_hazards(
        self,
        raw_records: list[RawHazardText],
    ) -> dict[str, list[ExtractionCandidate]]:
        return {raw.record_id: self.extract_hazard(raw) for raw in raw_records}

    @abstractmethod
    def arbitrate_extraction(
        self, raw: RawHazardText, candidates: list[ExtractionCandidate]
    ) -> list[ExtractionCandidate]:
        raise NotImplementedError

    @abstractmethod
    def split_scene(self, record: StructuredRecord) -> SceneSplitCandidate:
        raise NotImplementedError

    def split_scenes(
        self,
        records: list[StructuredRecord],
    ) -> dict[str, SceneSplitCandidate]:
        return {record.record_id: self.split_scene(record) for record in records}

    @abstractmethod
    def arbitrate_scene_split(
        self, record: StructuredRecord, candidates: list[SceneSplitCandidate]
    ) -> SceneSplitCandidate:
        raise NotImplementedError

    @abstractmethod
    def name_cluster(self, payload: dict) -> dict:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    name: str
    model: str
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
