from __future__ import annotations

from industrial_hazard_analysis.models import (
    ChannelConfig,
    ChannelInput,
    RawHazardText,
    SceneSplitRecord,
    StructuredRecord,
)
from industrial_hazard_analysis.pipeline.channeling import build_channel_text
from industrial_hazard_analysis.pipeline.data_revision import (
    build_canonical_raw_text_groups,
    consolidate_units_by_raw_text,
    filter_and_deduplicate_units,
)
from industrial_hazard_analysis.providers.base import LLMProvider


def generate_no_consistency_records_by_provider(
    raw_records: list[RawHazardText],
    providers: list[LLMProvider],
    batch_size: int = 1,
) -> dict[str, list[SceneSplitRecord]]:
    records_by_provider = {}
    for index, provider in enumerate(providers, start=1):
        print(
            f"Ablation no-consistency extraction provider {index}/{len(providers)}: {provider.name}",
            flush=True,
        )
        records_by_provider[provider.name] = generate_no_consistency_records(
            raw_records,
            provider,
            batch_size=batch_size,
        )
        print(
            f"Ablation no-consistency extraction provider {provider.name}: "
            f"{len(records_by_provider[provider.name])} row(s)",
            flush=True,
        )
    return records_by_provider


def generate_no_consistency_records(
    raw_records: list[RawHazardText],
    provider: LLMProvider,
    batch_size: int = 1,
) -> list[SceneSplitRecord]:
    records: list[SceneSplitRecord] = []
    record_index = 1
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batches = list(_batches(raw_records, batch_size))
    for batch_index, raw_batch in enumerate(batches, start=1):
        print(
            f"  {provider.name} Appendix A batch {batch_index}/{len(batches)}: "
            f"{len(raw_batch)} raw row(s)",
            flush=True,
        )
        candidates_by_raw_id = provider.extract_hazards(raw_batch)
        for raw in raw_batch:
            for candidate_index, candidate in enumerate(
                candidates_by_raw_id.get(raw.record_id, []), start=1
            ):
                structured = StructuredRecord(
                    record_id=f"N-{provider.name}-{record_index:05d}",
                    raw_record_id=raw.record_id,
                    raw_text=raw.raw_text,
                    finding=candidate.finding,
                    object=candidate.object,
                    scene=candidate.scene,
                    accepted_by=f"no_consistency:{provider.name}",
                    arbitration_required=False,
                    metadata={
                        "candidate_count": 1,
                        "candidate_index": candidate_index,
                        "ablation": "no_consistency_arbitration",
                        "llm_batch_size": batch_size,
                    },
                )
                records.append(
                    SceneSplitRecord(
                        record_id=structured.record_id,
                        raw_record_id=structured.raw_record_id,
                        raw_text=structured.raw_text,
                        finding=structured.finding,
                        object=structured.object,
                        scene=structured.scene,
                        loc_detail="",
                        risk_scene="",
                        accepted_by=f"no_consistency:{provider.name}",
                        arbitration_required=False,
                        metadata=structured.metadata,
                    )
                )
                record_index += 1
    return records


def prepare_no_consistency_record_level(
    raw_records: list[RawHazardText],
    unit_records: list[SceneSplitRecord],
    *,
    provider_name: str,
) -> tuple[list[SceneSplitRecord], list[SceneSplitRecord]]:
    """Deduplicate A1 units and aggregate them to the main record grain."""

    raw_groups, _ = build_canonical_raw_text_groups(raw_records)
    revised = filter_and_deduplicate_units(
        raw_groups,
        unit_records,
        source_label=f"A1_{provider_name}",
    )
    kept_units = list(revised.kept_records)
    record_level_records = consolidate_units_by_raw_text(
        raw_groups,
        kept_units,
        source_label=f"A1_{provider_name}",
    )
    return kept_units, record_level_records


def _batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_ablation_inputs(
    config: dict,
    full_records: list[SceneSplitRecord],
    no_consistency_records_by_provider: dict[str, list[SceneSplitRecord]],
) -> tuple[dict[str, list[ChannelInput]], dict[str, ChannelConfig], dict[str, list[SceneSplitRecord]]]:
    experiment_definitions = config.get("ablation", {}).get("experiments", {})
    provider_names = config.get("ablation", {}).get("no_consistency_providers", [])
    inputs_by_experiment: dict[str, list[ChannelInput]] = {}
    configs_by_experiment: dict[str, ChannelConfig] = {}
    records_by_experiment: dict[str, list[SceneSplitRecord]] = {}

    for experiment_id, definition in experiment_definitions.items():
        fields = tuple(definition["fields"])
        use_consistency = bool(definition.get("use_consistency_arbitration", True))
        provider_variants = bool(definition.get("provider_variants", False))

        if use_consistency:
            _add_ablation_input(
                experiment_id,
                definition,
                fields,
                full_records,
                inputs_by_experiment,
                configs_by_experiment,
                records_by_experiment,
            )
            continue

        selected_providers = provider_names if provider_variants else provider_names[:1]
        for provider_name in selected_providers:
            variant_id = f"{experiment_id}-{provider_name}"
            source_records = no_consistency_records_by_provider[provider_name]
            _add_ablation_input(
                variant_id,
                definition,
                fields,
                source_records,
                inputs_by_experiment,
                configs_by_experiment,
                records_by_experiment,
                name_suffix=provider_name,
            )

    return inputs_by_experiment, configs_by_experiment, records_by_experiment


def _add_ablation_input(
    experiment_id: str,
    definition: dict,
    fields: tuple[str, ...],
    source_records: list[SceneSplitRecord],
    inputs_by_experiment: dict[str, list[ChannelInput]],
    configs_by_experiment: dict[str, ChannelConfig],
    records_by_experiment: dict[str, list[SceneSplitRecord]],
    *,
    name_suffix: str | None = None,
) -> None:
    name = definition.get("name")
    if name_suffix:
        name = f"{name}_{name_suffix}" if name else name_suffix
    configs_by_experiment[experiment_id] = ChannelConfig(
        experiment_id=experiment_id,
        fields=fields,
        name=name,
    )
    records_by_experiment[experiment_id] = source_records
    inputs_by_experiment[experiment_id] = [
        ChannelInput(
            experiment_id=experiment_id,
            record_id=record.record_id,
            input_text=build_channel_text(record, fields),
        )
        for record in source_records
    ]
