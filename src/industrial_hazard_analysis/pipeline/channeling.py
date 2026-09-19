from __future__ import annotations

from industrial_hazard_analysis.models import ChannelConfig, ChannelInput, SceneSplitRecord


def load_channel_configs(config: dict) -> list[ChannelConfig]:
    channels = config.get("experiments", {}).get("channels", {})
    return [
        ChannelConfig(
            experiment_id=experiment_id,
            fields=tuple(definition["fields"]),
            name=definition.get("name"),
        )
        for experiment_id, definition in channels.items()
    ]


def build_channel_text(record: SceneSplitRecord, fields: tuple[str, ...]) -> str:
    pieces: list[str] = []
    for field in fields:
        value = getattr(record, field)
        if isinstance(value, list):
            text = "; ".join(item for item in value if item)
        else:
            text = str(value).strip()
        if text:
            pieces.append(f"{field}: {text}")
    return " | ".join(pieces)


def generate_channel_inputs(
    records: list[SceneSplitRecord], channel_configs: list[ChannelConfig]
) -> dict[str, list[ChannelInput]]:
    result: dict[str, list[ChannelInput]] = {}
    for channel in channel_configs:
        result[channel.experiment_id] = [
            ChannelInput(
                experiment_id=channel.experiment_id,
                record_id=record.record_id,
                input_text=build_channel_text(record, channel.fields),
            )
            for record in records
        ]
    return result


def generate_baseline_inputs(records: list[SceneSplitRecord]) -> dict[str, list[ChannelInput]]:
    return {
        "B1": [
            ChannelInput("B1", record.record_id, record.raw_text)
            for record in records
        ]
    }

