from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .models import RawHazardText


def read_raw_texts(
    path: str | Path,
    *,
    sheet_name: str | int | None = None,
    id_column: str | None = None,
    text_column: str | None = None,
) -> list[RawHazardText]:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name or 0)
    else:
        df = pd.read_csv(path)

    if id_column not in df.columns:
        id_column = _first_existing_column(df, ["raw_record_id", "id", "ID", "编号", "序号"])
    if text_column not in df.columns:
        text_column = _first_existing_column(df, ["raw_text", "隐患描述", "隐患文本", "text"])
    if text_column is None:
        raise ValueError(
            "Input file missing text column. Provide --text-column or use one of: "
            "raw_text, 隐患描述, 隐患文本, text"
        )

    df = df[df[text_column].notna()].copy()
    return [
        RawHazardText(
            record_id=str(row[id_column]) if id_column else f"R{index + 1:05d}",
            raw_text=str(row[text_column]).strip(),
        )
        for index, row in df.reset_index(drop=True).iterrows()
        if str(row[text_column]).strip()
    ]


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def _normalize_value(value):
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_dataclass_csv(rows: Sequence[object] | Iterable[object], path: str | Path) -> None:
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
        return
    dict_rows = []
    for row in rows:
        data = asdict(row) if is_dataclass(row) else dict(row)
        dict_rows.append({key: _normalize_value(value) for key, value in data.items()})
    pd.DataFrame(dict_rows).to_csv(path, index=False, encoding="utf-8-sig")

