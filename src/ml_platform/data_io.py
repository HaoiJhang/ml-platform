from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import pandas as pd


def read_csv(source: str | Path | IO[Any]) -> pd.DataFrame:
    df = pd.read_csv(source)
    if df.empty:
        raise ValueError("CSV contains no rows.")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"CSV contains duplicate column names: {duplicates}")
    return df
