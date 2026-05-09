from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, Any

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv(source: str | Path | IO[Any]) -> pd.DataFrame:
    logger.info("Reading CSV source=%s", source if isinstance(source, (str, Path)) else "<file-like>")
    df = pd.read_csv(source)
    if df.empty:
        raise ValueError("CSV contains no rows.")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"CSV contains duplicate column names: {duplicates}")
    logger.info("CSV loaded rows=%d columns=%d", len(df), len(df.columns))
    return df
