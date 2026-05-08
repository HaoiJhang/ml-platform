from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    runs_dir: Path
    data_dir: Path
    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    return Settings(
        runs_dir=Path(os.getenv("ML_PLATFORM_RUNS_DIR", "runs")),
        data_dir=Path(os.getenv("ML_PLATFORM_DATA_DIR", "data")),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
