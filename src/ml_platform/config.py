from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
_LEGACY_PROVIDER_PREFIX = "OPEN" + "AI"


@dataclass(frozen=True)
class Settings:
    runs_dir: Path
    data_dir: Path
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)


def load_settings() -> Settings:
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv(f"{_LEGACY_PROVIDER_PREFIX}_API_KEY") or None
    llm_base_url = os.getenv("LLM_BASE_URL") or os.getenv(f"{_LEGACY_PROVIDER_PREFIX}_BASE_URL") or None
    llm_model = os.getenv("LLM_MODEL") or os.getenv(f"{_LEGACY_PROVIDER_PREFIX}_MODEL") or "gpt-4o-mini"
    settings = Settings(
        runs_dir=Path(os.getenv("ML_PLATFORM_RUNS_DIR", "runs")),
        data_dir=Path(os.getenv("ML_PLATFORM_DATA_DIR", "data")),
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )
    logger.info("Settings loaded runs_dir=%s llm_enabled=%s model=%s", settings.runs_dir, settings.llm_enabled, settings.llm_model)
    return settings
