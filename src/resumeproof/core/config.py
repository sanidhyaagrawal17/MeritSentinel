"""Core configuration and settings for ResumeProof."""

from pathlib import Path
import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    # Data and Storage
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("RESUMEPROOF_DATA_DIR", "data")))
    db_path: Path = Field(default_factory=lambda: Path(os.getenv("RESUMEPROOF_DB_PATH", "resumeproof.db")))

    # Configurable Mastery & Verification Thresholds
    effective_score_threshold: float = Field(
        default_factory=lambda: float(os.getenv("RESUMEPROOF_EFFECTIVE_SCORE_THRESHOLD", "2.5")),
        description="Minimum effective score required for a claim to become eligible (default 2.5, configurable)"
    )
    confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("RESUMEPROOF_CONFIDENCE_THRESHOLD", "0.5")),
        description="Minimum confidence required for a claim to become eligible (default 0.5, configurable)"
    )

    # Mastery Math Parameters
    max_aggregate_jump: float = Field(
        default_factory=lambda: float(os.getenv("RESUMEPROOF_MAX_AGGREGATE_JUMP", "1.0")),
        description="Maximum allowed change in aggregate score per new answer (1.0 cap; raw scores remain uncapped)"
    )
    half_life_days: float = Field(
        default=30.0,
        description="Half life for time decay in days"
    )
    prior_score: float = Field(
        default=2.0,
        description="Prior score used for Bayesian shrinkage (0-4 scale)"
    )
    prior_weight: float = Field(
        default=1.5,
        description="Prior weight for confidence denominator: n_eff / (n_eff + prior_weight)"
    )

    # LLM Settings
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("RESUMEPROOF_OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Base URL for Ollama or OpenAI-compatible server"
    )
    extractor_model: str = Field(
        default_factory=lambda: os.getenv("RESUMEPROOF_EXTRACTOR_MODEL", "qwen2.5:3b"),
        description="Model name for structured claim extraction"
    )
    interviewer_model: str = Field(
        default_factory=lambda: os.getenv("RESUMEPROOF_INTERVIEWER_MODEL", "qwen2.5:3b"),
        description="Model name for question generation and chat"
    )
    grader_model: str = Field(
        default_factory=lambda: os.getenv("RESUMEPROOF_GRADER_MODEL", "qwen2.5:7b"),
        description="Model name for deterministic answer grading"
    )
    use_fake_llm: bool = Field(
        default_factory=lambda: os.getenv("RESUMEPROOF_USE_FAKE_LLM", "false").lower() in ("true", "1", "yes"),
        description="Set to True in offline CI/testing to prevent any network or Ollama calls"
    )

    # Web UI & Security Settings
    ui_host: str = Field(
        default_factory=lambda: os.getenv("RESUMEPROOF_UI_HOST", "127.0.0.1"),
        description="Host interface to bind the Web UI (defaults strictly to 127.0.0.1 for local isolation)"
    )
    ui_port: int = Field(
        default_factory=lambda: int(os.getenv("RESUMEPROOF_UI_PORT", "8000")),
        description="Port for Web UI"
    )
    upload_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("RESUMEPROOF_UPLOAD_DIR", ".data_uploads")),
        description="Directory for temporary uploaded user resumes (git-ignored)"
    )
    max_upload_size_bytes: int = Field(
        default=5 * 1024 * 1024,
        description="Maximum allowed upload file size in bytes (5MB)"
    )


# Global singleton settings
settings = Settings()
