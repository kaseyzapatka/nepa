from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency installed in app env
    def load_dotenv(*_: object, **__: object) -> bool:
        return False


RAG_DIR = Path(__file__).resolve().parents[2]
PHASE2_DIR = RAG_DIR.parent
REPO_ROOT = PHASE2_DIR.parent


def _load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(RAG_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, str(default)).lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class RagConfig:
    repo_root: Path
    phase2_dir: Path
    rag_dir: Path
    data_dir: Path
    catalog_path: Path
    manifest_path: Path
    hf_repo: str
    hf_revision: str
    llm_provider: str
    llm_model: str
    llm_quality_model: str
    llm_fallback_model: str
    ollama_host: str
    llm_timeout_seconds: int
    llm_num_ctx: int
    llm_num_predict: int
    top_k_per_shard: int
    max_context_passages: int
    max_context_tokens: int
    require_source_labels: bool
    session_history_only: bool
    default_scope: str
    project_energy_type: str

    @property
    def projects_path(self) -> Path:
        return self.phase2_dir / "data" / "analysis" / "projects_combined.parquet"

    @property
    def documents_path(self) -> Path:
        return self.phase2_dir / "data" / "analysis" / "documents_combined.parquet"

    def pages_path(self, process_type: str) -> Path:
        return self.phase2_dir / "data" / "processed" / process_type.lower() / "pages.parquet"


def load_config() -> RagConfig:
    _load_env()

    data_dir = resolve_path(_env("NEPA_RAG_DATA_DIR", "phase2/rag/data"))
    catalog_path = resolve_path(_env("NEPA_RAG_CATALOG", str(data_dir / "rag_catalog.duckdb")))
    manifest_path = resolve_path(_env("NEPA_RAG_MANIFEST", str(data_dir / "manifest.json")))

    return RagConfig(
        repo_root=REPO_ROOT,
        phase2_dir=PHASE2_DIR,
        rag_dir=RAG_DIR,
        data_dir=data_dir,
        catalog_path=catalog_path,
        manifest_path=manifest_path,
        hf_repo=_env("NEPA_RAG_DB_HF_REPO"),
        hf_revision=_env("NEPA_RAG_DB_HF_REVISION", "main"),
        llm_provider=_env("LLM_PROVIDER", "ollama").lower(),
        llm_model=_env("LLM_MODEL", "gemma4:e2b"),
        llm_quality_model=_env("LLM_QUALITY_MODEL", "gemma4:e4b"),
        llm_fallback_model=_env("LLM_FALLBACK_MODEL", "gemma4:e2b"),
        ollama_host=_env("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 120),
        llm_num_ctx=_env_int("LLM_NUM_CTX", 4096),
        llm_num_predict=_env_int("LLM_NUM_PREDICT", 350),
        top_k_per_shard=_env_int("RAG_TOP_K_PER_SHARD", 3),
        max_context_passages=_env_int("RAG_MAX_CONTEXT_PASSAGES", 4),
        max_context_tokens=_env_int("RAG_MAX_CONTEXT_TOKENS", 2500),
        require_source_labels=_env_bool("RAG_REQUIRE_SOURCE_LABELS", True),
        session_history_only=_env_bool("RAG_SESSION_HISTORY_ONLY", True),
        default_scope=_env("RAG_DEFAULT_SCOPE", "selected_project"),
        project_energy_type=_env("RAG_PROJECT_ENERGY_TYPE", "Clean"),
    )
