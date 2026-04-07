from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Coinbase Support Agent"
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Qwen / OpenAI-compatible endpoint
    llm_base_url: str = Field(
        default="https://rsm-8430-finalproject.bjlkeng.io/v1",
        alias="LLM_BASE_URL",
    )
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="qwen3-30b-a3b-fp8", alias="LLM_MODEL")
    llm_timeout_s: float = Field(default=120.0, alias="LLM_TIMEOUT_S")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")

    # Embeddings (local)
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL",
    )

    # Paths
    data_dir: Path = Field(default=_REPO_ROOT / "data", alias="DATA_DIR")
    corpus_path: Path = Field(default=_REPO_ROOT / "data" / "corpus" / "articles.jsonl")
    chunks_path: Path = Field(default=_REPO_ROOT / "data" / "corpus" / "chunks.jsonl")
    faiss_index_path: Path = Field(default=_REPO_ROOT / "data" / "index" / "faiss.index")
    faiss_meta_path: Path = Field(default=_REPO_ROOT / "data" / "index" / "faiss_meta.jsonl")
    sqlite_path: Path = Field(default=_REPO_ROOT / "data" / "app.db", alias="SQLITE_PATH")

    # Auth (Streamlit / optional API)
    demo_password: str = Field(default="changeme", alias="DEMO_PASSWORD")
    demo_username: str = Field(default="demo", alias="DEMO_USERNAME")

    # Retrieval
    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")
    rerank_top_n: int = Field(default=4, alias="RERANK_TOP_N")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    return Settings()
