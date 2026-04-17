from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = Field(default_factory=lambda: Path('data'))
    raw_dir: Path = Field(default_factory=lambda: Path('data/raw'))
    processed_dir: Path = Field(default_factory=lambda: Path('data/processed'))
    index_dir: Path = Field(default_factory=lambda: Path('data/index'))
    chroma_dir: Path = Field(default_factory=lambda: Path('data/index/chroma'))

    sec_user_agent: str = 'TeslaQA your_email@example.com'

    # LLM settings: default to Qwen via DashScope OpenAI-compatible endpoint.
    llm_provider: str = 'dashscope'
    llm_model: str = 'qwen-plus'
    llm_api_key: str | None = None
    llm_base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

    # Backward-compatible aliases.
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # Embedding settings.
    embedding_backend: str = 'local'  # local | api
    embedding_model: str = 'BAAI/bge-small-en-v1.5'
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model_api: str = 'text-embedding-v3'

    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_top_k: int = 10

    @property
    def resolved_llm_api_key(self) -> str | None:
        return self.llm_api_key or self.openai_api_key

    @property
    def resolved_llm_base_url(self) -> str | None:
        return self.llm_base_url or self.openai_base_url

    @property
    def resolved_embedding_api_key(self) -> str | None:
        return self.embedding_api_key or self.resolved_llm_api_key

    @property
    def resolved_embedding_base_url(self) -> str | None:
        return self.embedding_base_url or self.resolved_llm_base_url


settings = Settings()
