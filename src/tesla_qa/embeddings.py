from __future__ import annotations

from typing import Iterable

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .settings import settings


class EmbeddingClient:
    def __init__(self) -> None:
        self.backend = settings.embedding_backend.lower()
        self.local_model: SentenceTransformer | None = None
        self.api_client: OpenAI | None = None
        if self.backend == 'local':
            self.local_model = SentenceTransformer(settings.embedding_model)
        elif self.backend == 'api':
            api_key = settings.resolved_embedding_api_key
            base_url = settings.resolved_embedding_base_url
            if not api_key:
                raise ValueError('Missing embedding API key. Set EMBEDDING_API_KEY or LLM_API_KEY.')
            self.api_client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f'Unsupported embedding backend: {self.backend}')

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if self.backend == 'local':
            assert self.local_model is not None
            return self.local_model.encode(text_list, normalize_embeddings=True).tolist()
        assert self.api_client is not None
        response = self.api_client.embeddings.create(
            model=settings.embedding_model_api,
            input=text_list,
        )
        return [item.embedding for item in response.data]
