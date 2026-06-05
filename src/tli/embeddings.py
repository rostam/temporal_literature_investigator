"""Pluggable multilingual embedder.

Anthropic does not provide an embeddings endpoint, so we use either:
  - "local": intfloat/multilingual-e5-large via sentence-transformers (offline, no key)
  - "voyage": Voyage AI multilingual embeddings (needs VOYAGE_API_KEY)

e5 models want "query:" / "passage:" prefixes — handled here so callers don't care.
"""
from __future__ import annotations

from functools import lru_cache

from . import config


class Embedder:
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalE5Embedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self._is_e5 = "e5" in model_name.lower()
        # Method was renamed across sentence-transformers versions.
        get_dim = (getattr(self._model, "get_embedding_dimension", None)
                   or self._model.get_sentence_embedding_dimension)
        self.dim = get_dim()

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if self._is_e5:
            texts = [f"{prefix}: {t}" for t in texts]
        vecs = self._model.encode(texts, normalize_embeddings=True,
                                  convert_to_numpy=True)
        return vecs.tolist()

    def embed_documents(self, texts):
        return self._encode(texts, "passage")

    def embed_query(self, text):
        return self._encode([text], "query")[0]


class VoyageEmbedder(Embedder):
    def __init__(self, model_name: str):
        import voyageai  # lazy import

        self._client = voyageai.Client()
        self._model = model_name
        self.dim = 1024  # voyage-3 default

    def embed_documents(self, texts):
        r = self._client.embed(texts, model=self._model, input_type="document")
        return r.embeddings

    def embed_query(self, text):
        r = self._client.embed([text], model=self._model, input_type="query")
        return r.embeddings[0]


def resolved_model(backend: str | None = None) -> str:
    backend = (backend or config.EMBED_BACKEND).lower()
    return config.VOYAGE_EMBED_MODEL if backend == "voyage" else config.LOCAL_EMBED_MODEL


@lru_cache(maxsize=4)
def get_embedder(backend: str | None = None, model: str | None = None) -> Embedder:
    """Cached embedder. Pass backend/model explicitly to match a prebuilt index
    (see rag.retrieve, which reads them from the collection metadata); omit to
    use the configured defaults (used when building the index)."""
    backend = (backend or config.EMBED_BACKEND).lower()
    if backend == "voyage":
        return VoyageEmbedder(model or config.VOYAGE_EMBED_MODEL)
    return LocalE5Embedder(model or config.LOCAL_EMBED_MODEL)
