"""Inspectable loopback-only Ollama embedding adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from rag_harness.models import Chunk


@dataclass(frozen=True)
class OllamaEmbeddingService:
    model: str
    dimension: int
    endpoint: str = "http://127.0.0.1:11434/api/embed"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
        }:
            raise ValueError("Ollama endpoint must use loopback HTTP")

    def preflight(self) -> None:
        parsed = urlparse(self.endpoint)
        tags_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"
        with self._open(tags_url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama preflight returned {response.status}")
            payload = json.load(response)
        installed_models = {
            model.get("name")
            for model in payload.get("models", [])
            if isinstance(model, dict)
        }
        if self.model not in installed_models:
            raise RuntimeError(f"Ollama model is unavailable: {self.model}")

    @staticmethod
    def _open(request: str | Request, *, timeout: float):
        opener = build_opener(ProxyHandler({}))
        return opener.open(request, timeout=timeout)

    def _embed(self, text: str, prefix: str) -> list[float]:
        payload = json.dumps(
            {
                "model": self.model,
                "input": f"{prefix}: {text}",
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with self._open(request, timeout=self.timeout_seconds) as response:
            result = json.load(response)

        vector = result["embeddings"][0]
        if len(vector) != self.dimension:
            raise RuntimeError(f"unexpected embedding dimension: {len(vector)}")
        return vector

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        return [self._embed(chunk.text, "search_document") for chunk in chunks]

    def embed_query(self, query_text: str) -> list[float]:
        return self._embed(query_text, "search_query")
