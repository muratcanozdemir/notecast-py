"""Ollama LLM client — stdlib urllib, no SDK dependencies."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(Exception):
    pass


class Ollama:
    def __init__(self, base_url: str = "http://localhost:11434",
                 gen_model: str = "llama3.2:1b",
                 embed_model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.gen_model = gen_model
        self.embed_model = embed_model

    # ── health ──────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            return False

    # ── generate ────────────────────────────────────────────────────

    def generate(self, prompt: str, *, system: str = "",
                 model: str | None = None,
                 json_mode: bool = False) -> str:
        body: dict[str, Any] = {
            "model": model or self.gen_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            body["system"] = system
        if json_mode:
            body["format"] = "json"

        data = self._post("/api/generate", body)
        return data["response"]

    def generate_json(self, prompt: str, *, system: str = "",
                      model: str | None = None) -> Any:
        raw = self.generate(prompt, system=system, model=model, json_mode=True)
        return json.loads(raw)

    # ── embed ───────────────────────────────────────────────────────

    def embed(self, text: str, *, model: str | None = None) -> list[float]:
        data = self._post("/api/embed", {
            "model": model or self.embed_model,
            "input": text,
        })
        return data["embeddings"][0]

    def embed_batch(self, texts: list[str],
                    *, model: str | None = None) -> list[list[float]]:
        data = self._post("/api/embed", {
            "model": model or self.embed_model,
            "input": texts,
        })
        return data["embeddings"]

    # ── internal ────────────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            raise OllamaError(f"HTTP {e.code}: {msg}") from e
        except urllib.error.URLError as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url} — is it running?"
            ) from e
