from __future__ import annotations

import io
import json
import urllib.error

import pytest

from notecast.llm import Ollama, OllamaError


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_ping_true_when_reachable(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=5: _FakeResponse({})
    )
    assert Ollama().ping() is True


def test_ping_false_on_url_error(monkeypatch):
    def raise_error(req, timeout=5):
        raise urllib.error.URLError("no route")
    monkeypatch.setattr("urllib.request.urlopen", raise_error)
    assert Ollama().ping() is False


def test_generate_returns_response_text(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=120: _FakeResponse({"response": "hello"}),
    )
    assert Ollama().generate("prompt") == "hello"


def test_generate_json_parses_response(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=120: _FakeResponse({"response": '{"a": 1}'}),
    )
    assert Ollama().generate_json("prompt") == {"a": 1}


def test_embed_returns_first_vector(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=120: _FakeResponse({"embeddings": [[1.0, 2.0]]}),
    )
    assert Ollama().embed("text") == [1.0, 2.0]


def test_embed_batch_returns_all_vectors(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=120: _FakeResponse({"embeddings": [[1.0], [2.0]]}),
    )
    assert Ollama().embed_batch(["a", "b"]) == [[1.0], [2.0]]


def test_post_raises_ollama_error_on_http_error(monkeypatch):
    def raise_http_error(req, timeout=120):
        raise urllib.error.HTTPError(
            "http://x", 500, "boom", {}, io.BytesIO(b"server exploded")
        )
    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    with pytest.raises(OllamaError, match="HTTP 500"):
        Ollama().generate("prompt")


def test_post_raises_ollama_error_on_url_error(monkeypatch):
    def raise_url_error(req, timeout=120):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr("urllib.request.urlopen", raise_url_error)
    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        Ollama().generate("prompt")


def test_base_url_trailing_slash_is_stripped():
    client = Ollama(base_url="http://localhost:11434/")
    assert client.base_url == "http://localhost:11434"
