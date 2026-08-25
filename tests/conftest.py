from __future__ import annotations

import pytest
from click.testing import CliRunner

from notecast import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    yield c
    c.close()


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Point the CLI at an isolated on-disk database for the duration of a test."""
    db_path = tmp_path / "notecast.db"
    monkeypatch.setenv("NOTECAST_DB", str(db_path))
    return db_path


class FakeOllama:
    """Duck-typed stand-in for notecast.llm.Ollama — no network calls."""

    def __init__(self, *, json_responses=None, embed_dim=4, reachable=True):
        self._json_responses = list(json_responses or [])
        self.embed_dim = embed_dim
        self.reachable = reachable
        self.calls: list[str] = []

    def ping(self) -> bool:
        return self.reachable

    def generate_json(self, prompt: str, *, system: str = "", model=None):
        self.calls.append(prompt)
        if not self._json_responses:
            raise AssertionError("FakeOllama: no more queued responses")
        response = self._json_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def embed(self, text: str, *, model=None) -> list[float]:
        # deterministic pseudo-embedding derived from text length/hash
        h = abs(hash(text))
        return [((h >> (8 * i)) & 0xFF) / 255.0 for i in range(self.embed_dim)]


@pytest.fixture
def fake_ollama():
    return FakeOllama
