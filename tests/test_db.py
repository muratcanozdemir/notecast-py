from __future__ import annotations

import sqlite3

import pytest

from notecast import db

# ── notes ───────────────────────────────────────────────────────────

def test_add_and_get_note(conn):
    nid = db.add_note(conn, "Title", "content here", source_path="/tmp/x.md")
    note = db.get_note(conn, nid)
    assert note is not None
    assert note.title == "Title"
    assert note.content == "content here"
    assert note.status == "pending"
    assert note.source_path == "/tmp/x.md"
    assert note.topics == []
    assert note.vector == []


def test_get_note_missing_returns_none(conn):
    assert db.get_note(conn, "does-not-exist") is None


def test_update_note_serializes_json_fields(conn):
    nid = db.add_note(conn, "T", "c")
    db.update_note(conn, nid, summary="s", topics=["a", "b"], vector=[1.0, 2.0], status="processed")
    note = db.get_note(conn, nid)
    assert note.summary == "s"
    assert note.topics == ["a", "b"]
    assert note.vector == [1.0, 2.0]
    assert note.status == "processed"


def test_update_note_ignores_unknown_fields(conn):
    nid = db.add_note(conn, "T", "c")
    db.update_note(conn, nid, bogus="nope", id="other-id")
    note = db.get_note(conn, nid)
    assert note.id == nid  # id was not overwritten


def test_update_note_noop_with_no_valid_fields(conn):
    nid = db.add_note(conn, "T", "c")
    db.update_note(conn, nid, bogus="nope")  # should not raise


def test_delete_note(conn):
    nid = db.add_note(conn, "T", "c")
    db.delete_note(conn, nid)
    assert db.get_note(conn, nid) is None


def test_list_notes_filters_by_status(conn):
    a = db.add_note(conn, "A", "c")
    b = db.add_note(conn, "B", "c")
    db.update_note(conn, b, status="processed")
    pending = db.list_notes(conn, status="pending")
    assert [n.id for n in pending] == [a]
    everything = db.list_notes(conn)
    assert {n.id for n in everything} == {a, b}


def test_status_counts(conn):
    db.add_note(conn, "A", "c")
    b = db.add_note(conn, "B", "c")
    db.update_note(conn, b, status="processed")
    counts = db.status_counts(conn)
    assert counts == {"pending": 1, "processed": 1}


# ── search / FTS5 ───────────────────────────────────────────────────

def test_search_notes_matches_title_and_content(conn):
    db.add_note(conn, "Kubernetes basics", "pods and services")
    db.add_note(conn, "Unrelated", "gardening tips")
    results = db.search_notes(conn, "kubernetes")
    assert len(results) == 1
    assert results[0].title == "Kubernetes basics"


@pytest.mark.parametrize("query", [
    "port:8080",       # colon looks like an FTS5 column filter
    'unterminated"',    # stray quote
    "foo AND",          # dangling boolean operator
    "(unbalanced",       # unbalanced paren
    "a-b*c^",           # assorted operator characters
])
def test_search_notes_does_not_raise_on_special_characters(conn, query):
    db.add_note(conn, "Some Note", "content")
    # must not raise sqlite3.OperationalError
    results = db.search_notes(conn, query)
    assert isinstance(results, list)


def test_search_notes_empty_query_returns_no_results(conn):
    db.add_note(conn, "Some Note", "content")
    assert db.search_notes(conn, "") == []
    assert db.search_notes(conn, "   ") == []


# ── themes ──────────────────────────────────────────────────────────

def test_add_and_list_themes(conn):
    tid = db.add_theme(conn, "Tech", description="desc", is_base=True)
    themes = db.list_themes(conn)
    assert len(themes) == 1
    assert themes[0].id == tid
    assert themes[0].is_base is True


def test_get_theme_by_name_case_insensitive(conn):
    db.add_theme(conn, "Tech")
    assert db.get_theme_by_name(conn, "tech") is not None
    assert db.get_theme_by_name(conn, "TECH") is not None
    assert db.get_theme_by_name(conn, "missing") is None


def test_add_theme_with_parent_creates_edge(conn):
    parent = db.add_theme(conn, "Parent")
    child = db.add_theme(conn, "Child", parent_id=parent)
    assert db.theme_parents(conn, child) == [parent]
    assert db.theme_children(conn, parent) == [child]
    assert db.all_edges(conn) == [(child, parent)]


def test_add_theme_duplicate_name_raises(conn):
    db.add_theme(conn, "Tech")
    with pytest.raises(sqlite3.IntegrityError):
        db.add_theme(conn, "Tech")


def test_delete_theme_cascades_edges(conn):
    parent = db.add_theme(conn, "Parent")
    child = db.add_theme(conn, "Child", parent_id=parent)
    db.delete_theme(conn, parent)
    assert db.theme_parents(conn, child) == []


def test_theme_note_count_and_assignment(conn):
    tid = db.add_theme(conn, "Tech")
    nid = db.add_note(conn, "T", "c")
    assert db.theme_note_count(conn, tid) == 0
    db.assign_note_theme(conn, nid, tid)
    assert db.theme_note_count(conn, tid) == 1
    assert db.note_theme_ids(conn, nid) == [tid]
    assert [n.id for n in db.get_theme_notes(conn, tid)] == [nid]
    db.unassign_note_theme(conn, nid, tid)
    assert db.theme_note_count(conn, tid) == 0


def test_assign_note_theme_is_idempotent(conn):
    tid = db.add_theme(conn, "Tech")
    nid = db.add_note(conn, "T", "c")
    db.assign_note_theme(conn, nid, tid)
    db.assign_note_theme(conn, nid, tid)  # should not raise or duplicate
    assert db.theme_note_count(conn, tid) == 1


# ── config ──────────────────────────────────────────────────────────

def test_get_config_returns_default_when_unset(conn):
    assert db.get_config(conn, "ollama_url") == "http://localhost:11434"


def test_get_config_unknown_key_returns_empty_string(conn):
    assert db.get_config(conn, "totally_unknown_key") == ""


def test_set_config_overrides_default(conn):
    db.set_config(conn, "gen_model", "mistral:7b")
    assert db.get_config(conn, "gen_model") == "mistral:7b"


def test_set_config_upsert_overwrites(conn):
    db.set_config(conn, "gen_model", "a")
    db.set_config(conn, "gen_model", "b")
    assert db.get_config(conn, "gen_model") == "b"


def test_all_config_merges_defaults_and_overrides(conn):
    db.set_config(conn, "gen_model", "custom")
    merged = db.all_config(conn)
    assert merged["gen_model"] == "custom"
    assert merged["embed_model"] == db.DEFAULTS["embed_model"]


# ── pipeline counters ───────────────────────────────────────────────

def test_counters_increment_and_reset(conn):
    assert db.get_counter(conn, "classify_runs") == 0
    db.increment_counter(conn, "classify_runs")
    db.increment_counter(conn, "classify_runs")
    assert db.get_counter(conn, "classify_runs") == 2
    db.reset_counter(conn, "classify_runs")
    assert db.get_counter(conn, "classify_runs") == 0
