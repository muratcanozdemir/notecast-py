from __future__ import annotations

import pytest

from notecast import cli, db


def invoke(runner, *args, **kwargs):
    return runner.invoke(cli.cli, list(args), **kwargs)


# ── add / add-batch ─────────────────────────────────────────────────

def test_add_note_from_file(runner, cli_env, tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# My Title\n\nbody text")
    result = invoke(runner, "add", str(f))
    assert result.exit_code == 0
    assert "My Title" in result.output


def test_add_rejects_directory_path(runner, cli_env, tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    result = invoke(runner, "add", str(d))
    assert result.exit_code != 0


def test_add_process_reports_unreachable_ollama(runner, cli_env, tmp_path, monkeypatch):
    f = tmp_path / "note.md"
    f.write_text("content, no heading")
    monkeypatch.setattr("notecast.llm.Ollama.ping", lambda self: False)
    result = invoke(runner, "add", str(f), "--process")
    assert result.exit_code == 1
    assert "cannot reach Ollama" in result.output


def test_add_batch_dry_run_counts_matching_files(runner, cli_env, tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.txt").write_text("b body")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")
    result = invoke(runner, "add-batch", str(tmp_path), "--dry-run")
    assert result.exit_code == 0
    assert "would add 2 note(s)" in result.output


def test_add_batch_handles_whitespace_after_comma_in_ext(runner, cli_env, tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.txt").write_text("b body")
    result = invoke(runner, "add-batch", str(tmp_path), "--ext", ".md, .txt", "--dry-run")
    assert result.exit_code == 0
    assert "would add 2 note(s)" in result.output


def test_add_batch_excludes_glob_patterns(runner, cli_env, tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "README.md").write_text("# R")
    result = invoke(runner, "add-batch", str(tmp_path), "--dry-run", "-x", "README.md")
    assert result.exit_code == 0
    assert "would add 1 note(s), skipped 1" in result.output


# ── status ──────────────────────────────────────────────────────────

def test_status_shows_zero_notes_initially(runner, cli_env):
    result = invoke(runner, "status")
    assert result.exit_code == 0
    assert "notes: 0" in result.output


def test_version_matches_package_metadata(runner):
    import notecast
    result = invoke(runner, "--version")
    assert result.exit_code == 0
    assert notecast.__version__ in result.output


# ── theme ───────────────────────────────────────────────────────────

def test_theme_add_and_list(runner, cli_env):
    result = invoke(runner, "theme", "add", "Tech", "--base")
    assert result.exit_code == 0
    result = invoke(runner, "theme", "list")
    assert "Tech" in result.output
    assert "[base]" in result.output


def test_theme_add_duplicate_name_fails_gracefully(runner, cli_env):
    invoke(runner, "theme", "add", "Tech")
    result = invoke(runner, "theme", "add", "Tech")
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_theme_add_with_unknown_parent_errors(runner, cli_env):
    result = invoke(runner, "theme", "add", "Child", "--parent", "NoSuchParent")
    assert result.exit_code == 1
    assert "not found" in result.output


def test_theme_remove(runner, cli_env):
    invoke(runner, "theme", "add", "Tech")
    result = invoke(runner, "theme", "remove", "Tech")
    assert result.exit_code == 0
    result = invoke(runner, "theme", "list")
    assert "no themes" in result.output


def test_theme_remove_missing_theme(runner, cli_env):
    result = invoke(runner, "theme", "remove", "Ghost")
    assert "not found" in result.output


def test_theme_list_shows_parent(runner, cli_env):
    invoke(runner, "theme", "add", "Parent", "--base")
    invoke(runner, "theme", "add", "Child", "--parent", "Parent")
    result = invoke(runner, "theme", "list")
    assert "Child" in result.output
    assert "← Parent" in result.output


# ── delete ──────────────────────────────────────────────────────────

def test_delete_no_match(runner, cli_env):
    result = invoke(runner, "delete", "nothing")
    assert "no match" in result.output


def test_delete_multiple_matches_lists_them(runner, cli_env, tmp_path):
    (tmp_path / "a.md").write_text("# Kubernetes A")
    (tmp_path / "b.md").write_text("# Kubernetes B")
    invoke(runner, "add", str(tmp_path / "a.md"))
    invoke(runner, "add", str(tmp_path / "b.md"))
    result = invoke(runner, "delete", "kubernetes")
    assert "multiple matches" in result.output


def test_delete_single_match_confirms_and_deletes(runner, cli_env, tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Unique Title")
    invoke(runner, "add", str(f))
    result = invoke(runner, "delete", "Unique Title", input="y\n")
    assert "deleted" in result.output
    conn = db.connect(str(cli_env))
    assert db.list_notes(conn) == []


# ── scan ────────────────────────────────────────────────────────────

def test_scan_reports_unreachable_ollama(runner, cli_env, monkeypatch):
    monkeypatch.setattr("notecast.llm.Ollama.ping", lambda self: False)
    result = invoke(runner, "scan", "--stage", "process")
    assert result.exit_code == 1
    assert "cannot reach Ollama" in result.output


def test_scan_process_stage(runner, cli_env, tmp_path, monkeypatch):
    f = tmp_path / "note.md"
    f.write_text("content")
    invoke(runner, "add", str(f))
    monkeypatch.setattr("notecast.llm.Ollama.ping", lambda self: True)
    monkeypatch.setattr(
        "notecast.llm.Ollama.generate_json",
        lambda self, prompt, **kw: {"summary": "s", "keywords": ["k"]},
    )
    monkeypatch.setattr("notecast.llm.Ollama.embed", lambda self, text, **kw: [0.1, 0.2])
    result = invoke(runner, "scan", "--stage", "process")
    assert result.exit_code == 0
    assert "processed 1" in result.output


def test_scan_consolidate_stage_needs_no_ollama(runner, cli_env, monkeypatch):
    # consolidate never calls Ollama — ping should not even be consulted
    monkeypatch.setattr(
        "notecast.llm.Ollama.ping",
        lambda self: (_ for _ in ()).throw(AssertionError("ping should not be called")),
    )
    result = invoke(runner, "scan", "--stage", "consolidate")
    assert result.exit_code == 0


# ── search --similar ────────────────────────────────────────────────

def test_search_similar_shows_related_notes(runner, cli_env, tmp_path):
    (tmp_path / "a.md").write_text("# Target Note")
    (tmp_path / "b.md").write_text("# Other Note")
    invoke(runner, "add", str(tmp_path / "a.md"))
    invoke(runner, "add", str(tmp_path / "b.md"))
    conn = db.connect(str(cli_env))
    notes = db.list_notes(conn)
    for n in notes:
        db.update_note(conn, n.id, vector=[1.0, 0.0], status="processed")
    result = invoke(runner, "search", "Target Note", "--similar")
    assert result.exit_code == 0
    assert "similar to" in result.output


# ── graph ───────────────────────────────────────────────────────────

def test_graph_no_themes(runner, cli_env):
    result = invoke(runner, "graph")
    assert result.exit_code == 0
    assert "no themes to graph" in result.output


def test_graph_missing_dot_executable_reports_friendly_error(runner, cli_env, monkeypatch, tmp_path):
    invoke(runner, "theme", "add", "Tech", "--base")
    monkeypatch.setenv("PATH", "")  # hide the `dot` executable, if present
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = invoke(runner, "graph")
    assert result.exit_code == 1
    assert "graphviz" in result.output.lower()


# ── search ──────────────────────────────────────────────────────────

def test_search_no_results(runner, cli_env):
    result = invoke(runner, "search", "nothing")
    assert result.exit_code == 0
    assert "no results" in result.output


def test_search_finds_note(runner, cli_env, tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Kubernetes Basics\n\npods and services")
    invoke(runner, "add", str(f))
    result = invoke(runner, "search", "kubernetes")
    assert result.exit_code == 0
    assert "Kubernetes Basics" in result.output


@pytest.mark.parametrize("query", ["port:8080", 'bad"query', "trailing AND"])
def test_search_does_not_crash_on_special_characters(runner, cli_env, query):
    result = invoke(runner, "search", query)
    assert result.exit_code == 0


# ── config ──────────────────────────────────────────────────────────

def test_config_get_all_defaults(runner, cli_env):
    result = invoke(runner, "config", "get")
    assert result.exit_code == 0
    assert "ollama_url" in result.output


def test_config_set_and_get(runner, cli_env):
    invoke(runner, "config", "set", "gen_model", "mistral:7b")
    result = invoke(runner, "config", "get", "gen_model")
    assert "mistral:7b" in result.output


# ── retry-failed / reset ────────────────────────────────────────────

def test_retry_failed_requeues_notes(runner, cli_env, tmp_path):
    f = tmp_path / "note.md"
    f.write_text("content")
    invoke(runner, "add", str(f))
    conn = db.connect(str(cli_env))
    note_id = db.list_notes(conn)[0].id
    db.update_note(conn, note_id, status="failed")
    result = invoke(runner, "retry-failed")
    assert "re-enqueued 1 note(s)" in result.output
    assert db.get_note(conn, note_id).status == "pending"


def test_reset_full_deletes_everything(runner, cli_env, tmp_path):
    f = tmp_path / "note.md"
    f.write_text("content")
    invoke(runner, "add", str(f))
    invoke(runner, "theme", "add", "Base", "--base")
    result = invoke(runner, "reset", "--full", input="y\n")
    assert result.exit_code == 0
    assert "full reset complete" in result.output
    conn = db.connect(str(cli_env))
    assert db.list_notes(conn) == []


def test_reset_soft_requeues_scanned_notes(runner, cli_env, tmp_path):
    f = tmp_path / "note.md"
    f.write_text("content")
    invoke(runner, "add", str(f))
    conn = db.connect(str(cli_env))
    note_id = db.list_notes(conn)[0].id
    db.update_note(conn, note_id, status="scanned")
    result = invoke(runner, "reset")
    assert "soft reset" in result.output
    assert db.get_note(conn, note_id).status == "processed"
