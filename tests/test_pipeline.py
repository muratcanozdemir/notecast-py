from __future__ import annotations

from notecast import db, pipeline
from notecast.llm import OllamaError

# ── vector math ─────────────────────────────────────────────────────

def test_cosine_sim_identical_vectors():
    assert pipeline.cosine_sim([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_sim_orthogonal_vectors():
    assert pipeline.cosine_sim([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_sim_handles_empty_or_mismatched():
    assert pipeline.cosine_sim([], [1.0]) == 0.0
    assert pipeline.cosine_sim([1.0, 2.0], [1.0]) == 0.0


def test_cosine_sim_handles_zero_vector():
    assert pipeline.cosine_sim([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_find_similar_filters_by_threshold_and_k():
    target = [1.0, 0.0]
    close = db.Note(id="a", title="a", content="", vector=[0.9, 0.1])
    far = db.Note(id="b", title="b", content="", vector=[0.0, 1.0])
    no_vec = db.Note(id="c", title="c", content="", vector=[])
    result = pipeline.find_similar(target, [close, far, no_vec], k=5, threshold=0.5)
    assert result == [close]


def test_find_similar_respects_k():
    target = [1.0, 0.0]
    notes = [db.Note(id=str(i), title=str(i), content="", vector=[1.0, 0.0]) for i in range(5)]
    result = pipeline.find_similar(target, notes, k=2, threshold=0.0)
    assert len(result) == 2


def test_cluster_notes_empty():
    assert pipeline.cluster_notes([]) == []


def test_cluster_notes_groups_similar_vectors():
    a = db.Note(id="a", title="a", content="", vector=[1.0, 0.0])
    b = db.Note(id="b", title="b", content="", vector=[0.99, 0.01])
    c = db.Note(id="c", title="c", content="", vector=[0.0, 1.0])
    clusters = pipeline.cluster_notes([a, b, c], threshold=0.9)
    ids = [sorted(n.id for n in cluster) for cluster in clusters]
    assert ["a", "b"] in ids
    assert ["c"] in ids


def test_cluster_notes_without_vectors_each_own_cluster():
    a = db.Note(id="a", title="a", content="")
    b = db.Note(id="b", title="b", content="")
    clusters = pipeline.cluster_notes([a, b])
    assert len(clusters) == 2


# ── process ─────────────────────────────────────────────────────────

def test_process_notes_success_updates_note(conn, fake_ollama):
    nid = db.add_note(conn, "T", "content")
    llm = fake_ollama(json_responses=[{"summary": "s", "keywords": ["k1", "k2"]}])
    n = pipeline.process_notes(conn, llm, verbose=False)
    assert n == 1
    note = db.get_note(conn, nid)
    assert note.status == "processed"
    assert note.summary == "s"
    assert note.topics == ["k1", "k2"]
    assert len(note.vector) == llm.embed_dim


def test_process_notes_marks_failed_on_llm_error(conn, fake_ollama):
    nid = db.add_note(conn, "T", "content")
    llm = fake_ollama(json_responses=[OllamaError("boom")])
    n = pipeline.process_notes(conn, llm, verbose=False)
    assert n == 0
    assert db.get_note(conn, nid).status == "failed"


def test_process_notes_no_pending_returns_zero(conn, fake_ollama):
    llm = fake_ollama()
    assert pipeline.process_notes(conn, llm) == 0


def test_process_notes_non_list_keywords_becomes_empty(conn, fake_ollama):
    db.add_note(conn, "T", "content")
    llm = fake_ollama(json_responses=[{"summary": "s", "keywords": "not-a-list"}])
    pipeline.process_notes(conn, llm)
    note = db.list_notes(conn)[0]
    assert note.topics == []


# ── classify ────────────────────────────────────────────────────────

def test_classify_notes_assigns_matching_theme(conn, fake_ollama):
    tid = db.add_theme(conn, "Tech", is_base=True)
    nid = db.add_note(conn, "T", "c")
    db.update_note(conn, nid, status="processed", summary="s")
    llm = fake_ollama(json_responses=[{nid: ["Tech"]}])
    n = pipeline.classify_notes(conn, llm)
    assert n == 1
    note = db.get_note(conn, nid)
    assert note.status == "scanned"
    assert db.note_theme_ids(conn, nid) == [tid]


def test_classify_notes_no_notes_or_themes_returns_zero(conn, fake_ollama):
    llm = fake_ollama()
    assert pipeline.classify_notes(conn, llm) == 0


def test_classify_notes_unmatched_theme_name_leaves_note_processed(conn, fake_ollama):
    db.add_theme(conn, "Tech", is_base=True)
    nid = db.add_note(conn, "T", "c")
    db.update_note(conn, nid, status="processed")
    llm = fake_ollama(json_responses=[{nid: ["NoSuchTheme"]}])
    n = pipeline.classify_notes(conn, llm)
    assert n == 0
    assert db.get_note(conn, nid).status == "processed"


def test_classify_notes_batch_failure_is_isolated(conn, fake_ollama):
    db.add_theme(conn, "Tech", is_base=True)
    nid = db.add_note(conn, "T", "c")
    db.update_note(conn, nid, status="processed")
    llm = fake_ollama(json_responses=[OllamaError("boom")])
    n = pipeline.classify_notes(conn, llm)
    assert n == 0
    assert db.get_note(conn, nid).status == "processed"


# ── organize ────────────────────────────────────────────────────────

def test_organize_themes_skips_below_threshold(conn, fake_ollama):
    tid = db.add_theme(conn, "Tech", is_base=True)
    nid = db.add_note(conn, "T", "c")
    db.update_note(conn, nid, status="scanned")
    db.assign_note_theme(conn, nid, tid)
    db.set_config(conn, "split_threshold", "5")
    llm = fake_ollama()
    n = pipeline.organize_themes(conn, llm)
    assert n == 0


def test_organize_themes_splits_when_over_threshold(conn, fake_ollama):
    tid = db.add_theme(conn, "Tech", is_base=True)
    db.set_config(conn, "split_threshold", "1")
    titles = []
    for i in range(2):
        nid = db.add_note(conn, f"Note{i}", "content")
        db.update_note(conn, nid, status="scanned")
        db.assign_note_theme(conn, nid, tid)
        titles.append(f"Note{i}")

    llm = fake_ollama(json_responses=[{
        "subtopics": [
            {"name": "SubA", "titles": [titles[0]]},
            {"name": "SubB", "titles": [titles[1]]},
        ]
    }])
    n = pipeline.organize_themes(conn, llm)
    assert n == 2
    names = {t.name for t in db.list_themes(conn)}
    assert {"SubA", "SubB"} <= names


# ── consolidate ─────────────────────────────────────────────────────

def test_would_cycle_detects_direct_and_transitive(conn):
    a = db.add_theme(conn, "A")
    b = db.add_theme(conn, "B", parent_id=a)  # b's parent is a
    c = db.add_theme(conn, "C", parent_id=b)  # c's parent is b (a is c's grandparent)
    d = db.add_theme(conn, "D")  # unrelated root
    # a is already an ancestor of b and c, so making a a *child* of either cycles
    assert pipeline._would_cycle(conn, child=a, parent=b) is True
    assert pipeline._would_cycle(conn, child=a, parent=c) is True
    # d is unrelated to a's ancestry, so this would not cycle
    assert pipeline._would_cycle(conn, child=a, parent=d) is False


def test_consolidate_adds_cooccurrence_edge(conn):
    a = db.add_theme(conn, "A", is_base=True)
    b = db.add_theme(conn, "B", is_base=True)
    nid = db.add_note(conn, "T", "c")
    db.assign_note_theme(conn, nid, a)
    db.assign_note_theme(conn, nid, b)
    result = pipeline.consolidate_themes(conn)
    assert result["edges_added"] == 1
    assert (a, b) in db.all_edges(conn)


def test_consolidate_prunes_empty_non_base_theme(conn):
    db.add_theme(conn, "Empty")
    result = pipeline.consolidate_themes(conn)
    assert result["pruned"] == 1
    assert db.list_themes(conn) == []


def test_consolidate_never_prunes_base_theme(conn):
    db.add_theme(conn, "Base", is_base=True)
    result = pipeline.consolidate_themes(conn)
    assert result["pruned"] == 0
    assert len(db.list_themes(conn)) == 1


def test_consolidate_does_not_create_cycle(conn):
    a = db.add_theme(conn, "A", is_base=True)
    b = db.add_theme(conn, "B", is_base=True, parent_id=a)
    # give B all of A's notes too, so overlap could suggest A -> B,
    # but B -> A already exists and A -> B would cycle.
    nid = db.add_note(conn, "T", "c")
    db.assign_note_theme(conn, nid, a)
    db.assign_note_theme(conn, nid, b)
    pipeline.consolidate_themes(conn)
    assert (a, b) not in db.all_edges(conn)


# ── auto_scan ───────────────────────────────────────────────────────

def test_auto_scan_processes_pending_notes(conn, fake_ollama):
    db.add_note(conn, "T", "c")
    llm = fake_ollama(json_responses=[{"summary": "s", "keywords": []}])
    report = pipeline.auto_scan(conn, llm)
    assert report["processed"] == 1
    assert report["classified"] == 0
