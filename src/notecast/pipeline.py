"""Four-stage note pipeline: process → classify → organize → consolidate."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from typing import Any

from notecast import db
from notecast.llm import Ollama, OllamaError


# ── vector math (stdlib only) ──────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def find_similar(target: list[float], notes: list[db.Note],
                 k: int = 10, threshold: float = 0.3) -> list[db.Note]:
    scored = [
        (n, cosine_sim(target, n.vector))
        for n in notes if n.vector
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, s in scored[:k] if s >= threshold]


def cluster_notes(notes: list[db.Note],
                  threshold: float = 0.55) -> list[list[db.Note]]:
    """Simple greedy agglomerative clustering by vector similarity."""
    if not notes:
        return []
    remaining = list(notes)
    clusters: list[list[db.Note]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        still_remaining = []
        for n in remaining:
            if seed.vector and n.vector and cosine_sim(seed.vector, n.vector) >= threshold:
                cluster.append(n)
            else:
                still_remaining.append(n)
        remaining = still_remaining
        clusters.append(cluster)
    return clusters


# ── stage 1: process ───────────────────────────────────────────────

def _summarise_prompt(title: str, content: str, language: str, context: str) -> str:
    ctx = f"\nDomain context: {context}" if context else ""
    return (
        f"Analyse this note and return a JSON object with exactly two keys:\n"
        f'- "summary": a concise 2-3 sentence summary\n'
        f'- "keywords": a list of 3-8 keywords or key phrases\n'
        f"\nLanguage: {language}{ctx}\n"
        f"\n--- NOTE ---\nTitle: {title}\n\n{content}\n--- END ---\n"
        f"\nReturn ONLY valid JSON."
    )


def process_notes(conn: sqlite3.Connection, llm: Ollama, *,
                  verbose: bool = False) -> int:
    """Process all pending notes: summarise + keywords + embed."""
    pending = db.list_notes(conn, status="pending")
    if not pending:
        return 0

    language = db.get_config(conn, "language")
    context = db.get_config(conn, "context")
    processed = 0

    for note in pending:
        if verbose:
            print(f"  processing: {note.title}", file=sys.stderr)
        try:
            result = llm.generate_json(
                _summarise_prompt(note.title, note.content, language, context),
                system="You are a note analysis assistant. Return only valid JSON.",
            )
            summary = result.get("summary", "")
            keywords = result.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []

            embed_text = f"{note.title} {note.content} {' '.join(keywords)}"
            vector = llm.embed(embed_text)

            db.update_note(conn, note.id,
                           summary=summary, topics=keywords,
                           vector=vector, status="processed")
            processed += 1
        except (OllamaError, json.JSONDecodeError, KeyError) as e:
            if verbose:
                print(f"  FAILED {note.title}: {e}", file=sys.stderr)
            db.update_note(conn, note.id, status="failed")

    return processed


# ── classify ───────────────────────────────────────────────────────

def _classify_prompt(themes: list[db.Theme],
                     notes: list[db.Note]) -> str:
    theme_list = "\n".join(
        f"- {t.name}" + (f" ({t.description})" if t.description else "")
        for t in themes
    )
    note_list = "\n".join(
        f'- id="{n.id}" title="{n.title}" summary="{n.summary}"'
        for n in notes
    )
    return (
        f"Available themes:\n{theme_list}\n\n"
        f"Notes to classify:\n{note_list}\n\n"
        f"Assign each note to 1-3 themes. Return a JSON object mapping "
        f"note IDs to lists of theme names. Use ONLY the theme names listed above.\n"
        f"Return ONLY valid JSON."
    )


def classify_notes(conn: sqlite3.Connection, llm: Ollama, *,
                   batch_size: int = 20, verbose: bool = False) -> int:
    """Assign processed notes to existing themes → status=scanned."""
    notes = db.list_notes(conn, status="processed")
    themes = db.list_themes(conn)
    if not notes or not themes:
        return 0

    classified = 0
    for i in range(0, len(notes), batch_size):
        batch = notes[i:i + batch_size]
        if verbose:
            print(f"  classifying batch of {len(batch)}", file=sys.stderr)

        try:
            result = llm.generate_json(
                _classify_prompt(themes, batch),
                system="You are a note classification assistant. Return only valid JSON.",
            )
            theme_lookup = {t.name.lower(): t for t in themes}
            for note in batch:
                assigned_names = result.get(note.id, [])
                if not isinstance(assigned_names, list):
                    assigned_names = [assigned_names]
                assigned_any = False
                for name in assigned_names:
                    theme = theme_lookup.get(str(name).lower())
                    if theme:
                        db.assign_note_theme(conn, note.id, theme.id)
                        assigned_any = True
                if assigned_any:
                    db.update_note(conn, note.id, status="scanned")
                    classified += 1
                elif verbose:
                    print(f"  no valid theme for: {note.title}", file=sys.stderr)
        except (OllamaError, json.JSONDecodeError) as e:
            if verbose:
                print(f"  classify batch failed: {e}", file=sys.stderr)

    if classified:
        db.increment_counter(conn, "classify_runs")
    return classified


# ── organize ───────────────────────────────────────────────────────

def _organize_prompt(theme: db.Theme, notes: list[db.Note]) -> str:
    note_list = "\n".join(
        f'- "{n.title}": {n.summary or n.content[:200]}'
        for n in notes
    )
    return (
        f'Theme "{theme.name}" has {len(notes)} notes:\n{note_list}\n\n'
        f"These notes may cover distinct sub-areas. Propose 2-4 subtopic names "
        f"that would meaningfully subdivide this theme. For each subtopic, list "
        f"the note titles that belong to it.\n\n"
        f'Return JSON: {{"subtopics": [{{"name": "...", "titles": ["..."]}}]}}\n'
        f"Return ONLY valid JSON."
    )


def organize_themes(conn: sqlite3.Connection, llm: Ollama, *,
                    verbose: bool = False) -> int:
    """Split overloaded themes into subtopics → notes status=organized."""
    threshold = int(db.get_config(conn, "split_threshold"))
    themes = db.list_themes(conn)
    splits = 0

    for theme in themes:
        notes = db.get_theme_notes(conn, theme.id)
        scanned = [n for n in notes if n.status == "scanned"]
        if len(notes) < threshold or not scanned:
            continue

        if verbose:
            print(f"  organizing: {theme.name} ({len(notes)} notes)", file=sys.stderr)

        try:
            result = llm.generate_json(
                _organize_prompt(theme, notes),
                system="You are a knowledge organization assistant. Return only valid JSON.",
            )
            subtopics = result.get("subtopics", [])
            if not isinstance(subtopics, list) or len(subtopics) < 2:
                continue

            title_to_note = {n.title.lower(): n for n in notes}
            for st in subtopics:
                st_name = st.get("name", "").strip()
                if not st_name:
                    continue
                existing = db.get_theme_by_name(conn, st_name)
                if existing:
                    st_id = existing.id
                else:
                    st_id = db.add_theme(conn, st_name, parent_id=theme.id)
                    splits += 1
                for title in st.get("titles", []):
                    note = title_to_note.get(title.lower())
                    if note:
                        db.assign_note_theme(conn, note.id, st_id)
                        db.update_note(conn, note.id, status="organized")
        except (OllamaError, json.JSONDecodeError) as e:
            if verbose:
                print(f"  organize failed for {theme.name}: {e}", file=sys.stderr)

    if splits:
        db.increment_counter(conn, "organize_runs")
    return splits


# ── consolidate ────────────────────────────────────────────────────

def consolidate_themes(conn: sqlite3.Connection, *,
                       cooccurrence_threshold: float = 0.5,
                       verbose: bool = False) -> dict[str, int]:
    """Structural pass — no LLM needed.

    - Detect co-occurrence: if ≥50% of theme A's notes also appear in
      theme B, add a parent edge A→B.
    - Prune empty non-base themes.
    """
    themes = db.list_themes(conn)
    edges_added = 0
    pruned = 0

    # co-occurrence detection
    theme_notes: dict[str, set[str]] = {}
    for t in themes:
        rows = conn.execute(
            "SELECT note_id FROM note_themes WHERE theme_id=?", (t.id,)
        ).fetchall()
        theme_notes[t.id] = {r["note_id"] for r in rows}

    existing_edges = set(db.all_edges(conn))

    for t in themes:
        my_notes = theme_notes.get(t.id, set())
        if not my_notes:
            continue
        for other in themes:
            if other.id == t.id:
                continue
            other_notes = theme_notes.get(other.id, set())
            if not other_notes:
                continue
            overlap = len(my_notes & other_notes) / len(my_notes)
            if overlap >= cooccurrence_threshold:
                edge = (t.id, other.id)
                if edge not in existing_edges:
                    # verify no cycle
                    if not _would_cycle(conn, child=t.id, parent=other.id):
                        conn.execute(
                            "INSERT OR IGNORE INTO theme_edges(child_id, parent_id) "
                            "VALUES (?,?)", edge,
                        )
                        existing_edges.add(edge)
                        edges_added += 1
                        if verbose:
                            print(f"  edge: {t.name} → {other.name}", file=sys.stderr)

    # prune empty non-base themes
    for t in themes:
        if t.is_base:
            continue
        count = db.theme_note_count(conn, t.id)
        children = db.theme_children(conn, t.id)
        if count == 0 and not children:
            db.delete_theme(conn, t.id)
            pruned += 1
            if verbose:
                print(f"  pruned: {t.name}", file=sys.stderr)

    conn.commit()
    if edges_added or pruned:
        db.increment_counter(conn, "consolidate_runs")

    return {"edges_added": edges_added, "pruned": pruned}


def _would_cycle(conn: sqlite3.Connection, child: str, parent: str) -> bool:
    """BFS from parent upward — if we reach child, adding this edge cycles."""
    visited: set[str] = set()
    queue = [parent]
    while queue:
        current = queue.pop(0)
        if current == child:
            return True
        if current in visited:
            continue
        visited.add(current)
        for pid in db.theme_parents(conn, current):
            queue.append(pid)
    return False


# ── auto-scan ──────────────────────────────────────────────────────

def auto_scan(conn: sqlite3.Connection, llm: Ollama, *,
              verbose: bool = False) -> dict[str, Any]:
    """Run whichever pipeline stages are ready, in order."""
    report: dict[str, Any] = {}

    # always process pending first
    p = process_notes(conn, llm, verbose=verbose)
    report["processed"] = p

    # classify if enough processed notes
    classify_every = int(db.get_config(conn, "classify_every"))
    counts = db.status_counts(conn)
    if counts.get("processed", 0) >= classify_every:
        c = classify_notes(conn, llm, verbose=verbose)
        report["classified"] = c
    else:
        report["classified"] = 0

    # organize if enough classify runs
    organize_after = int(db.get_config(conn, "organize_after"))
    if db.get_counter(conn, "classify_runs") >= organize_after:
        o = organize_themes(conn, llm, verbose=verbose)
        report["splits"] = o
        db.reset_counter(conn, "classify_runs")
    else:
        report["splits"] = 0

    # consolidate if enough organize runs
    consolidate_after = int(db.get_config(conn, "consolidate_after"))
    if db.get_counter(conn, "organize_runs") >= consolidate_after:
        r = consolidate_themes(conn, verbose=verbose)
        report["consolidate"] = r
        db.reset_counter(conn, "organize_runs")
    else:
        report["consolidate"] = {"edges_added": 0, "pruned": 0}

    return report
