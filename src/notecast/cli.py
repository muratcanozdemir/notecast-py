"""CLI entry point — click-based, no server."""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path

import click

from notecast import db
from notecast.llm import Ollama

_HEADING_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)

def _conn() -> db.sqlite3.Connection:
    conn = db.connect()
    db.init_db(conn)
    return conn


def _llm(conn: db.sqlite3.Connection) -> Ollama:
    return Ollama(
        base_url=db.get_config(conn, "ollama_url"),
        gen_model=db.get_config(conn, "gen_model"),
        embed_model=db.get_config(conn, "embed_model"),
    )



def _extract_title(filepath: Path, base_dir: Path | None = None) -> str:
    """Extract a title from a markdown file.

    Priority:
      1. First '# Heading' in the file
      2. Parent directory + stem (e.g. 'observability/alerting')
         if the filename is generic (index, readme, overview, etc.)
      3. Bare stem
    """
    generic_names = {"index", "readme", "overview", "summary", "introduction", "intro"}

    try:
        text = filepath.read_text(errors="replace")
    except OSError:
        return filepath.stem

    match = _HEADING_RE.search(text[:2048])
    if match:
        return match.group(1).strip()

    stem = filepath.stem.lower()
    if stem in generic_names and filepath.parent != filepath.parent.parent:
        # use parent dir(s) relative to base_dir for context
        if base_dir:
            rel = filepath.parent.relative_to(base_dir)
            parts = [p for p in rel.parts if p not in (".", "..")]
            if parts:
                return "/".join(parts)
        return filepath.parent.name

    return filepath.stem


# ── root ────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="notecast")
def cli() -> None:
    """NoteCast — local note engine with LLM-driven knowledge graph."""


# ── add ─────────────────────────────────────────────────────────────

@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--title", "-t", default=None, help="Override title (default: auto-detect from heading).")
@click.option("--process", "-p", is_flag=True, help="Immediately process after adding.")
def add(path: str, title: str | None, process: bool) -> None:
    """Add a note from a file."""
    p = Path(path)
    content = p.read_text(errors="replace")
    t = title or _extract_title(p)
    conn = _conn()
    nid = db.add_note(conn, t, content, source_path=str(p.resolve()))
    click.echo(f"added {nid}: {t}")

    if process:
        from notecast.pipeline import process_notes
        llm = _llm(conn)
        if not llm.ping():
            click.echo("error: cannot reach Ollama", err=True)
            raise SystemExit(1)
        n = process_notes(conn, llm, verbose=True)
        click.echo(f"processed {n} note(s)")


@cli.command("add-batch")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--ext", default=".md,.txt", help="Comma-separated extensions to include.")
@click.option("--exclude", "-x", multiple=True,
              help="Glob patterns to exclude (repeatable). E.g. -x SUMMARY.md -x '**/README.md'")
@click.option("--dry-run", is_flag=True, help="Show what would be added without adding.")
def add_batch(directory: str, ext: str, exclude: tuple[str, ...], dry_run: bool) -> None:
    """Add all matching files from a directory."""
    exts = {e if e.startswith(".") else f".{e}" for e in (part.strip() for part in ext.split(","))}
    d = Path(directory)
    conn = _conn()
    count = 0
    skipped = 0
    for f in sorted(d.rglob("*")):
        if not (f.suffix in exts and f.is_file()):
            continue
        rel = str(f.relative_to(d))
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f.name, pat) for pat in exclude):
            skipped += 1
            continue
        title = _extract_title(f, base_dir=d)
        if dry_run:
            click.echo(f"  [dry] {title}  ← {rel}")
        else:
            content = f.read_text(errors="replace")
            nid = db.add_note(conn, title, content, source_path=str(f.resolve()))
            click.echo(f"  {nid}: {title}")
        count += 1
    verb = "would add" if dry_run else "added"
    click.echo(f"{verb} {count} note(s), skipped {skipped}")


# ── scan ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--stage", type=click.Choice(
    ["auto", "process", "classify", "organize", "consolidate"]),
    default="auto", help="Run a specific stage or auto-detect.")
@click.option("-v", "--verbose", is_flag=True)
def scan(stage: str, verbose: bool) -> None:
    """Run pipeline stages."""
    from notecast.pipeline import (
        auto_scan,
        classify_notes,
        consolidate_themes,
        organize_themes,
        process_notes,
    )

    conn = _conn()
    llm = _llm(conn)

    if stage != "consolidate" and not llm.ping():
        click.echo("error: cannot reach Ollama", err=True)
        raise SystemExit(1)

    if stage == "auto":
        report = auto_scan(conn, llm, verbose=verbose)
        click.echo(json.dumps(report, indent=2))
    elif stage == "process":
        n = process_notes(conn, llm, verbose=verbose)
        click.echo(f"processed {n}")
    elif stage == "classify":
        n = classify_notes(conn, llm, verbose=verbose)
        click.echo(f"classified {n}")
    elif stage == "organize":
        n = organize_themes(conn, llm, verbose=verbose)
        click.echo(f"splits created: {n}")
    elif stage == "consolidate":
        r = consolidate_themes(conn, verbose=verbose)
        click.echo(json.dumps(r))


# ── status ──────────────────────────────────────────────────────────

@cli.command()
def status() -> None:
    """Show pipeline status and counts."""
    conn = _conn()
    counts = db.status_counts(conn)
    total = sum(counts.values())
    click.echo(f"notes: {total}")
    for s in ("pending", "processed", "scanned", "organized", "failed"):
        click.echo(f"  {s:12s} {counts.get(s, 0)}")
    click.echo()
    themes = db.list_themes(conn)
    click.echo(f"themes: {len(themes)}")
    base = [t for t in themes if t.is_base]
    click.echo(f"  base: {len(base)}")
    click.echo(f"  derived: {len(themes) - len(base)}")


# ── retry ───────────────────────────────────────────────────────────

@cli.command("retry-failed")
def retry_failed() -> None:
    """Re-enqueue failed notes as pending."""
    conn = _conn()
    failed = db.list_notes(conn, status="failed")
    for n in failed:
        db.update_note(conn, n.id, status="pending")
    click.echo(f"re-enqueued {len(failed)} note(s)")


# ── delete ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
def delete(query: str) -> None:
    """Delete a note by title search."""
    conn = _conn()
    matches = db.search_notes(conn, query)
    if not matches:
        click.echo("no match")
        return
    if len(matches) > 1:
        for m in matches:
            click.echo(f"  {m.id}: {m.title}")
        click.echo("multiple matches — refine your query")
        return
    note = matches[0]
    if click.confirm(f"delete '{note.title}'?"):
        db.delete_note(conn, note.id)
        click.echo("deleted")


# ── theme ───────────────────────────────────────────────────────────

@cli.group()
def theme() -> None:
    """Manage themes."""


@theme.command("list")
def theme_list() -> None:
    """List all themes."""
    conn = _conn()
    themes = db.list_themes(conn)
    if not themes:
        click.echo("no themes — add base themes with: notecast theme add <name> --base")
        return
    for t in themes:
        count = db.theme_note_count(conn, t.id)
        base = " [base]" if t.is_base else ""
        parents = db.theme_parents(conn, t.id)
        parent_info = ""
        if parents:
            parent_names = []
            for pid in parents:
                pt = db.get_theme(conn, pid)
                if pt:
                    parent_names.append(pt.name)
            parent_info = f" ← {', '.join(parent_names)}"
        click.echo(f"  {t.id}  {t.name} ({count}){base}{parent_info}")


@theme.command("add")
@click.argument("name")
@click.option("--base", is_flag=True, help="Mark as base (anchor) theme.")
@click.option("--desc", default=None, help="Description.")
@click.option("--parent", default=None, help="Parent theme name.")
def theme_add(name: str, base: bool, desc: str | None, parent: str | None) -> None:
    """Create a theme."""
    conn = _conn()
    parent_id = None
    if parent:
        pt = db.get_theme_by_name(conn, parent)
        if not pt:
            click.echo(f"parent theme '{parent}' not found", err=True)
            raise SystemExit(1)
        parent_id = pt.id
    if db.get_theme_by_name(conn, name):
        click.echo(f"theme '{name}' already exists", err=True)
        raise SystemExit(1)
    tid = db.add_theme(conn, name, description=desc, is_base=base, parent_id=parent_id)
    click.echo(f"created {tid}: {name}")


@theme.command("remove")
@click.argument("name")
def theme_remove(name: str) -> None:
    """Remove a theme by name."""
    conn = _conn()
    t = db.get_theme_by_name(conn, name)
    if not t:
        click.echo("not found")
        return
    if t.is_base and not click.confirm(f"'{t.name}' is a base theme — really delete?"):
        return
    db.delete_theme(conn, t.id)
    click.echo(f"removed {t.name}")


# ── graph ───────────────────────────────────────────────────────────

@cli.command()
@click.option("-o", "--output", default="notecast-graph", help="Output filename (without extension).")
@click.option("--format", "fmt", default="svg", type=click.Choice(["svg", "png", "pdf"]))
def graph(output: str, fmt: str) -> None:
    """Render the theme DAG to SVG/PNG/PDF."""
    import graphviz

    from notecast.graph import render_dag
    conn = _conn()
    themes = db.list_themes(conn)
    if not themes:
        click.echo("no themes to graph")
        return
    try:
        out = render_dag(conn, output=output, fmt=fmt)
    except graphviz.backend.execute.ExecutableNotFound:
        click.echo(
            "error: graphviz 'dot' executable not found — "
            "install it (apt install graphviz / brew install graphviz)",
            err=True,
        )
        raise SystemExit(1)
    click.echo(f"written: {out}")


# ── config ──────────────────────────────────────────────────────────

@cli.group()
def config() -> None:
    """View or set configuration."""


@config.command("get")
@click.argument("key", required=False)
def config_get(key: str | None) -> None:
    """Show config (all or specific key)."""
    conn = _conn()
    if key:
        click.echo(f"{key} = {db.get_config(conn, key)}")
    else:
        for k, v in sorted(db.all_config(conn).items()):
            click.echo(f"  {k:20s} {v}")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a config value."""
    conn = _conn()
    db.set_config(conn, key, value)
    click.echo(f"{key} = {value}")


# ── search ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--similar", "-s", is_flag=True, help="Also show semantically similar notes.")
@click.option("-n", default=5, help="Max results.")
def search(query: str, similar: bool, n: int) -> None:
    """Search notes by text (FTS5)."""
    conn = _conn()
    results = db.search_notes(conn, query)[:n]
    if not results:
        click.echo("no results")
        return
    for note in results:
        themes = db.note_theme_ids(conn, note.id)
        theme_names = []
        for tid in themes:
            t = db.get_theme(conn, tid)
            if t:
                theme_names.append(t.name)
        tags = f" [{', '.join(theme_names)}]" if theme_names else ""
        click.echo(f"  {note.id}  {note.title} ({note.status}){tags}")
        if note.summary:
            click.echo(f"           {note.summary[:120]}")

    if similar and results[0].vector:
        from notecast.pipeline import find_similar
        all_notes = db.list_notes(conn)
        others = [n for n in all_notes if n.id != results[0].id]
        sim = find_similar(results[0].vector, others, k=n)
        if sim:
            click.echo(f"\nsimilar to '{results[0].title}':")
            for s in sim:
                click.echo(f"  {s.id}  {s.title}")


# ── reset ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--full", is_flag=True, help="Delete all data (notes + themes).")
def reset(full: bool) -> None:
    """Reset pipeline state (or everything with --full)."""
    conn = _conn()
    if full:
        if not click.confirm("delete ALL notes and non-base themes?"):
            return
        conn.execute("DELETE FROM note_themes")
        conn.execute("DELETE FROM notes")
        conn.execute("DELETE FROM themes WHERE is_base=0")
        conn.execute("DELETE FROM theme_edges")
        conn.commit()
        click.echo("full reset complete")
    else:
        # soft reset: re-queue scanned → processed
        conn.execute("UPDATE notes SET status='processed' WHERE status='scanned'")
        for name in ("classify_runs", "organize_runs", "consolidate_runs"):
            db.reset_counter(conn, name)
        conn.commit()
        click.echo("soft reset: scanned notes re-queued, counters cleared")
