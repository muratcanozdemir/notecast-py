"""Render the theme DAG as Graphviz SVG."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import graphviz

from notecast import db


def render_dag(conn: sqlite3.Connection, output: str = "notecast-graph",
               fmt: str = "svg") -> Path:
    """Build DOT graph from themes and render to file. Returns output path."""
    themes = db.list_themes(conn)
    edges = db.all_edges(conn)
    theme_map = {t.id: t for t in themes}

    dot = graphviz.Digraph(
        "notecast",
        format=fmt,
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "transparent",
            "fontname": "Helvetica",
            "pad": "0.5",
            "nodesep": "0.6",
            "ranksep": "0.8",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "style": "filled",
            "shape": "box",
            "penwidth": "1.2",
        },
        edge_attr={
            "color": "#666666",
            "arrowsize": "0.7",
        },
    )

    # compute depths via BFS from roots
    children_map: dict[str, list[str]] = {t.id: [] for t in themes}
    parent_set: set[str] = set()
    for child_id, parent_id in edges:
        children_map.setdefault(parent_id, []).append(child_id)
        parent_set.add(child_id)

    roots = [t.id for t in themes if t.id not in parent_set]
    depths: dict[str, int] = {}
    queue = [(r, 0) for r in roots]
    while queue:
        tid, d = queue.pop(0)
        if tid in depths:
            continue
        depths[tid] = d
        for cid in children_map.get(tid, []):
            queue.append((cid, d + 1))

    # color palette by depth
    palette = ["#4A90D9", "#6AB04C", "#F0932B", "#EB4D4B", "#9B59B6", "#1ABC9C"]

    for t in themes:
        count = db.theme_note_count(conn, t.id)
        depth = depths.get(t.id, 0)
        color = palette[depth % len(palette)]

        label = f"{t.name}\\n({count})"
        attrs: dict[str, str] = {
            "fillcolor": color,
            "fontcolor": "white",
        }
        if t.is_base:
            attrs["penwidth"] = "2.5"
            attrs["pencolor"] = "#333333"
        if count == 0:
            attrs["fillcolor"] = "#CCCCCC"
            attrs["fontcolor"] = "#666666"
            attrs["style"] = "filled,dashed"

        dot.node(t.id, label, **attrs)

    for child_id, parent_id in edges:
        if child_id in theme_map and parent_id in theme_map:
            dot.edge(parent_id, child_id)

    out_path = dot.render(output, cleanup=True)
    return Path(out_path)
