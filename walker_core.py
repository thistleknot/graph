"""walker_core.py -- the walker's UI-free logic: clip, labels, partitions, assess.

Spec: .spec/specs/graph-explorer/design.md 6.21(c) (walker split, UI wiring only)
Task: playbook.md T36

Imports NOTHING from streamlit, plotly or networkx and opens NO connection at
import time: `python -c "import walker_core"` must exit 0 with the database
down. Every function takes conn/run/state as parameters -- there is no module
state here. Spring layout deliberately stays in walker_app (design 6.21(a)).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import evidence
import graph_tools as gt
import interpret


def clip(text: str, n: int) -> str:
    """Never cut inside a word (design 6.3). Clip at the last whitespace
    before n and mark the cut; short text is returned untouched."""
    text = text or ""
    if len(text) <= n:
        return text
    head = text[:n]
    cut = head.rsplit(None, 1)[0] if " " in head else head
    return cut + " …"


def load_labels(path: Path, run_id: str) -> dict:
    """Draft labels for THIS run, or {}. Absent file, unreadable file and
    malformed JSON are all the normal empty state.

    cid is RUN-LOCAL: a labels file drafted against another run is not stale,
    it is wrong -- it put 'early electrical science history' on the
    Moroccan elections. A foreign file returns {"__stale_run__": <its run_id>} so the
    caller can say so; it never returns that file's labels.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if str(data.get("run_id")) != str(run_id):
        return {"__stale_run__": data.get("run_id")}
    return {c["cid"]: c for c in data.get("communities", []) if c.get("label")}


def src_of_map(conn, run, ords) -> dict:
    """{ord: source-or-None} for `ords`. One line, two call sites in the UI
    (the 3D tab and the Map partitions table) that each had their own copy."""
    return {o: gt.source_of(gt.node(conn, run, o)) for o in ords}


def partition_rows(ds: dict, src_of: dict) -> list[dict]:
    """One row per dendrite CHUNK chain: size, source mix, and the chain
    members' own BM25-salient vocabulary ranked by how many members carry it
    (top 12; 'term(n)' when n > 1). Pure: `ds` is dendrite_state's output,
    `src_of` comes from src_of_map. No DB, no st."""
    sal = ds["sal"]
    rows = []
    for ci, chain in enumerate(ds["chunks"]["chains"]):
        tc = Counter()
        mix = Counter()
        for o in chain:
            for t in (sal.get(o, {}).get("top") or []):
                tc[t] += 1
            mix[src_of.get(o) or "?"] += 1
        rows.append({
            "chain": ci + 1,
            "chunks": len(chain),
            "sources": " ".join(f"{k}:{v}" for k, v in mix.most_common()),
            "salient terms (carried by N members)":
                ", ".join(f"{t}({n})" if n > 1 else t
                          for t, n in tc.most_common(12)),
        })
    return rows


def assess(conn, run, bundle, ev, ds, q, *, embed=None) -> tuple[dict, str | None]:
    """ONE model call (I13) over the walk, then the best-effort neo4j mirror.
    Returns (reason-result, mirror_error-or-None); the caller owns the spinner,
    the session_state write and how the error is shown.

    NEO4J_MIRROR=0 skips the mirror entirely (T17 behaviour, unchanged).
    """
    rr = interpret.reason(conn, run, bundle, ev.terms, ev.concept, embed=embed,
                          judge=True, pw=ev.pathways, digest=ev.digest)
    err = None
    if os.environ.get("NEO4J_MIRROR", "1") != "0":
        err = evidence.mirror_walk(bundle, ev, ds, prompt=q)
    return rr, err
