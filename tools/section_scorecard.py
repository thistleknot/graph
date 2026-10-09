"""section_scorecard.py -- label-free numbers that compare one section map with another, so a change is judged by measurement and not by eye.

Spec: approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md, task T158 (playbook.md Layer 39). Operator 2026-10-06: "plan a proper fix".

For each map (a community label per section) it reports:
    communities, size>=10 communities, singletons          how fragmented the map is
    oversize communities and the share of sections in them  how much of the map is grab-bag; bound = sqrt(m) (m = fused edges), the order of the
                                                            modularity resolution limit (Fortunato and Barthelemy 2007, PNAS); the constant is
                                                            uncited, so the bound is a yardstick shared by every map, not a derived fact
    top-paper share (median over size>=10) and the count at >= 0.5   how paper-bound the communities are (the paper-mixed map: 131 of 258)
    top-heading share (median over size>=10)                how heading-bound they are (measured 0.05 on the cross-paper map: the grab-bags are
                                                            genre, which headings do not show)
    seed-ARI                                                the consensus gate value stored with the map's state

The read rubric (`write_rubric`) is written ONCE and never overwritten, so it is frozen before any community is read.

Run:  python tools\\section_scorecard.py .tmp\\sections_map_state.npz .tmp\\sections_xp_map_state.npz ...
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FUSED_EDGES = 1_587_270                          # the cross-paper map's fused edge count; the yardstick for every map, so it is the same number in every row
BIG = 10                                         # a community this size or larger is a candidate topic; below it is noise or a leaf
RUBRIC = ".tmp/section_rubric.md"
_LEAD = re.compile(r"^\s*(?:[A-Za-z]\.|\d+(?:\.\d+)*\.?|[IVX]+\.)?\s*")


def heading_kind(title: str) -> str:
    """Guarantee: the heading with its numbering and punctuation removed, lower-cased ('3.2 Related Work' -> 'related work'); '' when nothing is left."""
    return re.sub(r"[^a-z ]", "", _LEAD.sub("", title).lower()).strip()


def codes(values) -> np.ndarray:
    """Guarantee: one integer per value, equal values sharing one (ids in order of first appearance). A dictionary, not np.unique: np.unique makes a fixed-width
    string array as wide as the LONGEST value, and one section's heading is 5,039 characters, which turned 80,642 headings into a 1.5 GB array."""
    seen: dict = {}
    return np.array([seen.setdefault(v, len(seen)) for v in values], dtype=np.int64)


def _top_share(values: np.ndarray, members: np.ndarray) -> float:
    v = values[members]
    return float(np.bincount(v).max() / len(v)) if len(v) else 0.0


def scorecard(lab: np.ndarray, doc_ids, titles, seed_ari: float | None = None, m_edges: int = FUSED_EDGES) -> dict:
    """Require: lab, doc_ids, titles of equal length (one entry per section). Guarantee: the numbers described in the module docstring, as a dict."""
    lab = np.asarray(lab)
    N = len(lab)
    assert len(doc_ids) == N and len(titles) == N, "one doc id and one title per section"
    paper = codes(doc_ids)
    kind_text = [heading_kind(t) for t in titles]
    kind = codes(kind_text)
    has_kind = np.array([k != "" for k in kind_text])
    bound = float(np.sqrt(m_edges))
    size = np.bincount(lab)
    big = np.where(size >= BIG)[0]
    tp, tk = [], []
    for c in big:
        members = np.where(lab == c)[0]
        tp.append(_top_share(paper, members))
        tk.append(_top_share(kind, members[has_kind[members]]))
    tp, tk = np.array(tp), np.array(tk)
    over = size > bound
    return {"sections": N, "communities": int(len(size)), "size_ge_10": int(len(big)), "singletons": int((size == 1).sum()),
            "oversize": int(over.sum()), "oversize_share": float(size[over].sum() / N), "bound": bound,
            "top_paper_median": float(np.median(tp)) if len(tp) else 0.0, "paper_ge_50": int((tp >= 0.5).sum()),
            "top_heading_median": float(np.median(tk)) if len(tk) else 0.0, "heading_ge_50": int((tk >= 0.5).sum()),
            "seed_ari": seed_ari, "largest": int(size.max())}


def write_rubric(path: str = RUBRIC) -> bool:
    """Guarantee: the read rubric exists at `path`; returns True when written now, False when one was already there (never overwritten: it is frozen
    before any community is read)."""
    if os.path.exists(path):
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Read rubric for section communities (frozen 2026-10-06, before any community was read for the fix)\n\n"
                 "Read the exemplar sections of a community and give ONE verdict:\n\n"
                 "- **topic**: a person could name its subject in a few words and the sections are about that subject (e.g. tool-calling benchmarks).\n"
                 "- **genre**: the sections share a document type or register, not a subject (prompt templates, generic introduction/conclusion prose).\n"
                 "- **mixed**: two or more unrelated subjects, or no nameable subject.\n\n"
                 "A community that is one paper's own outline is a **paper**, not a topic: count it as genre.\n"
                 "Read 10 per map, spread across sizes (the same ranks for every map). Quote one line per community. Do not re-grade after seeing the scorecard.\n")
    return True


def format_rows(rows: list[tuple[str, dict]]) -> str:
    cols = [("communities", "communities"), ("size_ge_10", "size>=10"), ("singletons", "singletons"), ("largest", "largest"), ("oversize", "oversize"),
            ("oversize_share", "in oversize"), ("top_paper_median", "paper med"), ("paper_ge_50", "paper>=.5"),
            ("top_heading_median", "heading med"), ("seed_ari", "seed-ARI")]
    head = "%-26s" % "map" + "".join("%12s" % h for _, h in cols)
    lines = [head, "-" * len(head)]
    for name, s in rows:
        cells = []
        for k, _ in cols:
            v = s[k]
            cells.append("%12s" % ("n/a" if v is None else ("%.3f" % v if isinstance(v, float) else v)))
        lines.append("%-26s" % name[:26] + "".join(cells))
    return "\n".join(lines)


def main(paths: list[str]) -> None:
    import section_corpus as sc
    recs = sc.load()
    doc_ids, titles = [r["doc_id"] for r in recs], [r["section_title"] for r in recs]
    rows = []
    for p in paths:
        z = np.load(p)
        ari = float(z["cons_ari"]) if "cons_ari" in z.files else None
        rows.append((os.path.basename(p).replace("_map_state.npz", ""), scorecard(z["lab"], doc_ids, titles, ari)))
    print(format_rows(rows))
    print("oversize bound = sqrt(%d) = %.0f sections (the same yardstick for every row)" % (FUSED_EDGES, np.sqrt(FUSED_EDGES)))
    print("rubric %s" % ("written now: " + RUBRIC if write_rubric() else "already frozen: " + RUBRIC))


if __name__ == "__main__":
    main(sys.argv[1:])
