"""section_corpus.py -- the nodes of the section map: every header block of every paper, filtered to the usable ones, cached.

Spec: operator 2026-10-05 ("derive the graph based on sections", approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md, step A1)
and domain_corpora.py guard C13. Task: playbook.md T149.

    python -u tools\\section_corpus.py          writes .tmp/arxiv_sections.pkl and prints what was kept and why the rest was not

The cache is {"records": [usable section records], "stats": usable_sections' stats, "papers": n}. A record is {doc_id, section_idx,
chunk_idx 0, section_title, is_reference, is_junk, body_chars, text = header + body}; (doc_id, section_idx) is the key the
section_lengths.csv of 2026-10-05 uses.
"""
from __future__ import annotations

import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain_corpora import load_arxiv_papers, section_records, usable_sections

CACHE = os.path.join(ROOT, ".tmp", "arxiv_sections.pkl")


def build(path: str = CACHE, root: str | None = None) -> dict:
    """Guarantee: the cache at `path` holds every usable section of the latest version of every paper; returns it."""
    papers = load_arxiv_papers(root=root)
    recs = section_records([p["doc_id"] for p in papers], [p["text"] for p in papers])
    kept, st = usable_sections(recs)
    out = {"records": kept, "stats": st, "papers": len(papers)}
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def load(path: str = CACHE) -> list[dict]:
    """Guarantee: the usable section records, in order."""
    return pickle.load(open(path, "rb"))["records"]


def index_text(rec: dict) -> str:
    """Spec: approved plan T166 (arm C, operator 2026-10-06: a heading is a stop-phrase: "keep the text, just exclude it from embedding").
    Guarantee: the text that is POOLED and TOKENISED for a section: its body, without the leading `#` heading line the record's `text` starts with.
    A text whose first line is not a heading is returned whole. The record itself is not changed: `text` keeps the heading for display and the markdown.
    Measured 2026-10-06: a heading is 1.5% of all words but more than 5% of the words of 23.8% of sections (the short ones)."""
    first, sep, rest = rec["text"].partition("\n")
    return rest.lstrip("\n") if sep and first.lstrip().startswith("#") else rec["text"]


if __name__ == "__main__":
    t0 = time.time()
    out = build()
    st = out["stats"]
    print("papers %d | sections %d | empty %d | reference %d | short (< %.0f chars) %d | junk %d | KEPT %d  (%.0fs) -> %s"
          % (out["papers"], st["all"], st["empty"], st["reference"], st["floor"], st["short"], st["junk"], st["kept"], time.time() - t0, CACHE))
