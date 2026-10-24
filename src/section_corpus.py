"""Build the usable section corpus cache from a selected Markdown root.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| SOURCE | Read the latest Markdown version per paper from the chosen root. | [mandatory] | R1 |
| SECTION | Split documents into stable paper/section records and apply usability filters. | [mandatory] | R2 |
| CACHE | Save records and source statistics for the map pipeline. | [contract] | R3 |

R1 `--root` points to the top-level `papers/post_processed` directory; default uses `domain_corpora.ARXIV_DIR`.
R2 Each record retains its `(doc_id, section_idx)` key and whole section text.
R3 `--output` selects a cache path; the default remains `.tmp/arxiv_sections.pkl`.

Preconditions: Markdown files use recognized arXiv filenames and the target cache directory is writable.
Failure modes: malformed source files or filesystem errors propagate without replacing another cache path.
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
    import argparse
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Build the usable arXiv section cache.")
    parser.add_argument("--root", help="Markdown papers/post_processed directory")
    parser.add_argument("--output", default=CACHE, help="cache output path")
    args = parser.parse_args()
    out = build(path=args.output, root=args.root)
    st = out["stats"]
    print("papers %d | sections %d | empty %d | reference %d | short (< %.0f chars) %d | junk %d | KEPT %d  (%.0fs) -> %s"
          % (out["papers"], st["all"], st["empty"], st["reference"], st["floor"], st["short"], st["junk"], st["kept"], time.time() - t0, CACHE))
