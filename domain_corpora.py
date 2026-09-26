"""domain_corpora.py -- loaders for the two term-analysis corpora.

NO GOVERNING SPEC for this module's own requirements. Basis: operator
instruction 2026-09-26 ("incorporating neoplatonic texts as well as arxiv
papers ... the intent behind the two datasets is domain specific term
analysis"), planned in
C:\\Users\\user\\.claude\\plans\\i-ve-been-thinking-about-quiet-cray.md.
Task: playbook.md T104. Amends .spec/specs/graph-term-selection/requirements.md
(see that file's scope boundary). Promote these guards to REQs before anything
depends on them.

Returns the `(doc_ids, docs, sources)` triple in `ingest_mixed.py`'s established
shape, source-prefixed per chunkgraph R20. It does NOT fit a ChunkGraph and does
NOT persist: this is a term-analysis sidecar with no edges (and therefore none of
the measured n^2 pair-RAM wall).

    STAGE      MECHANISM
    EXTRACT    pymupdf text blocks, de-hyphenated, reading order (D1)
    UNITS      markdown -> blank-line paragraphs, ATX headings dropped (D2)
    RULER      pack target fitted on chars/section where sections exist (D3)
    PACK       consecutive whole units up to target, never split a unit (D4)
    SOURCES    split by document size so one fit is not pooled over two
               registers, per chunkgraph R19 (D5)

EARS guards, measured 2026-09-26 on the live corpora:

D1  WHERE a document is a PDF, units SHALL be pymupdf text blocks, sorted by
    (top, left), with soft hyphens joined across line breaks.
    Rejected: font-size heading detection, to emit '##' and reuse
    chunkgraph.sections. FALSIFIED on this corpus -- of 20 PDFs, 8 have ZERO
    bold heading candidates, body-size-by-char-mass lands on FOOTNOTE text in
    the Lewy book (8.2pt, making 20.1 lines/page look like headings), and
    italic is body text in 3 files (7.7-9.1 lines/page). A detector that fires
    on nothing for 40% of the corpus gives those files one giant section and a
    constant paragraph count -- the exact defect chunkgraph R23(a) documents.

D2  Block granularity is NOT paragraph granularity and varies by PDF producer.
    Measured p90/p50 of block char length: A_Pagan_Saint 100/93 and
    Plato_and_the_Art 95/90 (blocks are LINES); Plotinus_Philosophy_of_Religion
    3597/2485 and Porphyry_the_Apostate 2043/409 (blocks are PARAGRAPHS).
    The system SHALL therefore NOT assume a unit is a paragraph.

D3  The pack target SHALL be fitted with chunkgraph._bc_center on chars per
    SECTION, measured where real sections exist, and SHALL NOT be a constant.
    Measured on Lewy_Chaldaean_Oracles_repaired.md -- 53 ATX headings, 30
    sections carrying paragraphs (23 headings are back-to-back and excluded
    per the reduce_overlaps skill): chars/section p50=7316 -> m=7310,
    lam=-0.080. Fitted on a neoplatonic file for the neoplatonic sources; the
    arxiv sources fit their own. A band fitted on one corpus is a HYPOTHESIS
    about the other, not a finding (constitution Article XIV).

D4  A chunk SHALL be consecutive whole units packed until the next unit would
    exceed the target, and SHALL NEVER split a unit. This satisfies the
    operator's constraint that cuts land on boundaries (2026-09-26: "it will
    split not around end of paragraph boundaries") without needing to know
    whether a unit is a line or a paragraph (D2). Consequence: a chunk may
    reach target + max_unit; measured worst case 7310 + 5596 = 12,906 chars.

D5  WHERE documents of one domain differ in register by an order of magnitude,
    they SHALL carry distinct source labels so chunkgraph R19's per-source fit
    is not pooled. Measured: 18 neoplatonic PDFs at 4-36 pages, then 531 and
    631 pages -- p50=12, max=631. Labels: `neop_article`, `neop_book`.

D6  Extraction is deterministic and model-free. Reruns SHALL be byte-identical;
    a rerun that differs is a defect, not variance.
"""
from __future__ import annotations

import glob
import os
import re
import statistics

import numpy as np

from chunkgraph import _bc_center

NEOP_DIR = "C:/Users/user/Documents/wiki/spiritual/neoplatonism"
ARXIV_DIR = "C:/Users/user/arxiv_id_lists/papers/post_processed"

# D5: page count separating the two neoplatonic registers. Measured p50=12,
# with the only two outliers at 531 and 631 pages.
BOOK_PAGES = 100

_DEHYPH = re.compile(r"(\w)-\n(\w)")
_NL = re.compile(r"[ \t]*\n[ \t]*")
_BLANK = re.compile(r"\n[ \t]*\n")
_ATX = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S")
_IMG = re.compile(r"^<image[^>]*>$")


def pdf_units(path: str, cap_pages: int | None = None) -> tuple[list[str], int]:
    """D1: pymupdf text blocks, de-hyphenated, in reading order.

    Returns (units, n_pages_scanned). Import is local so the module loads
    without pymupdf when only the markdown path is used.
    """
    import fitz

    doc = fitz.open(path)
    total = len(doc)
    n = total if cap_pages is None else min(total, cap_pages)
    out = []
    for pno in range(n):
        blocks = [b for b in doc[pno].get_text("blocks") if b[6] == 0]
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        for b in blocks:
            t = _NL.sub(" ", _DEHYPH.sub(r"\1\2", b[4])).strip()
            if t:
                out.append(t)
    doc.close()
    return out, total


def md_units(text: str) -> list[str]:
    """D2: blank-line paragraphs, ATX heading lines and image placeholders dropped."""
    out = []
    for para in _BLANK.split(text):
        keep = [ln for ln in para.split("\n")
                if ln.strip() and not _ATX.match(ln) and not _IMG.match(ln.strip())]
        if not keep:
            continue
        t = _NL.sub(" ", "\n".join(keep)).strip()
        if t:
            out.append(t)
    return out


def md_sections(text: str) -> list[list[str]]:
    """Paragraph lists between ATX headings; zero-paragraph sections excluded.

    Mirrors chunkgraph.sections for ATX markdown. chunkgraph._is_heading matches
    wikitext (' = = X = = ') only, so it cannot be reused here as it stands.
    """
    secs, cur, n_head = [], [], 0
    for ln in text.split("\n"):
        if _ATX.match(ln):
            n_head += 1
            if cur:
                secs.append("\n".join(cur))
            cur = []
        else:
            cur.append(ln)
    if cur:
        secs.append("\n".join(cur))
    if not n_head:
        # D3: a document with no headings has NO sections. Returning the whole
        # document as one section is the defect chunkgraph R23(a) documents --
        # it reports a constant section size and licenses a fabricated ruler.
        return []
    out = []
    for s in secs:
        ps = md_units(s)
        if ps:
            out.append(ps)
    return out


def derive_pack_target(section_docs: list[str]) -> dict:
    """D3: fit the pack target on chars per section, via the incumbent _bc_center.

    Require: at least one document containing real ATX sections.
    Guarantee: returns {target, hi, lam, n_sections, p50}; `target` is the
    Box-Cox median of chars per section, never a constant.
    """
    chars = []
    for text in section_docs:
        for ps in md_sections(text):
            chars.append(sum(len(p) for p in ps))
    if not chars:
        raise ValueError("no ATX sections found -- cannot fit a pack target (D3)")
    arr = np.array(sorted(chars), dtype=float)
    m, hi, lam = _bc_center(arr)
    return {"target": float(m), "hi": float(hi), "lam": lam,
            "n_sections": len(chars), "p50": float(statistics.median(chars))}


def pack(units: list[str], target: float) -> list[str]:
    """D4: consecutive whole units up to `target` chars. Never splits a unit.

    Guarantee (conservation): "\\n\\n".join(pack(u, t)) == "\\n\\n".join(u).
    Every unit appears exactly once, in order, with no overlap.
    """
    out, cur, size = [], [], 0
    for u in units:
        if cur and size + len(u) > target:
            out.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(u)
        size += len(u)
    if cur:
        out.append("\n\n".join(cur))
    return out


def _read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def load_neop(cap_pages: int | None = None) -> tuple[list[str], list[str], list[str]]:
    """The neoplatonic corpus: 20 PDFs via pymupdf plus the repaired markdown.

    D5: PDFs at or above BOOK_PAGES are `neop_book`, the rest `neop_article`.
    The raw (unrepaired) Lewy markdown is skipped -- it is the same book as the
    repaired one, and both would double every term's df.
    """
    doc_ids, docs, sources = [], [], []

    md = os.path.join(NEOP_DIR, "Lewy_Chaldaean_Oracles_repaired.md").replace("\\", "/")
    if os.path.exists(md):
        doc_ids.append("neop_book/Lewy_Chaldaean_Oracles_repaired")
        docs.append(_read(md))
        sources.append("neop_book")

    seen_sizes = set()
    for p in sorted(glob.glob(os.path.join(NEOP_DIR, "*.pdf").replace("\\", "/"))):
        size = os.path.getsize(p)
        if size in seen_sizes:      # the ' (1)' duplicates are byte-identical
            continue
        seen_sizes.add(size)
        stem = os.path.basename(p)[:-4]
        if stem.startswith("Hans Lewy"):
            continue                # same book as the markdown above
        units, pages = pdf_units(p, cap_pages=cap_pages)
        if not units:
            continue
        src = "neop_book" if pages >= BOOK_PAGES else "neop_article"
        doc_ids.append("%s/%s" % (src, stem))
        docs.append("\n\n".join(units))
        sources.append(src)

    assert len(doc_ids) == len(docs) == len(sources)
    for did, src in zip(doc_ids, sources):
        assert did.startswith(src + "/")
    return doc_ids, docs, sources


def load_arxiv(n_docs: int | None = None,
               stride: int = 1) -> tuple[list[str], list[str], list[str]]:
    """The arxiv corpus: docling markdown under papers/post_processed.

    Only top-level *.md are taken; `partitioned/` holds a prior chunking of the
    same text and would duplicate every term.
    """
    pat = os.path.join(ARXIV_DIR, "*.md").replace("\\", "/")
    paths = sorted(p for p in glob.glob(pat) if not p.endswith(".md.fbak"))
    paths = paths[::stride]
    if n_docs is not None:
        paths = paths[:n_docs]
    doc_ids, docs, sources = [], [], []
    for p in paths:
        text = _read(p)
        if not text.strip():
            continue
        doc_ids.append("arxiv/" + os.path.basename(p)[:-3])
        docs.append(text)
        sources.append("arxiv")
    assert len(doc_ids) == len(docs) == len(sources)
    return doc_ids, docs, sources


def units_for(text: str, source: str) -> list[str]:
    """Units for an already-loaded document. PDFs arrive pre-joined by load_neop."""
    if source == "arxiv" or "#" in text[:2000]:
        return md_units(text)
    return [u for u in _BLANK.split(text) if u.strip()]


def chunk_corpus(doc_ids, docs, sources, target: float):
    """Pack every document, carrying doc_id and source onto each chunk (R20/R8).

    Guarantee: conservation per document, asserted by the caller's diagnostic.
    """
    out_ids, out_texts, out_srcs = [], [], []
    for did, text, src in zip(doc_ids, docs, sources):
        units = units_for(text, src)
        for i, ch in enumerate(pack(units, target)):
            out_ids.append(did)
            out_texts.append(ch)
            out_srcs.append(src)
    return out_ids, out_texts, out_srcs
