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

ARXIV SECTION CHUNKER (operator instruction 2026-10-02; supersedes D3/D4 for
`arxiv` only -- the neop path above is unchanged). NO GOVERNING SPEC.

    STAGE    MECHANISM
    CLEAN    drop base64 data-URI image lines and <image> placeholders (C1)
    SECTION  split on ATX header lines; a section = header + body (C2)
    FIT      size = newlines in the text; log of body sizes, zeros dropped;
             m = exp(median), hi = exp(median + 2*1.4826*MAD) (C3)
    SPLIT    section body > hi -> RecursiveCharacterTextSplitter(size=hi,
             overlap=m) measured in the same size unit (C4)
    TRIM     de-overlap of the pieces of ONE split section, by where each piece
             sits in the section body (C5)
    MERGE    consecutive units absorbed while the chunk is < m and stays <= hi,
             never across documents or the references boundary; conservation of
             the units is asserted first (C6, C9)
    HEADER   every chunk starts with an ATX header; a continuation piece gets its
             section header prepended; header-less text gets `# <doc_id>` (C7)
    DISTINCT identical chunk texts kept once, first document wins (C8)

C1  Lines carrying `data:image` SHALL be removed before any measurement. Measured:
    613 lines > 20,000 chars, the longest 1,042,020, all base64 PNG; one section
    of 16.7M chars made the old pack target meaningless for 293 chunks (V2).
C2  Size SHALL be newline count, with a line longer than `lcap` counting as
    1 + len//lcap lines, `lcap` = exp(median + 2*1.4826*MAD) of log line length.
    Without this a single-line giant has size 0 and can never be split.
C3  Zero-size bodies SHALL be excluded from the fit (bare headers, one-line
    sections) and SHALL still be chunked.
C4  Overlap SHALL equal m (the median); chunk size SHALL equal hi; the splitter's
    separators SHALL be `\n#` followed by the library defaults (operator 2026-10-02).
C5  De-overlap SHALL run only between pieces the splitter produced from one
    section and SHALL remove (previous piece end - next piece start) characters,
    read from the pieces' positions in the section body, never from comparing
    text. Rejected, both measured on the live corpus: difflib (2212_01762 'D.2.
    Benchmark results', repeated table rows: chose an interior repeat, removed 0
    of 129 duplicated chars) and exact suffix/prefix matching (1403_7426
    'References': trimmed a coincidental '|' between two non-overlapping pieces).
    WHERE no admissible position exists -- degenerate repetition such as
    ', of-\n' x thousands in 2408_03402 and Pragmatic, where every shift of a
    piece is an identical substring -- the section SHALL be split with overlap 0
    (disjoint by construction, so nothing is trimmed) and counted as `no_overlap`.
C7  Every chunk SHALL start with an ATX header (continuation pieces get their
    section header prepended; header-less text gets `# <doc_id>`).
C8  Identical chunk texts SHALL be kept once, first document winning.
C9  References SHALL NOT be dropped. Every chunk SHALL carry `section_title` (its
    first section's header text) and `is_reference` (the title IS references /
    bibliography / works cited / literature cited / citations, after optional
    numbering), so retrieval can exclude them (operator 2026-10-02). A reference
    section SHALL NOT merge with a non-reference one, so the flag is true for the
    whole chunk. Known gap: reference lists that docling splits under spurious
    headers (e.g. '## [135] T. A. Estlin...') are NOT flagged.
C10 Extraction junk SHALL be flagged `is_junk`, never dropped, and excluded wherever
    references are (operator 2026-10-03: junk formed its own communities and served as
    exemplars). A chunk is junk when it carries docling's image-OCR marker `latexi`
    (the `<latexit sha1_base64=...>` blob of a LaTeX-rendered image, 233 chunks), OR
    has no letter or digit at all, OR at least JUNK_MIN_LINES (8) of its lettered lines
    hold JUNK_LONE_SHARE (0.9) or more of them to two or fewer letters/digits (plot tick
    labels, letter-per-line soup, `, of-` repeats). Rejected, each measured on the live
    corpus because it overlapped legitimate math or algorithm pseudocode: share of
    in-vocabulary words (flagged 17.6% of chunks, ordinary prose among them), share of
    one-character tokens (legitimate proofs reach 0.47), share of short lines counting
    unlettered ones (code fences and table padding), and the same share below 0.9.
    Known gap: junk that carries neither the marker nor the extreme line profile.
C11 A paper's identity SHALL be its arXiv id without the version (plus a `_methods` suffix for the methods
    extract, which is a part of the paper, not a version of it); the version SHALL be a secondary key,
    1 when the file name carries none (operator 2026-10-03: "use arxiv paper itself as key, but version as
    overwriting mechanism ... presume v1 if no v info is provided"). Where several versions of a key are on
    disk only the highest is read. A stored paper is replaced only by a HIGHER version; the same or a
    lower one is not read.
C12 A section SHALL be re-aggregated from its chunks by reaggregate_section: header handled by chunk_idx, no
    overlap trimming (none is stored, and a trim would delete repeats the source contains).
C6  Conservation SHALL hold per document: non-whitespace text of the units equals
    non-whitespace text of header+body of every section, exactly once.
C13 The SECTION map (operator 2026-10-05: "derive the graph based on sections") SHALL take its nodes from section_records:
    one record per header block of the cleaned markdown BEFORE any splitting or merging, text = header + body. The usable
    nodes (usable_sections) SHALL exclude, in this order, empty bodies, reference sections, bodies shorter than
    exp(median - 2*1.4826*MAD) of the log body length of the non-empty non-reference sections (the chunker's own robust
    bound, 106 characters on the 2026-10-05 corpus), junk (C10), and PDF font-glyph debris: a section at least GLYPH_SHARE
    (0.5) of whose non-space characters are `glyph<c=..,font=..>` codes (1,900 of 82,542 sections, 2.3%, measured 2026-10-05:
    the exact dense pass ranked their pairs as the strongest and 91% of its top edges were such near-duplicates; there is a gap
    in the share between 0.5 (1,900 sections) and 0.2 (2,016), so the cut is not on the edge of a cluster). Each exclusion
    SHALL be counted in the returned stats.
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


_ARXIV_STEM = re.compile(r"^(?P<id>\d{4}_\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?_\d{7})(?:v(?P<v>\d+))?(?P<part>_methods)?$")


def arxiv_stem_key(stem: str) -> tuple[str, int]:
    """C11. Guarantee: (key, version). The key is the arXiv id WITHOUT its version, kept with a `_methods`
    suffix because a methods file is a second extract of the same paper, not a version of it (all 301
    on disk sit beside a main file of the same id). The version is the `v<n>` after the id, 1 when the
    stem carries none. A stem that is not an arXiv id (a book) is its own key, version 1."""
    m = _ARXIV_STEM.match(stem)
    if not m:
        return stem, 1
    return m.group("id") + (m.group("part") or ""), int(m.group("v") or 1)


def arxiv_files(root: str | None = None) -> list[tuple[str, int, str]]:
    """C11. Guarantee: [(key, version, path)] sorted by key, ONE file per key: the highest version present.
    Backups (*.md.fbak) are not read."""
    pat = os.path.join(root or ARXIV_DIR, "*.md").replace("\\", "/")
    best: dict[str, tuple[int, str]] = {}
    for p in glob.glob(pat):
        if p.endswith(".md.fbak"):
            continue
        key, ver = arxiv_stem_key(os.path.basename(p)[:-3])
        if key not in best or ver > best[key][0]:
            best[key] = (ver, p)
    return [(k, best[k][0], best[k][1]) for k in sorted(best)]


def load_arxiv_papers(root: str | None = None, stored: dict | None = None, n_docs: int | None = None,
                      stride: int = 1) -> list[dict]:
    """C11. Guarantee: [{doc_id, version, path, text, upgrade}], one per paper key, the latest version on
    disk, sorted by key. `stored` maps a doc_id ("arxiv/<key>") to the version already held: a paper whose
    file version is not HIGHER is not read and not returned (same version: already ingested; lower: the
    held one is newer). `upgrade` is True where a lower version is held and this one will replace it.
    Empty files are dropped."""
    stored = stored or {}
    files = arxiv_files(root)[::stride]
    if n_docs is not None:
        files = files[:n_docs]
    out = []
    for key, ver, path in files:
        doc_id = "arxiv/" + key
        if ver <= stored.get(doc_id, 0):
            continue
        text = _read(path)
        if text.strip():
            out.append({"doc_id": doc_id, "version": ver, "path": path, "text": text, "upgrade": doc_id in stored})
    return out


def load_arxiv(n_docs: int | None = None,
               stride: int = 1,
               skip: frozenset | set = frozenset(),
               root: str | None = None) -> tuple[list[str], list[str], list[str]]:
    """The arxiv corpus: docling markdown under papers/post_processed, the latest version of each paper.

    Only top-level *.md are taken; `partitioned/` holds a prior chunking of the
    same text and would duplicate every term. `skip` is a set of doc_ids ("arxiv/<key>", version
    stripped) not to read. `root` replaces ARXIV_DIR. See load_arxiv_papers for versions.
    """
    papers = load_arxiv_papers(root=root, n_docs=n_docs, stride=stride)
    papers = [p for p in papers if p["doc_id"] not in skip]
    return [p["doc_id"] for p in papers], [p["text"] for p in papers], ["arxiv"] * len(papers)


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


# ------------------------------------------------------- arxiv section chunker ----
_DATA_IMG = re.compile(r"data:image")
_NONWS = re.compile(r"\s+")


def clean_md(text: str) -> str:
    """C1. Guarantee: text without NULs, base64 image lines and <image> placeholders."""
    keep = [ln for ln in text.replace("\0", "").split("\n")
            if not _DATA_IMG.search(ln) and not _IMG.match(ln.strip())]
    return "\n".join(keep)


def _robust_hi(values) -> tuple[float, float]:
    """Guarantee: (exp(median), exp(median + 2*1.4826*MAD)) of log(values > 0)."""
    from scipy.stats import median_abs_deviation
    x = np.log(np.asarray([v for v in values if v > 0], float))
    med = float(np.median(x))
    return float(np.exp(med)), float(np.exp(med + 2 * 1.4826 * median_abs_deviation(x)))


def fit_line_cap(docs: list[str]) -> int:
    """C2. Guarantee: lcap, the log-MAD upper bound of non-empty line length."""
    lens = [len(ln) for d in docs for ln in d.split("\n") if ln.strip()]
    return max(1, int(round(_robust_hi(lens)[1])))


def make_size(lcap: int):
    """C2. Guarantee: size(text) = newlines + sum(len(line)//lcap)."""
    def size(s: str) -> int:
        return s.count("\n") + sum(len(ln) // lcap for ln in s.split("\n"))
    return size


def header_sections(text: str, doc_id: str) -> list[tuple[str, str]]:
    """C2, C7. Guarantee: [(header_line, body)] in source order. Text before the
    first header gets the synthetic header `# <doc_id>`; empty preambles are dropped."""
    parts, head, cur = [], None, []
    for ln in text.split("\n"):
        if _ATX.match(ln):
            parts.append((head, "\n".join(cur).strip("\n")))
            head, cur = ln.strip(), []
        else:
            cur.append(ln)
    parts.append((head, "\n".join(cur).strip("\n")))
    out = []
    for h, b in parts:
        if h is None:
            if not b.strip():
                continue
            h = "# " + doc_id
        out.append((h, b))
    return out


def fit_arxiv(docs: list[str]) -> dict:
    """C2, C3. Guarantee: {lcap, m, hi, n_sections, n_zero, hi_frac}; m and hi are
    integers in newline units, hi > m, fitted on non-zero body sizes only."""
    size = make_size(fit_line_cap(docs))
    sizes = [size(b) for d in docs for _, b in header_sections(d, "x") if b.strip()]
    nz = [s for s in sizes if s > 0]
    m, hi = _robust_hi(nz)
    m, hi = max(1, int(round(m))), int(round(hi))
    hi = max(hi, m + 1)
    return {"lcap": fit_line_cap(docs), "m": m, "hi": hi, "n_sections": len(sizes),
            "n_zero": len(sizes) - len(nz), "hi_frac": sum(s > hi for s in nz) / len(nz)}


_REF_TITLE = re.compile(
    r"^\W*(?:[0-9ivxIVX]+[\.\):]?\s+|appendix\s+\w+[\.:]?\s+)?"
    r"(?:selected\s+|cited\s+)?(?:references?(?:\s+(?:list|cited|and\s+notes))?|bibliography|"
    r"works\s+cited|literature\s+cited|citations?|notes\s+and\s+references)\W*$", re.I)


def header_title(header: str) -> str:
    """C9. Guarantee: the header line without its leading #s."""
    return re.sub(r"^\s*#{1,6}\s+", "", header).strip()


def is_reference(header: str) -> bool:
    """C9. Guarantee: True when the section TITLE IS a references phrase (after optional
    numbering), so retrieval can exclude it. Title only, never body; a title that merely
    starts with the word ('Citation Intent Classification') is content, not references."""
    return _REF_TITLE.match(header_title(header)) is not None


JUNK_LONE_SHARE, JUNK_MIN_LINES = 0.9, 8
_ALNUM = re.compile(r"\w")


def is_junk(text: str) -> bool:
    """C10. Guarantee: True for image-OCR debris (the `latexi` marker), a chunk with no
    letter or digit, or one whose lettered lines are almost all one or two characters long
    (at least JUNK_MIN_LINES of them, JUNK_LONE_SHARE or more of them short). Lines with
    no letter or digit (code fences, table padding) never count either way."""
    if "latexi" in text:
        return True
    n = [k for k in (len(_ALNUM.findall(ln)) for ln in text.split("\n")) if k]
    if not n:
        return True
    return len(n) >= JUNK_MIN_LINES and sum(k <= 2 for k in n) / len(n) >= JUNK_LONE_SHARE


def with_junk_flag(records: list[dict]) -> list[dict]:
    """C10. Guarantee: every record carries `is_junk` (set in place from its text where a cache
    written before the flag existed lacks it); returns `records`."""
    for r in records:
        r.setdefault("is_junk", is_junk(r["text"]))
    return records


def retrievable(records: list[dict]) -> list[dict]:
    """C9, C10. Guarantee: the records that are neither references nor junk, in order."""
    return [r for r in with_junk_flag(records) if not r["is_reference"] and not r["is_junk"]]


def section_records(doc_ids: list[str], docs: list[str]) -> list[dict]:
    """C13. Guarantee: one record per header block of every document, in order: {doc_id, section_idx (the position among
    the document's header blocks, the key tools/section_lengths used), chunk_idx 0, section_title, is_reference, body_chars,
    text = header + body}. Nothing is dropped here; usable_sections decides."""
    out = []
    for did, d in zip(doc_ids, docs):
        for si, (h, body) in enumerate(header_sections(clean_md(d), did)):
            out.append({"doc_id": did, "section_idx": si, "chunk_idx": 0, "section_title": header_title(h),
                        "is_reference": is_reference(h), "body_chars": len(body),
                        "text": h + ("\n\n" + body if body.strip() else "")})
    return out


GLYPH_SHARE = 0.5
_GLYPH = re.compile(r"glyph\s*(?:<[^>]*>|&lt;.*?&gt;|\[[^\]]*\])", re.I)


def glyph_share(text: str) -> float:
    """C13. Guarantee: the share of `text`'s non-space characters that sit inside `glyph<...>` font-glyph codes (0 for an empty text)."""
    n = len(re.sub(r"\s+", "", text))
    return sum(len(re.sub(r"\s+", "", m.group(0))) for m in _GLYPH.finditer(text)) / n if n else 0.0


def usable_sections(records: list[dict]) -> tuple[list[dict], dict]:
    """C13. Guarantee: (the records that are non-empty, not references, at least the robust lower bound long, not junk and not glyph
    debris, in order; stats {all, empty, reference, short, junk, glyph, floor, kept} counting each exclusion once in that order)."""
    from scipy.stats import median_abs_deviation
    body = np.array([r["body_chars"] for r in records if r["body_chars"] > 0 and not r["is_reference"]], float)
    x = np.log(body)
    floor = float(np.exp(np.median(x) - 2 * 1.4826 * median_abs_deviation(x)))
    st = {"all": len(records), "empty": 0, "reference": 0, "short": 0, "junk": 0, "glyph": 0, "floor": floor}
    kept = []
    for r in with_junk_flag(records):
        if r["body_chars"] == 0:
            st["empty"] += 1
        elif r["is_reference"]:
            st["reference"] += 1
        elif r["body_chars"] < floor:
            st["short"] += 1
        elif r["is_junk"]:
            st["junk"] += 1
        elif glyph_share(r["text"]) >= GLYPH_SHARE:
            st["glyph"] += 1
        else:
            kept.append(r)
    st["kept"] = len(kept)
    return kept, st


def make_splitter(fit: dict, overlap: int | None = None):
    """C4. Guarantee: RecursiveCharacterTextSplitter at size hi, overlap m (or `overlap`),
    measured in the C2 unit, with `\\n#` placed before the library defaults so a header
    line is the first boundary tried."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=fit["hi"], chunk_overlap=fit["m"] if overlap is None else overlap,
        length_function=make_size(fit["lcap"]), separators=["\n#", "\n\n", "\n", " ", ""])


def tile_pieces(body: str, pieces: list[str], m: int, size) -> tuple[list[str], int]:
    """C5. Guarantee: (pieces re-cut so they tile `body` with no span stored twice,
    chars removed). Overlap is read from WHERE each piece sits in `body`
    (previous end - next start), never from comparing text. A candidate position
    is accepted only if it leaves no gap beyond whitespace AND implies an overlap
    of at most m+1 units -- the splitter cannot have produced more. Text matching
    was tried and REJECTED on the live corpus: difflib's longest common substring
    chose an interior repeat on a repeated-row table (2212_01762 'D.2. Benchmark
    results'); exact suffix/prefix matching trimmed a coincidental '|' between
    non-overlapping pieces (1403_7426 'References'); and plain str.find took the
    EARLIER of two identical 303-char figure blocks (2311_04257 '3.3.
    Modality-Adaptive Module'), claiming 164 chars of overlap that were not there.
    No candidate satisfying both conditions is an error, not a fallback."""
    out, pos, end, trimmed = [], -1, 0, 0
    for i, p in enumerate(pieces):
        s = body.find(p, pos + 1)
        while s >= 0:
            if s >= end:
                if not body[end:s].strip():
                    break                               # contiguous: only whitespace between
                s = -1                                  # past the gap budget: no later one can fit
                break
            if size(body[s:end]) <= m + 1:
                break                                   # a plausible overlap
            s = body.find(p, s + 1)
        assert s >= 0, "no admissible position for piece %d of %d" % (i, len(pieces))
        cut = max(0, end - s)
        out.append(p[cut:])
        trimmed += cut
        pos, end = s, max(end, s + len(p))
    return out, trimmed


def chunk_arxiv_doc(doc_id: str, text: str, fit: dict, splitter=None) -> tuple[list[str], dict]:
    """C1-C7 for ONE document. Guarantee: (chunks, stats); every chunk starts with an
    ATX header; conservation (C6) asserted before return."""
    size = make_size(fit["lcap"])
    m, hi = fit["m"], fit["hi"]
    splitter = splitter or make_splitter(fit)
    secs = header_sections(clean_md(text), doc_id)
    units, trimmed, n_split, n_flat = [], 0, 0, 0   # unit = (header, text, continuation, section_idx, chunk_idx)
    for si, (h, body) in enumerate(secs):
        if size(body) <= hi:
            units.append((h, h + ("\n\n" + body if body.strip() else ""), False, si, 0))
            continue
        n_split += 1
        try:
            pieces, k = tile_pieces(body, splitter.split_text(body), m, size)
            placed = _NONWS.sub("", "".join(pieces)) == _NONWS.sub("", body)
        except AssertionError:
            placed = False
        if not placed:                          # C5: provenance unrecoverable -> split without overlap
            n_flat += 1
            pieces, k = make_splitter(fit, overlap=0).split_text(body), 0
        trimmed += k
        for i, p in enumerate(pieces):
            if not p.strip():
                continue
            units.append((h, h + "\n\n" + p if i == 0 else p, i > 0, si, i))
    expect = _NONWS.sub("", "".join(h + b for h, b in secs))
    assert _NONWS.sub("", "".join(u[1] for u in units)) == expect, \
        "C6 conservation failed for " + doc_id
    chunks, cur = [], None                      # cur = [header, text, size, continuation, section_idx, chunk_idx]
    for h, t, cont, si, ci in units:
        s = size(t)
        if (cur is not None and cur[2] < m and cur[2] + s <= hi
                and is_reference(cur[0]) == is_reference(h)):       # C9: never merge across the references boundary
            cur[1] += "\n\n" + t
            cur[2] += s + 2
        else:
            if cur is not None:
                chunks.append(cur)
            cur = [h, t, s, cont, si, ci]
    if cur is not None:
        chunks.append(cur)
    out = []                                    # (text, section_idx, chunk_idx, section_title, is_reference)
    for h, t, s, cont, si, ci in chunks:
        t = h + "\n\n" + t if cont else t
        assert _ATX.match(t.split("\n", 1)[0]), "C7 chunk without a header in " + doc_id
        out.append((t, si, ci, header_title(h), is_reference(h)))
    return out, {"sections": len(secs), "split": n_split, "trimmed_chars": trimmed, "no_overlap": n_flat}


def chunk_arxiv(doc_ids: list[str], docs: list[str], fit: dict | None = None):
    """C1-C9. Guarantee: (records, fit, stats). Identical chunk texts are kept once (first
    document wins). A record is {doc_id, section_idx, chunk_idx, section_title,
    is_reference, is_junk, text}; (doc_id, section_idx, chunk_idx) is the business key -- the
    section the chunk's first unit came from and that unit's piece index inside it
    (0 for an unsplit one). section_title is that section's header text."""
    docs = [clean_md(d) for d in docs]
    fit = fit or fit_arxiv(docs)
    splitter = make_splitter(fit)
    seen, records = set(), []
    tot = {"sections": 0, "split": 0, "trimmed_chars": 0, "no_overlap": 0, "duplicates": 0}
    for did, d in zip(doc_ids, docs):
        chunks, st = chunk_arxiv_doc(did, d, fit, splitter)
        for k in ("sections", "split", "trimmed_chars", "no_overlap"):
            tot[k] += st[k]
        for c, si, ci, title, ref in chunks:
            if c in seen:
                tot["duplicates"] += 1
                continue
            seen.add(c)
            records.append({"doc_id": did, "section_idx": si, "chunk_idx": ci,
                            "section_title": title, "is_reference": ref, "is_junk": is_junk(c), "text": c})
    keys = [(r["doc_id"], r["section_idx"], r["chunk_idx"]) for r in records]
    assert len(set(keys)) == len(keys), "business key (doc_id, section_idx, chunk_idx) is not unique"
    return records, fit, tot


SECTION_DISPLAY_CAP = 200_000       # chars of a section shown in full: p99.9 of section size is 79,000; one degenerate section is 16.7M
_NEW_ID = re.compile(r"^(\d{4})_(\d{4,5})$")
_OLD_ID = re.compile(r"^([a-z\-]+(?:\.[A-Z]{2})?)_(\d{7})$")


def arxiv_url(doc_id: str) -> str | None:
    """C11. Guarantee: the arXiv abstract page of a paper, None for anything that is not an arXiv id (a book). A
    `_methods` extract links to its paper."""
    stem = doc_id.removeprefix("arxiv/").removesuffix("_methods")
    if m := _NEW_ID.match(stem):
        return "https://arxiv.org/abs/%s.%s" % m.groups()
    if m := _OLD_ID.match(stem):
        return "https://arxiv.org/abs/%s/%s" % m.groups()
    return None


def reaggregate_section(chunks: list[tuple[int, str]]) -> str:
    """C12. Guarantee: ONE section's text rebuilt from its chunks, given as [(chunk_idx, text)] in chunk_idx
    order. A chunk with chunk_idx > 0 starts with the section header the chunker prepended (C7), so that first
    line and the blank after it are dropped; a chunk with chunk_idx 0 keeps its header, which is the section's
    own. Chunks are joined by a blank line. Nothing else is removed: no overlap is trimmed, because none is
    stored (C5 de-overlaps by position when it cuts). Measured on the live corpus, 72 of 1,245 neighbouring
    chunk pairs share an exact 12+ character span, and every sample is a repeat the SOURCE has
    ('<!-- formula-not-decoded -->', identical image-description lines); a reduce_overlaps pass would delete
    them. 324 of 400 documents reassemble to their markdown exactly (non-whitespace), the rest differing only
    by the `# <doc_id>` header C7 gives header-less text."""
    return "\n\n".join(t.partition("\n")[2].lstrip("\n") if c > 0 else t for c, t in chunks)
