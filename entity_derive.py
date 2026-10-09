"""entity_derive.py -- derive entities from raw text by information theory: no tagger, no NER, no pretrained model.

NO GOVERNING SPEC beyond operator instructions 2026-10-04 ("actual entities ... I don't want to rely on spacy ... autonomous
ways to derive entities using information theory such as DIRT, NPMI, PPMI" and "create a diagnostic set ... sweep over hyper
parms") and the approved plan C:\\Users\\user\\.claude\\plans\\derive-arxiv-entities-with-information-theory.md.
Tasks: playbook.md T130-T133. Skill: ~/.skills/unsupervised-entity-relations. PDMI in the request was a typo for PPMI.

    STAGE      MECHANISM                                                                              GUARD
    TOKENIZE   case-preserving; hyphenated and digit-bearing tokens whole; math, citation, cell cut    AE2, AE8
    CIPHER     per-paper leave-one-out cross-entropy; shifted-letter papers leave the counts           AE3
    COUNT      n-grams n<=4 inside a segment, apriori by support, int64 keys, df over PAPERS            AE1
    FEATURES   cohesion (min-split NPMI, G2), boundary entropy, form signature, burstiness, nesting      AE5-AE7
    ALIAS      acronym-expansion pairs, Schwartz-Hearst initials rule                                   AE9
    SCORE      per-track mid-ranks, a swept combination rule, a unigram quota                           AE4
    MATCH      a frozen inventory over new text, longest first, no overlap                              AE11, AE13

Guards (EARS; the prefix is AE because entities.py owns E1-E19):

AE1  WHEN a candidate occurs in fewer than `df_floor` PAPERS (not chunks) THEN it SHALL NOT be an entity. The floor is over
     papers so a single-document artifact, a whole mis-extracted book included, cannot qualify.
AE2  An n-gram SHALL NOT span a sentence, a math span, a citation or a table-cell boundary.
AE3  WHEN a paper's leave-one-out cross-entropy is more than CIPHER_Z robust z above the corpus THEN its tokens SHALL be left
     out of every count. Measured need: the book Machine-Learning-Systems was extracted with every letter shifted by one.
AE4  The same texts, parameters and seed SHALL give a byte-identical inventory (no randomness is used in this module).
AE5  WHEN a multiword candidate's minimum split NPMI is <= 0 THEN it SHALL score 0. PPMI/NPMI act as a demotion floor against
     rare-event noise, never as a term weight (the operator's standing rule).
AE6  WHEN a candidate begins or ends with a closed-class token (df over papers > CLOSED_DF_FRAC and capitalised share <
     CLOSED_P_CAP, derived from the corpus, no stoplist) THEN it SHALL score 0; a closed-class unigram scores 0.
AE7  WHEN one longer gram accounts for >= NEST_FRAC of a gram's occurrences THEN the shorter gram SHALL score 0 (C-value
     nesting, Frantzi & Ananiadou).
AE8  Capitalisation statistics SHALL use running prose only, never headings or table cells.
AE9  WHEN an acronym pair satisfies the Schwartz-Hearst initials rule THEN the acronym and its expansion SHALL be aliases of one
     entity.
AE10 (types, T133) Types SHALL be accepted only at seed-ARI >= 0.8, else class -1.
AE11 A frozen inventory SHALL only produce mentions over new text; it SHALL NOT change any statistic.
AE12 (store) Deleting a paper SHALL delete its mentions.
AE13 A token the inventory never saw SHALL NOT match.
AE6b WHEN a candidate begins or ends with a token that occurs in more than `universal_df` of ALL papers THEN it SHALL score 0, whatever
     its capitalisation. Measured at 2,521 papers: the (99.0% of papers, capitalised 2.4%), a (97.9%, 8.9%), of, and, in, to, for, is
     (all >= 97.8%) slip past the 2% capitalisation cut of AE6 and produced "the current" (1,362 papers), "the next", "a small",
     "the last"; model is in 93.8% and data in 88.7%, so "language model" and "training data" stay.
AE2b A docling glyph placeholder (glyph<...>, glyph[...]) and a run of font glyph ids (/gid00002/gid00030) SHALL be a boundary, as math is.
AE15 WHEN every token of a candidate is a single character (x y, R d, t t, i j) THEN it SHALL score 0: it is math notation left outside
     $...$, not an entity. Measured live: "x y" in 506 papers and "R d" in 431 ranked 6th and 12th by paper count. `min_tok_len`
     is a swept axis; 1 turns the guard off.
AE3b The cipher test compares a paper only with papers of similar token count (see cipher_outliers); a corpus-wide z flagged the short
     `_methods` extracts (55 of 62 flagged, 59 of 62 ordinary English).
AE14 WHEN a candidate's residual IDF over papers (Church & Gale: observed IDF minus the Poisson-expected IDF) is below
     `ridf_floor` THEN it SHALL score 0. A gram spread evenly across papers is vocabulary, not an entity. Measured need:
     "does not" (7,048 chunks) and "do not" (6,258) passed AE6 because "not" is capitalised mid-sentence 2.2% of the time,
     just over its 2% cut; their residual IDF is 0.24 and 0.33.
"""
from __future__ import annotations

import math
import re
from array import array
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

TOKENIZER_VERSION = "t3"                        # t3: glyph<...>, HTML-escaped glyph&lt;...&gt; and /gidNNNN placeholders are boundaries (live: 'glyph' 59,646 mentions under t1; 'glyph lt c' 58,318 still under t2)
MAX_N = 4
CIPHER_Z, CIPHER_MIN_TOKENS = 4.0, 200          # AE3
CLOSED_DF_FRAC, CLOSED_P_CAP, CLOSED_MIN_DEN = 0.30, 0.02, 5   # AE6
NEST_FRAC = 0.95                                # AE7
RIDF_FLOOR = 0.5                                # AE14: measured gap on 421 papers: does 0.19, do 0.26, "does not" 0.24, "do not" 0.33 | group 1.09, true 0.91, attention 0.83

LOWER, TITLE, UPPER, MIXED, ALNUM, NUM, OTHER = range(7)
CAPITALISH = (TITLE, UPPER, MIXED, ALNUM)
PROSE, HEADING, CELL = 0, 1, 2

_BOUNDARY = "\x00"
_MATH_BLOCK = re.compile(r"\$\$.*?\$\$", re.S)
_MATH_INLINE = re.compile(r"\$[^$\n]{1,200}\$")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_GLYPH = re.compile(r"glyph\s*(?:<[^>]*>|&lt;.*?&gt;|\[[^\]]*\])", re.I)      # docling's placeholder for a glyph it could not decode: glyph<c=3,font=/ABCD+Cambria>, glyph&lt;c=29&gt; (HTML-escaped), glyph[lscript]
_GID = re.compile(r"(?:/gid\d+)+")                                # a font's glyph ids left in the text: /gid00002/gid00030
_CITE_NUM = re.compile(r"\[\s*\d+(?:\s*[,\u2013\-]\s*\d+)*\s*\]")
_CITE_AUTH = re.compile(r"\((?:[A-Z][A-Za-z\u00C0-\u017F'\-]+(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z\u00C0-\u017F'\-]+)?,?\s+"
                        r"(?:19|20)\d\d[a-z]?(?:;\s*)?)+\)")
_LINEBREAK_HYPHEN = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")
_TOKEN = re.compile(r"[A-Za-z0-9\u00C0-\u024F]+(?:[-_./'\u2019][A-Za-z0-9\u00C0-\u024F]+)*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_NO_SPLIT = ("et al.", "e.g.", "i.e.", "Fig.", "Figs.", "Eq.", "Eqs.", "Sec.", "Tab.", "vs.", "cf.", "approx.", "No.", "Alg.", "Ref.", "Refs.", "resp.")


# ---------------------------------------------------------------------------------------------- tokenizer
def shape_of(surf: str) -> int:
    """Guarantee: the surface-form class of one token: LOWER, TITLE, UPPER (>= 2 capitals, no lower), MIXED (camelCase or
    an internal capital: LoRA, DeepSeekMath), ALNUM (letters and digits: GSM8K, Llama-3), NUM, or OTHER."""
    letters = [c for c in surf if c.isalpha()]
    has_digit = any(c.isdigit() for c in surf)
    if not letters:
        return NUM if has_digit else OTHER
    if has_digit:
        return ALNUM
    if surf.isupper() and len(letters) >= 2:
        return UPPER
    if any(c.isupper() for c in surf[1:]):
        return MIXED
    return TITLE if surf[0].isupper() else LOWER


def _segments(text: str) -> list[tuple[str, str]]:
    """AE2, AE8. Guarantee: [(kind, string)] with kind prose|heading|cell. Math spans, HTML comments (the
    formula-not-decoded marker), and numeric or author-year citations become hard boundaries; a hyphen at a line break joins the
    word; markdown headings and table cells are their own segment kinds; table rule rows are dropped."""
    text = _LINEBREAK_HYPHEN.sub("", text)
    for pat in (_MATH_BLOCK, _COMMENT, _GLYPH, _GID, _MATH_INLINE, _CITE_NUM, _CITE_AUTH):
        text = pat.sub(" " + _BOUNDARY + " ", text)
    out: list[tuple[str, str]] = []
    para: list[str] = []

    def flush():
        if para:
            out.append(("prose", " ".join(para)))
            para.clear()
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            flush()
            continue
        if s.startswith("#"):
            flush()
            out.append(("heading", s.lstrip("#").strip()))
        elif s.startswith("|"):
            flush()
            if set(s) <= set("|-: "):
                continue
            out.extend(("cell", c.strip()) for c in s.strip("|").split("|") if c.strip())
        else:
            para.append(s)
    flush()
    return out


def _sentences(s: str) -> list[str]:
    for a in _NO_SPLIT:
        s = s.replace(a, a.replace(".", "\x01"))
    return [p.replace("\x01", ".") for p in _SENT_SPLIT.split(s) if p.strip()]


def tokenize_text(text: str, hyphen_split: bool = False) -> list[tuple[str, list[tuple[str, str, int, bool]]]]:
    """AE2, AE8. Guarantee: [(kind, [(surface, folded, shape, sentence_initial)])], one entry per segment; no entry spans a
    sentence, math, citation or cell boundary. Hyphenated tokens stay whole (`Llama-3.1-8B`, `fine-tuning`); with
    `hyphen_split` a hyphenated token WITHOUT a digit is split into its parts. A trailing possessive 's is dropped.
    camelCase is never split."""
    out = []
    for kind, string in _segments(text):
        for piece in string.split(_BOUNDARY):
            for sent in _sentences(piece):
                toks: list[tuple[str, str, int, bool]] = []
                for m in _TOKEN.finditer(sent):
                    surf = re.sub(r"['\u2019]s$", "", m.group())
                    parts = surf.split("-") if (hyphen_split and "-" in surf and not any(c.isdigit() for c in surf)) else [surf]
                    for p in parts:
                        if p:
                            toks.append((p, p.lower(), shape_of(p), not toks))
                if toks:
                    out.append((kind, toks))
    return out


@dataclass
class Corpus:
    """Token arrays for a set of chunks. `seg` changes at every sentence/math/citation/cell boundary (AE2)."""
    fold_ids: np.ndarray
    surf_ids: np.ndarray
    shape: np.ndarray
    sent_init: np.ndarray
    seg: np.ndarray
    seg_chunk: np.ndarray
    seg_paper: np.ndarray
    seg_kind: np.ndarray
    fold_vocab: list[str]
    surf_vocab: list[str]
    n_chunks: int
    n_papers: int
    papers: list


def tokenize_corpus(texts: list[str], paper_ids: list, hyphen_split: bool = False) -> Corpus:
    """Guarantee: the Corpus of `texts` (chunk i is texts[i], belonging to paper_ids[i]); papers are numbered in order of first appearance."""
    fold_ix: dict[str, int] = {}
    surf_ix: dict[str, int] = {}
    fold_vocab: list[str] = []
    surf_vocab: list[str] = []
    folds, surfs, shapes, inits, segs = array("i"), array("i"), array("b"), array("b"), array("i")
    seg_chunk, seg_paper, seg_kind = array("i"), array("i"), array("b")
    kind_code = {"prose": PROSE, "heading": HEADING, "cell": CELL}
    paper_ix: dict = {}
    for ci, (text, pid) in enumerate(zip(texts, paper_ids)):
        pi = paper_ix.setdefault(pid, len(paper_ix))
        for kind, toks in tokenize_text(text, hyphen_split):
            s = len(seg_chunk)
            seg_chunk.append(ci)
            seg_paper.append(pi)
            seg_kind.append(kind_code[kind])
            for surf, fold, shape, init in toks:
                fi = fold_ix.get(fold)
                if fi is None:
                    fi = fold_ix[fold] = len(fold_vocab)
                    fold_vocab.append(fold)
                si = surf_ix.get(surf)
                if si is None:
                    si = surf_ix[surf] = len(surf_vocab)
                    surf_vocab.append(surf)
                folds.append(fi)
                surfs.append(si)
                shapes.append(shape)
                inits.append(1 if init else 0)
                segs.append(s)
    arr = lambda a, dt: np.frombuffer(a, dtype=dt).copy() if len(a) else np.zeros(0, dt)
    return Corpus(arr(folds, np.int32), arr(surfs, np.int32), arr(shapes, np.int8), arr(inits, np.int8).astype(bool), arr(segs, np.int32),
                  arr(seg_chunk, np.int32), arr(seg_paper, np.int32), arr(seg_kind, np.int8), fold_vocab, surf_vocab,
                  len(texts), len(paper_ix), list(paper_ix))


# ------------------------------------------------------------------------------------------------ cipher
def paper_cross_entropy(c: Corpus) -> tuple[np.ndarray, np.ndarray]:
    """AE3. Guarantee: (bits per token under a unigram model that LEAVES THE PAPER OUT, tokens per paper). Leaving the paper out is
    what makes a mis-extracted paper stand out: its own shifted tokens are unseen anywhere else."""
    V, P = len(c.fold_vocab), c.n_papers
    tok_paper = c.seg_paper[c.seg].astype(np.int64)
    ids = c.fold_ids.astype(np.int64)
    total = np.bincount(ids, minlength=V).astype(np.float64)
    n_tok = np.bincount(tok_paper, minlength=P).astype(np.float64)
    N = float(len(ids))
    ce = np.zeros(P)
    order = np.argsort(tok_paper, kind="stable")
    bounds = np.searchsorted(tok_paper[order], np.arange(P + 1))
    for p in range(P):
        sel = ids[order[bounds[p]:bounds[p + 1]]]
        if len(sel) == 0:
            continue
        u, k = np.unique(sel, return_counts=True)
        prob = (total[u] - k + 1.0) / (N - len(sel) + V)
        ce[p] = float(-(k * np.log2(prob)).sum() / len(sel))
    return ce, n_tok


CIPHER_BINS, CIPHER_MIN_PEERS = 8, 20


def cipher_outliers(c: Corpus, z_cut: float = CIPHER_Z, min_tokens: int = CIPHER_MIN_TOKENS) -> np.ndarray:
    """AE3. Guarantee: bool per paper, True where its leave-one-out cross-entropy is more than z_cut robust z-scores (median, 1.4826 MAD)
    above its PEERS: the papers of similar length (up to CIPHER_BINS equal-count bins by token count, each with at least CIPHER_MIN_PEERS
    papers). Per-token surprisal of a short paper is higher than a long one's for ordinary English, so a corpus-wide z flagged the short
    `_methods` extracts (measured live: 55 of 62 flagged, 59 of 62 ordinary English). Papers under min_tokens are never flagged."""
    ce, n_tok = paper_cross_entropy(c)
    flags = np.zeros(c.n_papers, bool)
    idx = np.flatnonzero(n_tok >= min_tokens)
    if len(idx) < 3:
        return flags
    order = idx[np.argsort(n_tok[idx], kind="stable")]
    for peers in np.array_split(order, max(1, min(CIPHER_BINS, len(order) // CIPHER_MIN_PEERS))):
        med = np.median(ce[peers])
        mad = 1.4826 * np.median(np.abs(ce[peers] - med))
        flags[peers[(ce[peers] - med) / (mad if mad > 0 else 1.0) > z_cut]] = True
    return flags


# -------------------------------------------------------------------------------------------- n-gram counts
@dataclass
class NgramTable:
    """Grams of one length n that survived the support floor. `pos_gid[i]` is the gram that STARTS at token i (-1 if none)."""
    n: int
    start: np.ndarray
    count: np.ndarray
    df_papers: np.ndarray
    df_chunks: np.ndarray
    pos_gid: np.ndarray
    n_windows: int


def _df(group_of_pos: np.ndarray, gid: np.ndarray, G: int) -> np.ndarray:
    """Guarantee: for each of the G grams, how many distinct `group_of_pos` values contain it."""
    pair = np.unique(group_of_pos.astype(np.int64) * G + gid)
    return np.bincount(pair % G, minlength=G)


def count_ngrams(c: Corpus, max_n: int = MAX_N, min_sup: int = 3, exclude_papers: np.ndarray | None = None) -> list[NgramTable]:
    """AE1, AE2, AE3. Guarantee: tables[n-1] for n = 1..max_n. A gram needs `min_sup` occurrences (apriori: it is only extended
    from a surviving (n-1)-gram and a surviving last token, so n=4 keys never overflow int64: key = dense(n-1)-gram id * G1 + dense
    unigram id). A window never crosses a segment boundary, and tokens of excluded papers are not counted."""
    T = len(c.fold_ids)
    seg = c.seg
    tok_paper = c.seg_paper[seg] if T else np.zeros(0, np.int32)
    tok_chunk = c.seg_chunk[seg] if T else np.zeros(0, np.int32)
    live = np.ones(T, bool) if exclude_papers is None else ~exclude_papers[tok_paper]
    V = len(c.fold_vocab)
    ids = c.fold_ids.astype(np.int64)
    cnt1 = np.bincount(ids[live], minlength=V)
    keep1 = cnt1 >= min_sup
    dense1 = -np.ones(V, np.int64)
    dense1[keep1] = np.arange(int(keep1.sum()))
    uni = np.where(live, dense1[ids], -1) if T else np.zeros(0, np.int64)
    G1 = int(keep1.sum())
    tables: list[NgramTable] = []

    def finish(n, gid, start_pos_per_gram, n_windows):
        G = len(start_pos_per_gram)
        pos = np.flatnonzero(gid >= 0)
        g = gid[pos]
        tables.append(NgramTable(n, start_pos_per_gram, np.bincount(g, minlength=G), _df(tok_paper[pos], g, G), _df(tok_chunk[pos], g, G),
                                 gid.astype(np.int32), n_windows))

    pos1 = np.flatnonzero(uni >= 0)
    first = np.zeros(G1, np.int64)
    if G1:
        order = np.argsort(uni[pos1], kind="stable")
        first_idx = np.searchsorted(uni[pos1][order], np.arange(G1))
        first = pos1[order][first_idx]
    finish(1, uni.astype(np.int64), first, int(live.sum()))
    prev = uni.astype(np.int64)
    for n in range(2, max_n + 1):
        gid = -np.ones(T, np.int64)
        if T >= n:
            i = np.arange(T - n + 1)
            ok = (prev[i] >= 0) & (uni[i + n - 1] >= 0) & (seg[i] == seg[i + n - 1])
            cand = i[ok]
            n_windows = int(((seg[i] == seg[i + n - 1]) & live[i] & live[i + n - 1]).sum())
        else:
            cand, n_windows = np.zeros(0, np.int64), 0
        if len(cand) == 0:
            finish(n, gid, np.zeros(0, np.int64), n_windows)
            prev = gid
            continue
        keys = prev[cand] * max(G1, 1) + uni[cand + n - 1]
        uniq, first_idx, inv, cnt = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
        survive = cnt >= min_sup
        dense = -np.ones(len(uniq), np.int64)
        dense[survive] = np.arange(int(survive.sum()))
        gid[cand] = dense[inv]
        finish(n, gid, cand[first_idx[survive]], n_windows)
        prev = gid
    return tables


# --------------------------------------------------------------------------------------------- statistics
def npmi_vec(c_xy, c_x, c_y, n_xy, n_x, n_y):
    """Guarantee: (npmi, ppmi) arrays. npmi = pmi / -log p(x,y), ppmi = max(pmi, 0). Where p(x,y) == 1 the result is (0, 0), the
    incumbent's degenerate rule (entities.npmi_ppmi returns (0.0, 0.0) when joint >= n). With equal sample sizes this equals
    entities.npmi_ppmi element by element: tests/test_entity_derive.py pins that."""
    p_xy = np.asarray(c_xy, np.float64) / n_xy
    p_x = np.asarray(c_x, np.float64) / n_x
    p_y = np.asarray(c_y, np.float64) / n_y
    pmi = np.log(p_xy / (p_x * p_y))
    denom = -np.log(p_xy)
    npmi = np.where(denom > 0, pmi / np.where(denom > 0, denom, 1.0), 0.0)
    ppmi = np.where(denom > 0, np.maximum(pmi, 0.0), 0.0)
    return npmi, ppmi


def g2_2x2(a, b, c, d):
    """Guarantee: Dunning's log-likelihood ratio G2 of the 2x2 table [[a, b], [c, d]] (arrays of counts), 0 where a cell is empty
    in both margins; chi-square(1) significance at p<0.001 is 10.83."""
    a, b, c, d = (np.asarray(v, np.float64) for v in (a, b, c, d))
    n = a + b + c + d

    def term(o, r, k):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(o > 0, o * np.log(o * n / (r * k)), 0.0)
    return 2.0 * (term(a, a + b, a + c) + term(b, a + b, b + d) + term(c, c + d, a + c) + term(d, c + d, b + d))


def entropy_by_group(group: np.ndarray, symbol: np.ndarray, G: int) -> np.ndarray:
    """Guarantee: Shannon entropy in bits of the distribution of `symbol` within each of the G groups (one row per occurrence)."""
    out = np.zeros(G)
    if len(group) == 0:
        return out
    order = np.lexsort((symbol, group))
    g, s = group[order], symbol[order]
    new = np.r_[True, (g[1:] != g[:-1]) | (s[1:] != s[:-1])]
    run_g = g[new]
    run_c = np.diff(np.r_[np.flatnonzero(new), len(g)]).astype(np.float64)
    tot = np.bincount(g, minlength=G).astype(np.float64)
    p = run_c / tot[run_g]
    return np.bincount(run_g, weights=-p * np.log2(p), minlength=G)


@dataclass
class Features:
    """Flat candidate table across n = 1..max_n; row r is gram (n[r], gram[r]) whose first occurrence starts at token start[r]."""
    n: np.ndarray
    gram: np.ndarray
    start: np.ndarray            # first occurrence: the token ids that spell the gram
    modal: np.ndarray            # first occurrence of its most frequent spelling: the surface form
    tf: np.ndarray
    df_papers: np.ndarray
    df_chunks: np.ndarray
    npmi_min: np.ndarray
    g2_min: np.ndarray
    h_left: np.ndarray
    h_right: np.ndarray
    h_var: np.ndarray
    p_cap: np.ndarray
    cap_den: np.ndarray
    ridf: np.ndarray
    nest: np.ndarray
    max_tok_len: np.ndarray      # characters in the gram's longest token (AE15)
    df_first: np.ndarray         # paper df of the gram's first token
    pcap_first: np.ndarray       # capitalised share (non-initial prose) of the gram's first token
    df_last: np.ndarray
    pcap_last: np.ndarray
    n_papers: int = 0


def _hash_variants(c: Corpus, pos: np.ndarray, n: int) -> np.ndarray:
    h = c.surf_ids[pos].astype(np.uint64)
    with np.errstate(over="ignore"):
        for k in range(1, n):
            h = h * np.uint64(1000003) + c.surf_ids[pos + k].astype(np.uint64)
    return h.astype(np.int64)


def compute_features(c: Corpus, tables: list[NgramTable]) -> Features:
    """AE5-AE8. Guarantee: every feature for every surviving gram; the capitalised share uses running prose only (AE8)."""
    T = len(c.fold_ids)
    seg = c.seg
    D = max(c.n_papers, 1)
    prose = (c.seg_kind[seg] == PROSE) if T else np.zeros(0, bool)
    capish = np.isin(c.shape, CAPITALISH)
    uni = tables[0]
    G1 = len(uni.count)
    # closed-class tokens (AE6), derived from the corpus: ubiquitous across papers and never capitalised mid-sentence
    pos_all = np.flatnonzero(uni.pos_gid >= 0)
    g_all = uni.pos_gid[pos_all].astype(np.int64)
    mask = prose[pos_all] & ~c.sent_init[pos_all]
    den = np.bincount(g_all[mask], minlength=G1)
    num = np.bincount(g_all[mask & capish[pos_all]], minlength=G1)
    pcap1 = np.where(den >= CLOSED_MIN_DEN, num / np.maximum(den, 1), 1.0)
    tok_len = np.array([len(w) for w in c.fold_vocab], np.int64)
    cols = {k: [] for k in ("n", "gram", "start", "modal", "tf", "dfp", "dfc", "npmi", "g2", "hl", "hr", "hv", "pcap", "cden", "ridf", "nest",
                            "mtl", "dfa", "pca", "dfz", "pcz")}
    for t in tables:
        n, G = t.n, len(t.count)
        if G == 0:
            continue
        pos = np.flatnonzero(t.pos_gid >= 0)
        g = t.pos_gid[pos].astype(np.int64)
        # boundary entropy: the distribution of the token left and right of the gram, a segment edge being its own symbol
        left = np.where((pos > 0) & (seg[np.maximum(pos - 1, 0)] == seg[pos]), c.fold_ids[np.maximum(pos - 1, 0)].astype(np.int64), -1)
        r_ix = np.minimum(pos + n, T - 1)
        right = np.where((pos + n < T) & (seg[r_ix] == seg[pos]), c.fold_ids[r_ix].astype(np.int64), -1)
        hl, hr = entropy_by_group(g, left, G), entropy_by_group(g, right, G)
        # form signature: entropy of the spelling variants, and the share of non-initial prose occurrences that are capitalised throughout
        var = _hash_variants(c, pos, n)
        hv = entropy_by_group(g, var, G)
        all_cap = capish[pos]
        for k in range(1, n):
            all_cap = all_cap & capish[pos + k]
        m = prose[pos] & ~c.sent_init[pos]
        cden = np.bincount(g[m], minlength=G)
        pcap = np.bincount(g[m & all_cap], minlength=G) / np.maximum(cden, 1)
        # modal surface form per gram
        order = np.lexsort((var, g))
        gs, vs = g[order], var[order]
        newrun = np.r_[True, (gs[1:] != gs[:-1]) | (vs[1:] != vs[:-1])]
        run_start = np.flatnonzero(newrun)
        run_cnt = np.diff(np.r_[run_start, len(gs)])
        run_g = gs[run_start]
        best = np.lexsort((-run_cnt, run_g))
        first_of_g = np.r_[True, run_g[best][1:] != run_g[best][:-1]]
        modal_pos = pos[order][run_start[best][first_of_g]]
        # cohesion: the weakest split of the gram (Bouma NPMI; Dunning G2), n >= 2
        npmi_min = np.zeros(G)
        g2_min = np.zeros(G)
        if n >= 2:
            npmi_min[:] = np.inf
            g2_min[:] = np.inf
            for k in range(1, n):
                lt, rt = tables[k - 1], tables[n - k - 1]
                lid = lt.pos_gid[t.start].astype(np.int64)
                rid = rt.pos_gid[t.start + k].astype(np.int64)
                assert (lid >= 0).all() and (rid >= 0).all(), "apriori invariant broken"
                cx, cy = lt.count[lid], rt.count[rid]
                npm, _ = npmi_vec(t.count, cx, cy, max(t.n_windows, 1), max(lt.n_windows, 1), max(rt.n_windows, 1))
                a = t.count.astype(np.float64)
                gg = g2_2x2(a, np.maximum(cx - a, 0), np.maximum(cy - a, 0), np.maximum(t.n_windows - cx - cy + a, 0))
                npmi_min = np.minimum(npmi_min, npm)
                g2_min = np.minimum(g2_min, gg)
        # nesting (AE7): the largest share of this gram's occurrences taken by one extension that has it as prefix or suffix
        nest = np.zeros(G)
        if n < len(tables) and len(tables[n].count):
            up = tables[n]
            for ref in (up.start, up.start + 1):                       # the extension's prefix gram, then its suffix gram
                sub = t.pos_gid[ref].astype(np.int64)
                okm = sub >= 0
                np.maximum.at(nest, sub[okm], up.count[okm].astype(np.float64))
            nest = np.minimum(nest / np.maximum(t.count, 1), 1.0)
        # burstiness: residual IDF over papers (Church & Gale): observed IDF minus the Poisson-expected IDF
        ridf = -np.log2(np.maximum(t.df_papers, 1) / D) + np.log2(1 - np.exp(-t.count / D))
        # the document frequency and capitalised share of its first and last token, so AE6 can be applied (and swept) at scoring time
        first_g = uni.pos_gid[t.start].astype(np.int64)
        last_g = uni.pos_gid[t.start + n - 1].astype(np.int64)
        cols["n"].append(np.full(G, n)); cols["gram"].append(np.arange(G)); cols["start"].append(t.start)
        cols["tf"].append(t.count); cols["dfp"].append(t.df_papers); cols["dfc"].append(t.df_chunks)
        cols["npmi"].append(npmi_min); cols["g2"].append(g2_min); cols["hl"].append(hl); cols["hr"].append(hr)
        cols["hv"].append(hv); cols["pcap"].append(pcap); cols["cden"].append(cden); cols["ridf"].append(ridf)
        cols["nest"].append(nest); cols["modal"].append(modal_pos)
        cols["mtl"].append(np.max([tok_len[c.fold_ids[t.start + k]] for k in range(n)], axis=0))
        cols["dfa"].append(uni.df_papers[first_g]); cols["pca"].append(pcap1[first_g])
        cols["dfz"].append(uni.df_papers[last_g]); cols["pcz"].append(pcap1[last_g])
    cat = lambda k, dt=None: (np.concatenate(cols[k]) if cols[k] else np.zeros(0)).astype(dt) if dt else (np.concatenate(cols[k]) if cols[k] else np.zeros(0))
    return Features(cat("n", np.int8), cat("gram", np.int64), cat("start", np.int64), cat("modal", np.int64), cat("tf", np.int64),
                    cat("dfp", np.int64), cat("dfc", np.int64), cat("npmi"), cat("g2"), cat("hl"), cat("hr"), cat("hv"), cat("pcap"),
                    cat("cden", np.int64), cat("ridf"), cat("nest"), cat("mtl", np.int64), cat("dfa", np.int64), cat("pca"),
                    cat("dfz", np.int64), cat("pcz"), c.n_papers)


# --------------------------------------------------------------------------------------------------- score
@dataclass(frozen=True)
class Params:
    """The swept axes. `rule` is how the per-feature mid-ranks combine; `uni_quota` is the share of the budget given to the
    unigram track (None = one merged ranking); `drop` names features removed (ablation)."""
    df_floor: int = 3
    rule: str = "geomean"          # geomean | min | meanrank | cohesion
    uni_quota: float | None = 0.6
    nest_demote: bool = True
    closed_df_frac: float = CLOSED_DF_FRAC     # AE6: a token in more than this share of papers and almost never capitalised is closed-class
    ridf_floor: float = RIDF_FLOOR             # AE14: a candidate below this residual IDF is vocabulary
    min_tok_len: int = 2                       # AE15: a gram whose longest token is shorter than this is math notation (1 = off)
    universal_df: float = 0.97                 # AE6b: a token in more than this share of papers carries no identity, however it is capitalised (>= 1.0 = off)
    drop: frozenset = frozenset()


MULTI_FEATURES = ("npmi", "g2", "bh", "ridf", "cap")
UNI_FEATURES = ("cap", "ridf", "bh")


def mid_rank(x: np.ndarray) -> np.ndarray:
    """Guarantee: average rank / N, in (0, 1]; ties share a rank, so a feature piled up at one value cannot zero a geometric mean."""
    return rankdata(x, method="average") / max(len(x), 1) if len(x) else np.zeros(0)


def score(f: Features, p: Params = Params()) -> np.ndarray:
    """AE1, AE5, AE6, AE7. Guarantee: one score per candidate, in [0, 1]; 0 means excluded. Multiword and unigram candidates are
    ranked within their own track (cohesion is undefined for one token). PPMI/NPMI is a demotion floor (AE5), never a weight."""
    s = np.zeros(len(f.n))
    eligible = f.df_papers >= p.df_floor                                              # AE1
    uni = f.n == 1
    D = max(f.n_papers, 1)
    closed_first = ((f.df_first > p.closed_df_frac * D) & (f.pcap_first < CLOSED_P_CAP)) | (f.df_first > p.universal_df * D)
    closed_last = ((f.df_last > p.closed_df_frac * D) & (f.pcap_last < CLOSED_P_CAP)) | (f.df_last > p.universal_df * D)
    eligible &= ~(closed_first | closed_last)                                         # AE6: a closed-class unigram, or a gram edged by one
    eligible &= uni | (f.npmi_min > 0)                                                # AE5
    eligible &= f.ridf >= p.ridf_floor                                                # AE14
    eligible &= f.max_tok_len >= p.min_tok_len                                        # AE15
    if p.nest_demote:
        eligible &= f.nest < NEST_FRAC                                                # AE7
    bh = np.minimum(f.h_left, f.h_right)
    cap = f.p_cap * np.exp2(-f.h_var)                                                 # rigid AND capitalised mid-sentence
    for track_mask, names in ((~uni, MULTI_FEATURES), (uni, UNI_FEATURES)):
        idx = np.flatnonzero(track_mask & eligible)
        if len(idx) == 0:
            continue
        raw = {"npmi": f.npmi_min, "g2": np.log1p(f.g2_min), "bh": bh, "ridf": f.ridf, "cap": cap}
        used = [k for k in names if k not in p.drop]
        if p.rule == "cohesion":
            used = ["npmi"] if (track_mask is not uni and "npmi" in used) else (["cap"] if "cap" in used else used[:1])
        if not used:
            continue
        ranks = np.vstack([mid_rank(raw[k][idx]) for k in used])
        if p.rule == "min":
            s[idx] = ranks.min(axis=0)
        elif p.rule == "meanrank":
            s[idx] = ranks.mean(axis=0)
        else:                                                                         # geomean, and cohesion over its single feature
            s[idx] = np.exp(np.log(ranks).mean(axis=0))
    return s


def select(f: Features, s: np.ndarray, k: int, p: Params = Params()) -> np.ndarray:
    """AE4. Guarantee: indices of the top-k scoring candidates (score > 0), best first, ties broken by row index so the result is
    deterministic. With a unigram quota, ceil(k * quota) come from the unigram track (spare slots pass to the other track)."""
    live = np.flatnonzero(s > 0)
    order = live[np.lexsort((live, -s[live]))]
    if p.uni_quota is None:
        return order[:k]
    uni = order[f.n[order] == 1]
    multi = order[f.n[order] != 1]
    want_u = min(int(math.ceil(k * p.uni_quota)), len(uni))
    want_m = min(k - want_u, len(multi))
    want_u = min(k - want_m, len(uni))
    pick = np.r_[uni[:want_u], multi[:want_m]]
    return pick[np.lexsort((pick, -s[pick]))]


# --------------------------------------------------------------------------------------------- acronyms
def _sh_long_form(short: str, long_text: str) -> str | None:
    """Schwartz & Hearst 2003: walk the short form right to left, matching each character in the long form; the first character of
    the short form must start a word. Returns the matching long form, or None."""
    s_i, l_i = len(short) - 1, len(long_text) - 1
    while s_i >= 0:
        ch = short[s_i].lower()
        if not ch.isalnum():
            s_i -= 1
            continue
        while (l_i >= 0 and long_text[l_i].lower() != ch) or (s_i == 0 and l_i > 0 and long_text[l_i - 1].isalnum()):
            l_i -= 1
        if l_i < 0:
            return None
        l_i -= 1
        s_i -= 1
    return long_text[long_text.rfind(" ", 0, l_i + 1) + 1:]


def acronym_pairs(sentences: list[str]) -> list[tuple[str, str]]:
    """AE9. Guarantee: [(abbreviation, expansion)] from "expansion (ABBR)" patterns that satisfy the Schwartz-Hearst initials rule
    (the abbreviation has a letter, at most two words, and the expansion is longer and at most min(|a|+5, 2|a|) words); a plural
    abbreviation (LLMs) is tried singular as well. Rule-based, no model. "(see Fig. 2)" and "(Smith et al., 2020)" give nothing."""
    out = []
    for sent in sentences:
        for m in re.finditer(r"\(([^()\s][^()]{0,14})\)", sent):
            short = m.group(1).strip()
            if len(short) < 2 or not any(ch.isalpha() for ch in short) or len(short.split()) > 2 or not short[0].isalnum():
                continue
            if not any(ch.isupper() or ch.isdigit() for ch in short):
                continue
            words = sent[:m.start()].split()
            for cand in (short, short[:-1] if short.endswith("s") and short[:-1].isupper() else None):
                if not cand or len(cand) < 2:
                    continue
                maxw = min(len(cand) + 5, len(cand) * 2)
                long_text = " ".join(words[-maxw:])
                lf = _sh_long_form(cand, long_text)
                if lf and len(lf) > len(cand) and len(lf.split()) <= maxw:
                    out.append((short, lf))
                    break
    return out


def alias_groups(fold: list[str], pairs: list[tuple[str, str]]) -> np.ndarray:
    """AE9. Guarantee: canonical[i] for each inventory entry i: the entry itself, or (when both an acronym and its expansion are
    entries) the acronym's index, so both spellings count as one entity. Union-find, lowest index wins ties."""
    ix = {f: i for i, f in enumerate(fold)}
    parent = list(range(len(fold)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for abbr, exp in pairs:
        a, e = ix.get(abbr.lower()), ix.get(" ".join(t.lower() for t in re.findall(_TOKEN, exp)))
        if a is not None and e is not None:
            ra, re_ = find(a), find(e)
            if ra != re_:
                lo, hi = (ra, re_) if ra < re_ else (re_, ra)
                parent[hi] = lo
    return np.array([find(i) for i in range(len(fold))], np.int64)


# ---------------------------------------------------------------------------------------- frozen matching
@dataclass
class Inventory:
    """A frozen entity inventory: what matching needs (folded token sequences) and what was measured when it was derived."""
    fold: list[str]
    surface: list[str]
    n: np.ndarray
    score: np.ndarray
    df_papers: np.ndarray
    df_chunks: np.ndarray
    tf: np.ndarray
    canonical: np.ndarray
    tokenizer_version: str = TOKENIZER_VERSION


def inventory_from(c: Corpus, f: Features, s: np.ndarray, idx: np.ndarray, canonical: np.ndarray | None = None) -> Inventory:
    """Guarantee: the Inventory of the selected candidates, in `idx` order (best first); their spellings are built here, not for
    every candidate."""
    spell = lambda vocab, ids, p0, n: " ".join(vocab[ids[p0 + k]] for k in range(n))
    fold = [spell(c.fold_vocab, c.fold_ids, int(f.start[i]), int(f.n[i])) for i in idx]
    surface = [spell(c.surf_vocab, c.surf_ids, int(f.modal[i]), int(f.n[i])) for i in idx]
    return Inventory(fold, surface, f.n[idx].astype(np.int64), s[idx], f.df_papers[idx], f.df_chunks[idx], f.tf[idx],
                     np.arange(len(idx), dtype=np.int64) if canonical is None else canonical)


_MULT = np.uint64(1000003)


def _roll(ids: np.ndarray, n: int, i: np.ndarray) -> np.ndarray:
    h = ids[i].astype(np.uint64) + np.uint64(1)
    with np.errstate(over="ignore"):
        for k in range(1, n):
            h = h * _MULT + (ids[i + k].astype(np.uint64) + np.uint64(1))
    return h


def match_spans(inv: Inventory, c: Corpus) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AE11, AE13. Guarantee: (pos, entity, n) of every match in `c`, in ascending token position -- found longest gram first, left to
    right, with no token used twice. Only the inventory's own tokens are known, so a token it never saw cannot match (AE13); no
    statistic is read or changed (AE11)."""
    vocab = {}
    for e in inv.fold:
        for tok in e.split(" "):
            vocab.setdefault(tok, len(vocab))
    mp = np.array([vocab.get(f, -1) for f in c.fold_vocab], np.int64)
    ids = mp[c.fold_ids] if len(c.fold_ids) else np.zeros(0, np.int64)
    T = len(ids)
    taken = np.zeros(T, bool)
    rows = []
    for n in sorted({int(x) for x in inv.n}, reverse=True):
        entries = np.flatnonzero(inv.n == n)
        keys = np.array([_roll(np.array([vocab[t] for t in inv.fold[e].split(" ")], np.int64), n, np.array([0]))[0] for e in entries], np.uint64)
        order = np.argsort(keys, kind="stable")
        skeys, sent = keys[order], entries[order]
        if T < n:
            continue
        i = np.arange(T - n + 1)
        win_ok = (c.seg[i] == c.seg[i + n - 1])
        for k in range(n):
            win_ok &= ids[i + k] >= 0
        cand = i[win_ok]
        if len(cand) == 0:
            continue
        h = _roll(ids, n, cand)
        loc = np.searchsorted(skeys, h)
        loc[loc >= len(skeys)] = 0
        hit = skeys[loc] == h
        for pos, ent in zip(cand[hit], sent[loc[hit]]):                      # ascending position: leftmost first
            if not taken[pos:pos + n].any():
                taken[pos:pos + n] = True
                rows.append((int(pos), int(ent), n))
    if not rows:
        z = np.zeros(0, np.int64)
        return z, z, z
    arr = np.array(sorted(rows), np.int64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def match_frozen(inv: Inventory, c: Corpus) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AE11, AE13. Guarantee: (chunk, entity, count) rows -- how often each inventory entry occurs in each chunk of `c` -- from
    match_spans."""
    pos, ent, _ = match_spans(inv, c)
    if len(pos) == 0:
        return pos, ent, pos
    key = c.seg_chunk[c.seg[pos]].astype(np.int64) * (len(inv.fold) + 1) + ent
    uk, cnt = np.unique(key, return_counts=True)
    return uk // (len(inv.fold) + 1), uk % (len(inv.fold) + 1), cnt


def derive(texts: list[str], paper_ids: list, params: Params = Params(), min_sup: int = 3, max_n: int = MAX_N, k: int = 5000,
           hyphen_split: bool = False) -> tuple[Inventory, Features, np.ndarray, Corpus, np.ndarray]:
    """AE1-AE9. Guarantee: (inventory, features, score, corpus, cipher_flags): tokenize, drop cipher papers from the counts, count,
    compute features, score, select the top k. Deterministic (AE4)."""
    c = tokenize_corpus(texts, paper_ids, hyphen_split)
    flags = cipher_outliers(c)
    f = compute_features(c, count_ngrams(c, max_n, min_sup, exclude_papers=flags))
    s = score(f, params)
    idx = select(f, s, k, params)
    return inventory_from(c, f, s, idx), f, s, c, flags


# ------------------------------------------------------------------------------------------------ types
TYPE_SEED_ARI = 0.8                              # AE10
TYPE_MIN_OCC = 5                                 # an entity with fewer occurrences has no context evidence: class -1
CTX_OFFSETS = (-2, -1, 1, 2)                     # tokens read around a match, inside one segment
SPELL_DIM = 2048


def _l2(X: "sp.csr_matrix") -> "sp.csr_matrix":
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    return (sp.diags(1.0 / np.maximum(nrm, 1e-12)) @ X).tocsr()


def context_vectors(inv: Inventory, c: Corpus, min_occ: int = TYPE_MIN_OCC) -> tuple["sp.csr_matrix", np.ndarray]:
    """AE10. Guarantee: (X, occ). X is (entities, 4 * vocabulary): PPMI of each entity against the token at each of CTX_OFFSETS from
    its matches (context frequencies smoothed by ^0.75, Levy et al. 2015), L2 rows; PPMI is the weight, the demotion floor of the
    operator's standing rule. An entity with fewer than `min_occ` matches has a zero row. `occ` counts matches per entity."""
    pos, ent, n = match_spans(inv, c)
    E, V, T = len(inv.fold), len(c.fold_vocab), len(c.fold_ids)
    occ = np.bincount(ent, minlength=E)
    rows, cols = [], []
    for slot, off in enumerate(CTX_OFFSETS):
        q = pos + off if off < 0 else pos + n - 1 + off
        ok = (q >= 0) & (q < T)
        sel = np.flatnonzero(ok)
        sel = sel[c.seg[q[sel]] == c.seg[pos[sel]]]
        rows.append(ent[sel])
        cols.append(slot * V + c.fold_ids[q[sel]].astype(np.int64))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    C = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(E, len(CTX_OFFSETS) * V)).tocsr().tocoo()
    tot = C.data.sum()
    pe = np.asarray(sp.coo_matrix(C).tocsr().sum(1)).ravel() / tot
    pc = np.asarray(sp.coo_matrix(C).tocsr().sum(0)).ravel() ** 0.75
    pc = pc / pc.sum()
    pmi = np.log((C.data / tot) / (pe[C.row] * pc[C.col]))
    keep = (pmi > 0) & (occ[C.row] >= min_occ)
    X = sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)
    return _l2(X), occ


def spelling_vectors(inv: Inventory) -> "sp.csr_matrix":
    """AE10. Guarantee: (entities, SPELL_DIM) L2 rows of hashed character trigrams of the surface form (case kept, so GRPO and grpo
    differ) plus shape flags. zlib.crc32 keeps the hash identical across runs (AE4)."""
    import zlib
    rows, cols = [], []
    for i, surf in enumerate(inv.surface):
        feats = ["c:" + ("^" + surf + "$")[j:j + 3] for j in range(max(len(surf) - 1, 1))]
        if surf.isupper():
            feats.append("f:allcaps")
        if any(ch.isdigit() for ch in surf):
            feats.append("f:digit")
        if "-" in surf:
            feats.append("f:hyphen")
        if " " in surf:
            feats.append("f:multiword")
        if surf[:1].isupper() and not surf.isupper():
            feats.append("f:title")
        for ft in feats:
            rows.append(i)
            cols.append(zlib.crc32(ft.encode()) % SPELL_DIM)
    X = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(inv.surface), SPELL_DIM)).tocsr()
    return _l2(X)


def _leiden(G: "sp.csr_matrix", res: float, seed: int) -> np.ndarray:
    import igraph
    import leidenalg
    T = sp.triu(G, 1).tocoo()
    g = igraph.Graph(n=G.shape[0], edges=list(zip(T.row.tolist(), T.col.tolist())))
    g.es["weight"] = T.data.tolist()
    return np.array(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                             resolution_parameter=float(res), seed=seed).membership)


def type_entities(inv: Inventory, c: Corpus, k_nn: int = 15, resolutions=(0.5, 1.0, 2.0, 4.0), seeds=range(8),
                  w_ctx: float = 1.0, w_spell: float = 0.5, seed_ari: float = TYPE_SEED_ARI, min_occ: int = TYPE_MIN_OCC) -> dict:
    """AE10. Guarantee: {classes, tried, resolution, ari}. Entities are typed by their contexts (PPMI over the tokens beside each match)
    and their spelling; a cosine kNN graph over the entities that have at least `min_occ` matches is partitioned by Leiden at each
    resolution over `seeds`. The FINEST resolution whose mean pairwise seed-ARI is >= `seed_ari` is taken; when none passes, every class
    is -1 (a partition that does not repeat is not a type). An entity without enough matches is -1. `tried` lists (resolution, classes, ARI)."""
    from sklearn.metrics import adjusted_rand_score
    Xc, occ = context_vectors(inv, c, min_occ)
    X = _l2(sp.hstack([Xc * w_ctx, spelling_vectors(inv) * w_spell]).tocsr())
    use = np.flatnonzero(occ >= min_occ)
    classes = -np.ones(len(inv.fold), np.int64)
    out = {"classes": classes, "tried": [], "resolution": None, "ari": None}
    if len(use) < 3:
        return out
    S = (X[use] @ X[use].T).toarray()
    np.fill_diagonal(S, 0.0)
    k = min(k_nn, len(use) - 1)
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    r = np.repeat(np.arange(len(use)), k)
    G = sp.csr_matrix((np.take_along_axis(S, top, 1).ravel(), (r, top.ravel())), shape=S.shape)
    G.data = np.maximum(G.data, 0.0)
    G.eliminate_zeros()
    G = G.maximum(G.T)
    best = None
    for res in resolutions:
        labs = [_leiden(G, res, s) for s in seeds]
        ari = float(np.mean([adjusted_rand_score(labs[a], labs[b]) for a in range(len(labs)) for b in range(a + 1, len(labs))]))
        ncl = int(labs[0].max() + 1)
        out["tried"].append((res, ncl, ari))
        if ari >= seed_ari and (best is None or ncl > best[1]):
            best = (res, ncl, ari, labs[0])
    if best is not None:
        classes[use] = best[3]
        out["resolution"], out["ari"] = best[0], best[2]
    return out
