"""Salient-term vocabulary selection for compact Postgres trigram indexes.

Production entry point: select_vocab_unified() (single-pass composite objective).
Legacy entry point: select_vocab() (per-mode BM25+MMR).

Depends on: scipy (stats, sparse, interpolate), numpy, nltk.stem (optional, for stemming),
wordfreq (optional, only for background='wordfreq')

PIPELINE (canonical spec — stages in execution order; validated 2026-08)
========================================================================
INGEST
  tokenize          alnum-aware, keeps part numbers          [mandatory] R1
  punct normalize   curly quotes, dashes -> ascii            [mandatory] R1
  divisor normalize all divisors (space, special char) ->    [mandatory] R1
                    ONE separator token; tokens stay split —
                    trigram matching handles separator
                    variance (measured 100% all variants)
  digit-run collapse separator removed INSIDE digit-bearing  [opt: typo
                    runs after divisor pass: ABC-123==abc123  tolerance] R1
                    sole justification: typo recall 100% vs
                    0% split (short fragments lack trigram
                    redundancy to absorb an error)
  length filter     box-cox -> median +/- mad_k*MAD          [mandatory]

CLEAN (opt)
  correct           multi-pass noisy-channel, bigram-        [opt, strict] R2 R3 R4
                    plausibility gate (raw count table),
                    alpha-only — digit tokens NEVER touched
  stem              snowball, BEFORE scoring, alpha-only     [opt, recommended] R3
                    (roots pool df; +4pts recall, frees ~18% of slots)

SCORE (on intact text — nothing removed yet)                              R5
  bm25              within-corpus discrimination             [mandatory]
  keyness prior     log-ratio keyness (AntConc/Sketch        [opt, recommended]
                    Engine convention) vs background: rank
                    list (Zipf) or {word:count} dict (own
                    sample via sample_background); llr flag
                    adds Dunning G2 evidence gate [opt, OFF:
                    measured 47%% vs 54%% recall — optimizes
                    distinctiveness, not query coverage];
                    utility = keyness*sqrt(df), else
                    max_bm25*sqrt(df)

ELIGIBILITY (masks applied AFTER scoring, never to scoring inputs)        R5
  df band           df_min <= df <= df_max_frac*N            [mandatory]
                    (kills corpus glue/boilerplate)
  pooled df floor   df_min vs signature-pooled df: residue-  [opt, OFF:    R8
                    pair productivity gate (>= 3 other        unmeasured]
                    cores), corpus-native, no lexicon;
                    upper band per-term
  stopwords         excluded from selection, not scoring     [opt]
  anomaly           always-caps alpha terms; alpha length    [opt] R3
                    > median + 3*1.4826*MAD; digit-exempt

SPLINE REFINEMENT (between ELIGIBILITY and SELECT)
  gamm spline       cubic spline of log(df) vs log(max_bm); [opt, recommended]
                    demotes terms below -1 std residual
                    (lower utility than expected for their
                    frequency band); spline_eligibility()

SELECT
  mmr               lamb*utility - (1-lamb)*max_corr,        [mandatory]
                    corr = Pearson on binary presence
                    (row-coverage overlap; diversity+utility);
                    static half (utility+keyness) hoisted out
                    of the loop, exclusions burned in as -inf
  waterline         chars/row budget (select_vocab_unified)  [mandatory]
                    or N terms (select_vocab legacy mode)
                    (bm25+stem hit 95% content coverage at
                    k=131 on test corpus — see dispositions.md)

SERVE
  salient column    GIN gin_trgm_ops, partial index          (DDL below)
  normalize_query   same divisor/collapse + same stem flag   [contract] R6
  levenshtein       re-rank of trigram candidates only       [opt]

TERM FILE NAMING
--------------------------------------------------------------------------
  terms0.txt = blend (default; min-rank merge of modes 1-4)
  terms1.txt = BM25+MMR
  terms2.txt = GIST (BM25+stem+MMR)
  terms3.txt = keyness+MMR
  terms4.txt = IF+MMR (isolation-forest utility variant)
All files are rank-ordered by selection order (position = importance), NOT alphabetical.

GUARDS & CONTRACTS
---------------------------------------------------------------------------
R1  The system SHALL preserve digit-bearing identifiers as single tokens
    and canonicalize them by casefolding + separator-stripping.
R2  IF a correction pass produces no changes, or max_passes is reached,
    THEN correction SHALL terminate (fixed-point guard).
R3  IF a token contains a digit, THEN correction, stemming, and anomaly
    filtering SHALL NOT touch it: edit-distance-1 neighbors of identifiers
    are distinct valid identifiers, not typos.
R4  WHERE correct is enabled, only pure-alpha unattested (df == 1) tokens
    are candidates; gate = attested context, unique winner, 2x margin.
R5  No eligibility mask SHALL alter scoring inputs; masks act on
    selection only.
R6  WHEN a query is issued, the caller SHALL pass it through
    normalize_query() with the same stem flag the index was built with —
    the stem flag is part of the index contract, not a per-query option.
R7  IF no eligible terms remain after masking, THEN select_vocab SHALL
    raise ValueError rather than emit an empty vocabulary.
R8  WHERE pooled_df is enabled, families SHALL pool only under residue
    pairs attested across >= min_productivity other cores; digit terms
    and singleton cores SHALL NOT pool; the df_max_frac ceiling stays
    per-term (pooling over the glue cutoff is the over-merge failure).
R9  WHERE no background is supplied, select_vocab_unified SHALL subtract
    no prior — keyness contributes zero and selection is BM25 + diversity.
    A prior is opt-in (background='wordfreq', a rank list, or a count
    dict); neither wordfreq nor a hardcoded word list SHALL ship as a
    silent default. Rationale: the prior encodes a register assumption
    about the corpus, and a wrong assumption is worse than none.

BACKGROUND PRIOR ALTERNATIVES
------------------------------
- None: **Default.** No prior. Keyness contributes zero; selection is BM25 +
  diversity. Correct when the corpus register is unknown or not general
  English — an ill-fitting prior misranks more than no prior does.
- wordfreq library (rspeer/wordfreq): opt-in via `background='wordfreq'`
  (= `top_n_list('en', 10000)`) — plugs into keyness()'s existing rank-Zipf
  path. Measured on an email corpus: catches that register (advise r6111,
  regards r5161, sincerely r8486; domain terms OOV). That result is
  register-specific evidence, not a general default. 10k vs 50k unmeasured.
- sample_background(): empirical prior from a held-out slice of the same
  corpus. Works today; measures deviation from YOUR distribution rather
  than general English. Best when you have a clear reference segment.
  Slice must be identified by metadata/provenance, never by content
  analysis against a reference — that's circular.

COMPLEXITY RATIONALE
--------------------
No single library does "BM25 + keyness + sparse MMR + chars/row budget +
GAMM spline eligibility" in one call. sklearn has BM25-like TF-IDF; gensim
has BM25 scoring; neither has keyness, sparse-correlation MMR, or
spline-based eligibility. The individual pieces use scipy and numpy. The
composition (this module, ~450 lines) is domain-specific glue that
cannot be replaced by a pip install.

CLOSED (rejected on evidence — do not re-litigate without new data)
-------------------------------------------------------------------
Multi-pass blend consensus as production path (replaced by unified
single-pass: same result, one BM25 pass, no redundant MMR loops);
fixed df_max_frac=0.10 upper cutoff (replaced by GAMM spline — the
spline adapts to the actual utility-vs-frequency curve rather than a
hard fraction); isolation forest over BM25 stats (score shadows log-df,
r ~= 0.9); post-selection lemmatization (measured no-op on size and recall);
wordninja (separator variance covered by canonicalize); pre-scoring
stopword removal (distorts the df/BM25 statistics scoring depends on);
ensemble of background priors (shared written-register bias reinforced,
not canceled — needs a diverse member to reopen); medoid/slice reference
machinery (no slices in this problem yet); ZCA whitening (covariance
singular at S slices << V terms; MMR owns decorrelation at selection);
markovify phrase salience (real signal, separate project — not this index);
robust-z standardized keyness as utility (measured: 83% same vocab,
df-weighted recall 51% vs 63% raw — the median shift trades frequent
moderately-key terms for rare high-z ones, shrinking query coverage;
z-scores stay valid for CROSS-background comparison, not for utility);
wordfreq rejection reversed 2026-08 (operator waived install-network
constraint), then demoted from default to opt-in (a prior encodes a
register assumption; wrong assumption ranks worse than none — R9);
hardcoded _EN_TOP fallback (no resolution past rank 100, caught zero
email platitudes; replaced by wordfreq); bare snowball-key
pooling (ungated collisions: provenance→proven, providence→provid);
WordNet derivational gate for pooling (superseded by corpus-native
signature gate — covers business jargon WordNet doesn't, one less data
dependency).

Preconditions: docs is a non-empty list[str]; background is rank-ordered
best-first. Failure modes: ValueError on empty/degenerate corpus (R7);
box-cox requires positive lengths (guaranteed by dropping empty docs).

APPLYING THIS TO OTHER PROJECTS
================================
This module is reusable for any project that needs a compact trigram index
over a text corpus in Postgres. The approach generalizes as follows:

1. ASSESS THE PROBLEM
   - You have N documents stored in Postgres (or going in).
   - You want fast fuzzy/substring search but can't afford a full trigram
     index on raw text (too large, too slow at scale).
   - Solution: select a VOCABULARY of salient terms per row, store in a
     dedicated column, index only that column with gin_trgm_ops.

2. ADAPT THE PIPELINE
   a. TOKENIZE: adjust TOKEN_RE for your domain. The default `[a-z0-9]+`
      with separator handling covers technical identifiers and natural text.
      If your corpus is code, keep more punctuation. If it's prose, this
      regex works as-is.
   b. CLEAN: stemming helps (+4pts recall, -18% vocab size) for English
      prose. Skip it for identifiers, codes, or multilingual text.
   c. SCORE: BM25 is always the base. Keyness is off unless you pass a
      background prior (background='wordfreq' for general English, a rank
      list, or sample_background from a reference segment). A prior
      suppresses its own register's noise and boosts domain terms — only
      add one when that register actually matches your corpus.
   d. ELIGIBILITY: tune df_min (2 is conservative) and df_max_frac (0.10
      kills corpus-wide glue). Enable anomaly filter for messy OCR/email
      text. Enable spline refinement when corpus is large (>10K docs).
   e. SELECT: MMR with Pearson correlation is the diversity mechanism.
      lambda=0.7 balances utility vs coverage overlap. The waterline is
      your size dial — set it by measuring index size vs query latency.
   f. SERVE: create the GIN index, wire normalize_query() into your query
      path. The stem flag is part of the index contract.

3. TUNE THE WATERLINE
   - Measure: chars_per_row * N_rows * 3 (trigram expansion) ≈ index size.
   - Our finding: 315 chars/row → ~5 GB at 13M rows. Adjust linearly.
   - Below 100 chars/row: recall degrades. Above 500: diminishing returns,
     index bloat.

4. VALIDATE
   - Run 50-100 known queries against the trigram column.
   - Measure recall@20 and p50/p95 latency.
   - Compare blend vocab vs individual modes if you want to tune weights.

5. DEPENDENCIES
   - numpy, scipy (stats, sparse, interpolate) — core math
   - nltk.stem.snowball — optional, for stemming
   - wordfreq — optional, only when background='wordfreq'
   - psycopg/sqlalchemy — for DB reads (not imported here, caller provides docs)
   - No torch, no transformers, no GPU. Runs on any Python 3.10+ environment.

6. WHAT TO COPY
   - This file (salient_grams.py) is self-contained. Drop it into any project.
   - Wire it: fetch docs from your DB, call select_vocab_unified(), write the
     returned salient column back, create the GIN index (DDL at bottom of file).
   - At query time: pass user input through normalize_query() before hitting
     the index.

MEASURED AT CHUNK GRANULARITY (appended 2026-08; does not supersede the above,
which is calibrated for whole documents)
-------------------------------------------------------------------------------
- The chars/row waterline does not bind when rows are chunks rather than
  documents. contribution = len(t)*df/N, and chunk-level df is small, so
  running_chars reached only 166-191 against budgets of 315/2000/8000 —
  all three returned the identical vocabulary. Consequence: MMR ranks but
  never excludes, so the diversity term is inert and the selection loop
  costs one sparse mat-vec per eligible term for an ordering that set()
  discards. Size chunk vocabularies with the df band and spline instead;
  short-circuit the loop when sum(len(t)*df/N) over eligible terms cannot
  reach the target.
- Feeding select_vocab_unified()['terms'] into a chunk-chunk retrieval graph
  (chunkgraph.ChunkGraph(vocab=...)): 3832 -> 1758 terms, edges +2%,
  homophily flat (0.85 -> 0.84), nonzero similarity density 0.163 -> 0.198,
  chunk coverage 1.00. Eligibility does the work; selection is near-identity.
"""
import math
import re
import numpy as np
from scipy import sparse, stats

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_/.'][a-z0-9]+)*")   # keeps XR-2201B, o'brien, 3.5mm
SEP_RE = re.compile(r"[-_/.\s']+")
PUNCT_MAP = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                           "\u2014": " ", "\u2013": " ", "\u00a0": " "})


def canonical(token: str, collapse: bool = True) -> str:
    """Canonical form shared by index-time and query-time.

    Base rule: divisors normalize to a single separator (tokens split there).
    collapse=True additionally removes separators INSIDE digit-bearing runs
    (ABC-123 == abc123) — sole justification is typo recall (100% vs 0%
    split); disable if identifier typo tolerance is out of scope.
    """
    t = token.lower()
    if any(c.isdigit() for c in t):
        return SEP_RE.sub("" if collapse else " ", t)
    return SEP_RE.sub("'", t) if "'" in t else t


def tokenize(text: str, collapse: bool = True) -> list:
    return [canonical(t, collapse)
            for t in TOKEN_RE.findall(text.translate(PUNCT_MAP).lower())]


def normalize_query(q: str, stem: bool = False, collapse: bool = True) -> str:
    """Apply to every user query before hitting the salient column.

    Require: stem AND collapse must match the flags the index was built
    with — both are part of the index contract, not per-query options.
    """
    ts = tokenize(q, collapse)
    if stem:
        from nltk.stem.snowball import SnowballStemmer
        _sb = SnowballStemmer("english")
        ts = [_sb.stem(t) if t.isalpha() else t for t in ts]
    return " ".join(ts)


def length_filter(docs: list, mad_k: float = 2.0):
    """Box-cox lengths, keep within median +/- mad_k * MAD. Returns kept docs + mask."""
    docs = [d for d in docs if d and d.strip()]
    lens = np.array([len(d) for d in docs], float)
    if len(docs) < 10 or lens.std() == 0:
        return docs, np.ones(len(docs), bool)
    bc, _ = stats.boxcox(lens)
    med, mad = np.median(bc), stats.median_abs_deviation(bc)
    mask = np.abs(bc - med) <= mad_k * max(mad, 1e-9)
    return [d for d, m in zip(docs, mask) if m], mask


def bm25_matrix(token_docs: list, k1: float = 1.5, b: float = 0.75):
    """Sparse doc x term BM25 matrix. Returns (BM, TF, terms, df)."""
    vocab, rows, cols, vals = {}, [], [], []
    for i, ts in enumerate(token_docs):
        cnt = {}
        for t in ts:
            cnt[t] = cnt.get(t, 0) + 1
        for t, c in cnt.items():
            j = vocab.setdefault(t, len(vocab))
            rows.append(i); cols.append(j); vals.append(c)
    if not vocab:
        raise ValueError("empty vocabulary after tokenization")
    N, V = len(token_docs), len(vocab)
    terms = np.array(sorted(vocab, key=vocab.get))
    TF = sparse.csr_matrix((vals, (rows, cols)), shape=(N, V))
    df = np.asarray((TF > 0).sum(0)).ravel()
    dl = np.array([len(t) for t in token_docs], float)
    avgdl = dl.mean() if dl.mean() > 0 else 1.0
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    C = TF.tocoo()
    bm = idf[C.col] * C.data * (k1 + 1) / (C.data + k1 * (1 - b + b * dl[C.row] / avgdl))
    BM = sparse.csr_matrix((bm, (C.row, C.col)), shape=(N, V))
    return BM, TF, terms, df


def sample_background(docs: list) -> dict:
    """Empirical background from a sample of documents: {word: count}.

    Use when you don't want to presume any external English distribution —
    pass a slice of your own corpus (another year, another segment, a held-out
    sample) and keyness measures deviation from THAT observed distribution.
    """
    counts = {}
    for d in docs:
        for t in tokenize(d):
            counts[t] = counts.get(t, 0) + 1
    return counts


def llr_gate(terms, TF, background: dict, min_g2: float = 6.63):
    """Dunning log-likelihood significance gate (G2 vs a COUNT background).

    Returns boolean keep-mask: True where the frequency difference is
    statistically believable (G2 >= min_g2; 6.63 ~ p<.01, 3.84 ~ p<.05).
    Convention: LLR gates for evidence, log-ratio ranks by effect size
    (Dunning 1993; AntConc / Sketch Engine practice). Require: background
    is {word: count} — rank-only lists carry no sample size to test against.
    """
    a = np.asarray(TF.sum(0)).ravel()                 # term count, corpus
    A = a.sum()
    b = np.array([background.get(t, 0) for t in terms], float)
    B = sum(background.values()) or 1
    E1, E2 = A * (a + b) / (A + B), B * (a + b) / (A + B)
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = 2 * (np.where(a > 0, a * np.log(a / E1), 0)
                  + np.where(b > 0, b * np.log(b / E2), 0))
    return g2 >= min_g2


def keyness(terms, TF, background, oov_rank: int = 30000):
    """log2 ratio of corpus frequency to background frequency.

    background: either a rank-ordered best-first word list (Zipf-from-rank,
    e.g. Google 10k) or a {word: count} dict of an observed distribution
    (e.g. from sample_background) — real counts beat rank-Zipf when you
    have them. OOV terms get a floor; digit-bearing identifiers are
    maximally un-background by construction.
    """
    counts = np.asarray(TF.sum(0)).ravel()
    total = counts.sum()
    if isinstance(background, dict):
        btot = sum(background.values()) or 1
        p_bg = np.array([(background.get(t, 0) + 0.5) / btot for t in terms])
    else:
        H = sum(1.0 / r for r in range(1, len(background) + 1))
        rank = {w: r for r, w in enumerate(background, 1)}
        p_bg = np.array([1.0 / rank.get(t, oov_rank) / H for t in terms])
    return np.log2((counts + 0.5) / total / p_bg)


def anomaly_mask(terms, kept_docs, length_k: float = 3.0, caps_frac: float = 0.9):
    """Term-level anomaly filter: eligibility mask, never a scoring input.

    Flags pure-alpha terms that are (a) predominantly ALL-CAPS in the raw
    surface text (boilerplate/shouting; >= caps_frac of occurrences), or
    (b) longer than median + length_k * 1.4826 * MAD of alpha term lengths
    (sigma-scaled upper prediction bound). Guarantee: digit-bearing terms
    are never flagged (R13 — identifiers are conventionally caps/long).
    Returns boolean array, True = keep.
    """
    upper, total = {}, {}
    for d in kept_docs:
        for m in re.findall(r"[A-Za-z]{2,}", d.translate(PUNCT_MAP)):
            key = m.lower()
            total[key] = total.get(key, 0) + 1
            if m.isupper():
                upper[key] = upper.get(key, 0) + 1
    alpha = np.array([t.isalpha() for t in terms])
    lens = np.array([len(t) for t in terms], float)
    a_lens = lens[alpha]
    bound = np.median(a_lens) + length_k * 1.4826 * stats.median_abs_deviation(a_lens)
    too_long = alpha & (lens > bound)
    capsy = np.array([alpha[i] and total.get(t, 0) > 0
                      and upper.get(t, 0) / total[t] >= caps_frac
                      for i, t in enumerate(terms)])
    return ~(too_long | capsy)


def signature_pooled_df(terms, df_arr, min_core: int = 4, min_productivity: int = 3):
    """Family-pooled df for the df_min eligibility floor (R8), corpus-native.

    Require: terms/df_arr aligned as returned by bm25_matrix.
    Guarantee: digit-bearing terms never pool (R3); result >= df_arr
    elementwise; scoring inputs untouched (R5); no lexicon, no nltk data.
    Maintain: two terms pool only when their residue pair recurs across
    >= min_productivity OTHER cores (Goldsmith signature productivity) —
    attestation gate, corpus-relative. Known residue: a false pair pools
    when the corpus itself productively attests the pattern (e.g.
    provenance/proven under a productive (∅,ance)); bounded because the
    pool feeds only the df floor — rescued terms still must win selection.
    """
    from collections import defaultdict
    alpha = [t for t in terms if t.isalpha() and len(t) > min_core]
    cores = defaultdict(dict)                      # core -> {residue -> term}
    for t in alpha:
        for i in range(min_core, len(t) + 1):
            cores[t[:i]][t[i:]] = t
    pair_cores = defaultdict(set)                  # residue pair -> attesting cores
    for core, res in cores.items():
        rs = sorted(res)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                pair_cores[(rs[i], rs[j])].add(core)
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for core, res in cores.items():
        rs = sorted(res)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if len(pair_cores[(rs[i], rs[j])]) - 1 >= min_productivity:
                    parent[find(res[rs[i]])] = find(res[rs[j]])
    keys = [find(t) if t in parent else t for t in terms]
    fam_df = {}
    for k, d in zip(keys, df_arr):
        fam_df[k] = fam_df.get(k, 0) + int(d)
    return np.array([fam_df[k] for k in keys], float)


def select_vocab(docs: list, waterline: int = 500, background: list = None,
                 lamb: float = 0.7, df_min: int = 2, df_max_frac: float = 0.10,
                 mad_k: float = 2.0, stopwords: frozenset = None,
                 correct: bool = False, stem: bool = False,
                 anomaly: bool = False, collapse: bool = True,
                 llr: bool = False,
                 pooled_df: bool = False):
    """Full pipeline. Returns dict with terms, scores, salient column values, kept docs.

    Utility = keyness * sqrt(df) when background given, else max_bm25 * sqrt(df).
    Diversity = Pearson correlation of binary presence vectors (row-coverage
    overlap), applied greedily: pick argmax lamb*util - (1-lamb)*max_corr.
    """
    kept, _ = length_filter(docs, mad_k)
    token_docs = [tokenize(d, collapse) for d in kept]
    if correct:
        token_docs, _ = correct_corpus(token_docs)
    if stem:
        from nltk.stem.snowball import SnowballStemmer
        _sb = SnowballStemmer("english")
        token_docs = [[_sb.stem(t) if t.isalpha() else t for t in ts] for ts in token_docs]
    BM, TF, terms, df = bm25_matrix(token_docs)
    N, V = TF.shape
    df_floor = signature_pooled_df(terms, df) if pooled_df else df
    eligible = (df_floor >= df_min) & (df <= max(df_min, df_max_frac * N))
    if stopwords:  # excluded from SELECTION only — scoring sees intact text
        eligible &= np.array([t not in stopwords for t in terms])
    if anomaly:    # eligibility mask (R10 principle): after scoring
        eligible &= anomaly_mask(terms, kept)
    max_bm = BM.tocsc().max(0).toarray().ravel()
    if background:
        if llr and isinstance(background, dict):
            # Dunning evidence gate — OPT-IN: measured 47% vs 53.7% recall
            # against the plain df floor (it evicts frequent-but-undistinctive
            # terms users still query). Right tool for glossaries, not indexes.
            eligible &= llr_gate(terms, TF, background)
        util_raw = keyness(terms, TF, background) * np.sqrt(df)
    else:
        util_raw = max_bm * np.sqrt(df)
    util = np.where(eligible, util_raw, -np.inf)
    finite = util[np.isfinite(util)]
    if finite.size == 0:
        raise ValueError("no eligible terms — loosen df band")
    util_n = np.where(np.isfinite(util), util / max(finite.max(), 1e-9), -np.inf)

    P = (TF > 0).astype(float).tocsc()
    mu = np.asarray(P.mean(0)).ravel()
    sd = np.sqrt(np.clip(mu * (1 - mu), 1e-12, None))
    selected, max_corr = [], np.zeros(V)
    target = min(waterline, int(eligible.sum()))
    for _ in range(target):
        score = lamb * util_n - (1 - lamb) * max_corr
        score[selected] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break
        selected.append(j)
        cov = np.asarray(P.T.dot(P[:, j].toarray()).ravel()) / N - mu * mu[j]
        max_corr = np.maximum(max_corr, cov / (sd * sd[j]))
    sel = set(selected)
    salient_col = [" ".join(sorted({t for t in set(ts) if t in
                   {terms[j] for j in sel}})) for ts in token_docs]
    order = {j: k for k, j in enumerate(selected)}
    return {
        "terms": [terms[j] for j in selected],
        "utility": [float(util_raw[j]) for j in selected],
        "df": [int(df[j]) for j in selected],
        "mmr_rank": [order[j] for j in selected],
        "docs": kept,
        "salient": salient_col,
    }


def spline_eligibility(df_arr, max_bm_arr, eligible, demote_threshold: float = -1.0):
    """GAMM-inspired eligibility refinement via smoothing spline.

    Fits a cubic spline to log(df) vs log(max_bm25) for eligible terms.
    Terms with residuals below demote_threshold * std(residuals) are demoted
    — they have lower utility than expected for their frequency band.

    Used by: select_vocab_unified (step 4b)
    Depends on: scipy.interpolate.UnivariateSpline

    Args:
        df_arr: Document frequency array (all terms).
        max_bm_arr: Max BM25 score array (all terms).
        eligible: Boolean eligibility mask (modified in place).
        demote_threshold: Residual z-score below which terms are demoted.

    Returns:
        int: Number of terms demoted.
    """
    from scipy.interpolate import UnivariateSpline
    elig_idx = np.where(eligible)[0]
    if len(elig_idx) <= 100:
        return 0
    log_df_e = np.log1p(df_arr[elig_idx])
    log_bm_e = np.log1p(max_bm_arr[elig_idx])
    sort_order = np.argsort(log_df_e)
    x_sorted = log_df_e[sort_order]
    y_sorted = log_bm_e[sort_order]
    s_factor = len(x_sorted) * 0.5
    spline = UnivariateSpline(x_sorted, y_sorted, s=s_factor, k=3)
    predicted = spline(log_df_e)
    residuals = log_bm_e - predicted
    resid_std = max(residuals.std(), 1e-9)
    demote_mask = residuals < (demote_threshold * resid_std)
    n_demoted = int(demote_mask.sum())
    if 0 < n_demoted < len(elig_idx) * 0.5:
        eligible[elig_idx[demote_mask]] = False
        return n_demoted
    return 0


def select_vocab_unified(docs: list, target_chars_per_row: float = 315,
                         background: list = None,
                         w_bm25: float = 0.4, w_keyness: float = 0.3,
                         w_diversity: float = 0.3,
                         df_min: int = 2, df_max_frac: float = 0.10,
                         mad_k: float = 2.0, stopwords: frozenset = None,
                         anomaly: bool = True, collapse: bool = True,
                         spline_refine: bool = True,
                         pooled_df: bool = False):
    """Single-pass unified selection with composite objective.

    Combines BM25 utility, keyness suppression, and diversity into one scoring
    function. One BM25 pass, one MMR iteration loop. Fills to chars/row waterline.

    Objective per iteration:
        score(t) = w_bm25 * bm25_n(t) + w_keyness * keyness_n(t) - w_diversity * max_corr(t)

    With a background prior, keyness_n is negative for terms common in that
    prior's register (suppresses "thank", "please") and positive for domain
    terms (boosts "corrosion", "fuselage"); OOV/digit-bearing terms get
    keyness=0 (neutral, scored only by BM25).

    With background=None (default) no prior is subtracted: keyness_n is 0 for
    every eligible term, so the objective reduces to BM25 + diversity (R9).

    Used by: Local term generation
    Depends on: bm25_matrix, keyness, anomaly_mask, spline_eligibility

    Args:
        docs: Raw document texts.
        target_chars_per_row: Waterline budget (315 = ~5 GB trigram at 5yr).
        background: Keyness prior, opt-in. None (default) subtracts no prior.
            'wordfreq' loads top_n_list('en', 10000). Otherwise a rank-ordered
            best-first word list, or a {word: count} dict.
        w_bm25: Weight for BM25 utility component.
        w_keyness: Weight for keyness component (English suppression/domain boost).
        w_diversity: Weight for MMR diversity penalty.
        df_min: Minimum document frequency for eligibility.
        df_max_frac: Maximum df as fraction of N for eligibility.
        mad_k: MAD multiplier for doc length filter.
        stopwords: Optional explicit stopword set.
        anomaly: Apply caps/length anomaly filter.
        collapse: Digit-run separator collapse.
        spline_refine: Apply GAMM spline eligibility refinement.
        pooled_df: Test the df_min floor against signature-pooled df (R8) —
            residue pairs attested across >= 3 other cores, corpus-native.
            Upper band stays per-term. Structural near-no-op under stem=True.

    Returns:
        dict: terms, utility, df, keyness_scores, chars_per_row, docs, salient.
    """
    kept, _ = length_filter(docs, mad_k)
    token_docs = [tokenize(d, collapse) for d in kept]
    BM, TF, terms, df_arr = bm25_matrix(token_docs)
    N, V = TF.shape

    df_floor = signature_pooled_df(terms, df_arr) if pooled_df else df_arr
    eligible = (df_floor >= df_min) & (df_arr <= max(df_min, df_max_frac * N))
    if stopwords:
        eligible &= np.array([t not in stopwords for t in terms])
    if anomaly:
        eligible &= anomaly_mask(terms, kept)

    max_bm = BM.tocsc().max(0).toarray().ravel()

    # GAMM spline refinement
    if spline_refine:
        spline_eligibility(df_arr, max_bm, eligible)

    bm25_util = max_bm * np.sqrt(df_arr)
    bm25_util = np.where(eligible, bm25_util, -np.inf)
    bm25_finite = bm25_util[np.isfinite(bm25_util)]
    if bm25_finite.size == 0:
        raise ValueError("no eligible terms after filtering")
    bm25_n = np.where(np.isfinite(bm25_util),
                      bm25_util / max(bm25_finite.max(), 1e-9), -np.inf)

    if background == "wordfreq":                      # R9 opt-in
        from wordfreq import top_n_list
        background = top_n_list("en", 10000)
    elif isinstance(background, str):
        raise ValueError(f"unknown background preset {background!r}; "
                         "pass 'wordfreq', a rank list, a count dict, or None")

    if background is None:
        # R9: no prior by default. Zero contribution is rank-neutral (a
        # constant shifts every eligible score equally), so selection is
        # BM25 + diversity only. key_raw stays 0.0 for reported diagnostics.
        key_raw = np.zeros(V)
        keyness_n = np.where(eligible, 0.0, -np.inf)
    else:
        key_raw = keyness(terms, TF, background)
        key_abs_max = max(np.abs(key_raw[np.isfinite(key_raw)]).max(), 1e-9)
        keyness_n = np.zeros(V)
        for j in range(V):
            if not eligible[j]:
                keyness_n[j] = -np.inf
            elif terms[j].isalpha() and len(terms[j]) >= 3:
                keyness_n[j] = key_raw[j] / key_abs_max
            else:
                keyness_n[j] = 0.0

    P = (TF > 0).astype(float).tocsc()
    mu = np.asarray(P.mean(0)).ravel()
    sd = np.sqrt(np.clip(mu * (1 - mu), 1e-12, None))

    selected: list = []
    max_corr = np.zeros(V)
    running_chars = 0.0

    n_eligible = int(eligible.sum())
    # Static half of the objective: utility and keyness do not change across
    # iterations — only max_corr does. Hoisted out of the loop, with every
    # exclusion (ineligible, selected, waterline-rejected) burned in as -inf,
    # which also removes the per-iteration re-mask and the O(k^2) rescan of
    # `selected`. Selection order and output are unchanged.
    base = np.where(eligible, w_bm25 * bm25_n + w_keyness * keyness_n, -np.inf)

    for _ in range(n_eligible):
        score = base - w_diversity * max_corr

        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break

        contribution = len(terms[j]) * (df_arr[j] / N)
        if running_chars + contribution > target_chars_per_row:
            eligible[j] = False
            base[j] = -np.inf
            continue

        selected.append(j)
        base[j] = -np.inf
        running_chars += contribution

        cov = np.asarray(P.T.dot(P[:, j].toarray()).ravel()) / N - mu * mu[j]
        corr = cov / (sd * max(sd[j], 1e-12))
        max_corr = np.maximum(max_corr, corr)

    # --- Post-selection: box-cox length round-off ---
    keep_set = {terms[j] for j in selected}
    if len(keep_set) > 10:
        sorted_sel = sorted(keep_set)
        sel_lens = np.array([len(t) for t in sorted_sel], dtype=float)
        if sel_lens.std() > 0 and sel_lens.min() > 0:
            bc_lens, _ = stats.boxcox(sel_lens)
            bc_med = np.median(bc_lens)
            bc_mad = stats.median_abs_deviation(bc_lens)
            bc_upper = bc_med + 3.0 * 1.4826 * max(bc_mad, 1e-9)
            keep_set = {t for t, bc in zip(sorted_sel, bc_lens) if bc <= bc_upper}

    # Build salient column values
    sel_set = keep_set
    salient_col = [" ".join(sorted({t for t in set(ts) if t in sel_set}))
                   for ts in token_docs]
    # Term-level diagnostics
    term_to_idx = {t: i for i, t in enumerate(terms)}
    selected_terms = sorted(keep_set)

    return {
        "terms": selected_terms,
        "utility": [float(bm25_util[term_to_idx[t]]) for t in selected_terms],
        "keyness": [float(key_raw[term_to_idx[t]]) for t in selected_terms],
        "df": [int(df_arr[term_to_idx[t]]) for t in selected_terms],
        "chars_per_row": running_chars,
        "n_eligible": n_eligible,
        "n_selected": len(keep_set),
        "docs": kept,
        "salient": salient_col,
    }


DDL = """-- salient trigram index (apply normalize_query() to every query string)
ALTER TABLE {table} ADD COLUMN IF NOT EXISTS salient text;
-- populate salient from application code via select_vocab()['salient']
CREATE INDEX IF NOT EXISTS {table}_salient_trgm ON {table}
  USING gin (salient gin_trgm_ops) WHERE salient IS NOT NULL;
-- query pattern:
--   SET pg_trgm.word_similarity_threshold = 0.5;
--   SELECT * FROM {table} WHERE %(normalized_query)s <%% salient;
-- optional exact re-rank on the candidate set (index does the narrowing):
--   ORDER BY levenshtein(%(normalized_query)s, salient) LIMIT 20;
"""


STOPWORDS_EN = frozenset("""a about above after again against all am an and any are as at be because
been before being below between both but by cannot could did do does doing down during each few for
from further had has have having he her here hers herself him himself his how i if in into is it its
itself me more most my myself no nor not of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while who whom why will
with you your yours yourself yourselves""".split())


def _edits1(word: str) -> set:
    """All strings at edit distance 1 (deletion, transposition, substitution, insertion)."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return ({L + R[1:] for L, R in splits if R} |
            {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1} |
            {L + c + R[1:] for L, R in splits if R for c in letters} |
            {L + c + R for L, R in splits for c in letters})


def correct_corpus(token_docs: list, df_confident: int = 3, min_len: int = 4,
                   max_passes: int = 3):
    """Multi-pass noisy-channel correction of pure-alpha tokens against the
    corpus's own confident vocabulary, gated by bigram context plausibility.

    Require: token_docs already canonicalized. Guarantee: digit-bearing tokens
    are NEVER modified (edit-1 neighbors of identifiers are distinct valid
    identifiers, not typos). Maintain: passes stop at fixed point or
    max_passes. Returns (corrected_token_docs, corrections_log_per_pass).
    """
    docs = [list(ts) for ts in token_docs]
    logs = []
    for _ in range(max_passes):
        df, bigram = {}, {}
        for ts in docs:
            for t in set(ts):
                df[t] = df.get(t, 0) + 1
            for a, b in zip(["<s>"] + ts, ts + ["</s>"]):
                bigram[(a, b)] = bigram.get((a, b), 0) + 1
        confident = {t for t, d in df.items() if d >= df_confident and t.isalpha()}
        changes = []
        for ts in docs:
            for i, t in enumerate(ts):
                if (not t.isalpha() or len(t) < min_len
                        or df.get(t, 0) >= df_confident or t in confident):
                    continue
                cands = _edits1(t) & confident
                if not cands:
                    continue
                if df.get(t, 0) > 1:
                    continue                      # attested >1x: plausibly a real word
                prev = ts[i - 1] if i > 0 else "<s>"
                nxt = ts[i + 1] if i + 1 < len(ts) else "</s>"

                def ctx(w):
                    return bigram.get((prev, w), 0) + bigram.get((w, nxt), 0)

                scored = sorted(((ctx(c), c) for c in cands), reverse=True)
                best_score, best = scored[0]
                runner = scored[1][0] if len(scored) > 1 else 0
                if best_score >= 2 and best_score > 2 * max(ctx(t), runner):
                    changes.append((t, best))     # unique, context-attested winner only
                    ts[i] = best
        logs.append(changes)
        if not changes:
            break
    return docs, logs