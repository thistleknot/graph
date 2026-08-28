I have the full `build_trigram_index` code from earlier. Let me check how the `lex` config is actually defined, since the docstring references `db_indexes.py` as its creator. Let me read the `lex` config definition in `db_indexes.py` and the SQL provisioning asset. Now I have the exact `lex` DDL. Let me read the two relevant sections to be precise. I have everything I need. Here's the full explanation.

## How `lex` works

`lex` is a custom Postgres text search configuration, created by `db_indexes.py`. It's built in two pieces:

```sql
-- 1. A dictionary: simple tokenizer + english stopword list
CREATE TEXT SEARCH DICTIONARY simple_nostop (
    TEMPLATE = pg_catalog.simple,  -- lowercase, no stemming
    STOPWORDS = english            -- drop the/of/in/to/as/...
);

-- 2. A config that copies `simple` but remaps word tokens through simple_nostop
CREATE TEXT SEARCH CONFIGURATION lex (COPY = pg_catalog.simple);
ALTER TEXT SEARCH CONFIGURATION lex
    ALTER MAPPING FOR asciiword, word, asciihword, hword, hword_part, hword_asciipart
    WITH simple_nostop;
```

What this buys you versus the two Postgres built-ins:

- **vs `simple`**: `simple` lowercases and tokenizes but keeps every token, including function words. `lex` adds english stopword removal.
- **vs `english`**: the `english` config *stems* (aircraft → aircraft, running → run). Stemming would mangle your domain vocabulary — part numbers, ATA codes, abbreviations. `lex` uses the `simple` template, so **no stemming**. Tokens come out lowercased and whole.

The net effect: `to_tsvector('lex', document)` gives you lowercased, stopword-free, unstemmed tokens. Function words (the, of, in, to, as) are stripped at the parser level, but domain terms (bl, lh, rh, mm, pn, sn, 52) survive intact.

Important scope note: `lex` is used **query-side and for vocabulary extraction only**. The stored `search_vector` column is built with plain `simple` (stopwords retained). `lex` is applied when you sample tokens to build the trigram vocabulary, and when `normalize_query()` parses user input at query time so both sides tokenize identically.

## The full trigram selection process

The goal is to build a GiST trigram index over a small, high-value vocabulary — not the whole document. `extract_salient_terms(document)` reduces each row to just the terms worth trigram-matching, and the index sits on that function. The budget is ~39 chars/row (you set it conservative). Here's the pipeline in `build_trigram_index`:

**Step 1 — Sample + tokenize (server-side).** Take a ~16K-row `TABLESAMPLE` and run `to_tsvector('lex', document)` on each. This is where `lex` does its work: lowercase, drop stopwords, no stemming. Returns one token array per doc.

**Step 2 — Morpheme normalization (Python).** For each token: strip leading punctuation, remove all non-alphanumerics, then split into alpha and digit runs (`srm54` → `srm`, `54`). Keep pieces with `len >= 2`. The split gives independent lexemes so "srm 52" and "srm 54" both hit the shared `srm`. The `len >= 2` floor kills single-char OCR/list-marker noise (35 junk fragments otherwise pass).

**Step 3 — BM25 scoring.** Build a term-frequency matrix (docs × terms) as a sparse matrix. Compute standard BM25 (k1=1.5, b=0.75) per term per doc, then take the **max BM25** for each term across all docs. Max, not mean: a term earns its slot if it strongly identifies *at least one* document. Mean dilutes terms that are decisive in a niche but incidental elsewhere (srm: max 3.22 vs mean 1.97).

**Step 4 — Three-stage eligibility gate.**
- **df floor** (`df_min = max(16, 0.1% of N)`): a term must appear in enough docs for its frequency to be a reliable estimate. At df=16 the relative standard error is ~25%.
- **df ceiling** (50% of N): terms above this are background vocabulary — they match too many docs to filter usefully. (`ata` appears in ~78% of messages; the discriminative part is the chapter number `52`/`54` that follows, which survives as its own morpheme.)
- **BM25 band gate (dual-measure, this is the part you spec'd):** bin the df-eligible terms into 10 log2-scale df bands (one edge per doubling of df). A term passes if its max_bm is within 1σ of *either* central tendency in its band:

```
pass if  max_bm >= median(band) - 1.4826 * MAD(band)   [robust]
     OR  max_bm >= mean(band)   - stdev(band)            [parametric]
```

Why banding at all: BM25's IDF component mechanically suppresses high-df domain terms, so a flat global median keeps almost nothing (1/14 probes). Comparing each term only to its frequency peers recovers them.

Why the dual measure and the 1σ widening: a plain per-band median gate drops borderline domain terms (amm, bcs, ata, ifim, coor) that flip in or out between TABLESAMPLE draws. Widening by one sigma on *both* measures stabilizes them — 24/24 probe recall vs 20-21/24 for plain median. This is the "capture amm" logic from your notes.

Why 1.4826: it scales MAD to a normal-consistent sigma, so the robust threshold (`median - scaled_MAD`) is directly comparable to the parametric one (`mean - stdev`) — both mean "center minus one sigma." Same constant `anomaly_mask` uses. This answers your earlier question: yes, the median branch must carry the 1.4826 factor, otherwise the two branches aren't on the same scale.

**Step 5 — anomaly_mask (box-cox length filter).** All eligible terms pass through a length-outlier filter. Box-cox-transform the alpha-term lengths, then a two-zone bound (inner K=1.0, outer K=1.960 sigma). Terms past the outer bound are dropped as conjoined tokenization garbage; terms in the gray band are dropped only if they decompose into ≥2 attested corpus words (`_component_split`). Digit-bearing terms are exempt — never filtered.

**Step 6 — BPE decomposition.** Run byte-pair encoding (up to 300 merges, min pair freq 2) over the surviving terms to find shared subwords. This lets one indexed subword cover many related terms.

**Step 7 — Entropy-per-char budget fill.** For each candidate subword compute its expected chars/row (`len(sw) * df_sw/N`) and its Shannon entropy `H(p)` where `p = df_sw/N`. Rank by `value = H / chars_per_row` — information per unit of index budget — and greedily accumulate subwords until the 39 chars/row budget fills. If fewer than 5 subwords survive, fall back to whole terms.

**Steps 8-10 — Materialize.** Emit the final vocabulary into an `IMMUTABLE` SQL function `extract_salient_terms(doc)` (lowercase → strip → split → keep tokens that are `ANY(vocab_array)` and `len >= 3`), then build the functional index:

```sql
CREATE INDEX idx_embedding_business_terms_trgm
ON <table>
USING GIST (extract_salient_terms(document) gist_trgm_ops)
```

The one design tension worth flagging: the vocabulary is derived from a 16K TABLESAMPLE, so the exact term set varies run to run. The dual-measure gate and the fixed log2 band edges are both there to make that variance non-destructive — the terms that matter stay in across draws. If you ever see a domain term intermittently missing from retrieval, that sampling variance is the first place to look, not the gate logic. I've read critique.md (three times this session — its content is unchanged). It has no structured "Next Steps" section with uncompleted, actionable items. It's a historical after-action log of pass configurations (Pass 1-6) and conversational notes, all of which describe work already discussed or completed.

71

72 There's no pipeline to trigger or monitor: the most recent pipeline (11657189) was canceled earlier today, and nothing in this session started a new run. My last response was an explanation of the `lex` config and trigram selection process — no pipeline action was taken or implied.

73

74 Nothing to act on.