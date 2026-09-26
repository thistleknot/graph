"""domain_terms.py -- per-domain term selection for domain-specific term analysis.

NO GOVERNING SPEC for this module's own requirements. Basis: operator
instruction 2026-09-26 (log2 over BM25 into equal-count bands; dual centre/sigma
union; df used ONLY for the >50% mask and the df_min floor; outlier mask; BPE
over survivors), planned in
C:\\Users\\user\\.claude\\plans\\i-ve-been-thinking-about-quiet-cray.md.
Task: playbook.md T105.

AMENDS .spec/specs/graph-term-selection/requirements.md:
  - R2.1's df-banding is SET ASIDE for equal-count bands on the log2 BM25 score,
    on operator evidence from prior work outside this repo. Consequence stated
    plainly: df-banding exists to hold idf near-constant within a band, which is
    what made R2.5's idf-suppression claim testable. Score-banding CANNOT test
    that claim, so R2.5 is NOT evaluated here and the band gate does not ship on
    its evidence.
  - R2.2's disjunction and R2.3's 1.4826 scaling are kept unchanged.
  - The scope line `OUT: ... BPE or subword decomposition` is amended to IN, in
    the restricted form of trigram/trigram.md:58 (BPE over surviving terms).
  - The df ceiling here is 50%, not R1.6's 10%. Different job: R1.6 is hub
    control for the graph walk; this is a term table with no walk. Precedent for
    50%: gt_terms.second_order's nomen pool.

    STAGE      MECHANISM
    TOKENIZE   stoplist.tokenize -- the ONE tokenizer (chunkgraph R25)
    BM25       salient_grams.bm25_matrix -> (BM, TF, terms, df), doc x term
    COLLAPSE   per-term max over chunks. NO sqrt(df) reweighting (T1)
    TRANSFORM  log2 of the collapsed score (T2)
    BAND       equal-count bands, adjacent bands under MIN_BAND merged (T3)
    SELECT     admit if score >= MIN(mean - f*sd, median - f*1.4826*MAD) (T4)
    MASK       df > 50% of chunks (T5), then length/caps outliers (T6)
    BPE        over surviving term strings, vocab = n * scale_factor (T7)

EARS guards:

T1  The per-term collapse SHALL be max over chunks and SHALL NOT be multiplied
    by sqrt(df). salient_grams.py:657 does apply that weight; adopting it would
    re-admit df as a score term, which the operator's design excludes. Floor
    worth stating: BM25's own idf = log(1 + (N-df+.5)/(df+.5)) contains df by
    definition, so the score is never df-free -- what is excluded is ADDITIONAL
    df weighting and df-based banding.

T2  The score SHALL be log2-transformed before band statistics are computed.
    The transform is INERT at the band cut -- quantiles are invariant under any
    monotone transform, so deciles of log2(x) hold the same terms as deciles of
    x. It is NOT inert at selection: mean and sd are not transform-equivariant,
    and median - k*MAD is not either. So the centre/sigma statistics SHALL be
    computed on the log2 values, which is the only place the transform changes
    an answer.

T3  Bands SHALL be equal-count over the log2 score, and any band with fewer than
    MIN_BAND members SHALL be merged into its neighbour. Rationale: robust
    statistics need members -- chunkgraph._bc_center already refuses to fit below
    len(x) >= 8. The merge rule is the reduce_overlaps skill's own
    ("MERGE consecutive sections with < m ... accumulating until they reach m")
    applied to bands rather than sections.

T4  WITHIN a band, a term SHALL be admitted when its log2 score clears EITHER
    the parametric bound (mean - factor*sd) OR the robust bound
    (median - factor*1.4826*MAD). Equivalent to >= MIN of the two. The
    disjunction SHALL NOT be reduced to one branch (spec R2.2). MAD is scaled by
    1.4826 so both bounds express "centre minus one sigma" (spec R2.3).

T5  The df mask SHALL be applied AFTER scoring, never before. Mechanism, stated
    because the usual phrasing is broader than the truth: idf(t) and tf(t,c) are
    per-term, so removing term u cannot perturb term t's score. The ONLY channel
    by which pre-filtering biases BM25 is the length normaliser -- dropping
    tokens from the stream moves len(d) and avgdl. Post-masking is therefore
    arithmetically identical to never having considered those terms.

T6  Length/caps outliers SHALL be removed with salient_grams.anomaly_mask, the
    INCUMBENT (median + length_k*1.4826*MAD over alpha term lengths, digit-bearing
    terms exempt per its R13). It is applied BEFORE the cardinality cut so freed
    slots go to other terms and the vocab still reaches its target size.
    NO compound-word detector ships. Segmentability does not discriminate --
    `therapist` -> `the`+`rapist`, `together` -> `to`+`get`+`her`. df does: an
    OCR join is a hapax by construction (measured: "Wepropose" in
    papers/post_processed/1301_3781.md), so the df_min floor removes the class
    before this mask runs, and genuine compounds are BPE's job (T7).

T7  BPE SHALL be trained over the SURVIVING term strings only, never the raw
    corpus, with target vocab = n_survivors * scale_factor. A second corpus-wide
    tokenizer would diverge from stoplist.tokenize, which spec 1.2 calls a
    defect, and chunkgraph T101 records the live consequence: every phrase the
    index merges is unreachable from an unmerged query.

T8  Every stage SHALL be individually disableable so a change is attributable to
    one stage (spec R8.4), and the pipeline SHALL be deterministic -- no model
    call anywhere, so a rerun that differs is a defect, not variance.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from salient_grams import anomaly_mask, bm25_matrix
from stoplist import tokenize

# Operator's design parameters.
DF_MIN = 2                 # eligibility floor; also the OCR-join filter (T6)
DF_MAX_FRAC = 0.50         # the "whales" mask (T5)
N_BANDS = 10               # deciles; 4 is the documented alternate
MIN_BAND = 8               # T3, matching chunkgraph._bc_center's own floor
SIGMA_FACTOR = 1.0         # T4, tunable
LENGTH_K = 3.0             # T6, the incumbent anomaly_mask default
SCALE_FACTORS = (2, 3, 5)  # T7 sweep

# PROBE SETS -- fixed here BEFORE any measurement, per spec R2.2's acceptance
# ("The probe set SHALL NOT be drawn from what the gate admits"). Domain terms a
# reader would expect the analysis to surface; recall of these is the gate's test.
PROBE = {
    "neop": (
        "plotinus enneads hypostasis henad theurgy demiurge emanation nous "
        "intelligible ineffable procession participation intellect porphyry "
        "iamblichus proclus chaldaean oracles ascent purification contemplation "
        "dyad monad platonic soul daemon divinization"
    ).split(),
    "arxiv": (
        "transformer attention embedding gradient retrieval benchmark corpus "
        "encoder decoder tokenizer perplexity pretraining inference latency "
        "recall precision baseline ablation dataset optimizer neural "
        "convolutional recurrent supervised"
    ).split(),
}


def tokenize_chunks(texts: list[str]) -> list[list[str]]:
    """T: the one tokenizer, applied to every chunk."""
    return [tokenize(t) for t in texts]


def score_terms(token_docs: list[list[str]]) -> dict:
    """BM25 -> per-term max collapse -> log2. Returns aligned arrays.

    T1: max over chunks, no sqrt(df).
    T2: log2 applied here; band statistics consume `log_score`.
    """
    BM, TF, terms, df = bm25_matrix(token_docs)
    n_chunks = TF.shape[0]
    max_bm = BM.tocsc().max(0).toarray().ravel()
    with np.errstate(divide="ignore"):
        log_score = np.log2(np.where(max_bm > 0, max_bm, np.nan))
    return {"terms": terms, "df": df, "max_bm": max_bm,
            "log_score": log_score, "n_chunks": n_chunks, "n_terms": len(terms)}


def make_bands(log_score: np.ndarray, eligible: np.ndarray,
               n_bands: int = N_BANDS, min_band: int = MIN_BAND) -> list[np.ndarray]:
    """T3: equal-count bands over the log2 score, small bands merged forward.

    Returns a list of index arrays into the full term axis.
    """
    idx = np.where(eligible & np.isfinite(log_score))[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(log_score[idx], kind="stable")]
    raw = [b for b in np.array_split(order, n_bands) if b.size]
    merged: list[np.ndarray] = []
    for band in raw:
        if merged and merged[-1].size < min_band:
            merged[-1] = np.concatenate([merged[-1], band])
        else:
            merged.append(band)
    # a trailing short band merges backwards
    while len(merged) > 1 and merged[-1].size < min_band:
        merged[-2] = np.concatenate([merged[-2], merged.pop()])
    return merged


def select_in_bands(log_score: np.ndarray, bands: list[np.ndarray],
                    factor: float = SIGMA_FACTOR) -> dict:
    """T4: admit on the union of the parametric and robust lower bounds.

    Returns {admitted: bool array, rows: per-band diagnostics}.
    """
    admitted = np.zeros(log_score.shape, dtype=bool)
    rows = []
    for bi, band in enumerate(bands):
        v = log_score[band]
        mean, sd = float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        med = float(np.median(v))
        mad = float(stats.median_abs_deviation(v))
        par = mean - factor * sd
        rob = med - factor * 1.4826 * mad
        thr = min(par, rob)
        keep = band[v >= thr]
        admitted[keep] = True
        rows.append({
            "band": bi, "n": int(band.size), "lo": float(v.min()),
            "hi": float(v.max()), "mean": mean, "sd": sd, "median": med,
            "mad": mad, "parametric": par, "robust": rob, "threshold": thr,
            "binds": "parametric" if par <= rob else "robust",
            "admitted": int(keep.size),
            "frac": float(keep.size) / float(band.size),
        })
    return {"admitted": admitted, "rows": rows}


def select(texts: list[str], *, n_bands: int = N_BANDS,
           factor: float = SIGMA_FACTOR, df_max_frac: float = DF_MAX_FRAC,
           length_k: float = LENGTH_K, use_bands: bool = True,
           use_df_mask: bool = True, use_anomaly: bool = True) -> dict:
    """The full per-domain selection. T8: every stage independently disableable."""
    token_docs = tokenize_chunks(texts)
    sc = score_terms(token_docs)
    terms, df, log_score = sc["terms"], sc["df"], sc["log_score"]
    n_chunks = sc["n_chunks"]

    eligible = (df >= DF_MIN) & np.isfinite(log_score)
    stages = {"tokenized": int(sc["n_terms"]), "df_min": int(eligible.sum())}

    if use_anomaly:                                     # T6, before the cut
        eligible &= anomaly_mask(terms, texts, length_k=length_k)
        stages["anomaly"] = int(eligible.sum())

    if use_bands:                                       # T3 + T4
        bands = make_bands(log_score, eligible, n_bands=n_bands)
        band_out = select_in_bands(log_score, bands, factor=factor)
        admitted = eligible & band_out["admitted"]
        rows = band_out["rows"]
    else:                                               # global control
        v = log_score[eligible]
        thr = min(float(np.mean(v)) - factor * float(np.std(v, ddof=1)),
                  float(np.median(v)) - factor * 1.4826
                  * float(stats.median_abs_deviation(v)))
        admitted = eligible & (log_score >= thr)
        rows = []
    stages["banded" if use_bands else "global"] = int(admitted.sum())

    if use_df_mask:                                     # T5, last
        admitted &= df <= df_max_frac * n_chunks
        stages["df_mask"] = int(admitted.sum())

    sel = np.where(admitted)[0]
    return {
        "terms": terms, "df": df, "log_score": log_score, "max_bm": sc["max_bm"],
        "n_chunks": n_chunks, "selected": sel,
        "selected_terms": [str(terms[i]) for i in sel],
        "bands": rows, "stages": stages,
        "params": {"n_bands": n_bands, "factor": factor,
                   "df_min": DF_MIN, "df_max_frac": df_max_frac,
                   "length_k": length_k, "use_bands": use_bands,
                   "use_df_mask": use_df_mask, "use_anomaly": use_anomaly},
    }


def probe_recall(selected_terms: list[str], domain: str) -> dict:
    """Recall of the pre-registered probe set. The gate's discriminating test."""
    want = PROBE[domain]
    got = set(selected_terms)
    hit = [w for w in want if w in got]
    return {"n_probe": len(want), "n_hit": len(hit),
            "recall": len(hit) / len(want) if want else 0.0,
            "missed": [w for w in want if w not in got]}


def bpe_over_terms(selected_terms: list[str], scale_factor: int) -> dict:
    """T7: BPE trained on the surviving term strings, vocab = n * scale_factor."""
    from tokenizers import Tokenizer, models, trainers

    n = len(selected_terms)
    target = int(n * scale_factor)
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(vocab_size=target, min_frequency=1,
                                  special_tokens=["[UNK]"], show_progress=False)
    tok.train_from_iterator(selected_terms, trainer=trainer)
    vocab = tok.get_vocab()
    whole = sum(1 for t in selected_terms
                if len(tok.encode(t).tokens) == 1)
    pieces = [len(tok.encode(t).tokens) for t in selected_terms]
    return {"scale_factor": scale_factor, "n_terms": n, "target_vocab": target,
            "actual_vocab": len(vocab),
            "whole_single_token": whole,
            "whole_frac": whole / n if n else 0.0,
            "mean_pieces": float(np.mean(pieces)) if pieces else 0.0}
