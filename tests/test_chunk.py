"""Pins chunkgraph R17: DOCUMENT-level atoms sized by the corpus.

derive_chunk_params counts lines per document, Box-Cox, m = median,
hi = m + 2*MAD. A document at or under hi is one chunk; over hi it splits at
paragraph boundaries into windows of at most hi lines, a short tail merges
back, no overlap, conservation exact. Pure unit tests, a battery.

Spec: design.md §6.14 R19 · Task: playbook.md T4
Also pins R19: chunk params fitted once per source label, not pooled across
the whole corpus, so a minority source's threshold is not dominated by the
majority's line-count distribution.
Also pins R21: per-source-pair block normalization with one global cut
(design.md §6.14 R21 · Task: playbook.md T6)
"""
from __future__ import annotations

import inspect
import json
import random
import zlib
from pathlib import Path

import numpy as np
import pytest

from chunkgraph import ChunkGraph, _chunk, _paras, derive_chunk_params, derive_chunk_params_by_source, MIN_BLOCK_PAIRS


def _para(n_lines, tag="s", words=6):
    return "\n".join(" ".join(f"{tag}{i}w{k}" for k in range(words)) + " ." for i in range(n_lines))


def _doc(sizes, tag="p"):
    return "\n\n".join(_para(n, f"{tag}{j}_") for j, n in enumerate(sizes))


def _lines(text):
    return [l for p in _paras(text) for l in p]


# ---------------------------------------------------------- derive params


def test_params_count_lines_per_document():
    docs = [_doc([3, 4, 3]) for _ in range(7)] + [_doc([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])]
    pr = derive_chunk_params(docs)
    assert pr["unit"] == "lines"
    assert pr["n"] == 8
    assert 9 <= pr["m"] <= 12, pr                  # the bulk is 10-line docs
    assert pr["hi"] >= pr["m"]
    assert 0 <= pr["hi_frac"] <= 1 / 8 + 1e-9


def test_params_fall_back_to_chars_for_single_line_documents():
    docs = ["x " * (50 + 13 * i) for i in range(10)]     # no newlines anywhere
    pr = derive_chunk_params(docs)
    assert pr["unit"] == "chars" and pr["m"] >= 1 and pr["hi"] >= pr["m"]


def test_params_survive_tiny_and_empty_corpora():
    assert derive_chunk_params([_doc([2, 2])])["m"] >= 1
    assert derive_chunk_params(["", "   "])["m"] >= 1


# ---------------------------------------------------------- one document


def test_document_at_or_under_hi_is_one_chunk_verbatim_structure():
    doc = _doc([3, 4, 3])                                    # 10 lines
    out = _chunk(doc, m=8, hi=10)
    assert len(out) == 1
    assert out[0].count("\n\n") == 2, "paragraph breaks preserved"
    assert _lines(out[0]) == _lines(doc)


def test_document_over_hi_splits_at_paragraph_boundaries():
    doc = _doc([5, 5, 5, 5, 5, 5])                           # 30 lines
    out = _chunk(doc, m=8, hi=12)
    sizes = [len(_lines(c)) for c in out]
    assert sizes == [10, 10, 10], sizes                      # never cuts a paragraph
    assert all(c.count("\n\n") == 1 for c in out)


def test_short_tail_merges_back_into_previous_window():
    doc = _doc([6, 6, 6, 2])                                 # 20 lines
    out = _chunk(doc, m=5, hi=12)
    sizes = [len(_lines(c)) for c in out]
    assert sizes == [12, 8], sizes                           # 2-line tail (< m) merged


def test_tail_at_or_above_m_stays_separate():
    doc = _doc([6, 6, 6, 6])                                 # 24 lines
    out = _chunk(doc, m=5, hi=12)
    assert [len(_lines(c)) for c in out] == [12, 12]


def test_oversize_paragraph_is_cut_on_lines_never_words():
    doc = _doc([2, 30, 2])
    out = _chunk(doc, m=5, hi=12)
    words = set(doc.split())
    for c in out:
        assert len(_lines(c)) <= 12
        assert set(c.split()) <= words
    assert _lines("\n\n".join(out)) == _lines(doc)


def test_no_overlap_between_windows():
    doc = _doc([7] * 8)
    seen = set()
    for c in _chunk(doc, m=10, hi=20):
        ls = set(_lines(c))
        assert not (ls & seen)
        seen |= ls


# ---------------------------------------------------------- conservation


@pytest.mark.parametrize("sizes", [[3, 3], [40], [1] * 30, [9, 1, 9, 1, 9], [12, 12, 12, 3], [2, 50, 2]])
@pytest.mark.parametrize("m,hi", [(5, 12), (10, 20), (100, 170)])
def test_every_source_line_lands_in_exactly_one_chunk(sizes, m, hi):
    doc = _doc(sizes)
    out = _chunk(doc, m=m, hi=hi)
    assert [l for c in out for l in _lines(c)] == _lines(doc)
    assert all(c.strip() == c and c for c in out)


def test_chars_unit_cuts_on_lines_and_never_cuts_an_overlong_line():
    doc = "\n".join(['a' * 40, 'b' * 40, 'y' * 500, 'c' * 40])
    out = _chunk(doc, m=60, hi=200, unit='chars')
    got = [l for c in out for l in _lines(c)]
    assert got == doc.split("\n")                     # conservation
    assert 'y' * 500 in got, 'overlong line must survive uncut'
    assert len(out) >= 2, 'a 624-char doc over hi=200 must split'


def test_empty_and_whitespace_docs_yield_nothing():
    assert _chunk("", m=5, hi=12) == []
    assert _chunk("\n\n  \n", m=5, hi=12) == []


# ---------------------------------------------------------- per-source params (R19)


def test_fit_accepts_a_sources_keyword():
    params = inspect.signature(ChunkGraph.fit).parameters
    assert "sources" in params
    assert params["sources"].default is None


@pytest.mark.parametrize("docs", [
    [_doc([10]) for _ in range(12)],                                    # (a) uniform
    [_doc([4]) for _ in range(10)] + [_doc([60]) for _ in range(3)],    # (b) skewed mix
    ["x " * (50 + 13 * i) for i in range(10)],                          # (c) chars fallback
])
def test_single_source_matches_r17_exactly(docs):
    grouped = derive_chunk_params_by_source(docs)
    assert set(grouped) == {"default"}
    assert grouped["default"] == derive_chunk_params(docs)


def test_explicit_single_label_is_the_same_fit():
    docs = [_doc([10]) for _ in range(12)]
    grouped = derive_chunk_params_by_source(docs, sources=["brown"] * len(docs))
    assert set(grouped) == {"brown"}
    assert grouped["brown"] == derive_chunk_params(docs)


@pytest.mark.parametrize("mixes", [
    {"quotes": [_doc([1]) for _ in range(12)], "wiki": [_doc([80]) for _ in range(12)]},
    {"quotes": [_doc([1]) for _ in range(12)],
     "brown": [_doc([10]) for _ in range(12)],
     "wiki": [_doc([80]) for _ in range(12)]},
    {"quotes": [_doc([1]) for _ in range(30)], "wiki": [_doc([80]) for _ in range(3)]},
])
def test_groups_are_fitted_independently(mixes):
    docs, sources = [], []
    for src, ds in mixes.items():
        docs += ds; sources += [src] * len(ds)
    params = derive_chunk_params_by_source(docs, sources)
    for src, ds in mixes.items():
        assert params[src] == derive_chunk_params(ds)
    if "quotes" in mixes and "wiki" in mixes:
        assert params["quotes"]["hi"] < params["wiki"]["hi"]


def test_group_key_order_follows_first_appearance():
    docs = [_doc([80]), _doc([1]), _doc([80]), _doc([1])] * 3
    sources = ["wiki", "quotes", "wiki", "quotes"] * 3
    params = derive_chunk_params_by_source(docs, sources)
    assert list(params) == ["wiki", "quotes"]


def test_chunk_params_are_json_serializable():
    docs = ([_doc([1]) for _ in range(12)] + [_doc([10]) for _ in range(12)]
            + [_doc([80]) for _ in range(12)])
    sources = ["quotes"] * 12 + ["brown"] * 12 + ["wiki"] * 12
    params = derive_chunk_params_by_source(docs, sources)
    assert json.loads(json.dumps(params)) == params
    for group in params.values():
        for v in group.values():
            assert isinstance(v, (str, int, float, type(None)))


def test_tiny_group_still_yields_usable_params():
    docs = [_doc([1]), _doc([1])] + [_doc([10]) for _ in range(12)]
    sources = ["tiny", "tiny"] + ["big"] * 12
    params = derive_chunk_params_by_source(docs, sources)
    assert params["tiny"]["m"] >= 1
    assert params["tiny"]["hi"] >= params["tiny"]["m"]
    assert params["tiny"]["lam"] is None
    assert params["big"]["m"] >= 1


def test_each_document_is_chunked_against_its_own_source():
    quotes = [_doc([1]) for _ in range(12)]
    brown = [_doc([10]) for _ in range(12)]
    # wiki docs built to sit well above hi_wiki + m_wiki so at least one splits
    wiki = [_doc([20]) for _ in range(11)] + [_doc([20] * 20)]
    docs = quotes + brown + wiki
    sources = ["quotes"] * len(quotes) + ["brown"] * len(brown) + ["wiki"] * len(wiki)
    params = derive_chunk_params_by_source(docs, sources)

    split_seen = False
    for d, src in zip(docs, sources):
        cp = {k: params[src][k] for k in ("m", "hi", "unit")}
        out = _chunk(d, **cp)
        assert [l for c in out for l in _lines(c)] == _lines(d)          # conservation
        if src == "quotes":
            assert len(out) == 1, "a 1-line quote must never split"
        if src == "wiki" and len(out) > 1:
            split_seen = True
    assert split_seen, "at least one oversized wiki doc must split"


def test_brown_500_reproduces_m107_hi153():
    pytest.importorskip("nltk")
    try:
        import ingest_brown
        _, docs = ingest_brown.load_docs(500)
    except Exception as e:
        pytest.skip(f"Brown corpus not available: {e}")
    if len(docs) < 500:
        pytest.skip("Brown corpus not fully downloaded")
    flat = derive_chunk_params(docs)
    grouped = derive_chunk_params_by_source(docs)["default"]
    assert flat["m"] == 107 and flat["hi"] == 153 and round(flat["lam"], 3) == -0.441
    assert grouped["m"] == 107 and grouped["hi"] == 153 and round(grouped["lam"], 3) == -0.441


# ---------------------------------------------------------- R21: per-source-pair blocks


_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _mixed_corpus(seed, mix):
    """mix: {source_label: n_docs}. Per-source private vocab (`f"{src}priv{c}"`)
    so intra-source pairs share terms; a small shared bridge vocab gives
    cross-source pairs some (weaker, more variable) overlap too, so every
    block realized in the test has nonzero pairs to fit on. One doc == one
    chunk (single line, well under any per-source `hi`)."""
    rng = random.Random(seed)
    bridge = [f"bridgeword{c}" for c in _LETTERS[:6]]
    docs, sources = [], []
    for src, n in mix.items():
        priv = [f"{src}priv{c}" for c in _LETTERS[:14]]
        for _ in range(n):
            words = rng.sample(priv, rng.randint(6, 10)) + rng.sample(bridge, rng.randint(1, 3))
            rng.shuffle(words)
            reps = [w for w in words for _ in range(rng.randint(1, 3))]
            docs.append(" ".join(reps) + ".")
            sources.append(src)
    return docs, sources


def _fake_embed(texts, dim=32):
    """Deterministic hashed bag-of-tokens vectors, offline, no model download."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, zlib.crc32(tok.encode()) % dim] += 1.0
    return out


def _fit(docs, sources=None, embed=True, **kw):
    return ChunkGraph(embed_fn=_fake_embed if embed else None, phrases=False, **kw).fit(docs, sources=sources)


@pytest.mark.parametrize("mix", [
    {"solo": 40},   # (a) uniform-ish single source
    {"solo": 50},   # (b) larger corpus
    {"solo": 30},   # (c) smaller corpus
])
def test_single_source_run_never_enters_the_block_path(monkeypatch, mix):
    def _boom(self, *a, **k):
        raise AssertionError("_cut_blocks must not be called for a single-source run")
    monkeypatch.setattr(ChunkGraph, "_cut_blocks", _boom)
    docs, sources = _mixed_corpus(7, mix)
    for use_sources in (None, sources):
        cg = _fit(docs, sources=use_sources)
        assert cg.diagnostics["sparse"]["blocks"] and len(cg.diagnostics["sparse"]["blocks"]) == 1
        if cg.sim_dense is not None:
            assert len(cg.diagnostics["dense"]["blocks"]) == 1


@pytest.mark.parametrize("mix", [{"solo": 40}, {"solo": 50}, {"solo": 30}])
def test_labelling_a_single_source_changes_nothing(mix):
    docs, sources = _mixed_corpus(11, mix)
    a = _fit(docs, sources=None)
    b = _fit(docs, sources=["brown"] * len(docs))
    assert np.array_equal(a.A, b.A)
    assert np.array_equal(a.A_sparse, b.A_sparse)
    assert np.array_equal(a.strength, b.strength)
    assert np.array_equal(a.D, b.D)
    assert a.blend_mode == b.blend_mode


@pytest.mark.parametrize("mix", [
    {"brown": 20, "quotes": 20, "wiki": 20},
    {"brown": 25, "quotes": 15, "wiki": 20},
    {"brown": 15, "quotes": 20, "wiki": 25},
])
def test_each_source_pair_block_gets_its_own_fit(mix):
    docs, sources = _mixed_corpus(3, mix)
    cg = _fit(docs, sources=sources, embed=False)
    labels = sorted(mix)
    expected = sorted({"|".join(sorted((a, b))) for a in labels for b in labels if a <= b})
    got = sorted(e["block"] for e in cg.diagnostics["sparse"]["blocks"])
    assert got == expected
    intra = [e for e in cg.diagnostics["sparse"]["blocks"] if e["block"].split("|")[0] == e["block"].split("|")[1]]
    assert all(e["fit"] == "own" for e in intra)
    lams = {e["lam"] for e in cg.diagnostics["sparse"]["blocks"] if e["lam"] is not None}
    assert len(lams) >= 2, f"expected distinct lambdas across blocks, got {lams}"


def test_a_degenerate_cross_block_falls_back_alone():
    docs, sources = _mixed_corpus(5, {"brown": 25, "quotes": 25})
    stub = "identicalstubterm identicalstubterm identicalstubterm."
    docs += [stub] * 12
    sources += ["oddball"] * 12
    cg = _fit(docs, sources=sources, embed=False)
    blocks = {e["block"]: e for e in cg.diagnostics["sparse"]["blocks"]}
    healthy_intra = ["brown|brown", "quotes|quotes"]
    for name in healthy_intra:
        if name in blocks:
            assert blocks[name]["fit"] == "own"
    oddball_cross = [e for e in cg.diagnostics["sparse"]["blocks"]
                      if "oddball" in e["block"] and e["block"] != "oddball|oddball"]
    for e in oddball_cross:
        assert e["fit"] in {"pooled", "rank"}
    assert cg.diagnostics["sparse"]["mode"] == "boxcox-blocks"


def test_a_small_block_inherits_the_pooled_fit():
    docs, sources = _mixed_corpus(9, {"brown": 25, "quotes": 25})
    tiny_docs, tiny_sources = _mixed_corpus(9, {"tiny": 3})
    docs += tiny_docs; sources += tiny_sources
    cg = _fit(docs, sources=sources, embed=False)
    pooled_lam = cg.diagnostics["sparse"]["lam"]
    for e in cg.diagnostics["sparse"]["blocks"]:
        if e["n_pairs"] < MIN_BLOCK_PAIRS:
            assert e["fit"] == "pooled", e
            assert e["lam"] == pooled_lam
        else:
            assert not (e["fit"] == "pooled" and e["n_pairs"] >= MIN_BLOCK_PAIRS)


@pytest.mark.parametrize("mix", [
    {"brown": 20, "quotes": 20, "wiki": 20},
    {"brown": 25, "quotes": 15, "wiki": 20},
    {"brown": 15, "quotes": 25, "wiki": 15},
])
def test_the_cut_is_one_global_threshold_in_z(mix):
    docs, sources = _mixed_corpus(17, mix)
    cg = _fit(docs, sources=sources, embed=False)
    tri = np.triu_indices(cg.n, 1)
    vals = cg.sim_sparse[tri]
    z, mask = cg._cut(vals, "probe")
    assert np.array_equal(mask, (z > cg.k_sigma) & (vals > 0))
    intra = [e["block"] for e in cg.diagnostics["sparse"]["blocks"] if e["block"].split("|")[0] == e["block"].split("|")[1]]
    assert len(intra) >= 2
    cutpoints = []
    for blkname in intra[:2]:
        lbl = blkname.split("|")[0]
        mask_lbl = np.array([s == lbl for s in sources])
        idx = np.array([i for i, (a, b) in enumerate(zip(*tri)) if mask_lbl[a] and mask_lbl[b]])
        v = vals[idx]
        v = v[v > 0]
        if len(v):
            cutpoints.append(v.min())
    assert len(cutpoints) == 2 and abs(cutpoints[0] - cutpoints[1]) > 1e-9


def test_blocks_partition_every_nonzero_pair_and_serialize():
    docs, sources = _mixed_corpus(23, {"brown": 20, "quotes": 20, "wiki": 20})
    cg = _fit(docs, sources=sources, embed=True)
    tri = np.triu_indices(cg.n, 1)
    total_nz = int((cg.sim_sparse[tri] > 0).sum())
    total_all = len(tri[0])
    assert sum(e["n_pairs"] for e in cg.diagnostics["sparse"]["blocks"]) == total_nz
    assert sum(e["n_pairs_all"] for e in cg.diagnostics["sparse"]["blocks"]) == total_all
    for e in cg.diagnostics["sparse"]["blocks"]:
        assert 0 <= e["edge_rate"] <= 1
    assert json.loads(json.dumps(cg.diagnostics)) == cg.diagnostics
    assert "blocks" in cg.diagnostics["sparse"]
    assert "blocks" in cg.diagnostics["dense"]


def test_a_mixed_run_still_serves():
    docs, sources = _mixed_corpus(29, {"brown": 20, "quotes": 20, "wiki": 20})
    cg = _fit(docs, sources=sources, embed=True)
    term = docs[0].split()[0]
    res = cg.query(term)
    assert "anchors" in res
    rows = cg.edges()
    assert isinstance(rows, list)


def test_merge_phrases_single_model_matches_per_doc_rebuild():
    # Pins the R9 hoisting fix (chunkgraph.py _merge_phrases): one Phrases model
    # built from the whole corpus must yield the same merged tokens as the old
    # per-document rebuild, which was O(n^2) and identical in output.
    pytest.importorskip("gensim")
    from gensim.models.phrases import Phrases, Phraser
    from chunkgraph import ChunkGraph
    toks = [
        ["new", "york", "is", "a", "city"],
        ["new", "york", "has", "boroughs"],
        ["quantum", "physics", "is", "hard"],
        ["quantum", "physics", "wins", "prizes"],
    ] * 5
    cg = ChunkGraph(phrases=True)
    cg.diagnostics = {}
    merged = cg._merge_phrases([list(t) for t in toks])
    old = [Phraser(Phrases(toks, min_count=5, threshold=0.4, scoring="npmi"))[t]
           for t in toks]
    assert merged == old
    assert cg.diagnostics["phrases"]["mode"] == "npmi"
