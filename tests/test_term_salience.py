"""Pins tools/term_salience.py guards TS1-TS6 on synthetic data: hand-computed edge shares, a hashing bag-of-words encoder in place of
MiniLM, and planted community structure. No model, no database.

Run:  pytest tests/test_term_salience.py -v
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import term_salience as ts


def _word_vec(w: str, d: int = 64) -> np.ndarray:
    seed = int(hashlib.md5(w.lower().encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=d)


def encode(texts: list[str]) -> np.ndarray:
    """A bag-of-words embedding: the sum of fixed random word vectors, L2-normalised. Nearly additive, like the ideal case."""
    out = []
    for t in texts:
        v = sum((_word_vec(w) for w in t.split()), np.zeros(64))
        out.append(v / max(np.linalg.norm(v), 1e-9))
    return np.array(out)


# ------------------------------------------------------------------------------------------------- TS1
def _rows(*dense):
    M = np.array(dense, np.float32)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return sp.csr_matrix(M)


def test_a_terms_share_of_an_edge_is_its_summand_over_the_cosine_TS1():
    X = _rows([0.6, 0.8, 0.0], [0.6, 0.0, 0.8], [0.6, 0.8, 0.0])
    nbrs = np.array([[0, 2, 1], [1, 0, 2], [2, 0, 1]])
    a = ts.edge_share_of_term(X, nbrs, 0)                      # term a is in all three rows
    assert a["chunks"] == 3 and a["edges"] == 6 and a["shared"] == 6
    want = sorted([0.36, 1.0, 1.0, 1.0, 0.36, 1.0])           # cos 1.0 for rows 0 and 2 (term carries 0.36 of it), cos 0.36 to row 1 (carries all)
    assert np.sort(a["shares"]).tolist() == pytest.approx(want)
    b = ts.edge_share_of_term(X, nbrs, 1)                      # term b is in rows 0 and 2 only
    assert b["chunks"] == 2 and b["edges"] == 4 and b["shared"] == 2
    assert b["shares"].tolist() == pytest.approx([0.64, 0.64])


def test_the_shares_of_all_terms_of_one_edge_sum_to_one_TS1():
    X = _rows([0.6, 0.8, 0.0], [0.6, 0.8, 0.5])
    nbrs = np.array([[1], [0]])
    total = sum(ts.edge_share_of_term(X, nbrs, j)["shares"].sum() for j in range(3))
    assert total == pytest.approx(2.0)                         # two directed edges, each summing to 1


def test_an_edge_with_no_overlap_gives_no_share_and_a_term_in_no_chunk_gives_nothing_TS1():
    X = _rows([1.0, 0.0], [0.0, 1.0])
    r = ts.edge_share_of_term(X, np.array([[1], [0]]), 0)
    assert r["chunks"] == 1 and r["edges"] == 1 and r["shared"] == 0 and len(r["shares"]) == 0
    empty = ts.edge_share_of_term(sp.csr_matrix((2, 3), dtype=np.float32), np.array([[1], [0]]), 2)
    assert empty["chunks"] == 0 and empty["edges"] == 0


# ------------------------------------------------------------------------------------------------- TS2
def test_a_term_confined_to_one_community_has_share_one_and_spread_zero_and_an_even_term_spreads_wide_TS2():
    lab = np.repeat(np.arange(4), 50)                          # 4 communities of 50
    P = np.zeros((200, 3))
    P[:30, 0] = 1                                              # confined to community 0
    P[np.arange(0, 200, 4), 1] = 1                             # one in every four chunks, spread evenly
    P[:5, 2] = 1                                               # below min_df
    k = ts.keyness_by_community(sp.csr_matrix(P), lab, min_df=20)
    assert k["ok"].tolist() == [True, True, False]
    assert k["best"][0] == 0 and k["share"][0] == 1.0 and k["spread"][0] == pytest.approx(0.0)
    assert k["g2"][0] > 50
    assert k["spread"][1] == pytest.approx(1.0, abs=2e-3) and k["share"][1] == pytest.approx(0.26) and k["g2"][1] < 1.0     # counts 13/12/12/13


def test_under_representation_never_picks_the_best_community_TS2():
    lab = np.array([0] * 100 + [1] * 100)
    P = np.zeros((200, 1))
    P[:80, 0] = 1                                              # 80 of the term's 100 chunks sit in community 0
    P[100:120, 0] = 1
    k = ts.keyness_by_community(sp.csr_matrix(P), lab, min_df=20)
    assert k["best"][0] == 0 and k["share"][0] == pytest.approx(0.8)


# ------------------------------------------------------------------------------------------------- TS3
@pytest.mark.parametrize("text,term,want", [
    ("We use GRPO here and GRPO there", "grpo", "We use here and there"),
    ("Policy gradient methods", "policy gradient", "methods"),
    ("the LoRAs differ from LoRA", "lora", "the LoRAs differ from"),
    ("nothing relevant", "grpo", None),
])
def test_remove_term_deletes_whole_words_case_insensitively_TS3(text, term, want):
    assert ts.remove_term(text, term) == want


def test_occlusion_gives_one_delta_per_chunk_that_holds_the_term_and_none_otherwise_TS3():
    texts = ["alpha GRPO beta gamma", "GRPO delta epsilon", "no match at all"]
    D = ts.occlusion_deltas(encode, texts, "GRPO")
    assert D.shape == (2, 64)
    e = encode(["alpha GRPO beta gamma", "alpha beta gamma"])
    assert D[0].tolist() == pytest.approx((e[0] - e[1]).tolist())
    assert ts.occlusion_deltas(encode, ["nothing"], "GRPO").shape == (0, 0)


# ------------------------------------------------------------------------------------------------- TS4
def _chunks(term: str, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    words = ["w%d" % i for i in range(200)]
    return [" ".join(list(rng.choice(words, size=14)) + [term]) for _ in range(n)]


def test_the_gate_separates_a_shared_term_direction_from_a_mismatched_null_TS4():
    deltas = {t: ts.occlusion_deltas(encode, _chunks(t, 12, i), t) for i, t in enumerate(["GRPO", "LoRA", "Mamba", "PagedAttention"])}
    g = ts.stability(deltas, seed=3)
    assert g["same"] > 0.6 and abs(g["diff"]) < 0.2            # a bag-of-words encoder is the additive case
    assert g["n_same"] == 4 * (12 * 11 // 2) and g["n_diff"] == 4 * 36          # mismatched pairs are capped by the 36 chunks of the other terms
    assert min(g["per_term"].values()) > 0.5 and set(g["per_term"]) == {"GRPO", "LoRA", "Mamba", "PagedAttention"}


def test_the_gate_reports_no_stability_when_the_deltas_are_random_TS4():
    rng = np.random.default_rng(0)
    g = ts.stability({"a": rng.normal(size=(15, 64)), "b": rng.normal(size=(15, 64)), "c": rng.normal(size=(15, 64))}, seed=1)
    assert abs(g["same"]) < 0.1 and abs(g["diff"]) < 0.1


def test_a_term_with_one_delta_cannot_enter_the_gate_TS4():
    g = ts.stability({"a": np.ones((1, 8)), "b": np.eye(8)[:3] + 0.1, "c": np.eye(8)[3:6] + 0.1}, seed=0)
    assert "a" not in g["per_term"]


# ------------------------------------------------------------------------------------------------- TS5
def test_the_token_share_vector_is_the_terms_part_of_the_pooled_vector_TS5():
    V = np.arange(24, dtype=float).reshape(6, 4)
    s = ts.token_share_vector(V, [1, 2])
    assert s.tolist() == pytest.approx(((V[1] + V[2]) / 6).tolist())
    assert (s + ts.token_share_vector(V, [0, 3, 4, 5])).tolist() == pytest.approx(V.mean(0).tolist())
    assert ts.token_share_vector(V, []).tolist() == [0.0] * 4


# ------------------------------------------------------------------------------------------------- TS6
def test_subtracting_the_right_term_direction_helps_and_a_wrong_term_does_not_TS6():
    terms = ["GRPO", "LoRA", "Mamba", "PagedAttention"]
    train = {t: ts.occlusion_deltas(encode, _chunks(t, 10, 10 + i), t) for i, t in enumerate(terms)}
    estar = {t: ts.term_direction(D) for t, D in train.items()}
    Ed, Ew, right, wrong = [], [], [], []
    for i, t in enumerate(terms):
        for x in _chunks(t, 8, 100 + i):
            Ed.append(encode([x])[0])
            Ew.append(encode([ts.remove_term(x, t)])[0])
            right.append(estar[t])
            wrong.append(estar[terms[(i + 1) % 4]])
    g = ts.subtraction_gain(np.array(Ed), np.array(Ew), np.array(right), np.array(wrong))
    assert g["n"] == 32 and g["right_gain"] > 0 and g["right_improved"] > 0.9
    assert g["wrong_gain"] < 0 and g["wrong_improved"] < 0.1


# ------------------------------------------------------------------------------------------------- TS7
def test_parse_sparsevec_reads_pgvectors_text_with_one_based_columns_TS7():
    assert ts.parse_sparsevec("{1:0.5,7:0.25,12:1}/200") == {0: 0.5, 6: 0.25, 11: 1.0}
    assert ts.parse_sparsevec("{}/200") == {}


VOCAB = {"graph": 0, "trans": 1, "##former": 2, "##s": 3, "kv": 4, "cache": 5, "attention": 6}


def test_a_word_is_its_own_column_and_a_longer_word_is_spelled_by_its_bpe_pieces_TS7():
    assert ts.term_columns(VOCAB, "graph") == [0]
    assert ts.term_columns(VOCAB, "transformer") == [1, 2]                       # trans + ##former, greedy longest leading piece first
    assert ts.term_columns(VOCAB, "transformers") == [1, 2, 3]
    assert ts.term_columns(VOCAB, "KV cache") == [4, 5]                          # a phrase is the columns of its words
    assert ts.term_columns(VOCAB, "zzz") is None                                 # not spellable from this vocabulary
    assert ts.term_columns(VOCAB, "graphzzz") is None


def test_self_shares_of_all_columns_sum_to_one_TS7():
    v = {0: 0.6, 1: 0.48, 2: 0.64}
    assert sum(ts.self_share(v, [c]) for c in v) == pytest.approx(1.0)
    assert ts.self_share(v, [0]) == pytest.approx(0.36 / (0.36 + 0.2304 + 0.4096))
    assert ts.self_share({}, [0]) == 0.0


def test_a_phrases_share_of_an_inner_product_is_exactly_the_sum_of_its_pieces_TS7():
    a = {0: 0.5, 1: 0.5, 2: 0.5, 5: 0.5}
    b = {0: 0.4, 1: 0.6, 2: 0.2, 6: 0.65}
    ip = 0.5 * 0.4 + 0.5 * 0.6 + 0.5 * 0.2
    pieces = [ts.pair_share(a, b, [c])[0] for c in (1, 2)]
    whole, got_ip = ts.pair_share(a, b, [1, 2])
    assert got_ip == pytest.approx(ip) and whole == pytest.approx(sum(pieces)) == pytest.approx((0.3 + 0.1) / ip)
    assert sum(ts.pair_share(a, b, [c])[0] for c in (0, 1, 2)) == pytest.approx(1.0)      # every shared column together carry all of it
    assert ts.pair_share(a, {9: 1.0}, [0]) == (0.0, 0.0)                                  # no overlap, no share


def test_community_profile_reports_spread_and_the_over_represented_communities_TS7():
    sizes = np.array([100, 100, 100, 100])
    p = ts.community_profile(np.array([30.0, 0, 0, 0]), sizes)
    assert p["n"] == 30 and p["spread"] == pytest.approx(0.0) and p["top"][0][0] == 0 and p["top"][0][2] == 1.0 and p["top"][0][4] > 50
    even = ts.community_profile(np.array([25.0, 25, 25, 25]), sizes)
    assert even["spread"] == pytest.approx(1.0) and even["top"] == []                     # evenly spread: no community over-represents it
    assert ts.community_profile(np.zeros(4), sizes) == {"n": 0, "spread": 0.0, "top": []}


def test_term_direction_is_the_mean_of_the_deltas_TS6():
    D = np.array([[1.0, 0.0], [3.0, 2.0]])
    assert ts.term_direction(D).tolist() == [2.0, 1.0]


def _terms_world():
    """60 sections in 3 communities of 20. 'tumor' is in 18 of community 0 and 1 elsewhere; 'tumor cells' always rides with it; 'the' is everywhere;
    'graph' is in 15 of community 1; 'rare' is in 2 of community 2 (under min_df)."""
    vocab = ["the", "tumor", "tumor cells", "graph", "rare"]
    P = np.zeros((60, 5))
    P[:, 0] = 1
    P[:18, 1] = 1; P[40, 1] = 1
    P[:18, 2] = 1
    P[20:35, 3] = 1
    P[40:42, 4] = 1
    lab = np.repeat([0, 1, 2], 20)
    return sp.csr_matrix(P), lab, vocab


def test_community_terms_rank_the_over_represented_terms_and_drop_the_ubiquitous_one():
    P, lab, vocab = _terms_world()
    got = ts.community_terms(P, lab, vocab, n_terms=3)
    assert [t for t, _ in got[0]] == ["tumor cells"]            # 'tumor' sits inside it: one slot, and the better-ranked one (18 of 18, none elsewhere) wins; 'the' is never over-represented
    assert [t for t, _ in got[1]] == ["graph"]
    assert got[0][0][1] > 20 and got[1][0][1] > 10              # a G2 far above 10.83 (chi-square(1), p < 0.001)
    assert got[2] == []                                          # 'rare' is under min_df, 'tumor' is under-represented there


def test_community_terms_take_a_count_per_community_and_the_label_length_follows_the_exemplars():
    P, lab, vocab = _terms_world()
    extra = np.zeros((60, 1)); extra[:20, 0] = 1
    P = sp.csr_matrix(np.hstack([P.toarray(), extra]))
    got = ts.community_terms(P, lab, vocab + ["lesion"], n_terms=np.array([1, 5, 5]))
    assert len(got[0]) == 1
    assert [ts.exemplar_term_count(n) for n in (1, 2, 3, 7)] == [3, 5, 8, 8]
