"""Pins src/section_embed.py: the vectorised pooling equals model2vec's own, without its truncation and pad-id failure.

Spec: approved plan step A2 (operator 2026-10-05), playbook.md T150. A synthetic word-level model, no disk; one test uses the saved
distilled model when it is on this machine.

Run:  pytest tests/test_section_embed.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model2vec import StaticModel
from tokenizers import Tokenizer, models, pre_tokenizers

import section_embed as se

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
SAVED = os.path.expanduser("~/models/m2v-jina-v5-nano-256")


def _model(rows: int | None = None, weights: bool = True) -> StaticModel:
    vocab = {w: i for i, w in enumerate(WORDS)}
    vocab["[UNK]"] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    n = len(vocab)
    rng = np.random.default_rng(0)
    m = StaticModel(rng.normal(size=(n, 6)).astype(np.float32), tok, normalize=True,
                    weights=rng.random(n).astype(np.float32) + 0.5 if weights else None)
    if rows is not None:                                                    # the constructor insists rows == tokens; distillation of a
        m.embedding = m.embedding[:rows]                                    # tokenizer with extra ids leaves fewer rows, so cut after
        m.weights = None if m.weights is None else m.weights[:rows]
    return m


TEXTS = ["alpha beta gamma", "delta", "alpha alpha alpha beta", "zeta eta theta alpha beta gamma delta epsilon", "gamma beta"]


def test_pool_equals_model2vec_encode_to_cosine_one_with_and_without_weights():
    for weights in (True, False):
        m = _model(weights=weights)
        got = se.pool(m, TEXTS, step=2)                                   # a step smaller than the corpus: batch seams are exercised
        want = np.vstack([m.encode([t], max_length=None) for t in TEXTS])
        cos = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
        assert np.allclose(cos, 1.0, atol=1e-5), cos
        assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)


def test_pool_is_the_same_when_texts_span_gather_blocks(monkeypatch):
    m = _model()
    texts = TEXTS + [" ".join(WORDS * 40), "", "eta"]
    whole = se.pool(m, texts)
    monkeypatch.setattr(se, "BLOCK", 7)                                     # a text of 320 tokens now spans 46 blocks, seams fall mid-text
    split = se.pool(m, texts, step=3)
    assert np.allclose(whole, split, atol=1e-5)
    want = np.vstack([m.encode([t], max_length=None) if t else np.zeros(6) for t in texts])
    assert np.allclose(split, want, atol=1e-5)


def test_pool_switches_off_a_tokenizers_padding_and_truncation_so_nothing_is_padded_or_cut():
    m = _model(weights=False)
    m.tokenizer.enable_padding(pad_id=m.embedding.shape[0] + 3, pad_token="[PAD]")       # an id with no row, like jina's 128004
    m.tokenizer.enable_truncation(max_length=4)                                          # and a cut far below the text length
    texts = ["alpha beta gamma delta epsilon zeta eta theta", "alpha"]
    got = se.pool(m, texts)
    ids = m.tokenize(texts, max_length=None)
    assert [len(x) for x in ids] == [8, 1]                                                # not padded to 8, not cut to 4
    ref = _model(weights=False)
    want = np.vstack([ref.encode([t], max_length=None) for t in texts])
    assert np.allclose(got, want, atol=1e-5)


def test_pool_does_not_truncate_at_512_tokens_where_encode_does():
    m = _model(weights=False)
    long = " ".join(["alpha"] * 600 + ["theta"] * 600)                      # the second half is past token 512
    full = se.pool(m, [long])[0]
    cut = m.encode([long])                                                  # model2vec's default: max_length 512
    only_alpha = se.pool(m, [" ".join(["alpha"] * 1200)])[0]
    assert not np.allclose(full, only_alpha, atol=1e-3)                    # the pooled vector sees theta
    assert np.allclose(cut / np.linalg.norm(cut), only_alpha, atol=1e-3)   # encode did not: it saw only the first 512 alphas


def test_pool_gives_the_zero_vector_for_an_empty_text_and_keeps_neighbours_intact():
    m = _model()
    got = se.pool(m, ["alpha beta", "", "gamma delta"], step=3)
    assert np.allclose(got[1], 0.0)
    assert np.allclose(got[0], m.encode(["alpha beta"])[0], atol=1e-5) and np.allclose(got[2], m.encode(["gamma delta"])[0], atol=1e-5)


def test_pool_survives_ids_with_no_row_where_model2vec_raises():
    m = _model(rows=5)                                                      # rows for alpha..epsilon only; zeta, eta, theta, [UNK] have none
    with pytest.raises(IndexError):
        m.encode(["alpha zeta"], max_length=None)
    got = se.pool(m, ["alpha zeta", "alpha"])
    assert np.allclose(got[0], got[1], atol=1e-5)                          # zeta added nothing, so the direction is alpha's


def test_center_removes_the_shared_mean_vector_and_leaves_unit_rows():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(40, 16)).astype(np.float32) + 3.0                 # every row shares a large common component
    X, mu = se.center(E)
    assert np.allclose(mu, E.mean(0)) and np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    D = E - E.mean(0)
    want = (D / np.linalg.norm(D, axis=1, keepdims=True)) @ (D / np.linalg.norm(D, axis=1, keepdims=True)).T
    assert np.allclose(X @ X.T, want, atol=1e-5)                           # the cosine of the rows' deviations from the sample mean
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    assert (En @ En.T).mean() > 0.9 > (X @ X.T).mean() + 0.5               # uncentred cosine is ~1 for everything: the mean vector is what centring removes
    assert not np.allclose(X @ X.T, np.corrcoef(E), atol=1e-2)             # NOT np.corrcoef, which removes each ROW's own mean


@pytest.mark.skipif(not os.path.isdir(SAVED), reason="the saved distilled model is not on this machine")
def test_pool_matches_the_saved_distilled_model_on_real_text_including_its_pad_ids():
    m = StaticModel.from_pretrained(SAVED)
    texts = ["Retrieval-augmented generation retrieves passages before answering.", "x", "", "word " * 700 + "end", "<|finetune_right_pad_id|> weird token"]
    got = se.pool(m, texts)
    rows = m.embedding.shape[0]
    checked = 0
    for t, g in zip(texts, got):
        ids = m.tokenize([t], max_length=None)[0]
        if not ids or any(i >= rows for i in ids):
            continue                                                        # model2vec itself cannot encode these; pool's rule is pinned above
        want = m.encode([t], max_length=None)[0]
        assert np.dot(g, want) / (np.linalg.norm(g) * np.linalg.norm(want)) > 0.99999
        checked += 1
    assert checked >= 3
