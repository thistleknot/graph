"""Exercises salient_grams.signature_pooled_df() and its eligibility wiring (R8),
plus the opt-in keyness prior (R9).

Pooling is corpus-native: two terms pool only when the residue pair splitting
them off a shared core recurs across >= min_productivity OTHER cores
(Goldsmith signature productivity). There is no lexicon and no nltk data
dependency — the "no lexicon" guarantee is itself asserted below.

Note the gate is strictly more conservative than the superseded snowball/
WordNet tiers: a lone pair (convert/conversion with nothing else attesting
the same residue pair) does NOT pool. The wiring fixture therefore supplies a
genuinely attested pattern.

The wiring tests drive the real select_vocab()/select_vocab_unified() so the
flag is proven to change selection, not just the returned array.

Run:  pytest tests/test_salient_grams.py -v      (no container needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import salient_grams as sg


# ------------------------------------------------- signature productivity


def test_pools_family_under_attested_residue_pair():
    """(e,ion) is attested by 5 cores -> 4 others -> clears min_productivity=3."""
    terms = np.array(["operate", "operation", "educate", "education",
                      "allocate", "allocation", "indicate", "indication",
                      "calculate", "calculation"])
    out = sg.signature_pooled_df(terms, np.ones(len(terms), float))
    assert list(out) == [2.0] * 10


def test_lone_pair_does_not_pool():
    """convert/conversion share core 'conver' but ('sion','t') has no other
    attesting core — the attestation gate refuses it."""
    out = sg.signature_pooled_df(
        np.array(["convert", "conversion", "pump", "shaft"]),
        np.ones(4, float))
    assert list(out) == [1.0, 1.0, 1.0, 1.0]


def test_productivity_threshold_is_the_gate():
    """Exactly at the threshold pools; one attesting core fewer does not."""
    at = ["operate", "operation", "educate", "education",
          "allocate", "allocation", "indicate", "indication"]   # 4 cores
    below = at[:6]                                              # 3 cores
    out_at = sg.signature_pooled_df(np.array(at), np.ones(len(at), float))
    out_below = sg.signature_pooled_df(np.array(below), np.ones(len(below), float))
    assert list(out_at) == [2.0] * 8          # 4 cores -> 3 others -> fires
    assert list(out_below) == [1.0] * 6       # 3 cores -> 2 others -> refused


def test_min_productivity_is_tunable():
    six = ["operate", "operation", "educate", "education", "allocate", "allocation"]
    terms = np.array(six)
    strict = sg.signature_pooled_df(terms, np.ones(6, float), min_productivity=3)
    loose = sg.signature_pooled_df(terms, np.ones(6, float), min_productivity=2)
    assert list(strict) == [1.0] * 6
    assert list(loose) == [2.0] * 6


def test_digit_bearing_terms_never_pool():
    """R3: identifiers are their own family regardless of surface overlap."""
    terms = np.array(["operate", "operation", "educate", "education",
                      "allocate", "allocation", "indicate", "indication",
                      "xr-2201b", "xr-2201c", "2201b"])
    out = sg.signature_pooled_df(terms, np.ones(len(terms), float))
    assert list(out[:8]) == [2.0] * 8
    assert list(out[8:]) == [1.0, 1.0, 1.0]


def test_result_is_elementwise_ge_input():
    """Stated guarantee: pooling only ever raises the floor."""
    terms = np.array(["operate", "operation", "educate", "education",
                      "allocate", "allocation", "indicate", "indication",
                      "pump", "xr-9"])
    df = np.array([1, 3, 1, 1, 2, 1, 1, 5, 4, 7], float)
    out = sg.signature_pooled_df(terms, df)
    assert np.all(out >= df)


def test_short_terms_below_min_core_never_pool():
    """len(t) must exceed min_core (4) to contribute a core at all."""
    terms = np.array(["cat", "cats", "dog", "dogs", "bat", "bats", "rat", "rats"])
    out = sg.signature_pooled_df(terms, np.ones(len(terms), float))
    assert list(out) == [1.0] * 8


def test_known_residue_false_pair_pools_when_corpus_attests_pattern():
    """Documented residue, asserted so it stays visible rather than surprising:
    provenance/proven pool under a productive (,ance) — 4 cores attest it."""
    terms = np.array(["provenance", "proven", "perform", "performance",
                      "accept", "acceptance", "attend", "attendance"])
    out = sg.signature_pooled_df(terms, np.ones(len(terms), float))
    assert out[0] == 2.0 and out[1] == 2.0


def test_no_lexicon_dependency(monkeypatch):
    """Guarantee: no nltk, no WordNet, no network — blocked imports must not
    break pooling."""
    for mod in list(sys.modules):
        if mod.startswith("nltk"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path])

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def blocked(name, *a, **kw):
        if name.startswith("nltk"):
            raise ImportError("nltk is not available")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", blocked)
    terms = np.array(["operate", "operation", "educate", "education",
                      "allocate", "allocation", "indicate", "indication"])
    out = sg.signature_pooled_df(terms, np.ones(8, float))
    assert list(out) == [2.0] * 8


# -------------------------------------------------------------- wiring


@pytest.fixture(scope="module")
def rescue_corpus():
    """5 docs, N=5. operate/operation sit at df=1 each (dead under a df_min=2
    floor) and are rescued only by pooling. The four attesting pairs
    (educate/education, allocate/allocation, indicate/indication,
    calculate/calculation) appear in every doc, so they are df=5 glue —
    present in the vocabulary to attest the (e,ion) pair, but never selectable
    themselves. 'the/pump/torque/and/shaft/drive' are glue at df=5, above the
    df_max_frac=0.6 ceiling (3.0).
    """
    attest = ("educate education allocate allocation indicate indication "
              "calculate calculation")
    glue = "the pump torque and shaft drive"
    return [
        f"{glue} operate {attest} housing valve seal bearing gasket",
        f"{glue} operation {attest} housing valve seal rotor blade",
        f"{glue} impeller {attest} housing manifold flange rotor blade",
        f"{glue} impeller {attest} casing manifold flange bearing gasket",
        f"{glue} impeller {attest} casing manifold seal bearing blade",
    ]


def _run(fn, corpus, **kw):
    return fn(corpus, background=None, df_min=2, df_max_frac=0.6, **kw)


def test_select_vocab_rescues_family_only_with_flag(rescue_corpus):
    off = _run(sg.select_vocab, rescue_corpus, waterline=50)["terms"]
    on = _run(sg.select_vocab, rescue_corpus, waterline=50, pooled_df=True)["terms"]
    assert "operate" not in off and "operation" not in off
    assert "operate" in on and "operation" in on
    # rescued-and-selected count > 0 (the promotion test)
    assert len(set(on) - set(off)) >= 2


def test_select_vocab_unified_rescues_family_only_with_flag(rescue_corpus):
    off = _run(sg.select_vocab_unified, rescue_corpus,
               target_chars_per_row=1000, anomaly=False, spline_refine=False)["terms"]
    on = _run(sg.select_vocab_unified, rescue_corpus,
              target_chars_per_row=1000, anomaly=False, spline_refine=False,
              pooled_df=True)["terms"]
    assert "operate" not in off and "operation" not in off
    assert "operate" in on and "operation" in on


def test_pooled_df_defaults_off_leaves_selection_unchanged(rescue_corpus):
    base = _run(sg.select_vocab, rescue_corpus, waterline=50)["terms"]
    explicit_off = _run(sg.select_vocab, rescue_corpus, waterline=50,
                        pooled_df=False)["terms"]
    assert base == explicit_off


def test_upper_band_stays_per_term(rescue_corpus):
    """Pooling must not push a family over the df_max_frac glue cutoff.

    'pump' is corpus glue (df == N). With a tight upper band it is excluded
    with the flag on exactly as with it off — the ceiling is per-term.
    """
    on = sg.select_vocab(rescue_corpus, waterline=50, background=None,
                         df_min=2, df_max_frac=0.6, pooled_df=True)["terms"]
    assert "pump" not in on
    assert "shaft" not in on
    # the attesting glue pairs are pooled (df 5+5=10) yet still excluded
    assert "educate" not in on and "education" not in on


# ------------------------------------------------- R9: prior is opt-in


def test_unified_subtracts_no_prior_by_default(rescue_corpus, monkeypatch):
    """R9: background=None SHALL subtract no prior — keyness() is never
    called and every reported keyness score is exactly 0.0."""
    calls = []
    real_keyness = sg.keyness
    monkeypatch.setattr(sg, "keyness",
                        lambda *a, **kw: (calls.append(a[2]),
                                          real_keyness(*a, **kw))[1])
    res = sg.select_vocab_unified(rescue_corpus, background=None, df_min=2,
                                  df_max_frac=0.6, target_chars_per_row=1000,
                                  anomaly=False, spline_refine=False)
    assert calls == []                       # no prior resolved, none applied
    assert res["terms"]                      # still selects a vocabulary
    assert res["keyness"] == [0.0] * len(res["terms"])


def test_wordfreq_prior_is_opt_in(rescue_corpus, monkeypatch):
    """R9: background='wordfreq' SHALL resolve to wordfreq top-10k."""
    seen = {}
    real_keyness = sg.keyness

    def spy(terms, TF, background, **kw):
        seen["bg"] = background
        return real_keyness(terms, TF, background, **kw)

    monkeypatch.setattr(sg, "keyness", spy)
    res = sg.select_vocab_unified(rescue_corpus, background="wordfreq",
                                  df_min=2, df_max_frac=0.6,
                                  target_chars_per_row=1000,
                                  anomaly=False, spline_refine=False)
    bg = seen["bg"]
    assert isinstance(bg, list) and len(bg) == 10000
    assert bg[0] == "the"
    # resolution well past rank 100 is the whole point of the swap
    assert bg.index("advise") + 1 == 6111
    # a real prior must actually move scores off the no-prior zero
    assert any(k != 0.0 for k in res["keyness"])


def test_unknown_background_preset_raises(rescue_corpus):
    """A misspelled preset must fail loudly, not silently score with no prior."""
    with pytest.raises(ValueError, match="unknown background preset"):
        sg.select_vocab_unified(rescue_corpus, background="wordfrequency",
                                df_min=2, df_max_frac=0.6,
                                target_chars_per_row=1000,
                                anomaly=False, spline_refine=False)


def test_no_hardcoded_fallback_list_ships():
    """R9: a hardcoded word list SHALL NOT ship as a silent fallback."""
    src = Path(sg.__file__).read_text(encoding="utf-8")
    assert "_EN_TOP" not in src.split("CLOSED (rejected on evidence")[0]
    assert not hasattr(sg, "_EN_TOP")


def test_explicit_background_still_honoured(rescue_corpus, monkeypatch):
    """An explicitly supplied prior must bypass the wordfreq default."""
    seen = {}
    real_keyness = sg.keyness

    def spy(terms, TF, background, **kw):
        seen["bg"] = background
        return real_keyness(terms, TF, background, **kw)

    monkeypatch.setattr(sg, "keyness", spy)
    mine = ["the", "pump", "torque"]
    sg.select_vocab_unified(rescue_corpus, background=mine, df_min=2,
                            df_max_frac=0.6, target_chars_per_row=1000,
                            anomaly=False, spline_refine=False)
    assert seen["bg"] == mine
