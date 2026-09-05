"""Pins R10: vocabulary selection as a constrained program.

Three rows can bind, and the correct greedy rule differs by which one does:

    weight-bound (chars/row)  -> rank by value DENSITY u/cost   (knapsack)
    cardinality-bound (K)     -> rank by VALUE                  (top-k)
    neither                   -> eligibility is the real limit

The distinction is not cosmetic. Density under a cardinality cap buys cheap
low-value terms a term-count budget has no reason to prefer; value under a
weight cap leaves value on the table. These tests hold both rules in place and
pin the diagnostic that says which one ran.

Fixtures are synthetic so the constraint arithmetic is exact and no database or
corpus download is involved.

Run:  pytest tests/test_vocab_constraints.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

import salient_grams as sg


TOPICS = [
    "purchasing department clerical personnel policies experienced",
    "jury election county grand investigation courthouse",
    "molecular symmetry structure crystal lattice diffraction",
    "hypothalamic autonomic regulation stimulation cortical thalamus",
    "textile machinery production looms spindles manufacturing",
    "fiscal appropriations legislature revenue expenditure treasury",
    "orchestra symphony conductor rehearsal repertoire soloist",
    "sediment stratigraphy outcrop limestone fossil deposition",
    "photosynthesis chloroplast membrane pigment absorption",
    "cathedral vaulting buttress transept nave masonry",
    "monsoon precipitation humidity barometric convection",
    "epistemology inference proposition premise deduction",
    "hydraulic turbine impeller cavitation reservoir penstock",
    "pediatric immunization antibody serum prophylaxis",
    "tessellation polygon vertices adjacency planar",
    "fermentation yeast enzyme substrate anaerobic",
    "cartography projection meridian latitude azimuth",
    "metallurgy annealing tempering alloy quenching",
    "phonology morpheme syllable consonant intonation",
    "arbitration grievance stewardship bargaining tribunal",
]


def corpus(per_topic=5):
    """Every topic word lands at df == per_topic, inside the default band.

    The default ceiling is df <= max(df_min, 0.10 * N); with 20 topics at 5
    documents each, N = 100 and the ceiling is 10, so df=5 terms survive while
    a corpus-wide filler would not. Getting this wrong masks every term and
    select_vocab_unified raises -- which is the correct behaviour (R7), just
    not what these tests are measuring.
    """
    docs = []
    for gi, topic in enumerate(TOPICS):
        bridge = TOPICS[(gi + 1) % len(TOPICS)].split()[0]   # df == 2*per_topic
        for _ in range(per_topic):
            docs.append(f"{topic} {bridge}")
    return docs


@pytest.fixture(scope="module")
def docs():
    return corpus()


# ---------------------------------------------------------------- which binds
def test_slack_budget_reports_eligibility(docs):
    """A budget that never fills is not a size dial, and must not claim to be."""
    r = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                spline_refine=False)
    assert r["binding_constraint"] == "eligibility"
    assert r["greedy_rule"] == "value"
    assert r["chars_per_row"] == pytest.approx(r["cost_if_all_eligible"], rel=1e-6)


def test_tight_budget_binds_and_switches_to_density(docs):
    full = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                   spline_refine=False)
    tight = full["cost_if_all_eligible"] / 3.0
    r = sg.select_vocab_unified(docs, target_chars_per_row=tight,
                                spline_refine=False)
    assert r["binding_constraint"] == "chars_per_row"
    assert r["greedy_rule"] == "density"
    assert r["chars_per_row"] <= tight + 1e-9
    assert r["n_selected"] < full["n_selected"]


def test_term_cap_binds_and_keeps_value_rule(docs):
    r = sg.select_vocab_unified(docs, target_chars_per_row=100000, max_terms=12,
                                spline_refine=False)
    assert r["binding_constraint"] == "max_terms"
    assert r["greedy_rule"] == "value"        # cardinality => top-k, not density
    assert r["n_selected"] <= 12


def test_whichever_is_reached_first_is_reported(docs):
    """With both rows tight, the label must name what actually stopped the loop
    -- otherwise you tune a dial that was never touching anything."""
    full = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                   spline_refine=False)
    r = sg.select_vocab_unified(docs,
                                target_chars_per_row=full["cost_if_all_eligible"] / 2,
                                max_terms=5, spline_refine=False)
    assert r["binding_constraint"] == "max_terms"
    assert r["n_selected"] <= 5


def test_term_cap_above_pool_does_not_bind(docs):
    a = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                spline_refine=False)
    b = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                max_terms=10**6, spline_refine=False)
    assert b["binding_constraint"] == "eligibility"
    assert b["n_selected"] == a["n_selected"]


# ---------------------------------------------------------------- hard filter
def test_length_cap_excludes_only_over_length_terms(docs):
    long_term = "supercalifragilisticexpialidociousandthensome"   # 44 chars
    plus = docs + [f"{long_term} {long_term} filler words here" for _ in range(4)]
    capped = sg.select_vocab_unified(plus, target_chars_per_row=100000,
                                     max_term_chars=35, spline_refine=False)
    assert all(len(t) <= 35 for t in capped["terms"])
    assert long_term not in capped["terms"]


def test_length_cap_is_inert_on_short_vocabulary(docs):
    """A pathology guard fires on nothing when there is no pathology."""
    a = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                max_term_chars=35, spline_refine=False)
    b = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                max_term_chars=10**6, spline_refine=False)
    assert a["terms"] == b["terms"]


# ---------------------------------------------------------------- cost model
def test_cost_carries_the_trigram_padding_tax(docs):
    """R10: cost is (len+1)*df/N. pg_trgm emits len+1 trigrams per token, so a
    model using len*df/N understates every term's index footprint."""
    r = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                spline_refine=False)
    n = len(r["docs"])
    naive = sum(len(t) * d / n for t, d in zip(r["terms"], r["df"]))
    padded = sum((len(t) + 1) * d / n for t, d in zip(r["terms"], r["df"]))
    assert r["chars_per_row"] > naive
    assert r["chars_per_row"] == pytest.approx(padded, rel=0.02)


def test_budget_is_never_exceeded(docs):
    full = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                   spline_refine=False)
    for frac in (0.2, 0.4, 0.7):
        b = full["cost_if_all_eligible"] * frac
        r = sg.select_vocab_unified(docs, target_chars_per_row=b,
                                    spline_refine=False)
        assert r["chars_per_row"] <= b + 1e-9


def test_diagnostics_echo_the_configured_rows(docs):
    r = sg.select_vocab_unified(docs, target_chars_per_row=777, max_terms=33,
                                max_term_chars=29, spline_refine=False)
    assert r["chars_budget"] == 777
    assert r["max_terms"] == 33
    assert r["max_term_chars"] == 29


def test_tighter_budget_never_yields_more_terms(docs):
    """Monotonicity: shrinking the budget cannot grow the vocabulary."""
    full = sg.select_vocab_unified(docs, target_chars_per_row=100000,
                                   spline_refine=False)
    prev = None
    for frac in (0.25, 0.5, 0.75, 1.0):
        r = sg.select_vocab_unified(docs,
                                    target_chars_per_row=full["cost_if_all_eligible"] * frac,
                                    spline_refine=False)
        if prev is not None:
            assert r["n_selected"] >= prev
        prev = r["n_selected"]
