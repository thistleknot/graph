"""Acceptance battery for the multi-source run (R19/R20/R21).

Spec: .spec/specs/graph-explorer/design.md §6.14 acceptance 5(a)-(c)
Task: playbook.md T9

(a) per-block intra edge rates within one order of magnitude, read from the run's
    stored diagnostics; (b) source-targeted prompts reach their own source;
(c) is covered by tests/test_sampler.py, tests/test_gist_walk.py and
    tests/test_walker_render.py passing unchanged in the same command — nothing
    is added here for it.

Prefers label `mixed-full`; falls back to `mixed-smoke` and SAYS SO (a UserWarning
plus the first line of .tmp/mixed_acceptance_report.txt).

DB-backed: skips cleanly, never fails, when Postgres :5433 is down.

Run:  pytest tests/test_mixed_acceptance.py -q
"""
from __future__ import annotations

import os
import warnings
from collections import Counter
from pathlib import Path

import pytest

import graph_tools as gt
import sampler as sp
from conftest import require_gt_conn

PREFERRED_LABEL = os.environ.get("MIXED_ACCEPTANCE_LABEL", "mixed-full")
FALLBACK_LABEL = "mixed-smoke"
SOURCES = ("brown", "quotes", "wiki")
EF = 24
MIN_DF = 3
N_PROMPTS = 3
RATIO_BOUND = 10.0

REPORT: list[str] = []


# --------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def live():
    conn = require_gt_conn()
    run = None
    used_fallback = False
    for label in (PREFERRED_LABEL, FALLBACK_LABEL):
        try:
            run = gt.get_run(conn, label)
            used_fallback = label != PREFERRED_LABEL
            break
        except LookupError:
            continue
    if run is None:
        conn.close()
        pytest.skip("neither mixed-full nor mixed-smoke has a live run")

    if used_fallback:
        warnings.warn(UserWarning(
            f"acceptance battery ran against {FALLBACK_LABEL!r}: "
            f"{PREFERRED_LABEL!r} has no live run yet (T8 still ingesting)"))

    REPORT.append(f"label={run.label} fallback={used_fallback}")
    yield conn, run
    conn.close()


@pytest.fixture(scope="module")
def mixed(live):
    conn, run = live
    mix = gt.run_sources(conn, run)
    missing = [s for s in SOURCES if mix.get(s, 0) == 0]
    if missing:
        pytest.skip(f"{run.label} is not a three-source run: {mix}")
    return mix


@pytest.fixture(scope="module")
def src_by_ord(live):
    conn, run = live
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ord, attrs->>'source' AS source FROM node WHERE run_id = %s",
            (run.run_id,))
        rows = cur.fetchall()
    return {r["ord"]: r["source"] for r in rows}


@pytest.fixture(scope="module", autouse=True)
def _evidence_report():
    yield
    try:
        import os
        os.makedirs(".tmp", exist_ok=True)
        with open(".tmp/mixed_acceptance_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(REPORT) + "\n")
    except OSError:
        pass


# --------------------------------------------------------- (a) edge rates
def test_intra_block_edge_rates_are_within_one_order_of_magnitude(live):
    conn, run = live
    with conn.cursor() as cur:
        cur.execute("SELECT diagnostics FROM live_run WHERE label = %s", (run.label,))
        row = cur.fetchone()
    diagnostics = (row and row["diagnostics"]) or {}

    intra_blocks = ["brown|brown", "quotes|quotes", "wiki|wiki"]
    checked = False
    for space in ("sparse", "dense"):
        space_diag = diagnostics.get(space)
        if not space_diag:
            continue
        checked = True
        by_block = {b["block"]: b for b in (space_diag.get("blocks") or [])}
        rates = {}
        for name in intra_blocks:
            b = by_block.get(name)
            assert b is not None, f"[{space}] intra block {name!r} missing from diagnostics"
            er = b.get("edge_rate")
            assert er is not None and er > 0.0, (
                f"[{space}] intra block {name!r} has edge_rate={er!r}")
            rates[name] = er
        ratio = max(rates.values()) / min(rates.values())
        fits = {name: by_block[name].get("fit") for name in intra_blocks}
        assert ratio <= RATIO_BOUND, (
            f"[{space}] intra edge rates out of range: {rates} fits={fits} "
            f"ratio={ratio:.3f} > {RATIO_BOUND}")
        REPORT.append(f"[{space}] rates={rates} fits={fits} ratio={ratio:.3f}")

    assert checked, "no sparse/dense diagnostics on this run"


# --------------------------------------------------- (b) source-targeted prompts
def distinctive_terms(conn, run, src_by_ord, source, k):
    ix = gt.corpus_index(conn, run)
    n_source = sum(1 for s in src_by_ord.values() if s == source)
    n_other = sum(1 for s in src_by_ord.values() if s is not None and s != source)
    n_source = n_source or 1
    n_other = n_other or 1

    scored = []
    for term, post in ix["post"].items():
        if gt.tokenize(term) != [term]:
            continue
        df_in = sum(1 for o in post if src_by_ord.get(o) == source)
        if df_in < MIN_DF:
            continue
        df_out = sum(1 for o in post if src_by_ord.get(o) not in (None, source))
        distinctiveness = (df_in / n_source) / ((df_out / n_other) + 1e-9)
        scored.append((distinctiveness, df_in, term))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [term for _, _, term in scored[:k]]


def prompts_for(terms, source):
    groups = [terms[0::3], terms[1::3], terms[2::3]]
    out = []
    for g in groups:
        if not g:
            continue
        if source == "quotes":
            out.append("a quote about " + " ".join(g))
        elif source == "wiki":
            out.append("how does wikipedia describe " + " ".join(g))
        else:
            out.append(" ".join(g))
    return out


def reach(conn, run, src_by_ord, prompt):
    bundle, _tele = sp.ef_evidence(conn, run, prompt, ef=EF, T=0.0, bridge_pairs=0)
    mix = Counter(src_by_ord.get(o) for o in bundle.sampled)
    REPORT.append(f"prompt={prompt!r} mix={dict(mix)} n={len(bundle.sampled)}")
    return mix, bundle


def _assert_reaches(conn, run, src_by_ord, prompt, target):
    mix, bundle = reach(conn, run, src_by_ord, prompt)
    assert bundle.sampled, f"{prompt!r}: empty bundle (no valid anchor, R1)"
    assert mix.get(target, 0) >= 1, (
        f"{prompt!r} targeted source={target} but the bundle surfaced {dict(mix)} "
        f"({len(bundle.sampled)} chunks, ef={EF}); run={run.label}")


def _run_battery(conn, run, src_by_ord, prompts, target):
    failures = []
    for prompt in prompts:
        try:
            _assert_reaches(conn, run, src_by_ord, prompt, target)
        except AssertionError as e:
            failures.append(str(e))
    assert not failures, "\n".join(failures)


def test_quotes_prompts_reach_quotes_chunks(live, mixed, src_by_ord):
    conn, run = live
    terms = distinctive_terms(conn, run, src_by_ord, "quotes", 9)
    prompts = prompts_for(terms, "quotes")
    assert len(prompts) >= N_PROMPTS, f"not enough distinctive quotes terms: {terms}"
    _run_battery(conn, run, src_by_ord, prompts, "quotes")


def test_wiki_prompts_reach_wiki_chunks(live, mixed, src_by_ord):
    conn, run = live
    terms = distinctive_terms(conn, run, src_by_ord, "wiki", 9)
    prompts = prompts_for(terms, "wiki")
    assert len(prompts) >= N_PROMPTS, f"not enough distinctive wiki terms: {terms}"
    _run_battery(conn, run, src_by_ord, prompts, "wiki")


def test_brown_prompts_still_anchor_in_brown(live, mixed, src_by_ord):
    conn, run = live
    ix = gt.corpus_index(conn, run)
    candidates = [
        "jury trial grand jury investigation",
        "school children teacher education",
        "molecular structure crystal",
        "church religious faith congregation",
    ]

    def df_in_brown(prompt):
        total = 0
        for term in gt.tokenize(prompt):
            post = ix["post"].get(term, {})
            total += sum(1 for o in post if src_by_ord.get(o) == "brown")
        return total

    used = [p for p in candidates if df_in_brown(p) >= 1]
    if len(used) < 3:
        extra_terms = distinctive_terms(conn, run, src_by_ord, "brown", 9)
        used += prompts_for(extra_terms, "brown")
    assert len(used) >= 3, f"fewer than 3 usable brown prompts: {used}"
    _run_battery(conn, run, src_by_ord, used, "brown")


def test_spec_literal_prompts(live, mixed, src_by_ord):
    conn, run = live
    ix = gt.corpus_index(conn, run)
    cases = [
        ("a quote about courage", "quotes"),
        ("how does wikipedia describe the history of the city", "wiki"),
    ]
    ran_any = False
    for prompt, target in cases:
        df_in = 0
        for term in gt.tokenize(prompt):
            post = ix["post"].get(term, {})
            df_in += sum(1 for o in post if src_by_ord.get(o) == target)
        if df_in == 0:
            REPORT.append(f"skip prompt={prompt!r}: topic absent from {target} in {run.label}")
            continue
        ran_any = True
        _assert_reaches(conn, run, src_by_ord, prompt, target)
    if not ran_any:
        pytest.skip("neither spec-literal prompt's topic is present in this run")
