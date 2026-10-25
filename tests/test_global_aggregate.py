"""B2/T1 contracts for canonical-paper exact lexical indexing (R1-R8)."""
from __future__ import annotations

import random
import sqlite3

import config
import psycopg
import pytest

import global_aggregate as ga


def synthetic_rows() -> list[dict]:
    """Return ordered rows spanning three communities and main/method pairs."""
    raw = [
        ("p01", 1, "alpha beta gamma delta epsilon zeta eta theta iota kappa model-XT42 transformer"),
        ("p01", 1, "alpha repeated"),
        ("p01_methods", 2, "alpha methods gamma model-XT42"),
        ("p02", 1, "alpha beta eta"),
        ("p03", 2, "beta gamma iota"),
        ("p04", 2, "delta eta theta"),
        ("p05", 3, "alpha epsilon kappa lambda"),
        ("p06", 2, "beta zeta theta omega"),
        ("p06_methods", 3, "iota methods"),
    ]
    rows = []
    per_doc: dict[str, int] = {}
    for doc_id, community, text in raw:
        section_idx = per_doc.get(doc_id, 0)
        per_doc[doc_id] = section_idx + 1
        rows.append({"paper_id": ga.canonical_paper_id(doc_id), "community_id": community,
                     "doc_id": doc_id, "section_idx": section_idx, "text": text})
    return sorted(rows, key=lambda row: (row["paper_id"], row["doc_id"], row["section_idx"]))


def test_full_df_stays_exact_outside_top_k_and_counts_canonical_papers_once():
    """R1-R5: route truncation cannot change exact per-paper document counts."""
    rows = synthetic_rows()
    papers = list(ga.iter_canonical_papers(rows))
    index = ga.build_index(rows, top_k=1)

    assert len(papers) == 6
    assert index.paper_community["p01"] == 1  # two main sections beat one methods section
    assert index.paper_community["p06"] == 2  # tied mode resolves to the lower community ID
    assert set(index.n_docs) == {1, 2, 3}
    terms = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa")
    for term in terms:
        direct = sum(term in paper.tokens for paper in papers)
        summed_df = sum(index.df[cid].get(term, 0) for cid in index.n_docs)
        assert summed_df == direct == index.global_df.get(term, 0)
        assert ga.exact_tally(index, term) == direct

    signature_terms = {term for rows in index.signature.values() for term, _score, _df in rows}
    omitted = next(term for term in index.global_df if term not in signature_terms)
    assert index.global_df[omitted] > 0
    assert ga.exact_tally(index, omitted) == index.global_df[omitted]
    assert ga.exact_tally(index, "never-seen") == 0


def test_t3_keeps_digit_bearing_identifier_as_one_exact_term():
    """R2/R7: the index and exact query share the t3 identifier token."""
    index = ga.build_index(synthetic_rows(), top_k=100)

    assert ga.normalize_single_term("MODEL-XT42") == "model-xt42"
    assert ga.exact_tally(index, "model-XT42") == 1
    with pytest.raises(ValueError, match="one t3 token"):
        ga.normalize_single_term("alpha beta")


def test_eval_cache_round_trip_and_source_identity_gate(tmp_path, monkeypatch):
    """R21: exact eval counts reload only for matching artifact and source identities."""
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"record_type":"manifest","build_id":1163}\n', encoding="utf-8")
    cache_path = tmp_path / "eval.sqlite"
    cache_key, artifact_sha = ga.eval_cache_identity(eval_path, 1163)
    index = ga.build_index(synthetic_rows(), top_k=500)

    ga.save_eval_index_cache(cache_path, cache_key, 1163, artifact_sha, index)
    loaded = ga.load_eval_index_cache(cache_path, cache_key, 1163, artifact_sha)
    assert loaded is not None
    assert loaded.n_docs == index.n_docs
    assert loaded.df == index.df
    assert loaded.global_df == index.global_df
    assert loaded.paper_community == index.paper_community
    assert loaded.paper_sha256 == index.paper_sha256

    changed_key = "changed-artifact-key"
    assert ga.load_eval_index_cache(cache_path, changed_key, 1163, artifact_sha) is None

    rows = [{**row, "community_id": index.paper_community[row["paper_id"]]}
            for row in synthetic_rows()]
    monkeypatch.setattr(ga, "iter_source_rows", lambda _conn, _build, _ids: iter(rows))
    expected = {paper_id: {"primary_community": community_id,
                           "text_sha256": index.paper_sha256[paper_id]}
                for paper_id, community_id in index.paper_community.items()}
    ga.verify_eval_source_identity(object(), 1163, expected)
    changed = dict(expected)
    paper_id = next(iter(changed))
    changed[paper_id] = {**changed[paper_id], "text_sha256": "stale"}
    with pytest.raises(RuntimeError, match="text hash mismatch"):
        ga.verify_eval_source_identity(object(), 1163, changed)


def test_eval_cache_rejects_incomplete_matching_checkpoint(tmp_path):
    """R21: a matching but unpublished SQLite checkpoint is corruption, not a cache miss."""
    cache_path = tmp_path / "eval.sqlite"
    with sqlite3.connect(cache_path) as db:
        for statement in ga.EVAL_CACHE_DDL:
            db.execute(statement)
        db.execute("INSERT INTO eval_cache_meta VALUES (?,?,?,?,0,0,0)",
                   ("key", 1163, ga.TOKENIZER_VERSION, "digest"))

    with pytest.raises(RuntimeError, match="incomplete or inconsistent"):
        ga.load_eval_index_cache(cache_path, "key", 1163, "digest")


def test_t4_synthetic_coverage_and_endpoint_variance_over_200_repeats():
    """R9/R10: estimated totals retain near-z coverage and endpoint uncertainty."""
    populations = {
        11: [0] * 80,
        22: [1] * 42 + [0] * 78,
        33: [1] * 100,
    }
    sample_sizes = {11: 16, 22: 20, 33: 25}
    known_total = sum(sum(values) for values in populations.values())
    rng = random.Random(731)
    covered = 0

    for _ in range(200):
        strata = {}
        for community_id, values in populations.items():
            sample = rng.sample(values, sample_sizes[community_id])
            strata[community_id] = {"N": len(values), "n": len(sample),
                                    "yes": sum(sample), "unsure": 0}
        result = ga.stratified_estimate(strata)
        covered += result.interval[0] <= known_total <= result.interval[1]
        assert result.variance_lower > 0
        assert result.variance_upper > 0

    assert abs(covered / 200.0 - 0.95) <= 0.06
    unsure = ga.stratified_estimate({1: {"N": 10, "n": 5, "yes": 1, "unsure": 2}})
    assert unsure.estimate == 4.0
    assert unsure.audit[0]["p_yes"] == 0.2
    assert unsure.audit[0]["p_possible"] == 0.6


def test_lexical_route_ranks_stably_expands_cutoff_ties_and_keeps_global_population():
    """R6/R7/R11: ties widen routing output but never alter exact global counts."""
    rows = []
    for community_id, prefix in ((1, "a"), (2, "b"), (3, "c")):
        rows.extend([
            {"paper_id": prefix + "1", "community_id": community_id, "doc_id": prefix + "1",
             "section_idx": 0, "text": "marker red"},
            {"paper_id": prefix + "2", "community_id": community_id, "doc_id": prefix + "2",
             "section_idx": 0, "text": "quiet blue"},
        ])
    index = ga.build_index(rows, top_k=10)

    route = ga.route_lexical(index, "marker", top_k=1, tie_margin=0.01)
    assert route["top_k"][0][0] == 1
    assert [cid for cid, _score in route["selected"]] == [1, 2, 3]
    assert route["tie_expanded"]
    assert ga.exact_tally(index, "marker") == 3

    unknown = ga.route_lexical(index, "unknown", top_k=1, tie_margin=0.1)
    assert len(unknown["selected"]) == 3
    assert unknown["matched_terms"] == []


def test_predicate_sampler_and_json_label_contract():
    """R12-R14: deterministic SRS and exact yes/no/unsure response schema."""
    population = {"a": 1, "b": 1, "c": 1, "d": 2, "e": 2}
    first = ga.sample_paper_ids(population, community_id=1, sample_n=8, seed=42)
    second = ga.sample_paper_ids(population, community_id=1, sample_n=8, seed=42)
    assert first == second
    assert set(first) == {"a", "b", "c"}

    for label in ("yes", "no", "unsure"):
        result = ga.classify_predicate(
            "contrastive learning", "[abstract]\nThe method uses InfoNCE.",
            llm_call=lambda _system, _user, label=label: ('{"label":"%s"}' % label, "fixture"),
        )
        assert result["label"] == label
        assert result["attempts"] == 1
    with pytest.raises(ValueError, match="exactly the label"):
        ga.parse_predicate_label('{"label":"no","reason":"extra"}')
    with pytest.raises(ValueError, match="not valid JSON"):
        ga.parse_predicate_label("not-json")
    corrected = ga.classify_predicate(
        "contrastive learning", "partial first pages", truncated=True,
        llm_call=lambda _system, _user: ('{"label":"no"}', "fixture"),
    )
    assert corrected["model_label"] == "no"
    assert corrected["label"] == "unsure"
    assert corrected["truncation_override"]
    system, user = ga.predicate_prompt("contrastive learning", "visible section", truncated=True)
    assert "truncated before relevant evidence" in system
    assert "TRUNCATED: absence cannot be inferred" in user


def test_semantic_tally_pilots_all_strata_respects_budget_and_reuses_labels(tmp_path):
    """R9/R12/R15/R16/R21: semantic draws are stratified, bounded, auditable, and checkpointed."""
    population = {"a1": 1, "a2": 1, "a3": 1, "b1": 2, "b2": 2, "b3": 2,
                  "c1": 3, "c2": 3, "c3": 3}
    labels = {"a1": "yes", "a2": "no", "a3": "unsure", "b1": "no", "b2": "no",
              "b3": "yes", "c1": "unsure", "c2": "no", "c3": "no"}
    call_ids = []

    def load(ids, _max_chars):
        return {paper_id: {"text": paper_id, "truncated": False} for paper_id in ids}

    def classify(_predicate, paper_text, _truncated):
        call_ids.append(paper_text)
        return {"label": labels[paper_text]}

    cfg = ga.SemanticTallyConfig(budget=6, pilot_budget=3, pilot_per_stratum_cap=1,
                                 batch_size=3, target_halfwidth=0.00001, seed=17,
                                 classifier_version="fixture", cache_path=tmp_path / "labels.sqlite")
    result = ga.semantic_tally(population, "predicate", load, classify, cfg, 1163)
    assert result.calls == len(call_ids) <= cfg.budget
    assert result.cache_hits == 0
    assert result.mode == "estimated"
    assert not result.converged
    assert {row["community_id"] for row in result.audit} == {1, 2, 3}
    assert all(row["n_sampled"] >= 1 for row in result.audit)
    assert len(call_ids) == len(set(call_ids))
    assert all(row["yes"] + row["no"] + row["unsure"] == row["n_sampled"] for row in result.audit)

    repeat_calls = []
    repeated = ga.semantic_tally(
        population, "predicate", load,
        lambda _predicate, paper_text, _truncated: (repeat_calls.append(paper_text) or {"label": labels[paper_text]}),
        cfg, 1163,
    )
    assert repeated.calls == 0
    assert repeated.cache_hits == len(call_ids)
    assert repeat_calls == []
    assert repeated.audit == result.audit


def test_semantic_tally_census_and_truncated_negative_are_explicit(tmp_path):
    """R10/R14/R15: census removes sampling error while truncation remains explicit uncertainty."""
    population = {"a": 1, "b": 1, "c": 2, "d": 2, "e": 3, "f": 3}
    truncated = {"a"}
    cfg = ga.SemanticTallyConfig(budget=10, pilot_budget=3, pilot_per_stratum_cap=1,
                                 batch_size=3, target_halfwidth=1e-8, seed=3,
                                 classifier_version="census", cache_path=tmp_path / "census.sqlite")
    result = ga.semantic_tally(
        population, "predicate",
        lambda ids, _max_chars: {paper_id: {"text": paper_id, "truncated": paper_id in truncated} for paper_id in ids},
        lambda _predicate, _text, _cut: {"label": "no"}, cfg, 1163,
    )

    assert not result.converged
    assert result.calls == len(population)
    assert result.interval == (0.0, 1.0)
    assert result.audit[0]["unsure"] == 1
    assert result.audit[0]["no"] == 1


def test_neyman_allocation_honors_floor_capacities_and_budget():
    """R14: optional reallocation remains bounded and retains the defensive floor."""
    allocation = ga.allocate_neyman({1: 2, 2: 20, 3: 20}, {1: 100.0, 2: 1.0, 3: 1.0}, 10, 0.3)

    assert sum(allocation.values()) == 10
    assert allocation[1] <= 2
    assert allocation[2] >= 1
    assert allocation[3] >= 1


def test_measured_neyman_reallocation_is_the_default():
    """R27: matched eval evidence promotes Neyman allocation while preserving an explicit control switch."""
    assert ga.SemanticTallyConfig().use_reallocation is True
    assert ga.SemanticTallyConfig(use_reallocation=False).use_reallocation is False


def test_sequential_intervals_cover_all_looks_and_adaptive_strata():
    """R23: finite-population CSs retain simultaneous coverage under optional stopping."""
    populations = {1: [0] * 15, 2: [1] * 8 + [0] * 12, 3: [1] * 12}
    truth = sum(sum(values) for values in populations.values())
    all_look_covered = 0
    for seed in range(200):
        rng = random.Random(seed)
        orders = {cid: rng.sample(values, len(values)) for cid, values in populations.items()}
        strata = {cid: {"N": len(values), "n": 1, "yes": orders[cid][0], "unsure": 0}
                  for cid, values in populations.items()}
        bounds = {}
        covered = True
        while True:
            result = ga.sequential_estimate(strata, prior_bounds=bounds)
            covered &= result.interval[0] <= truth <= result.interval[1]
            eligible = [cid for cid in strata if strata[cid]["n"] < strata[cid]["N"]]
            if not eligible:
                assert result.interval == (float(truth), float(truth))
                break
            # Data-dependent allocation stresses the actual optional-stopping assumption.
            cid = max(eligible, key=lambda c: strata[c]["N"] * (strata[c]["yes"] + .5) / (strata[c]["n"] + 1))
            sample = strata[cid]
            sample["yes"] += orders[cid][sample["n"]]
            sample["n"] += 1
        all_look_covered += covered
    assert all_look_covered / 200 >= 0.94


def test_interrupted_tally_keeps_each_completed_label(tmp_path):
    """R24: one later model failure must not discard earlier completed paid work."""
    population = {"a": 1, "b": 2, "c": 3}
    cfg = ga.SemanticTallyConfig(budget=3, pilot_budget=3, cache_path=tmp_path / "labels.sqlite")
    seen = []

    def failing_classifier(_predicate, text, _truncated):
        seen.append(text)
        if len(seen) == 2:
            raise RuntimeError("injected provider failure")
        return {"label": "no"}

    loader = lambda ids, _cap: {pid: {"text": pid, "truncated": False} for pid in ids}
    with pytest.raises(RuntimeError, match="injected"):
        ga.semantic_tally(population, "predicate", loader, failing_classifier, cfg, 1163)
    with sqlite3.connect(cfg.cache_path) as db:
        assert db.execute("SELECT count(*) FROM semantic_label_cache").fetchone()[0] == 1
    resumed = ga.semantic_tally(population, "predicate", loader,
                                lambda *_: {"label": "no"}, cfg, 1163)
    assert resumed.calls == 2
    assert resumed.converged


def test_global_qa_schema_ddl_is_idempotent_and_rolls_back_cleanly():
    """R6: the checked-in migration applies twice without touching source rows."""
    conn = psycopg.connect(config.DSN)
    try:
        with conn.transaction():
            ga.ensure_schema(conn)
            ga.ensure_schema(conn)
            names = conn.execute(
                "SELECT to_regclass('global_qa_build'), to_regclass('global_qa_doc'), "
                "to_regclass('global_qa_community'), to_regclass('global_qa_term_df'), "
                "to_regclass('global_qa_signature')"
            ).fetchone()
            assert all(names)
    finally:
        conn.rollback()
        conn.close()


def test_persistence_publishes_complete_marker_and_tally_validates_scope():
    """R3/R4/R6/R7: transactional persistence supports exact lookup only at matching scope."""
    conn = psycopg.connect(config.DSN)
    try:
        with conn.transaction():
            conn.execute("CREATE TEMP TABLE sect_build (build_id bigint PRIMARY KEY)")
            conn.execute("CREATE TEMP TABLE sect_node (build_id bigint, doc_id text, community integer)")
            conn.execute("CREATE TEMP TABLE global_qa_build (qa_build_id bigserial PRIMARY KEY, source_build_id bigint, "
                         "tokenizer_version text, signature_top_k integer, n_docs integer, complete boolean DEFAULT false)")
            conn.execute("CREATE TEMP TABLE global_qa_doc (qa_build_id bigint, paper_id text, community_id integer, text_sha256 text)")
            conn.execute("CREATE TEMP TABLE global_qa_community (qa_build_id bigint, community_id integer, n_docs integer)")
            conn.execute("CREATE TEMP TABLE global_qa_term_df (qa_build_id bigint, community_id integer, term text, df integer)")
            conn.execute("CREATE TEMP TABLE global_qa_signature (qa_build_id bigint, community_id integer, term text, keyness double precision, df integer)")
            source_id = 987654321
            conn.execute("INSERT INTO sect_build VALUES (%s)", (source_id,))
            conn.cursor().executemany("INSERT INTO sect_node VALUES (%s,%s,%s)",
                                      [(source_id, row["doc_id"], row["community_id"]) for row in synthetic_rows()])

            index = ga.build_index(synthetic_rows(), top_k=1)
            qa_id = ga.persist_index(conn, source_id, index, signature_top_k=1)
            assert conn.execute("SELECT complete FROM global_qa_build WHERE qa_build_id=%s", (qa_id,)).fetchone()[0]
            assert ga.lexical_tally(conn, source_id, "alpha")["count"] == index.global_df["alpha"]
            assert ga.lexical_tally(conn, source_id, "never-seen")["count"] == 0
            with pytest.raises(RuntimeError, match="no complete"):
                ga.lexical_tally(conn, source_id + 1, "alpha")
            with pytest.raises(ValueError, match="does not match"):
                ga.lexical_tally(conn, source_id, "alpha", tokenizer_version="wrong")
    finally:
        conn.rollback()
        conn.close()


def test_build_1163_eval_slice_matches_manifest():
    """R8: 1,000-paper source reconstruction must match all lexical gold tallies."""
    from pathlib import Path

    with psycopg.connect(config.DSN, connect_timeout=5) as conn:
        result = ga.verify_eval_slice(conn, Path(__file__).resolve().parents[1] /
                                      "docs" / "cookbook" / "global_eval_set.jsonl")
        cached = ga.verify_eval_slice(conn, Path(__file__).resolve().parents[1] /
                                      "docs" / "cookbook" / "global_eval_set.jsonl")

    assert result["n_papers"] == 1000
    assert cached["cache_hit"]
    assert len(result["lexical_counts"]) == 10
    assert result["routing_metrics"]["signature_top_k_100"]["n_questions"] == 10
    assert result["routing_metrics"]["signature_top_k_500"]["n_questions"] == 10
    for metrics in result["routing_metrics"].values():
        assert 0.0 <= metrics["tie_frequency"] <= 1.0
