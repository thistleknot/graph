"""Pins entity_derive.py's guards AE1-AE9, AE11 and AE13 on synthetic data (no corpus, no network, no database).

The n-gram counter is checked against a brute-force Python Counter; the vectorised NPMI against the incumbent scalar
entities.npmi_ppmi; the closed forms (entropy, G2, NPMI) against hand and scipy values.

Run:  pytest tests/test_entity_derive.py -v
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import entity_derive as ed


# ---------------------------------------------------------------------------------------------- helpers
def segs(text, **kw):
    return [(k, [t[0] for t in toks]) for k, toks in ed.tokenize_text(text, **kw)]


FUNC = "the of and in for with is are on to a we that this by from as it be".split()
CONTENT = "model data training results method approach performance task evaluation analysis system network learning loss".split()


def synthetic_papers(n_papers=40, seed=0):
    """40 papers, 2 chunks each. GRPO in papers 0-14, DeepSeekMath in 10-23, the phrase 'Group Relative Policy Optimization
    (GRPO)' in 0-19, Zorblax only in papers 0 and 1 (3 mentions). Everything else is function words and common content words.
    No planted entity is in more than half the papers: a term in nearly every paper is vocabulary (AE14)."""
    rng = np.random.default_rng(seed)
    texts, pids = [], []
    for p in range(n_papers):
        chunks = []
        for _ in range(2):
            sents = []
            for _ in range(4):
                w = [str(rng.choice(FUNC + CONTENT)) for _ in range(int(rng.integers(10, 18)))]
                w[0] = w[0].capitalize()
                if p < 15 and rng.random() < 0.5:
                    w.insert(int(rng.integers(2, len(w))), "GRPO")
                if 10 <= p < 24 and rng.random() < 0.5:
                    w.insert(int(rng.integers(2, len(w))), "DeepSeekMath")
                if p < 20 and rng.random() < 0.4:
                    i = int(rng.integers(2, len(w)))
                    w[i:i] = ["Group", "Relative", "Policy", "Optimization", "(GRPO)"]
                sents.append(" ".join(w) + ".")
            chunks.append(" ".join(sents))
        texts.extend(chunks)
        pids.extend(["p%d" % p] * 2)
    texts[0] += " We also tried Zorblax here. Then Zorblax again."
    texts[2] += " Later Zorblax won."
    return texts, pids


def row_of(c, f, gram_text):
    folded = gram_text.lower()
    for r in range(len(f.n)):
        n, s = int(f.n[r]), int(f.start[r])
        if " ".join(c.fold_vocab[c.fold_ids[s + k]] for k in range(n)) == folded:
            return r
    raise AssertionError("no candidate %r" % gram_text)


def make_features(rows, n_papers=100):
    base = dict(n=2, gram=0, start=0, modal=0, tf=10, df_papers=10, df_chunks=10, npmi_min=0.5, g2_min=20.0, h_left=2.0, h_right=2.0,
                h_var=0.0, p_cap=1.0, cap_den=10, ridf=1.0, nest=0.0, max_tok_len=5, df_first=1, pcap_first=1.0, df_last=1, pcap_last=1.0)
    cols = {k: np.array([r.get(k, v) for r in rows]) for k, v in base.items()}
    return ed.Features(**cols, n_papers=n_papers)


# ------------------------------------------------------------------------------------------------ tokenizer
@pytest.mark.parametrize("surf,shape", [("policy", ed.LOWER), ("Policy", ed.TITLE), ("GRPO", ed.UPPER), ("DeepSeekMath", ed.MIXED),
                                        ("LoRA", ed.MIXED), ("iPhone", ed.MIXED), ("GSM8K", ed.ALNUM), ("Llama-3.1-8B", ed.ALNUM),
                                        ("2024", ed.NUM), ("A", ed.TITLE), ("x", ed.LOWER)])
def test_shape_classes(surf, shape):
    assert ed.shape_of(surf) == shape


def test_names_with_digits_and_hyphens_stay_whole_and_sentences_split_without_breaking_abbreviations():
    got = segs("We evaluate Llama-3.1-8B on GSM8K. See Fig. 2 for details, e.g. this one.")
    assert got == [("prose", ["We", "evaluate", "Llama-3.1-8B", "on", "GSM8K"]),
                   ("prose", ["See", "Fig", "2", "for", "details", "e.g", "this", "one"])]


def test_math_comments_and_citations_are_hard_boundaries_AE2():
    assert segs("The $$x^2 + y$$ term and $z$ vary.") == [("prose", ["The"]), ("prose", ["term", "and"]), ("prose", ["vary"])]
    assert segs("before <!-- formula-not-decoded --> after") == [("prose", ["before"]), ("prose", ["after"])]
    assert segs("as shown in [12] and (Smith et al., 2020) we agree") == [("prose", ["as", "shown", "in"]), ("prose", ["and"]),
                                                                          ("prose", ["we", "agree"])]
    assert segs("see [1, 2] and [3-5] then") == [("prose", ["see"]), ("prose", ["and"]), ("prose", ["then"])]


def test_docling_glyph_placeholders_and_font_glyph_ids_are_boundaries_but_the_word_glyph_is_not_AE2b():
    got = segs("A glyph<c=3,font=/ABC+Cambria> mid glyph[lscript] text /gid00002/gid00030 end")
    assert got == [("prose", ["A"]), ("prose", ["mid"]), ("prose", ["text"]), ("prose", ["end"])]
    assert segs("the glyph shape") == [("prose", ["the", "glyph", "shape"])]
    assert segs("A glyph&lt;c=29&gt; mid GLYPH&lt;cmap:d835&gt; end") == [("prose", ["A"]), ("prose", ["mid"]), ("prose", ["end"])]   # the HTML-escaped form
    assert ed.TOKENIZER_VERSION == "t3"                                                    # a changed tokenizer must not match an older inventory


def test_a_token_in_nearly_every_paper_closes_a_gram_whatever_its_capitalisation_AE6b():
    f = make_features([dict(df_first=98, pcap_first=0.5), dict(df_last=98, pcap_last=0.5), dict(df_first=97, pcap_first=0.5),
                       dict(n=1, df_first=99, pcap_first=0.9, df_last=99, pcap_last=0.9)])
    s = ed.score(f)
    assert s[0] == 0 and s[1] == 0 and s[2] > 0 and s[3] == 0                              # 98 of 100 papers closes; 97 of 100 does not
    assert ed.score(f, ed.Params(universal_df=1.0))[0] > 0 and ed.score(f, ed.Params(universal_df=0.5))[2] == 0


def test_a_hyphen_at_a_line_break_joins_the_word_and_an_inline_hyphen_does_not():
    assert segs("pol-\nicy gradient") == [("prose", ["policy", "gradient"])]
    assert segs("fine-tuning helps") == [("prose", ["fine-tuning", "helps"])]
    assert segs("fine-tuning helps on Llama-3", hyphen_split=True) == [("prose", ["fine", "tuning", "helps", "on", "Llama-3"])]


def test_camel_case_is_never_split_and_a_possessive_is_dropped():
    toks = ed.tokenize_text("DeepSeekMath beats LoRA and the model's output")[0][1]
    assert [t[0] for t in toks] == ["DeepSeekMath", "beats", "LoRA", "and", "the", "model", "output"]
    assert [t[2] for t in toks][:3] == [ed.MIXED, ed.LOWER, ed.MIXED]
    assert [t[1] for t in toks][:1] == ["deepseekmath"] and [t[3] for t in toks][:2] == [True, False]


def test_headings_and_table_cells_are_their_own_segment_kinds_AE8():
    got = segs("## Results\n\n| Model | Score |\n|---|---|\n| GRPO | 0.5 |\n\nBody text here.")
    assert got == [("heading", ["Results"]), ("cell", ["Model"]), ("cell", ["Score"]), ("cell", ["GRPO"]), ("cell", ["0.5"]),
                   ("prose", ["Body", "text", "here"])]


def test_tokenize_corpus_numbers_papers_by_first_appearance_and_keeps_segments_apart():
    c = ed.tokenize_corpus(["Alpha beta. Gamma delta.", "## Head\n\nepsilon zeta"], ["pB", "pA"])
    assert c.papers == ["pB", "pA"] and c.n_papers == 2 and c.n_chunks == 2
    assert list(c.seg) == [0, 0, 1, 1, 2, 3, 3] and list(c.seg_chunk) == [0, 0, 1, 1]
    assert list(c.seg_kind) == [ed.PROSE, ed.PROSE, ed.HEADING, ed.PROSE] and list(c.seg_paper) == [0, 0, 1, 1]
    assert c.fold_vocab[c.fold_ids[0]] == "alpha" and c.surf_vocab[c.surf_ids[0]] == "Alpha"


# ---------------------------------------------------------------------------------------------- n-gram counts
def brute(texts, pids, max_n, min_sup, excluded=frozenset()):
    segments = []
    for ci, (t, pid) in enumerate(zip(texts, pids)):
        for _, toks in ed.tokenize_text(t):
            segments.append((pid, ci, [x[1] for x in toks]))
    surv, out = {}, {}
    for n in range(1, max_n + 1):
        cnt, dp, dc = Counter(), defaultdict(set), defaultdict(set)
        for pid, ci, toks in segments:
            if pid in excluded:
                continue
            for i in range(len(toks) - n + 1):
                g = tuple(toks[i:i + n])
                if n > 1 and (g[:-1] not in surv[n - 1] or (g[-1],) not in surv[1]):
                    continue
                cnt[g] += 1
                dp[g].add(pid)
                dc[g].add(ci)
        surv[n] = {g for g, c in cnt.items() if c >= min_sup}
        out[n] = {g: (cnt[g], len(dp[g]), len(dc[g])) for g in surv[n]}
    return out


def table_dict(c, tables):
    out = {}
    for t in tables:
        d = {}
        for gi in range(len(t.count)):
            s = int(t.start[gi])
            d[tuple(c.fold_vocab[c.fold_ids[s + k]] for k in range(t.n))] = (int(t.count[gi]), int(t.df_papers[gi]), int(t.df_chunks[gi]))
        out[t.n] = d
    return out


@pytest.mark.parametrize("seed,vocab,min_sup", [(0, 12, 2), (1, 40, 2), (2, 8, 3)])
def test_the_numpy_counter_equals_a_brute_force_counter_including_four_grams(seed, vocab, min_sup):
    rng = np.random.default_rng(seed)
    words = ["w%d" % i for i in range(vocab)]
    texts, pids = [], []
    for ci in range(30):
        sents = []
        for _ in range(int(rng.integers(2, 5))):
            w = [str(rng.choice(words, p=np.arange(vocab, 0, -1) / np.arange(vocab, 0, -1).sum())) for _ in range(int(rng.integers(5, 14)))]
            sents.append(w[0].capitalize() + " " + " ".join(w[1:]) + ".")
        texts.append(" ".join(sents))
        pids.append("p%d" % (ci % 7))
    c = ed.tokenize_corpus(texts, pids)
    assert table_dict(c, ed.count_ngrams(c, 4, min_sup)) == brute(texts, pids, 4, min_sup)


def test_tokens_of_an_excluded_paper_are_not_counted_AE3():
    texts = ["Alpha beta gamma alpha beta gamma."] * 6
    pids = ["pA", "pA", "pA", "pB", "pB", "pB"]
    c = ed.tokenize_corpus(texts, pids)
    ex = np.array([False, True])
    assert table_dict(c, ed.count_ngrams(c, 3, 2, exclude_papers=ex)) == brute(texts, pids, 3, 2, excluded={"pB"})


def test_a_gram_never_spans_a_segment_boundary_AE2():
    texts = ["Alpha beta. Gamma delta."] * 4
    c = ed.tokenize_corpus(texts, ["p%d" % i for i in range(4)])
    grams = table_dict(c, ed.count_ngrams(c, 3, 2))
    assert ("alpha", "beta") in grams[2] and ("gamma", "delta") in grams[2] and ("beta", "gamma") not in grams[2]
    assert grams[3] == {}


# ------------------------------------------------------------------------------------------------ statistics
def test_vectorised_npmi_equals_the_incumbent_scalar_function_elementwise():
    from entities import npmi_ppmi
    rng = np.random.default_rng(3)
    n = 1000
    a = rng.integers(5, 500, 200)
    b = rng.integers(5, 500, 200)
    j = np.minimum(rng.integers(5, 400, 200), np.minimum(a, b))
    npmi, ppmi = ed.npmi_vec(j, a, b, n, n, n)
    for i in range(200):
        want = npmi_ppmi(int(a[i]), int(b[i]), int(j[i]), n)
        assert npmi[i] == pytest.approx(want[0], abs=1e-12) and ppmi[i] == pytest.approx(want[1], abs=1e-12)
    full = ed.npmi_vec(np.array([10]), np.array([10]), np.array([10]), 10, 10, 10)          # both in every chunk: the degenerate rule
    assert float(full[0][0]) == 0.0 and float(full[1][0]) == 0.0 and npmi_ppmi(10, 10, 10, 10) == (0.0, 0.0)


def test_npmi_closed_forms_independent_is_zero_perfect_is_one_repelled_is_negative():
    n = 10_000
    ind, _ = ed.npmi_vec(np.array([100]), np.array([1000]), np.array([1000]), n, n, n)         # 0.1 * 0.1 * 10000 = 100
    perfect, _ = ed.npmi_vec(np.array([300]), np.array([300]), np.array([300]), n, n, n)
    neg, ppmi = ed.npmi_vec(np.array([10]), np.array([1000]), np.array([1000]), n, n, n)
    assert float(ind[0]) == pytest.approx(0.0, abs=1e-12) and float(perfect[0]) == pytest.approx(1.0, abs=1e-12)
    assert float(neg[0]) < 0 and float(ppmi[0]) == 0.0                                          # PPMI clips the negative side (AE5)


def test_g2_matches_scipy_loglikelihood_and_is_zero_for_independence():
    from scipy.stats import chi2_contingency
    rng = np.random.default_rng(4)
    for _ in range(5):
        t = rng.integers(3, 300, 4)
        want = chi2_contingency(t.reshape(2, 2), correction=False, lambda_="log-likelihood")[0]
        assert float(ed.g2_2x2(*t)) == pytest.approx(want, rel=1e-9)
    assert float(ed.g2_2x2(10, 20, 30, 60)) == pytest.approx(0.0, abs=1e-9)


def test_entropy_by_group_uniform_is_log2_k_and_deterministic_is_zero():
    g = np.array([0] * 4 + [1] * 4 + [2] * 6)
    s = np.array([5, 6, 7, 8] + [3, 3, 3, 3] + [1, 1, 1, 2, 2, 2])
    h = ed.entropy_by_group(g, s, 3)
    assert h[0] == pytest.approx(2.0) and h[1] == pytest.approx(0.0) and h[2] == pytest.approx(1.0)
    assert list(ed.entropy_by_group(np.zeros(0, np.int64), np.zeros(0, np.int64), 2)) == [0.0, 0.0]


def test_mid_rank_ties_share_a_rank_so_a_piled_up_feature_cannot_zero_a_geometric_mean():
    r = ed.mid_rank(np.array([0.0, 0.0, 0.0, 5.0]))
    assert list(r) == [0.5, 0.5, 0.5, 1.0] and (r > 0).all()


# ------------------------------------------------------------------------------------------------ features
def test_boundary_entropy_and_the_form_signature_separate_a_name_from_a_common_word():
    lefts = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    rights = ["india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]
    texts = ["We apply the %s GRPO %s here." % (a, b) for a, b in zip(lefts, rights)]
    texts += ["Then the policy improves."] * 8
    c = ed.tokenize_corpus(texts, ["p%d" % (i % 4) for i in range(16)])
    f = ed.compute_features(c, ed.count_ngrams(c, 2, 2))
    g, p = row_of(c, f, "grpo"), row_of(c, f, "policy")
    assert f.h_left[g] == pytest.approx(3.0) and f.h_right[g] == pytest.approx(3.0)            # 8 distinct neighbours: 3 bits
    assert f.h_left[p] == pytest.approx(0.0) and f.h_right[p] == pytest.approx(0.0)            # always 'the _ improves'
    assert f.p_cap[g] == 1.0 and f.cap_den[g] == 8 and f.h_var[g] == 0.0                       # rigid, capitalised mid-sentence
    assert f.p_cap[p] == 0.0 and f.cap_den[p] == 8


def test_capitalisation_is_read_from_running_prose_only_AE8():
    texts = ["## Overview\n\nSome text about methods."] * 6 + ["The overview shows methods."] * 6
    c = ed.tokenize_corpus(texts, ["p%d" % (i % 3) for i in range(12)])
    f = ed.compute_features(c, ed.count_ngrams(c, 1, 2))
    r = row_of(c, f, "overview")
    assert f.cap_den[r] == 6 and f.p_cap[r] == 0.0                  # the heading 'Overview' does not count; prose 'overview' is lowercase
    assert f.h_var[r] > 0                                            # but both spellings were seen


def test_a_variant_spelling_raises_the_form_entropy_and_one_spelling_leaves_it_at_zero():
    texts = ["We use BERT here.", "We use Bert here.", "We use BERT here.", "We use Bert here.", "We run LoRA here.", "We run LoRA there.",
             "We run LoRA again.", "We run LoRA too."]
    c = ed.tokenize_corpus(texts, ["p%d" % (i % 4) for i in range(8)])
    f = ed.compute_features(c, ed.count_ngrams(c, 1, 2))
    assert f.h_var[row_of(c, f, "bert")] == pytest.approx(1.0) and f.h_var[row_of(c, f, "lora")] == pytest.approx(0.0)


def test_cohesion_is_the_weakest_split_so_a_loose_extension_scores_below_its_tight_core():
    texts = ["Then alpha beta ends."] * 30 + ["Next gamma appears alone."] * 30 + ["Also alpha beta gamma together."] * 3
    c = ed.tokenize_corpus(texts, ["p%d" % (i % 9) for i in range(63)])
    f = ed.compute_features(c, ed.count_ngrams(c, 3, 3))
    core, ext = row_of(c, f, "alpha beta"), row_of(c, f, "alpha beta gamma")
    assert f.npmi_min[core] > 0.9 and f.npmi_min[ext] < f.npmi_min[core] - 0.3
    assert f.g2_min[core] > f.g2_min[ext] > 0


def test_nesting_is_the_share_of_a_grams_occurrences_one_extension_takes_AE7():
    texts = ["Then policy optimization algorithm runs."] * 10 + ["Other words appear."] * 10
    c = ed.tokenize_corpus(texts, ["p%d" % (i % 5) for i in range(20)])
    f = ed.compute_features(c, ed.count_ngrams(c, 3, 3))
    assert f.nest[row_of(c, f, "policy optimization")] == pytest.approx(1.0)
    assert f.nest[row_of(c, f, "optimization algorithm")] == pytest.approx(1.0)                # the suffix gram is nested too
    assert f.nest[row_of(c, f, "policy optimization algorithm")] == 0.0


def test_burstiness_is_higher_for_a_term_concentrated_in_few_papers():
    texts = ["Then spiky spiky spiky spiky spiky spiky here."] * 3 + ["Then flat appears once here."] * 12
    pids = ["p0"] * 3 + ["p%d" % i for i in range(1, 13)]
    c = ed.tokenize_corpus(texts, pids)
    f = ed.compute_features(c, ed.count_ngrams(c, 1, 2))
    assert f.ridf[row_of(c, f, "spiky")] > f.ridf[row_of(c, f, "flat")]
    assert f.df_papers[row_of(c, f, "spiky")] == 1 and f.df_papers[row_of(c, f, "flat")] == 12


# --------------------------------------------------------------------------------------------------- scoring
def test_the_paper_floor_demotes_below_and_admits_at_the_floor_AE1():
    f = make_features([dict(df_papers=2), dict(df_papers=3)])
    assert ed.score(f, ed.Params(df_floor=3)).tolist()[0] == 0.0 and ed.score(f, ed.Params(df_floor=3)).tolist()[1] > 0
    assert ed.score(f, ed.Params(df_floor=2)).tolist()[0] > 0


def test_a_multiword_gram_with_npmi_at_or_below_zero_scores_zero_but_a_unigram_has_no_cohesion_AE5():
    f = make_features([dict(npmi_min=0.0), dict(npmi_min=-0.2), dict(npmi_min=0.1), dict(n=1, npmi_min=0.0)])
    s = ed.score(f)
    assert s[0] == 0 and s[1] == 0 and s[2] > 0 and s[3] > 0


def test_a_gram_edged_by_a_closed_class_token_scores_zero_unless_the_threshold_is_raised_AE6():
    f = make_features([dict(df_first=50, pcap_first=0.0), dict(df_last=50, pcap_last=0.0), dict(df_first=50, pcap_first=0.5),
                       dict(df_first=10, pcap_first=0.0), dict(n=1, df_first=50, pcap_first=0.0, df_last=50, pcap_last=0.0)])
    s = ed.score(f, ed.Params(closed_df_frac=0.3))
    assert s[0] == 0 and s[1] == 0 and s[2] > 0 and s[3] > 0 and s[4] == 0
    assert ed.score(f, ed.Params(closed_df_frac=0.6))[0] > 0                                    # 50 <= 0.6 * 100: no longer closed


def test_a_gram_spread_evenly_across_papers_scores_zero_at_the_residual_idf_floor_AE14():
    f = make_features([dict(ridf=0.24), dict(ridf=0.499), dict(ridf=0.5), dict(n=1, ridf=0.19), dict(n=1, ridf=0.9)])
    s = ed.score(f)
    assert s[0] == 0 and s[1] == 0 and s[2] > 0 and s[3] == 0 and s[4] > 0
    assert ed.score(f, ed.Params(ridf_floor=0.0))[0] > 0                                        # the floor is a swept axis, not a constant


def test_a_function_word_pair_that_passes_the_capitalisation_cut_is_still_removed_by_burstiness_AE14():
    """'does not' is in every paper at a steady rate and 'not' is capitalised 3% of the time (over AE6's 2% cut)."""
    rng = np.random.default_rng(11)
    texts, pids = [], []
    for p in range(30):
        for _ in range(2):
            sents = []
            for _ in range(5):
                w = [str(rng.choice(CONTENT)) for _ in range(int(rng.integers(8, 14)))]
                w[0] = w[0].capitalize()
                w.insert(int(rng.integers(2, len(w))), "does not")
                sents.append(" ".join(w) + ".")
            texts.append(" ".join(sents))
            pids.append("p%d" % p)
    texts.append("Not only that. NOT again. Not so. Why Not here today. Then we Not far.")
    pids.append("p0")
    inv, f, s, c, flags = ed.derive(texts, pids, ed.Params(ridf_floor=-1.0, closed_df_frac=5.0, universal_df=1.0), min_sup=3, k=500)
    assert "does not" in inv.fold                                                              # with both guards off it qualifies
    inv2, *_ = ed.derive(texts, pids, ed.Params(closed_df_frac=5.0, universal_df=1.0), min_sup=3, k=500)
    assert "does not" not in inv2.fold                                                         # AE14 alone removes it


def test_a_gram_of_single_character_tokens_scores_zero_and_the_axis_can_turn_it_off_AE15():
    f = make_features([dict(max_tok_len=1), dict(max_tok_len=2), dict(n=1, max_tok_len=1), dict(n=1, max_tok_len=2)])
    s = ed.score(f)
    assert s[0] == 0 and s[1] > 0 and s[2] == 0 and s[3] > 0
    assert ed.score(f, ed.Params(min_tok_len=1))[0] > 0 and ed.score(f, ed.Params(min_tok_len=3))[1] == 0


def test_math_residue_across_many_papers_is_removed_end_to_end_AE15():
    rng = np.random.default_rng(21)
    texts, pids = [], []
    for p in range(30):
        w = [str(rng.choice(CONTENT)) for _ in range(12)]
        w.insert(int(rng.integers(1, len(w))), "x y")
        w.insert(int(rng.integers(1, len(w))), "Zorbquark Flux")
        texts.append(w[0].capitalize() + " " + " ".join(w[1:]) + ".")
        pids.append("p%d" % p)
    loose = ed.Params(ridf_floor=-1.0, closed_df_frac=5.0, universal_df=1.0, uni_quota=None, df_floor=2, min_tok_len=1)
    on = ed.Params(ridf_floor=-1.0, closed_df_frac=5.0, universal_df=1.0, uni_quota=None, df_floor=2)
    off_inv, *_ = ed.derive(texts, pids, loose, min_sup=3, k=500)
    on_inv, *_ = ed.derive(texts, pids, on, min_sup=3, k=500)
    assert "x y" in off_inv.fold and "x y" not in on_inv.fold
    assert "zorbquark flux" in on_inv.fold                                              # a real name is untouched


def test_the_cipher_test_compares_a_paper_with_papers_of_similar_length_AE3b():
    """Short papers sit on a rarer vocabulary than long ones (as the `_methods` extracts do), so a corpus-wide z cannot see the real cipher."""
    rng = np.random.default_rng(31)
    common = ["w%d" % i for i in range(120)]
    rare = ["r%d" % i for i in range(900)]
    pc = np.arange(120, 0, -1) / np.arange(120, 0, -1).sum()
    texts, pids = [], []
    for i in range(40):                                                                  # 40 long ordinary papers
        words = [str(rng.choice(common, p=pc)) for _ in range(1500)]
        texts.append(words[0].capitalize() + " " + " ".join(words[1:]) + ".")
        pids.append("long%d" % i)
    for i in range(40):                                                                  # 40 short papers, half of whose words are rare
        words = [str(rng.choice(rare)) if rng.random() < 0.5 else str(rng.choice(common, p=pc)) for _ in range(260)]
        texts.append(words[0].capitalize() + " " + " ".join(words[1:]) + ".")
        pids.append("short%d" % i)
    shifted = ["q%dz" % int(rng.integers(0, 20000)) for _ in range(1500)]               # a long paper of words found nowhere else
    texts.append(shifted[0].capitalize() + " " + " ".join(shifted[1:]) + ".")
    pids.append("cipher")
    c = ed.tokenize_corpus(texts, pids)
    flags = dict(zip(c.papers, ed.cipher_outliers(c)))
    assert flags["cipher"]
    assert not any(v for k, v in flags.items() if k.startswith("short")) and not any(v for k, v in flags.items() if k.startswith("long"))


def test_nesting_at_the_cut_demotes_and_the_switch_turns_it_off_AE7():
    f = make_features([dict(nest=0.95), dict(nest=0.949)])
    assert ed.score(f).tolist()[0] == 0.0 and ed.score(f).tolist()[1] > 0
    assert ed.score(f, ed.Params(nest_demote=False)).tolist()[0] > 0


def test_every_rule_scores_a_row_best_on_all_features_at_one_and_the_weakest_lowest():
    rows = [dict(npmi_min=0.2, g2_min=10, h_left=1, h_right=1, ridf=0.5, p_cap=0.1),
            dict(npmi_min=0.5, g2_min=50, h_left=2, h_right=2, ridf=1.5, p_cap=0.5),
            dict(npmi_min=0.9, g2_min=90, h_left=3, h_right=3, ridf=2.5, p_cap=0.9)]
    f = make_features(rows)
    for rule in ("geomean", "min", "meanrank", "cohesion"):
        s = ed.score(f, ed.Params(rule=rule))
        assert s[2] == pytest.approx(1.0) and s[0] == s.min() and s[0] < s[1] < s[2], rule


def test_the_min_rule_punishes_a_weak_feature_harder_than_the_geometric_mean_does():
    f = make_features([dict(npmi_min=0.9, g2_min=90, ridf=2.5, h_left=3, h_right=3, p_cap=0.0),
                       dict(npmi_min=0.5, g2_min=50, ridf=1.5, h_left=2, h_right=2, p_cap=0.5),
                       dict(npmi_min=0.2, g2_min=10, ridf=0.5, h_left=1, h_right=1, p_cap=0.9)])
    geo, mn = ed.score(f, ed.Params(rule="geomean")), ed.score(f, ed.Params(rule="min"))
    assert mn[0] == pytest.approx(1 / 3) and geo[0] > mn[0]                                      # best on four features, worst on one


def test_dropping_a_feature_and_the_cohesion_rule_change_the_ranking_so_those_axes_are_live():
    f = make_features([dict(npmi_min=0.9, ridf=0.6), dict(npmi_min=0.1, ridf=2.0)])
    base = ed.score(f)
    assert ed.score(f, ed.Params(drop=frozenset({"ridf"}))).tolist() != base.tolist()
    assert ed.score(f, ed.Params(rule="cohesion")).tolist()[0] > ed.score(f, ed.Params(rule="cohesion")).tolist()[1]


def test_select_takes_a_unigram_quota_and_passes_spare_slots_to_the_other_track_and_breaks_ties_by_row():
    f = SimpleNamespace(n=np.array([1, 1, 1, 2, 2, 2]))
    s = np.array([0.9, 0.8, 0.7, 0.95, 0.5, 0.4])
    assert ed.select(f, s, 4, ed.Params(uni_quota=0.5)).tolist() == [3, 0, 1, 4]
    assert ed.select(f, s, 4, ed.Params(uni_quota=None)).tolist() == [3, 0, 1, 2]
    assert sorted(ed.select(f, s, 5, ed.Params(uni_quota=0.9)).tolist()) == [0, 1, 2, 3, 4]    # 3 unigrams exist: the spare slot goes to multiword
    tie = SimpleNamespace(n=np.array([1, 1, 1]))
    assert ed.select(tie, np.array([0.5, 0.5, 0.5]), 2, ed.Params(uni_quota=None)).tolist() == [0, 1]
    assert ed.select(f, np.array([0, 0, 0, 0.1, 0, 0]), 4, ed.Params(uni_quota=None)).tolist() == [3]        # a score of 0 is never selected


# ---------------------------------------------------------------------------------------- end to end, synthetic
def test_planted_entities_are_found_and_function_words_and_nested_pieces_are_not():
    texts, pids = synthetic_papers()
    inv, f, s, c, flags = ed.derive(texts, pids, ed.Params(df_floor=3), min_sup=3, k=200)
    fold = set(inv.fold)
    assert {"grpo", "deepseekmath", "group relative policy optimization"} <= fold
    assert "the" not in fold and "model" not in fold and "the model" not in fold                # closed class, derived (AE6)
    assert "relative policy" not in fold and "group relative policy" not in fold                # nested pieces (AE7)
    assert "zorblax" not in fold and not flags.any()                                            # only 2 papers (AE1); no cipher paper here


def test_the_paper_floor_is_what_admits_a_two_paper_term_AE1():
    texts, pids = synthetic_papers()
    inv2, *_ = ed.derive(texts, pids, ed.Params(df_floor=2), min_sup=3, k=200)
    assert "zorblax" in inv2.fold
    inv3, *_ = ed.derive(texts, pids, ed.Params(df_floor=3), min_sup=3, k=200)
    assert "zorblax" not in inv3.fold


def test_raising_the_closed_class_threshold_lets_a_function_word_back_in_so_that_axis_is_live_AE6():
    texts, pids = synthetic_papers()
    base, *_ = ed.derive(texts, pids, ed.Params(), min_sup=3, k=5000)
    off, *_ = ed.derive(texts, pids, ed.Params(closed_df_frac=2.0, ridf_floor=-1.0, universal_df=1.0), min_sup=3, k=5000)
    assert "the" not in base.fold and "the" in off.fold


def test_the_same_input_and_parameters_give_a_byte_identical_inventory_AE4():
    texts, pids = synthetic_papers()
    a, *_ = ed.derive(texts, pids, ed.Params(), min_sup=3, k=100)
    b, *_ = ed.derive(texts, pids, ed.Params(), min_sup=3, k=100)
    digest = lambda inv: hashlib.md5(repr((inv.fold, inv.surface, inv.score.tolist(), inv.df_papers.tolist(), inv.tf.tolist())).encode()).hexdigest()
    assert digest(a) == digest(b) and len(a.fold) > 3


def test_the_surface_form_is_the_most_frequent_spelling_not_the_first_one_seen():
    texts = ["We use bert here."] + ["We use BERT here."] * 5 + ["Bert is good."] * 2
    c = ed.tokenize_corpus(texts, ["p%d" % (i % 4) for i in range(8)])
    f = ed.compute_features(c, ed.count_ngrams(c, 1, 2))
    inv = ed.inventory_from(c, f, np.ones(len(f.n)), np.array([row_of(c, f, "bert")]))
    assert inv.fold == ["bert"] and inv.surface == ["BERT"]


# ------------------------------------------------------------------------------------------------ acronyms
@pytest.mark.parametrize("sentence,expected", [
    ("We use Group Relative Policy Optimization (GRPO) for training.", [("GRPO", "Group Relative Policy Optimization")]),
    ("We study reinforcement learning (RL) here.", [("RL", "reinforcement learning")]),
    ("Several large language models (LLMs) fail.", [("LLMs", "large language models")]),
    ("It is a Mixture of Experts (MoE) design.", [("MoE", "Mixture of Experts")]),
])
def test_acronym_pairs_accepts_the_initials_rule(sentence, expected):
    assert ed.acronym_pairs([sentence]) == expected


@pytest.mark.parametrize("sentence", ["See the figure (see Fig. 2) above.", "As in prior work (Smith et al., 2020) we agree.",
                                      "Equation (1) holds.", "It goes left (left) then right.", "The results (Table 3) show gains.",
                                      "A part (b) of it.", "Use the thing (i.e. a thing) now."])
def test_acronym_pairs_rejects_parentheses_that_are_not_abbreviations(sentence):
    assert ed.acronym_pairs([sentence]) == []


def test_alias_groups_unify_an_acronym_with_its_expansion_only_when_both_are_entries_AE9():
    fold = ["grpo", "group relative policy optimization", "gsm8k", "policy"]
    pairs = [("GRPO", "Group Relative Policy Optimization"), ("MoE", "Mixture of Experts")]
    assert ed.alias_groups(fold, pairs).tolist() == [0, 0, 2, 3]                                # MoE is not an entry: nothing merges


# -------------------------------------------------------------------------------------------------- cipher
def test_a_paper_in_shifted_letters_is_flagged_and_ordinary_and_short_papers_are_not_AE3():
    rng = np.random.default_rng(5)
    vocab = ["w%d" % i for i in range(300)]
    p = np.arange(300, 0, -1) / np.arange(300, 0, -1).sum()
    texts, pids = [], []
    for i in range(25):
        for _ in range(2):
            words = [str(rng.choice(vocab, p=p)) for _ in range(200)]
            texts.append(words[0].capitalize() + " " + " ".join(words[1:]) + ".")
            pids.append("p%d" % i)
    shifted = ["x%dq" % int(rng.integers(0, 3000)) for _ in range(400)]                        # words found in no other paper
    texts.append(shifted[0].capitalize() + " " + " ".join(shifted[1:]) + ".")
    pids.append("cipher")
    texts.append("Few words only here.")
    pids.append("short")
    c = ed.tokenize_corpus(texts, pids)
    flags = dict(zip(c.papers, ed.cipher_outliers(c)))
    assert flags["cipher"] and not flags["short"] and not any(v for k, v in flags.items() if k.startswith("p"))
    ce, n_tok = ed.paper_cross_entropy(c)
    assert ce[c.papers.index("cipher")] > 1.5 * np.median(ce[:25])                              # 13.4 vs 8.1 bits measured


def test_a_flagged_paper_leaves_the_document_frequency_so_its_terms_cannot_qualify_AE1_AE3():
    rng = np.random.default_rng(6)
    base = ["Then the alpha beta runs and we see the result in the data."] * 4
    texts, pids = [], []
    for i in range(22):
        words = [str(rng.choice(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"] + FUNC)) for _ in range(250)]
        texts.append(words[0].capitalize() + " " + " ".join(words[1:]) + ".")
        pids.append("p%d" % i)
    cipher_words = ["Zq%dx" % int(rng.integers(0, 40)) for _ in range(400)]                    # a small repeated cipher vocabulary
    texts.append("Then " + " ".join(cipher_words) + ".")
    pids.append("cipher")
    inv, f, s, c, flags = ed.derive(texts, pids, ed.Params(df_floor=1), min_sup=3, k=5000)
    assert flags[c.papers.index("cipher")] and not any(w.startswith("zq") for w in inv.fold)


# --------------------------------------------------------------------------------------- frozen matching
def _inventory(*folds):
    return ed.Inventory(list(folds), [f.upper() for f in folds], np.array([len(f.split()) for f in folds], np.int64), np.ones(len(folds)),
                        np.ones(len(folds), np.int64), np.ones(len(folds), np.int64), np.ones(len(folds), np.int64), np.arange(len(folds), dtype=np.int64))


def test_match_frozen_takes_the_longest_gram_and_never_uses_a_token_twice():
    inv = _inventory("grpo", "group relative policy optimization", "policy")
    c = ed.tokenize_corpus(["Group Relative Policy Optimization (GRPO) improves policy.", "Our policy and GRPO and GRPO."], ["a", "b"])
    chunk, ent, cnt = ed.match_frozen(inv, c)
    got = {(int(a), int(b)): int(n) for a, b, n in zip(chunk, ent, cnt)}
    assert got == {(0, 1): 1, (0, 0): 1, (0, 2): 1, (1, 2): 1, (1, 0): 2}                      # 'policy' inside the 4-gram is not counted again


def test_match_frozen_never_matches_across_a_boundary_or_an_unseen_token_AE2_AE13():
    inv = _inventory("grpo", "policy optimization")
    c = ed.tokenize_corpus(["Policy. Optimization follows. A zzz GRPO wins.", "Policy $x$ optimization."], ["a", "b"])
    chunk, ent, cnt = ed.match_frozen(inv, c)
    assert {(int(a), int(b)): int(n) for a, b, n in zip(chunk, ent, cnt)} == {(0, 0): 1}        # only the GRPO; no gram spans '.' or math


def test_match_frozen_on_empty_input_and_an_unrelated_text_is_empty_and_changes_nothing_AE11():
    inv = _inventory("grpo")
    before = (list(inv.fold), inv.score.copy(), inv.tf.copy())
    for texts in ([], ["Nothing relevant here at all."]):
        c = ed.tokenize_corpus(texts, ["p"] * len(texts))
        chunk, ent, cnt = ed.match_frozen(inv, c)
        assert len(chunk) == len(ent) == len(cnt) == 0
    assert (inv.fold, inv.score.tolist(), inv.tf.tolist()) == (before[0], before[1].tolist(), before[2].tolist())


def test_match_frozen_agrees_with_the_counts_the_inventory_was_derived_from():
    texts, pids = synthetic_papers()
    inv, f, s, c, flags = ed.derive(texts, pids, ed.Params(df_floor=3), min_sup=3, k=200)
    chunk, ent, cnt = ed.match_frozen(inv, c)
    i = inv.fold.index("deepseekmath")
    assert int(cnt[ent == i].sum()) == int(inv.tf[i])                                           # same corpus, same count
    j = inv.fold.index("group relative policy optimization")
    assert int(cnt[ent == j].sum()) == int(inv.tf[j])


def test_match_spans_gives_position_entity_and_length_in_ascending_order_and_agrees_with_match_frozen():
    inv = _inventory("grpo", "group relative policy optimization")
    c = ed.tokenize_corpus(["We use GRPO. Group Relative Policy Optimization (GRPO) helps."], ["a"])
    pos, ent, n = ed.match_spans(inv, c)
    assert pos.tolist() == sorted(pos.tolist()) and len(pos) == 3
    assert [(int(e), int(k)) for e, k in zip(ent, n)] == [(0, 1), (1, 4), (0, 1)]
    chunk, e2, cnt = ed.match_frozen(inv, c)
    assert chunk.tolist() == [0, 0] and e2.tolist() == [0, 1] and cnt.tolist() == [2, 1]
    z = ed.match_spans(inv, ed.tokenize_corpus(["nothing here"], ["a"]))
    assert all(len(a) == 0 for a in z)


# ------------------------------------------------------------------------------------------------- types
def _typed_corpus(per=20, occ=8):
    """Datasets seen in 'evaluated on X benchmark', models in 'we fine-tune X with adapters'; names differ only by number."""
    rng = np.random.default_rng(2)
    fillers = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "kappa"]
    texts = []
    for k in range(per):
        for _ in range(occ):
            texts.append("%s %s evaluated on ds%02d benchmark %s." % (rng.choice(fillers), rng.choice(fillers), k, rng.choice(fillers)))
            texts.append("%s we fine-tune mdl%02d with adapters %s." % (rng.choice(fillers), k, rng.choice(fillers)))
    names = ["ds%02d" % k for k in range(per)] + ["mdl%02d" % k for k in range(per)]
    inv = ed.Inventory(names, [x.upper() for x in names], np.ones(2 * per, np.int64), np.ones(2 * per), np.ones(2 * per, np.int64),
                       np.ones(2 * per, np.int64), np.ones(2 * per, np.int64), np.arange(2 * per, dtype=np.int64))
    truth = np.array([0] * per + [1] * per)
    return inv, ed.tokenize_corpus(texts, ["p%d" % (i % 10) for i in range(len(texts))]), truth


def test_context_vectors_are_ppmi_rows_that_group_entities_by_what_surrounds_them_AE10():
    from sklearn.metrics import adjusted_rand_score  # noqa: F401  (imported to fail early if absent)
    inv, c, truth = _typed_corpus()
    X, occ = ed.context_vectors(inv, c)
    assert occ.tolist() == [8] * 40 and X.shape[0] == 40
    assert np.allclose(np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()), 1.0)
    S = (X @ X.T).toarray()
    same = S[np.ix_(truth == 0, truth == 0)][np.triu_indices(20, 1)]
    cross = S[np.ix_(truth == 0, truth == 1)].ravel()
    assert same.mean() > 0.5 and cross.mean() < 0.05                      # a dataset's contexts resemble another dataset's, not a model's


def test_an_entity_with_too_few_matches_has_a_zero_context_row_and_class_minus_one_AE10():
    inv, c, truth = _typed_corpus(per=6, occ=3)
    X, occ = ed.context_vectors(inv, c, min_occ=5)
    assert occ.max() == 3 and X.nnz == 0
    out = ed.type_entities(inv, c, min_occ=5)
    assert (out["classes"] == -1).all() and out["tried"] == [] and out["resolution"] is None


def test_the_spelling_view_puts_same_shaped_names_closer_than_different_shapes_AE10():
    inv = ed.Inventory(["grpo", "sft", "kv cache"], ["GRPO", "SFT", "KV cache"], np.array([1, 1, 2]), np.ones(3), np.ones(3, np.int64),
                       np.ones(3, np.int64), np.ones(3, np.int64), np.arange(3, dtype=np.int64))
    X = ed.spelling_vectors(inv)
    S = (X @ X.T).toarray()
    assert np.allclose(np.diag(S), 1.0) and S[0, 1] > S[0, 2] and S[0, 1] > S[1, 2]


def test_types_recover_the_planted_classes_and_pass_the_seed_stability_gate_AE10():
    from sklearn.metrics import adjusted_rand_score
    inv, c, truth = _typed_corpus()
    out = ed.type_entities(inv, c, resolutions=(0.1, 0.5), seeds=range(4))
    assert out["ari"] is not None and out["ari"] >= ed.TYPE_SEED_ARI and out["resolution"] in (0.1, 0.5)
    assert adjusted_rand_score(truth, out["classes"]) > 0.9 and (out["classes"] >= 0).all()
    again = ed.type_entities(inv, c, resolutions=(0.1, 0.5), seeds=range(4))
    assert again["classes"].tolist() == out["classes"].tolist()           # AE4: no randomness the seeds do not fix


def test_a_partition_that_does_not_pass_the_gate_gives_no_types_and_keeps_the_record_AE10():
    inv, c, truth = _typed_corpus()
    out = ed.type_entities(inv, c, resolutions=(0.5,), seeds=range(4), seed_ari=1.01)       # no partition can reach 1.01
    assert (out["classes"] == -1).all() and out["resolution"] is None and out["ari"] is None
    assert len(out["tried"]) == 1 and out["tried"][0][0] == 0.5
