"""section_screen.py -- score every arm of the section-map fix the same way, on a seeded sample of whole papers, so arms are compared on one table.

Spec: approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md, task T161 (screening: a full map is 12-25 minutes per arm, so every arm is run on
a seeded sample of WHOLE papers first and only the best two are run in full). Task: playbook.md T161, T166.

    python -u src\\section_screen.py baseline C_heading B_k8      runs the named arms (all of ARMS when none are named), appends to .tmp/screen_results.json

An arm is an override of DEFAULTS:
    strip_heading   leave the heading line out of what is pooled and tokenised (section_corpus.index_text; arm C)
    genre_k         remove the top-k within-paper axes from the dense view (section_genre; arm B), 0 = none
    lam             weight of the adjacent-pair view in the sparse rows (0 = the adjacency rerank off)
Each arm: dense view -> sparse view -> exact cross-paper edges in both -> fuse -> Leiden sweep, plateau pick, consensus -> scorecard. Same sample, same seeds,
same code for every arm; the baseline is re-run in the same round so no number is read from memory.

Beside the scorecard each row carries `same_heading_edges`: the share of fused edges whose two ends have the same normalised heading, and `base_rate`,
the chance of that for two random sections (the plan's arm E asks whether heading-caused edges survive).
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import section_corpus as sc
import section_embed as se
import section_genre as sg
import section_graph as gr
import section_scorecard as card

SAMPLE_N, SAMPLE_SEED = 20_000, 0
DENSE_NPY = os.path.join(ROOT, ".tmp", "arxiv_sections_jina256.npy")
MODEL = os.path.expanduser("~/models/m2v-jina-v5-nano-256")
RESULTS = os.path.join(ROOT, ".tmp", "screen_results.json")
DEFAULTS = {"strip_heading": False, "genre_k": 0, "lam": 1.0}
ARMS = {"baseline": {}, "C_heading": {"strip_heading": True}, "B_k2": {"genre_k": 2}, "B_k8": {"genre_k": 8}, "lam0": {"lam": 0.0}}


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def pick_sample(doc_ids, target: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> np.ndarray:
    """Guarantee: the sorted row indices of a seeded sample made of WHOLE papers: papers are taken in a seeded random order until their sections number at
    least `target`; every section of a chosen paper is in, none of an unchosen one. Deterministic for a seed."""
    paper = np.unique(np.asarray(doc_ids), return_inverse=True)[1]
    counts = np.bincount(paper)
    order = np.random.default_rng(seed).permutation(len(counts))
    take = order[np.cumsum(counts[order]) - counts[order] < target]              # a paper is taken while the running total BEFORE it is still short of the target
    return np.where(np.isin(paper, take))[0]


def arm_config(overrides: dict) -> dict:
    """Guarantee: DEFAULTS with `overrides` applied; an unknown key is an error, so a typo cannot silently run the baseline."""
    bad = set(overrides) - set(DEFAULTS)
    assert not bad, "unknown arm setting(s) %s" % sorted(bad)
    return {**DEFAULTS, **overrides}


def same_heading_share(i: np.ndarray, j: np.ndarray, kind: np.ndarray) -> tuple[float, float]:
    """Guarantee: (share of the pairs (i[t], j[t]) whose two sections have the same non-empty heading id in `kind` (-1 = empty heading), the chance of that
    for two random sections with a non-empty heading)."""
    ok = (kind[i] >= 0) & (kind[j] >= 0)
    share = float((kind[i][ok] == kind[j][ok]).mean()) if ok.any() else 0.0
    p = np.bincount(kind[kind >= 0]) / max(int((kind >= 0).sum()), 1)
    return share, float((p ** 2).sum())


def dense_view(rows: np.ndarray, recs: list[dict], paper: np.ndarray, cfg: dict, E_all: np.ndarray, model=None) -> np.ndarray:
    """Guarantee: the sample's dense rows, centred on the sample mean and L2-normalised; heading-stripped text is re-pooled with `model`, otherwise the
    cached pooled rows are used; with genre_k > 0 the top-k within-paper axes are projected out (section_genre)."""
    if cfg["strip_heading"]:
        E = se.pool(model, [sc.index_text(recs[r]) for r in rows])
    else:
        E = E_all[rows]
    X, _ = se.center(E)
    if cfg["genre_k"]:
        V, _ = sg.role_axes(X, paper, cfg["genre_k"])
        X = sg.remove_axes(X, V)
    return X


def run_arm(name: str, overrides: dict, ctx: dict, fixed_res: float | None = None) -> dict:
    """Guarantee: one scorecard row for the arm on the sample in `ctx` (recs, rows, E_all, model getter), with seconds taken, the resolution used and the
    same-heading edge share. With `fixed_res` every arm is clustered at the same resolution, so the partition columns compare graphs, not plateau picks."""
    import scipy.sparse as sp
    import section_map as smap
    import section_sparse as ss
    cfg = arm_config(overrides)
    t0 = time.time()
    recs, rows = ctx["recs"], ctx["rows"]
    sub = [recs[r] for r in rows]
    paper = np.unique([r["doc_id"] for r in sub], return_inverse=True)[1]
    X = dense_view(rows, recs, paper, cfg, ctx["E_all"], ctx["model"]() if cfg["strip_heading"] else None)
    log("%s: dense %s (%.0fs)" % (name, X.shape, time.time() - t0))
    Z, info = ss.build(sub, merges=10_000, lam=cfg["lam"], strip_heading=cfg["strip_heading"])
    log("%s: sparse %s (%.0fs)" % (name, Z.shape, time.time() - t0))
    reps = {}
    for rep, M in (("dense", X), ("sparse", Z)):
        cut = gr.fit_null(M, rows=min(2000, len(sub)))
        reps[rep] = {"X": M, "cut": cut, "edges": gr.stream_edges(M, cut, k_sigma=gr.K_SIGMA, topk=15, block=512, group=paper)}
    f = gr.fuse(len(sub), reps)
    log("%s: %d fused edges (%.0fs)" % (name, len(f["i"]), time.time() - t0))
    res = smap.communities_from_graph(f["G"], target_n=max(2, round(smap.m.TARGET_N * len(sub) / ctx["n_all"])), fixed_res=fixed_res)
    kind_text = [card.heading_kind(r["section_title"]) for r in sub]
    kind = np.where(np.array([k == "" for k in kind_text]), -1, card.codes(kind_text))
    share, base = same_heading_share(f["i"], f["j"], kind)
    row = card.scorecard(res["lab"], [r["doc_id"] for r in sub], [r["section_title"] for r in sub], res["cons_ari"], m_edges=len(f["i"]))
    row.update({"arm": name, "config": cfg, "fused_edges": int(len(f["i"])), "same_heading_edges": share, "base_rate": base, "seconds": round(time.time() - t0),
                "resolution": float(res["chosen"]), "resolution_fixed": fixed_res is not None})
    return row


def parse_args(argv: list[str]) -> tuple[list[str], float | None, int]:
    """Guarantee: (arm names, fixed resolution or None, sample seed). `res=3.42` sets the resolution every named arm is clustered at; `seed=1` draws a different
    whole-paper sample (the default is SAMPLE_SEED), which is how the noise floor of every column is measured; the rest are arm names, and an unknown name is
    an error."""
    fixed = [float(a.split("=", 1)[1]) for a in argv if a.startswith("res=")]
    seeds = [int(a.split("=", 1)[1]) for a in argv if a.startswith("seed=")]
    names = [a for a in argv if not a.startswith(("res=", "seed="))]
    bad = [n for n in names if n not in ARMS]
    assert not bad, "unknown arm(s) %s; known: %s" % (bad, sorted(ARMS))
    assert len(fixed) <= 1 and len(seeds) <= 1, "give res= and seed= at most once"
    return names, (fixed[0] if fixed else None), (seeds[0] if seeds else SAMPLE_SEED)


def main(argv: list[str]) -> None:
    names, fixed_res, seed = parse_args(argv)
    recs = sc.load()
    rows = pick_sample([r["doc_id"] for r in recs], seed=seed)
    log("sample: %d of %d sections, %d whole papers (seed %d)" % (len(rows), len(recs), len({recs[r]["doc_id"] for r in rows}), seed))
    state = {}
    def model():
        if "m" not in state:
            from model2vec import StaticModel
            state["m"] = StaticModel.from_pretrained(MODEL)
        return state["m"]
    ctx = {"recs": recs, "rows": rows, "E_all": np.load(DENSE_NPY), "model": model, "n_all": len(recs)}
    done = json.load(open(RESULTS, encoding="utf-8")) if os.path.exists(RESULTS) else {}
    for arm in names or list(ARMS):
        name = arm if seed == SAMPLE_SEED else "%s@seed%d" % (arm, seed)                  # a different sample never overwrites the default-sample row
        done[name] = run_arm(name, ARMS[arm], ctx, fixed_res)
        done[name]["sample_seed"] = seed
        json.dump(done, open(RESULTS, "w", encoding="utf-8"), indent=1, default=float)
        log("%s done: communities %d, size>=10 %d, singletons %d, oversize %d, in-oversize %.3f, paper>=.5 %d, seed-ARI %.3f, same-heading edges %.4f (chance %.4f)" % (
            name, done[name]["communities"], done[name]["size_ge_10"], done[name]["singletons"], done[name]["oversize"], done[name]["oversize_share"],
            done[name]["paper_ge_50"], done[name]["seed_ari"], done[name]["same_heading_edges"], done[name]["base_rate"]))
    print(card.format_rows([(n, d) for n, d in done.items()]))
    print("\nsame-heading share of fused edges (chance in brackets):")
    for n, d in done.items():
        print("  %-12s %.4f  (%.4f)  x%.0f chance | fused edges %d | %ds" % (n, d["same_heading_edges"], d["base_rate"], d["same_heading_edges"] / max(d["base_rate"], 1e-9), d["fused_edges"], d["seconds"]))


if __name__ == "__main__":
    main(sys.argv[1:])
