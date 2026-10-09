"""section_render.py -- the labelled community map of a saved section build: the PNG cards and the markdown, with Dunning terms, entities and paper titles.

Spec: operator 2026-10-07 ("the dunning terms should just be next to community labels"), 2026-10-08 (entities per community; paper names beside section headings).
No governing spec beyond those instructions. Promoted from a scratch script so the README has one entrypoint for the picture.

Reads  .tmp/sections_<tag>_map_state.npz, _exemplars.json, _summaries.json (optional) and Postgres (sect_community.entities, paper_title).
Writes .tmp/sections_<tag>_labelled_community_map.png, .tmp/sections_<tag>_labelled_communities.md, .tmp/sections_<tag>_dunning_terms.json.

Run:  python -u tools\\section_render.py [--tag xpa]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import arxiv_community_map as acm
import section_corpus as sc
import section_embed as se
import section_genre as sg
import section_store as sstore
import sparsevec_store as ss
import term_salience as ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xpa")
    tag = ap.parse_args().tag
    out_png = os.path.join(ROOT, ".tmp", "sections_%s_labelled_community_map.png" % tag)
    out_md = os.path.join(".tmp", "sections_%s_labelled_communities.md" % tag)
    acm.OUT_PNG, acm.UNIT, acm.OUT_MD = out_png, "sections", out_md

    recs = sc.load()
    t0 = time.time()
    X, _ = se.center(np.load(os.path.join(ROOT, ".tmp", "arxiv_sections_jina256.npy")))
    paper = np.unique([r["doc_id"] for r in recs], return_inverse=True)[1]
    V, _ = sg.role_axes(X, paper, sg.GENRE_K)
    frac = sg.genre_fraction(X, V)
    z = np.load(os.path.join(ROOT, ".tmp", "sections_%s_map_state.npz" % tag))
    st = {k: z[k] for k in z.files}
    st["chosen"], st["cons_ari"] = float(st["chosen"]), float(st["cons_ari"])
    score, flag = sg.genre_flags(st["lab"], frac)
    genre = {int(i) for i in np.where(flag)[0]}
    ex = {int(c): v for c, v in json.load(open(os.path.join(ROOT, ".tmp", "sections_%s_exemplars.json" % tag), encoding="utf-8")).items()}

    cv = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b[^\W\d_]{3,}\b", ngram_range=(1, 2), min_df=20, binary=True, dtype=np.float32)
    P = cv.fit_transform(sc.index_text(r) for r in recs).tocsr()
    vocab = cv.get_feature_names_out().tolist()
    print("term matrix %s, %d terms, %.0fs" % (P.shape, len(vocab), time.time() - t0))
    want = np.array([ts.exemplar_term_count(len(ex[c]["exemplars"])) for c in range(len(ex))])
    got = ts.community_terms(P, st["lab"], vocab, want)
    terms = {c: [t for t, _ in v] for c, v in got.items()}
    json.dump({str(c): v for c, v in got.items()}, open(os.path.join(ROOT, ".tmp", "sections_%s_dunning_terms.json" % tag), "w", encoding="utf-8"))

    summ = {}
    sp_ = os.path.join(ROOT, ".tmp", "sections_%s_summaries.json" % tag)
    if os.path.exists(sp_):                                  # LLM drafts whose source sections are STILL the community's exemplars; anything else is stale and skipped
        for d in json.load(open(sp_, encoding="utf-8")):
            c = d["community"]
            if d.get("status") == "draft" and c in ex and d["chunk_keys"] == [e["key"] for e in ex[c]["exemplars"]] and (d.get("terms") or []) == (terms.get(c) or []):
                summ[c] = d
    print("summaries used: %d" % len(summ))

    conn = ss.connect()
    build = conn.execute("SELECT max(build_id) FROM sect_build WHERE tag = %s AND live", (tag,)).fetchone()[0]
    ents = {int(c): ["%s %d" % (e[0], e[1]) for e in v] for c, v in conn.execute(
        "SELECT cid, entities FROM sect_community WHERE build_id = %s AND entities IS NOT NULL", (build,)).fetchall()}
    stats = {c: "%.0f%% of sections mention an entity · %s distinct · %s mentions" % (100 * v["covered"] / max(v["size"], 1), format(v["distinct"], ","), format(v["mentions"], ","))
             for c, v in sstore.community_entity_stats(conn, build).items()}
    titles = sstore.paper_titles(conn, sorted({recs[it["row"]]["doc_id"] for c in ex for it in ex[c]["exemplars"]}))
    print("entities for %d communities, stats for %d, paper titles for %d papers" % (len(ents), len(stats), len(titles)))

    n, cut = acm.write_sections_md(recs, recs, ex, summ, path=out_md, genre=genre, terms=terms, entities=ents, entity_stats=stats, titles=titles)
    print("MD", os.path.abspath(out_md), "| %d sections in full, %d cut" % (n - cut, cut))
    acm.render_cards(recs, st, ex, summ, genre=genre, terms=terms, entities={c: v[:5] for c, v in ents.items()}, entity_stats=stats, titles=titles)
    print("PNG", out_png)


if __name__ == "__main__":
    main()
