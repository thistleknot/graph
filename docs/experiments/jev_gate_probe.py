"""scratch: the Jev-Style gate on in-corpus questions. For the first N stage-B questions (a sentence taken from one section, truth = that section): score all 415 community summaries solo,
then report how many pass 0.50 (the question is rejected when none does), the score and rank of the community that CONTAINS the source section, and whether it passes.
Pre-registered: N = 20 first, widened to 60 the same day (the first 20 are kept); every question's full score list is stored so the cut can be calibrated and the gate put in a stage B loop. N = 60, threshold 0.50 (community_reranker.relevance_threshold), solo requests (batch size 1). Needs the Jev-Style server on 127.0.0.1:8765. No governing spec beyond plan.md."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.chdir(os.path.join(HERE, ".."))
import numpy as np
import community_reranker as cr
import ingest_arxiv_sparsevec as ing
import section_corpus as sc, section_graphrag as g, section_store as st, sparsevec_store as ss

N = 60
conn = ss.connect()
sg = g.SectionGraph(conn, "xpa")
recs = sc.load()
_, body = ing.gold(recs)
qs = []
for q, rel in body:
    exact = [o for o in rel if " ".join(q.split()) in " ".join(recs[o]["text"].split())]
    if exact:
        qs.append((q, exact[0]))
    if len(qs) == N:
        break
cands = [r[0] for r in conn.execute("SELECT cid FROM sect_community WHERE build_id = %s AND size > 1 AND status = 'draft' ORDER BY size DESC, cid", (sg.build,)).fetchall()]
OUT = os.path.join(HERE, "jev_gate_probe.json")
done = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
t0 = time.time()
for i, (q, o) in enumerate(qs):
    if str(i) in done and ("scores" in done[str(i)] or done[str(i)]["passed"] == 0):
        done[str(i)].setdefault("passers", [])                  # the first 20 stored only counts: a question that passed nothing has no passers; one that passed something is rescored below
        continue
    scores, _ = cr.score_communities(q, st.summary_batches(conn, sg.build, cands, cr.PAGE_SIZE))
    own = st.community_of(conn, sg.build, [o])[o]
    ranked = sorted(scores, key=lambda s: -s["score"])
    rank = [s["community"] for s in ranked].index(own) + 1 if own in {s["community"] for s in ranked} else None
    mine = next((s["score"] for s in scores if s["community"] == own), None)
    done[str(i)] = {"scores": [[s["community"], round(s["score"], 4)] for s in scores], "passers": [[s["community"], s["score"]] for s in ranked if s["score"] >= cr.relevance_threshold()], "q": q, "own": own, "own_score": mine, "own_rank": rank, "passed": sum(1 for s in scores if s["score"] >= cr.relevance_threshold()),
                    "best": ranked[0]["score"], "n": len(scores)}
    json.dump(done, open(OUT, "w", encoding="utf-8"))
    print("[%4.0fs] %d/%d passed %d best %.3f | own community c%s score %s rank %s of %d" % (time.time() - t0, i + 1, N, done[str(i)]["passed"], ranked[0]["score"], own, "%.3f" % mine if mine is not None else None, rank, len(scores)), flush=True)
json.dump(done, open(OUT, "w", encoding="utf-8"))
rows = [done[str(i)] for i in range(N)]
thr = cr.relevance_threshold()
rej = sum(1 for r in rows if r["passed"] == 0)
own_pass = sum(1 for r in rows if r["own_score"] is not None and r["own_score"] >= thr)
ranks = [r["own_rank"] for r in rows if r["own_rank"]]
print("\nquestions %d | rejected (no community passed %.2f): %d = %.2f | own community passes: %d = %.2f | own rank median %.0f, top-10: %d, top-1: %d" % (
    N, thr, rej, rej / N, own_pass, own_pass / N, float(np.median(ranks)), sum(1 for r in ranks if r <= 10), sum(1 for r in ranks if r == 1)))
