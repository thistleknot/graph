"""scratch: plan.md stage B with the Jev gate IN the loop, over the 60 gold questions of stage_b.py (truth = the exact source section).
Per question: communities scoring >= 0.50 solo (from jev_gate_probe.json) -> partial per passer (PARTIAL_PROMPT, RELEVANT: YES|NO) -> reduce -> global answer; no passer, or no partial that
says YES, REJECTS the question (no agent, no evidence). An answered question then runs the agent for hops 1, 2, 3 (queries=3, global view = the reduce output), as in stage_b.py.
Reported: the rejection rate; over ANSWERED questions only, exact-section recall of the walk at 0..3 searches against a same-size hybrid control, and over ALL 60 the recall of the whole system
(a rejected question counts as a miss) against the plain hybrid list of 18 sections. One number per arm with n; Wilson 95%. Pre-registered here before any arm is scored (2026-10-12).
Model replies are cached by prompt hash in .tmp/stage_b_gate_cache.json. No governing spec beyond plan.md."""
import hashlib, json, math, os, sys, time, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.chdir(os.path.join(HERE, ".."))
import numpy as np
import ingest_arxiv_sparsevec as ing
import section_corpus as sc, section_graphrag as g, section_store as st, sparsevec_store as ss
import summarize_clusters as scz

N, HOPS, KS = 60, (1, 2, 3), (0, 1, 2, 3)
AGENT = "anthropic/claude-haiku-5.5"
CACHE = os.path.join(HERE, "stage_b_gate_cache.json")
OUT = os.path.join(HERE, "stage_b_gate_runs.json")
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
t0 = time.time()
log = lambda m: print("[%4.0fs] %s" % (time.time() - t0, m), flush=True)


def chat(model: str, prompt: str) -> str:
    k = hashlib.sha256((model + "\0" + prompt).encode("utf-8")).hexdigest()
    if k not in cache:
        cache[k] = scz.chat(model, prompt)["content"]
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    return cache[k]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


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
probe = json.load(open(os.path.join(HERE, "jev_gate_probe.json"), encoding="utf-8"))
catalogue, facts = st.community_catalogue(sg.conn, sg.build), st.community_entity_lines(sg.conn, sg.build)
runs = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
for i, (q, o) in enumerate(qs):
    if str(i) in runs:
        continue
    passers = [int(c) for c, _ in probe[str(i)].get("passers", [])]
    partials, rejected_partials = [], []
    for c in passers:
        rec = st.summaries(sg.conn, sg.build, [c])[c]
        reply = chat(scz.MODEL, g.PARTIAL_PROMPT.format(question=q, cid=c, size=rec["size"], title=rec["title"], summary=rec["summary"]))
        relevant, text = g.parse_map(reply)
        (partials if relevant else rejected_partials).append((c, text if relevant else reply))
    if not partials:
        runs[str(i)] = {"rejected": True, "passers": passers, "own_in_passers": probe[str(i)]["own"] in passers}
        json.dump(runs, open(OUT, "w", encoding="utf-8"))
        log("question %d/%d REJECTED (passers %d)" % (i + 1, N, len(passers)))
        continue
    glob = chat(scz.MODEL, g.reduce_prompt(q, partials))
    got = sg.retrieve(q, 3)
    rec = {"rejected": False, "passers": passers, "partials": [c for c, _ in partials], "own_in_partials": probe[str(i)]["own"] in [c for c, _ in partials], "hops": {}}
    for H in HOPS:
        ex = g.Explorer(sg, got, hops=H, per_query=g.PER_QUERY)
        run = g.run_agent(q, glob, ex.initial(), ex, lambda p: chat(AGENT, p), 3, catalogue=catalogue, seed=zlib.crc32(q.encode()), mode="sample", facts=facts, hops=H, per_query=g.PER_QUERY)
        rec["hops"][str(H)] = {"nodes": [{"ord": n["ord"], "query": n.get("query", 1)} for n in run["nodes"].values()], "stop": run["stop"]}
    runs[str(i)] = rec
    json.dump(runs, open(OUT, "w", encoding="utf-8"))
    log("question %d/%d answered (partials %d)" % (i + 1, N, len(partials)))

rej = sum(1 for r in runs.values() if r["rejected"])
ans = [i for i in range(N) if not runs[str(i)]["rejected"]]
lines = ["questions %d | rejected %d = %.2f | answered %d | agent %s | gate cut 0.50 solo" % (N, rej, rej / N, len(ans), AGENT)]
lines.append("answered questions whose own community is among the kept partials: %d of %d" % (sum(1 for i in ans if runs[str(i)]["own_in_partials"]), len(ans)))
ranked = {i: sg.ranked(qs[i][0])["ranked"] for i in ans}
for H in HOPS:
    lines.append("\nHOPS %d (answered questions only, n=%d)" % (H, len(ans)))
    for k in KS:
        w = c = 0
        sizes = []
        for i in ans:
            ords = list(dict.fromkeys(n["ord"] for n in runs[str(i)]["hops"][str(H)]["nodes"] if n["query"] <= k + 1))
            sizes.append(len(ords))
            w += qs[i][1] in ords
            c += qs[i][1] in ranked[i][:len(ords)]
        lo, hi = wilson(w, len(ans)) if ans else (0, 0)
        lines.append("  searches %d | mean shown %.1f | walk exact %d/%d = %.2f [%.2f,%.2f] | hybrid same size %d/%d = %.2f" % (k, np.mean(sizes) if sizes else 0, w, len(ans), w / max(len(ans), 1), lo, hi, c, len(ans), c / max(len(ans), 1)))
sysrec = {H: sum(1 for i in ans if qs[i][1] in [n["ord"] for n in runs[str(i)]["hops"][str(H)]["nodes"]]) for H in HOPS}
plain = sum(1 for i in range(N) if qs[i][1] in sg.ranked(qs[i][0])["ranked"][:18])
lines.append("\nwhole system over ALL %d questions (a rejected question is a miss), every section the traversal saw: " % N + "; ".join("hops %d: %d = %.2f" % (H, v, v / N) for H, v in sysrec.items()) + " | plain hybrid top 18: %d = %.2f" % (plain, plain / N))
open(os.path.join(HERE, "stage_b_gate_result.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
log("done")
