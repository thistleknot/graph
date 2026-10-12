"""scratch: plan.md step 5b + 5c, scaled to 20 questions. Agent searches 0..3 x hops 1..3 against a same-size hybrid control; recall and cross-encoder precision.
PRE-REGISTERED before any arm is scored (written 2026-10-12):
  questions   the first 20 body-sentence queries of ingest_arxiv_sparsevec.gold; truth = the exact section the sentence came from (paper-level beside it)
  agent       claude-haiku-5.5 via summarize_clusters.chat, global view EMPTY (stage B tests the traversal, not the Jev gate), queries=3 per hop setting; the result for k searches is the union of
              the question's subgraph and the agent's first k searches (a 3-search run contains the 1- and 2-search runs as prefixes)
  precision   share of shown sections whose cross-encoder (ms-marco-MiniLM-L-6-v2) score for (question, section) is >= tau; tau = 25th percentile of the cross-encoder scores of the 20 gold sections
  control     the top n of the hybrid RRF list, n = how many sections that arm showed
  report      one number per arm with n, Wilson 95%, paired counts against the control
Model replies are cached by prompt hash in .tmp/stage_b_cache.json so a rerun is free. No governing spec beyond plan.md."""
import hashlib, json, math, os, sys, time, zlib
for n in ("torchvision", "torchaudio"):                       # this environment's torchvision/torchaudio do not match its torch; transformers only needs torch
    sys.modules[n] = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.chdir(os.path.join(HERE, ".."))
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ingest_arxiv_sparsevec as ing
import section_corpus as sc, section_graphrag as g, section_store as st, sparsevec_store as ss
import summarize_clusters as scz

N, HOPS, KS = 20, (1, 2, 3), (0, 1, 2, 3)
AGENT = "anthropic/claude-haiku-5.5"
CACHE = os.path.join(HERE, "stage_b_cache.json")
OUT = os.path.join(HERE, "stage_b_runs.json")
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
t0 = time.time()
log = lambda m: print("[%4.0fs] %s" % (time.time() - t0, m), flush=True)

tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "ce_minilm"))
ce = AutoModelForSequenceClassification.from_pretrained(os.path.join(HERE, "ce_minilm")).eval()
ce_cache: dict = {}


def ce_scores(q: str, texts: list[str]) -> list[float]:
    out = []
    for i in range(0, len(texts), 16):
        b = tok([q] * len(texts[i:i + 16]), texts[i:i + 16], padding=True, truncation="only_second", max_length=512, return_tensors="pt")
        with torch.no_grad():
            out += ce(**b).logits.squeeze(-1).tolist()
    return out


def ask(prompt: str) -> str:
    k = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if k not in cache:
        cache[k] = scz.chat(AGENT, prompt)["content"]
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
        qs.append((q, set(rel), exact[0]))
    if len(qs) == N:
        break
log("%d questions with an exact source section" % len(qs))

text_of = lambda o: " ".join(recs[o]["text"].split())[:2500]
gold_ce = [ce_scores(q, [text_of(e)])[0] for q, _, e in qs]
TAU = float(np.percentile(gold_ce, 25))
log("tau = 25th percentile of the gold sections' cross-encoder scores = %.3f (gold median %.3f, min %.3f)" % (TAU, np.median(gold_ce), min(gold_ce)))

catalogue, facts = st.community_catalogue(sg.conn, sg.build), st.community_entity_lines(sg.conn, sg.build)
runs = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
for i, (q, paper, exact) in enumerate(qs):
    for H in HOPS:
        key = "%d|%d" % (i, H)
        if key in runs:
            continue
        got = sg.retrieve(q, 3)
        ex = g.Explorer(sg, got, hops=H, per_query=g.PER_QUERY)
        run = g.run_agent(q, "", ex.initial(), ex, ask, 3, catalogue=catalogue, seed=zlib.crc32(q.encode()), mode="sample", facts=facts, hops=H, per_query=g.PER_QUERY)
        runs[key] = {"nodes": [{"ord": n["ord"], "query": n.get("query", 1), "hop": n["hop"]} for n in run["nodes"].values()],
                     "searches": sum(1 for t in run["trace"] if t["action"] == "SEARCH" and t["new"] is not None and t["result"].startswith("search")), "stop": run["stop"]}
        json.dump(runs, open(OUT, "w", encoding="utf-8"))
    log("question %d/%d done" % (i + 1, len(qs)))

# ---- score every arm
rows = {}
for i, (q, paper, exact) in enumerate(qs):
    ranked = sg.ranked(q)["ranked"]
    for H in HOPS:
        r = runs["%d|%d" % (i, H)]
        for k in KS:
            ords = list(dict.fromkeys(n["ord"] for n in r["nodes"] if n["query"] <= k + 1))
            n_shown = len(ords)
            ctrl = ranked[:n_shown]
            for name, shown in (("walk", ords), ("hybrid", ctrl)):
                need = [o for o in shown if (i, o) not in ce_cache]
                if need:
                    for o, s in zip(need, ce_scores(q, [text_of(o) for o in need])):
                        ce_cache[(i, o)] = s
                prec = float(np.mean([ce_cache[(i, o)] >= TAU for o in shown])) if shown else 0.0
                rows.setdefault((H, k, name), []).append({"exact": exact in shown, "paper": bool(paper & set(shown)), "prec": prec, "n": n_shown})
log("scored")

lines = ["tau %.3f | questions %d | agent %s | global view empty" % (TAU, len(qs), AGENT)]
for H in HOPS:
    lines.append("\nHOPS %d" % H)
    for k in KS:
        w, c = rows[(H, k, "walk")], rows[(H, k, "hybrid")]
        wn, cn = sum(x["exact"] for x in w), sum(x["exact"] for x in c)
        lo, hi = wilson(wn, len(w))
        only_w = sum(1 for a, b in zip(w, c) if a["exact"] and not b["exact"])
        only_c = sum(1 for a, b in zip(w, c) if b["exact"] and not a["exact"])
        lines.append("  searches %d | mean shown %.1f | walk exact %2d/%d = %.2f [%.2f,%.2f] | hybrid same size %2d/%d = %.2f | paired only-walk/only-hybrid %d/%d | paper walk %.2f hybrid %.2f | precision walk %.3f hybrid %.3f" % (
            k, np.mean([x["n"] for x in w]), wn, len(w), wn / len(w), lo, hi, cn, len(c), cn / len(c), only_w, only_c,
            np.mean([x["paper"] for x in w]), np.mean([x["paper"] for x in c]), np.mean([x["prec"] for x in w]), np.mean([x["prec"] for x in c])))
actual = {H: [runs["%d|%d" % (i, H)]["stop"] for i in range(len(qs))] for H in HOPS}
lines.append("\nagent stop reasons by hops: " + "; ".join("%d: %s" % (H, {s: v.count(s) for s in set(v)}) for H, v in actual.items()))
open(os.path.join(HERE, "stage_b_result.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
log("done")
