"""scratch: the blind answer comparison, scaled to the 25 gold questions the Jev gate answered (stage_b_gate_runs.json). Four arms per question, same answer prompt and model:
  A  hybrid top-12 sections alone, no global view
  B  A's retrieval replaced by the question's own retrieval (SectionGraph.items: typical sections + nearest subgraph sections), no global view
  C  B + the global view (gate -> partials -> reduce, cached)
  D  the whole agent traversal at hops 3 (every section it saw, nearest the question first) + the global view
Each final answer goes through keep_grounded_sentences. A model judge sees ONLY the question and the four answers, shuffled and relabelled X1-X4, twice per question in different orders,
and ranks them by: addresses the question directly; specific, substantive content (mechanisms, named methods, categories) rather than generic statements; no padding or disclaimers; citations present.
Reported: mean rank per arm (1 = best), first places, and paired sign counts. The judge is the same model family as the writer: a second opinion, not a verdict; no human read these.
Pre-registered here before any answer is generated (2026-10-12). Model replies cached in .tmp/blind_cmp_cache.json. No governing spec beyond plan.md."""
import hashlib, json, math, os, random, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.chdir(os.path.join(HERE, ".."))
import numpy as np
import ingest_arxiv_sparsevec as ing
import section_corpus as sc, section_graphrag as g, section_store as st, sparsevec_store as ss
import summarize_clusters as scz

CACHE = os.path.join(HERE, "blind_cmp_cache.json")
cache = json.load(open(os.path.join(HERE, "stage_b_gate_cache.json"), encoding="utf-8"))          # the partials and global answers the agent saw
if os.path.exists(CACHE):
    cache.update(json.load(open(CACHE, encoding="utf-8")))
t0 = time.time()
log = lambda m: print("[%4.0fs] %s" % (time.time() - t0, m), flush=True)


def chat(model: str, prompt: str) -> str:
    k = hashlib.sha256((model + "\0" + prompt).encode("utf-8")).hexdigest()
    if k not in cache:
        cache[k] = scz.chat(model, prompt)["content"]
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    return cache[k]


conn = ss.connect()
sg = g.SectionGraph(conn, "xpa")
recs = sc.load()
_, body = ing.gold(recs)
qs = []
for q, rel in body:
    exact = [o for o in rel if " ".join(q.split()) in " ".join(recs[o]["text"].split())]
    if exact:
        qs.append((q, exact[0]))
    if len(qs) == 60:
        break
runs = json.load(open(os.path.join(HERE, "stage_b_gate_runs.json"), encoding="utf-8"))
answered = [i for i in range(60) if not runs[str(i)]["rejected"]]
log("%d answered questions" % len(answered))
key_of = lambda o: "%s#%s" % (recs[o]["doc_id"], recs[o]["section_idx"])
ARMS = ("A hybrid only", "B + subgraph", "C + global", "D + agent")


def items_of(ords: list[int], role: str) -> list[dict]:
    return [{"key": key_of(o), "ord": o, "community": 0, "role": role, "text": recs[o]["text"]} for o in ords]


def write(q: str, glob: str, items: list[dict]) -> str:
    raw = chat(scz.MODEL, g.answer_prompt(q, glob, items, g.answer_chars(len(items))))
    return g.keep_grounded_sentences(raw, items)


answers = {}
for n, i in enumerate(answered):
    q = qs[i][0]
    r = runs[str(i)]
    partials = []
    for c in r["passers"]:
        rec = st.summaries(sg.conn, sg.build, [c])[c]
        relevant, text = g.parse_map(chat(scz.MODEL, g.PARTIAL_PROMPT.format(question=q, cid=c, size=rec["size"], title=rec["title"], summary=rec["summary"])))
        if relevant:
            partials.append((c, text))
    glob = chat(scz.MODEL, g.reduce_prompt(q, partials))
    got = sg.retrieve(q, 3)
    base = sg.items(got)
    ranked = sg.ranked(q)["ranked"]
    seen = list(dict.fromkeys(nd["ord"] for nd in r["hops"]["3"]["nodes"]))
    sim = st.similarity(sg.conn, sg.build, seen, got["qd"], got["col"])
    trav = items_of(sorted(seen, key=lambda o: (-sim[o], o)), "traversal")
    answers[i] = {ARMS[0]: write(q, "", items_of(ranked[:12], "hybrid")), ARMS[1]: write(q, "", base), ARMS[2]: write(q, glob, base), ARMS[3]: write(q, glob, trav)}
    log("question %d/%d answers written" % (n + 1, len(answered)))

JUDGE = """Question (a statement taken from a research paper, used as a query): {q}

Below are four answers to it from different systems, in random order. Rank them from best to worst. Judge only the text: (1) it addresses the question directly; (2) it gives specific, substantive content (mechanisms, named methods, categories, results) rather than generic statements; (3) no padding, hedging or disclaimers; (4) claims carry citations in square brackets.

{answers}

Reply with one line only, in this form: RANK: X2 > X1 > X4 > X3"""
rng = random.Random(0)
ranks = {a: [] for a in ARMS}
firsts = {a: 0 for a in ARMS}
per_q = []
for i in answered:
    q = qs[i][0]
    for rep in range(2):
        order = list(ARMS)
        rng.shuffle(order)
        label = {a: "X%d" % (k + 1) for k, a in enumerate(order)}
        block = "\n\n".join("%s:\n%s" % (label[a], answers[i][a]) for a in order)
        reply = chat(scz.MODEL, JUDGE.format(q=q, answers=block))
        m = re.search(r"RANK:\s*(X\d(?:\s*>\s*X\d){3})", reply)
        if not m:
            log("unparsed judge reply for question %d: %r" % (i, reply[:80]))
            continue
        seq = [s.strip() for s in m.group(1).split(">")]
        back = {v: k for k, v in label.items()}
        rk = {back[s]: pos + 1 for pos, s in enumerate(seq)}
        for a in ARMS:
            ranks[a].append(rk[a])
        firsts[back[seq[0]]] += 1
        per_q.append(rk)
n_judged = len(per_q)
lines = ["answered questions %d | judgements %d (2 shuffles each) | judge and writer %s" % (len(answered), n_judged, scz.MODEL)]
for a in ARMS:
    lines.append("  %-14s mean rank %.2f | first place %d of %d" % (a, np.mean(ranks[a]), firsts[a], n_judged))
for x, y in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[1]), (ARMS[3], ARMS[2]), (ARMS[3], ARMS[0])):
    w = sum(1 for rk in per_q if rk[x] < rk[y])
    l = sum(1 for rk in per_q if rk[x] > rk[y])
    lines.append("  %s ranked above %s: %d of %d judgements, below: %d" % (x, y, w, n_judged, l))
lines.append("  mean answer length (characters): " + "; ".join("%s %.0f" % (a, np.mean([len(answers[i][a]) for i in answered])) for a in ARMS))
open(os.path.join(HERE, "blind_cmp_result.txt"), "w", encoding="utf-8").write("\n".join(lines))
json.dump({str(i): answers[i] for i in answered}, open(os.path.join(HERE, "blind_cmp_answers.json"), "w", encoding="utf-8"))
print("\n".join(lines))
log("done")
