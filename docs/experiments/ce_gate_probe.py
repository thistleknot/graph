"""docs/experiments/ce_gate_probe.py: how the cross-encoder gate (section_graphrag G15) behaves on the 60 gold body-sentence questions, with no language model in the loop.
For each question: the CE_POOL best hybrid sections are scored by cross-encoder/ms-marco-MiniLM-L-6-v2 on (question, section); the gate passes a question when some section scores at/above CE_TAU.
Reported: questions with no passing section (the gate would reject them), questions whose exact source section is in the pool, and whose source section itself passes, plus the rank the cross-encoder gives it.
Needs the Postgres section store (build tag xpa) and ~/models/ce-ms-marco-minilm-l6. No governing spec beyond plan.md (step 5g)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np
import ingest_arxiv_sparsevec as ing
import section_corpus as sc, section_graphrag as g, sparsevec_store as ss

sg = g.SectionGraph(ss.connect(), "xpa")
recs = sc.load()
_, body = ing.gold(recs)
qs = []
for q, rel in body:
    exact = [o for o in rel if " ".join(q.split()) in " ".join(recs[o]["text"].split())]
    if exact:
        qs.append((q, exact[0]))
    if len(qs) == 60:
        break
EXTRA = ["How do LLM agents store and retrieve long-term memory?"]
rejected, in_pool, source_passes, source_rank, best = 0, 0, 0, [], []
for q, src in qs:
    pool = sg.retrieve(q, 3)["ranked"][:g.CE_POOL]
    marks = g.cross_scores(q, [" ".join(recs[o]["text"].split())[:g.CE_CHARS] for o in pool])
    best.append(max(marks))
    rejected += max(marks) < g.CE_TAU
    if src in pool:
        in_pool += 1
        m = marks[pool.index(src)]
        source_passes += m >= g.CE_TAU
        source_rank.append(1 + sum(1 for x in marks if x > m))
lines = ["questions %d | pool %d sections | pass score %.3f" % (len(qs), g.CE_POOL, g.CE_TAU),
         "gate rejects %d of %d questions (no section at/above the pass score); best section score per question: median %.2f, min %.2f" % (rejected, len(qs), np.median(best), min(best)),
         "source section in the pool for %d of %d; it passes for %d of those; cross-encoder rank of the source inside the pool: median %d, rank 1 in %d" % (in_pool, len(qs), source_passes, np.median(source_rank), sum(1 for r in source_rank if r == 1))]
lines.append("in-corpus questions rejected at each pass score: " + " | ".join("%.2f: %d of %d" % (c, sum(1 for b in best if b < c), len(best)) for c in (-4.0, -2.0, 0.0, 2.0, 5.116)))
OFF = ["How do I bake sourdough bread at home?", "Who won the 2022 football world cup?", "What is the capital of Australia?", "How do I change a flat tire on a car?", "What are the best hiking trails in Colorado?",
       "What is a good recipe for chicken soup?", "How many calories are in a banana?", "Which team won the last basketball championship?"]
off_best = []
for q in EXTRA + OFF:
    pool = sg.retrieve(q, 3)["ranked"][:g.CE_POOL]
    marks = g.cross_scores(q, [" ".join(recs[o]["text"].split())[:g.CE_CHARS] for o in pool])
    lines.append("%s question %r: %d of %d sections pass; best %.2f" % ("extra" if q in EXTRA else "off-topic", q, sum(1 for m in marks if m >= g.CE_TAU), len(pool), max(marks)))
    off_best += [max(marks)] if q in OFF else []
lines.append("off-topic questions (%d) accepted at each pass score: " % len(OFF) + " | ".join("%.2f: %d" % (c, sum(1 for b in off_best if b >= c)) for c in (-4.0, -2.0, 0.0, 2.0, 5.116)))
open(os.path.join(HERE, "ce_gate_probe_result.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
