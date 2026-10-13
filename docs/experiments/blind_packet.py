"""docs/experiments/blind_packet.py: builds the human blind-reading packet from blind_cmp_answers.json.
For each of the 25 answered gold questions it writes the four arms' answers under shuffled labels X1-X4 (blind_packet.md), the label-to-arm key (blind_key.json) and an empty score sheet (blind_scores.txt).
Needs the Postgres section store only to recover the query sentences. No governing spec beyond plan.md (blind comparison, human-read follow-up)."""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import ingest_arxiv_sparsevec as ing
import section_corpus as sc

ans = json.load(open(os.path.join(HERE, "..", "..", ".tmp", "blind_cmp_answers.json"), encoding="utf-8"))
recs = sc.load()
_, body = ing.gold(recs)
qs = []
for q, rel in body:
    if [o for o in rel if " ".join(q.split()) in " ".join(recs[o]["text"].split())]:
        qs.append(q)
    if len(qs) == 60:
        break
rng = random.Random(20261012)
key, out = {}, [
    "# Blind reading packet: which answer is best?\n\n",
    "For each of the %d items below, read the four answers (labelled X1-X4, random order) and rank them best to worst. Judge: (1) addresses the query directly; (2) specific, substantive content (mechanisms, named methods, categories, results) rather than generic statements; (3) no padding, hedging or disclaimers; (4) claims carry citations in square brackets. The query is a sentence lifted from one paper.\n\n" % len(ans),
    "Write one line per item in `blind_scores.txt`, for example `item 3: X2 > X1 > X4 > X3`. Do not open `blind_key.json` until all items are ranked. Then run `python docs/experiments/blind_score.py`.\n",
]
for n, (i, a) in enumerate(sorted(ans.items(), key=lambda kv: int(kv[0]))):
    arms = list(a)
    rng.shuffle(arms)
    lab = {arm: "X%d" % (k + 1) for k, arm in enumerate(arms)}
    key["item %d" % (n + 1)] = {v: k for k, v in lab.items()}
    out.append("\n---\n\n## Item %d\n\n**Query:** %s\n" % (n + 1, qs[int(i)]))
    for arm in arms:
        out.append("\n**%s**\n\n%s\n" % (lab[arm], a[arm].strip()))
open(os.path.join(HERE, "blind_packet.md"), "w", encoding="utf-8").write("".join(out))
json.dump(key, open(os.path.join(HERE, "blind_key.json"), "w", encoding="utf-8"), indent=1)
open(os.path.join(HERE, "blind_scores.txt"), "w", encoding="utf-8").write("".join("item %d: \n" % (n + 1) for n in range(len(ans))))
print(len(ans), "items,", len("".join(out)), "characters")
