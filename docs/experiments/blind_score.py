"""docs/experiments/blind_score.py: scores the human blind reading.
Reads blind_scores.txt (one `item N: X2 > X1 > X4 > X3` line per item) and blind_key.json (label -> arm) from this folder and prints the mean rank per arm
(1 = best), first places, and how often each arm ranks above the hybrid-only arm. Blank items are skipped. No governing spec beyond plan.md."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
key = json.load(open(os.path.join(HERE, "blind_key.json"), encoding="utf-8"))
ranks, first, above, n = {}, {}, {}, 0
for line in open(os.path.join(HERE, "blind_scores.txt"), encoding="utf-8"):
    m = re.match(r"item (\d+):\s*(X\d(?:\s*>\s*X\d){3})", line.strip())
    if not m:
        continue
    k = key["item %s" % m.group(1)]
    seq = [k[s.strip()] for s in m.group(2).split(">")]
    n += 1
    for pos, arm in enumerate(seq):
        ranks.setdefault(arm, []).append(pos + 1)
    first[seq[0]] = first.get(seq[0], 0) + 1
    base = seq.index("A hybrid only")
    for pos, arm in enumerate(seq):
        if pos < base:
            above[arm] = above.get(arm, 0) + 1
print("items ranked:", n)
for arm, r in sorted(ranks.items()):
    print("  %-14s mean rank %.2f | first place %d of %d | above hybrid-only in %d" % (arm, sum(r) / len(r), first.get(arm, 0), n, above.get(arm, 0)))
