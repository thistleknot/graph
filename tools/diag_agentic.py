"""diag_agentic.py -- the A10 gold diagnostic lane: runs the ReAct loop (react.py)
end to end against operator-authored gold rows and scores gold recall.

Spec: .spec/specs/graph-explorer/design.md 6.23 A9-A10
Task: playbook.md T63

Second lane, additive to tools/diag_rerun.py's frozen 20-row do-no-harm set
(A7) -- this file never touches ROWS/run_row there. No reimplementation of the
walk or the loop: this calls react.run() / interpret.answer() only and reports
what comes back (Article VI: observed, not re-derived).

G-class rows are {id, prompt, gold_terms}. Gold terms are checked against the
LIVE corpus at authoring time -- see the ILIKE count comment beside each row
(mixed-full-dual, run_id bfa594df-1238-448f-8e2b-d05136e6307a, 2026-09-06);
every row ships with >= 3 matching chunks, most far more.

Usage: PYTHONPATH=. python tools/diag_agentic.py <run-label>
"""
import sys
import time

import config
import evidence
import graph_tools as gt
import interpret
import react

# -- G-class rows: (id, prompt, gold_terms) -- counts are
# `SELECT count(*) FROM node WHERE run_id=<run> AND body ILIKE '%term%'`
# against mixed-full-dual at authoring time.
ROWS = [
    ("G1", "who is the most famous musician of the 1990's?",
     ["kurt cobain", "nirvana"]),                    # counts: 16, 44
    ("G2", "what damage did the hurricane cause",
     ["hurricane", "tropical cyclone"]),              # counts: 423, 261
    ("G3", "describe the attack on pearl harbor",
     ["pearl harbor", "battle of midway"]),           # counts: 106, 20
    ("G4", "what is a popular american tv sitcom",
     ["seinfeld", "simpsons"]),                       # counts: 40, 161
    ("G5", "who is a legendary american sports figure",
     ["muhammad ali", "babe ruth"]),                  # counts: 18, 24
]


def score_row(result, answer_text=None, gold_terms=None, bodies=None):
    """PURE (A9): pass rule is gold recall == 1.0 over the accumulated
    evidence BODIES (spec A9: "gold terms present in the final evidence
    bodies") -- the digest-based per-iteration number undermeasures, since a
    compact digest can omit a term its chunk body carries (first live run:
    G1 evid=0.00 by digest while the walk had reached cobain chunks).
    `bodies` is the list of accumulated chunks' body texts; when absent,
    falls back to the digest-based number. answer_text optional."""
    iters = result["iterations"]
    final = iters[-1]
    if bodies is not None and gold_terms:
        evid = react.gold_recall(bodies, gold_terms)
    else:
        evid = final.gold_recall_evidence if final.gold_recall_evidence is not None else 0.0
    ans = react.gold_recall(answer_text, gold_terms) if answer_text else 0.0
    passed = evid == 1.0
    return {"evid": evid, "ans": ans, "n_iters": result["n_iters"],
            "stop_reason": result["stop_reason"], "passed": passed}


def run_row(conn, run, rid, prompt, gold_terms, embed):
    t0 = time.time()
    result = react.run(conn, run, prompt, embed=embed, gold_terms=gold_terms,
                        max_iters=3, seed=0)
    dt = time.time() - t0

    answer_text = None
    answer_error = None
    answer_bundle, ev = result.get("answer_bundle"), result.get("ev")
    if answer_bundle is not None and ev is not None:
        try:
            # A13: entails-first bundle react.run() already assembled --
            # no re-derivation of the cap here (Article VI).
            ans = interpret.answer(conn, run, answer_bundle, ev.terms, ev.concept, embed=embed)
            answer_text = ans.get("answer") or None
            if not ans.get("ok"):
                answer_error = ans.get("error")
        except Exception as e:                                       # noqa: BLE001
            answer_error = f"{type(e).__name__}: {e}"

    bodies = None
    try:
        ords = result.get("ords") or []
        if ords:
            with conn.cursor() as cur:
                cur.execute("SELECT body FROM node WHERE run_id = %s AND ord = ANY(%s)",
                            (str(run.run_id), list(ords)))
                bodies = [r["body"] for r in cur.fetchall()]
    except Exception:                                        # noqa: BLE001
        bodies = None                     # fall back to digest-based number

    sc = score_row(result, answer_text, gold_terms, bodies=bodies)
    sc["elapsed_s"] = round(dt, 2)
    sc["answer_error"] = answer_error
    return sc


def main():
    argv = sys.argv[1:]
    if len(argv) != 1:
        print("usage: PYTHONPATH=. python tools/diag_agentic.py <run-label>",
              file=sys.stderr)
        sys.exit(2)
    label = argv[0]

    try:
        conn = gt.connect()
        run = gt.get_run(conn, label)
    except Exception as e:
        print(f"ERROR: could not resolve run {label!r}: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"run_id={run.run_id} label={run.label} n_chunks={run.n_chunks}")

    try:
        embed = evidence.load_embed(config.MODEL_DIR)
    except Exception as e:                                            # noqa: BLE001
        print(f"WARN: embed load failed, continuing without: {e!r}")
        embed = None

    n_pass = n_fail = 0
    not_tested = []
    for rid, prompt, gold_terms in ROWS:
        try:
            sc = run_row(conn, run, rid, prompt, gold_terms, embed)
        except Exception as e:                                       # noqa: BLE001
            print(f"{rid}: NOT-TESTED {type(e).__name__}: {e}")
            not_tested.append(rid)
            continue

        v = "PASS" if sc["passed"] else "FAIL"
        if sc["passed"]:
            n_pass += 1
        else:
            n_fail += 1
        extra = f" answer_error={sc['answer_error']}" if sc["answer_error"] else ""
        print(f"{rid} {v} evid={sc['evid']:.2f} ans={sc['ans']:.2f} "
              f"iters={sc['n_iters']} stop={sc['stop_reason']} "
              f"({sc['elapsed_s']}s){extra}")

    print("---")
    print(f"AGENTIC {n_pass}/{len(ROWS)} | FAIL {n_fail} | NOT-TESTED {not_tested}")
    print(f"run_id={run.run_id}")

    sys.exit(0 if not not_tested and n_fail == 0 else (2 if not_tested else 1))


if __name__ == "__main__":
    main()
