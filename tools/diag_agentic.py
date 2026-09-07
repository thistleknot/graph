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
import walker_core

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
    # T77 (A14(d) coverage hole): G1-G5 above all reach the loop because
    # diag_agentic calls react.run() unconditionally. Missing from every
    # diagnostic was the UI path where the BASE walk already has entails > 0
    # on a NON-superlative prompt -- pre-A15 the loop never ran there at all,
    # and that silent path is exactly what let the Gallagher regression ship.
    # G6-G8 are factual "what happened" prompts (not superlative) over topics
    # already verified above -- counts re-confirmed live, mixed-full-dual,
    # run_id bfa594df-1238-448f-8e2b-d05136e6307a, 2026-09-07.
    ("G6", "what happened during the attack on pearl harbor in 1941",
     ["pearl harbor", "pacific fleet"]),               # counts: 106, 46
    ("G7", "what damage does a hurricane cause to coastal areas",
     ["storm surge", "category 5"]),                   # counts: 106, 44
    ("G8", "what is the plot of the sitcom seinfeld",
     ["seinfeld", "jerry seinfeld"]),                  # counts: 40, 10
    # G9: SUPERLATIVE regression pin (A14/A16 at the tool level) -- this row
    # MUST be gated (gate=superlative) even after the loop runs, because no
    # entailing chunk ranks a population of hurricanes, only individual ones.
    ("G9", "what was the deadliest hurricane on record",
     ["deadliest hurricane", "galveston hurricane"]),  # counts: 3, 6
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


def gate_reason(result, prompt, answer_text):
    """T77: what walker_core.answer_gate would say about the FINAL,
    accumulated answer -- independent of react's own stop_reason. Mirrors
    the accumulated-evidence gate call in walker_app.py (A15's re-run of the
    SAME gate over the loop's evidence, not a re-derivation of its logic --
    Article VI). Returns a short label: "ok" (not gated), "zero-entails",
    "uncited", or "superlative".

    Spec: .spec/specs/graph-explorer/design.md 6.23 A14, A15
    Task: playbook.md T77
    """
    entails = result.get("entails") or []
    verdicts = result.get("verdicts") or {}
    ans_ords = walker_core.cited_ords(answer_text) if answer_text else []
    why = {o: v.get("why", "") for o, v in verdicts.items()}
    sup_entails = walker_core.count_population_superlatives(
        [why.get(o, "") for o in entails])
    gated, _ = walker_core.answer_gate(
        answer_text or "", len(entails), prompt=prompt,
        entail_ords=entails, answer_ords=ans_ords,
        superlative_entails=sup_entails, n_chunks=len(result.get("ords") or []))
    if not gated:
        return "ok"
    if len(entails) <= 0:
        return "zero-entails"
    if any(o not in entails for o in ans_ords):
        return "uncited"
    return "superlative"


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
    sc["gate"] = gate_reason(result, prompt, answer_text)
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

        # G9 (A14/A16 regression pin): this row MUST be gated as superlative
        # regardless of gold recall -- a passing recall over an ungated
        # crowned answer would BE the Gallagher regression (A15).
        row_passed = sc["passed"] and (rid != "G9" or sc["gate"] == "superlative")
        v = "PASS" if row_passed else "FAIL"
        if row_passed:
            n_pass += 1
        else:
            n_fail += 1
        extra = f" answer_error={sc['answer_error']}" if sc["answer_error"] else ""
        print(f"{rid} {v} evid={sc['evid']:.2f} ans={sc['ans']:.2f} "
              f"iters={sc['n_iters']} stop={walker_core.stop_reason_label(sc['stop_reason'])} "
              f"gate={sc['gate']} ({sc['elapsed_s']}s){extra}")

    print("---")
    print(f"AGENTIC {n_pass}/{len(ROWS)} | FAIL {n_fail} | NOT-TESTED {not_tested}")
    print(f"run_id={run.run_id}")

    sys.exit(0 if not not_tested and n_fail == 0 else (2 if not_tested else 1))


if __name__ == "__main__":
    main()
