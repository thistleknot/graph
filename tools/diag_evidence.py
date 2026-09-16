"""Deterministic evidence lane: does the gold term SURVIVE into what the judge reads?

Spec: .spec/specs/graph-explorer/design.md 6.28 (the per-unit evidence cap) ·
Task: playbook.md T95

WHY THIS EXISTS. The gold lane (tools/diag_agentic.py) scores the model's ANSWER,
so every reading carries the model's sampling variance. Measured 2026-09-16 on
ab-document, two identical n=3 runs of the same arm, same corpus:

    run 1   G1 0.333  G2 0.833  G4 0.500  G6 0.667  G7 0.167  G8 0.833  -> 0.556
    run 2   G1 0.167  G2 0.333  G4 0.500  G6 0.333  G7 0.167  G8 1.000  -> 0.417

The 6-row mean moved 0.139 between repeats, while the effect under test
(section vs document chunking) measured 0.083. The comparator's own noise was
LARGER than the difference it was asked to resolve, so no number of repeats can
settle the question -- that is a broken instrument, not a tie (rules/085).

WHAT THIS MEASURES INSTEAD. Everything up to the model call is deterministic:
the walk is seeded (sampler.ef_evidence, seed=0) and interpret.render_bundle is
deterministic by contract (I3, "same bundle -> same string"). So this lane asks
the question the LLM cannot blur:

    does the gold term appear in the evidence text the judge is handed?

No model call, no sampling, zero variance between runs. A difference here is a
real difference. It also targets exactly what 6.28 identified as the live axis:
the per-unit budget clips every unit, so this lane reports whether that clipping
is what loses the answer. Measured at the walk sizes this lane actually returns
(~90 units, so per_doc = 60000/90 = 645 chars, BELOW the 1,500 cap):

    ab-section   645 of a 1,816-char unit shown  -> 36%
    ab-document  645 of a 13,720-char unit shown -> 4.7%

Section shows the judge ~7x more of its own unit, and STILL both arms score the
same 14/18 gold survival -- so the clipping ratio is not what decides it.

It does NOT replace the gold lane: an answer can still be wrong from good
evidence. It bounds it from below. Evidence that never contained the gold cannot
produce an answer that names it, so a failure here is upstream of the model and
attributable without argument.

Usage: PYTHONPATH=. python tools/diag_evidence.py <run-label> [<run-label> ...]
"""
import sys

import config
import evidence as ev_mod
import graph_tools as gt
import interpret
import react
import sampler

# Same rows as tools/diag_agentic.py -- imported rather than restated so the two
# lanes can never drift apart on what "gold" means.
sys.path.insert(0, "tools")
from diag_agentic import ROWS                                      # noqa: E402


def evidence_for(conn, run, prompt, embed):
    """The deterministic half of the pipeline: seeded walk -> rendered evidence.
    Returns (evidence_text, n_chunks). No model call anywhere in this path."""
    params = dict(react.BASE_PARAMS)
    bnd, _tele = sampler.ef_evidence(conn, run, prompt, seed=0, **params)
    ev = ev_mod.assemble(conn, run, bnd, embed=embed)
    text = interpret.render_bundle(conn, run, bnd, ev.terms, ev.concept, embed=embed)
    return text, len(bnd.sampled)


def main():
    labels = sys.argv[1:]
    if not labels:
        print("usage: PYTHONPATH=. python tools/diag_evidence.py <run-label> ...",
              file=sys.stderr)
        sys.exit(2)

    conn = gt.connect()
    try:
        embed = ev_mod.load_embed(config.MODEL_DIR)
    except Exception as e:                                          # noqa: BLE001
        print(f"WARN: embed load failed, continuing without: {e!r}")
        embed = None

    for label in labels:
        run = gt.get_run(conn, label)
        print(f"\n########## {label} (run {str(run.run_id)[:8]}, "
              f"{run.n_chunks} chunks) ##########")
        print(f"{'row':5} {'chunks':>6} {'ev_chars':>9} {'gold_in_evidence':>18}  terms")
        tot, hit_tot = 0, 0
        for rid, prompt, gold_terms in ROWS:
            try:
                text, n = evidence_for(conn, run, prompt, embed)
            except Exception as e:                                  # noqa: BLE001
                print(f"{rid:5} NOT-TESTED {type(e).__name__}: {e}")
                continue
            low = text.lower()
            hits = [g for g in gold_terms if g.lower() in low]
            tot += len(gold_terms)
            hit_tot += len(hits)
            marks = " ".join(("+" if g in hits else "-") + g for g in gold_terms)
            print(f"{rid:5} {n:6d} {len(text):9,d} {len(hits):9d}/{len(gold_terms):<8d} {marks}")
        if tot:
            print(f"  GOLD SURVIVAL {hit_tot}/{tot} = {hit_tot / tot:.3f}  "
                  f"(deterministic -- rerunning this gives the same number)")


if __name__ == "__main__":
    main()
