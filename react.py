"""react.py -- agentic retrieval v0: a bounded ReAct loop over walk parameters.

The agent chooses ef_evidence knobs and query text BETWEEN walks; it never edits
chunks, edges or communities, and every individual walk stays the deterministic
sampler.ef_evidence call it already was (A1).

Spec: .spec/specs/graph-explorer/design.md 6.23 A1-A13
Task: playbook.md T61, T66
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from dataclasses import dataclass, field

import evidence
import interpret
import sampler

# --- A3/A4 constants ---------------------------------------------------------

MAX_ITERS = 3                 # ARBITRARY: chosen, not derived. Best practice is a
                               # budget calibrated on held-out task difficulty
                               # (6.26 T3(b), gold-lane). A4: iterations BEYOND
                               # the base walk
PROPOSE_TIMEOUT = 60.0         # COST-BOUND: a measured performance cap, not a
                               # quality threshold -- wall-clock budget on the
                               # propose call.
PROPOSE_MAX_TOKENS = 700       # COST-BOUND: a measured performance cap, not a
                               # quality threshold -- token/cost budget on the
                               # propose call.
ACTIONS = ("WIDEN", "REANCHOR", "PIVOT", "DEEPEN")   # A3
LADDER = ("REANCHOR", "DEEPEN", "WIDEN")             # A12: narrowing before
                                                      # amplification -- the
                                                      # matched filter beat
                                                      # WIDEN in every gold run
BOUNDS = {"ef": (16, 512), "k_anchor": (1, 12), "ring_top": (0, 8),
          "ring_per": (0, 24), "m": (1, 8), "bridge_pairs": (0, 8)}
BASE_PARAMS = {"ef": sampler.DEFAULT_EF, "m": sampler.DEFAULT_M,
               "k_anchor": sampler.DEFAULT_K_ANCHOR,
               "ring_top": sampler.DEFAULT_RING_TOP,
               "ring_per": sampler.DEFAULT_RING_PER,
               "bridge_pairs": sampler.DEFAULT_BRIDGE_PAIRS,
               "expand_aliases": False}
# BASE_PARAMS mirrors ef_evidence's own defaults so the base walk of the loop is
# byte-identical to today's single walk (A7). T/seed are not mutable knobs; seed
# is passed fixed by run().


@dataclass(frozen=True)
class IterationRecord:
    """A1 + A8 + A9: one row of the transcript the proposer (and the operator)
    sees. `action` is what PRODUCED this iteration (None for the base walk);
    `stop_reason` is set only on the final record."""
    i: int
    params: dict
    query: str
    n_chunks: int
    mean_score: float
    sdev_score: float
    entails: int
    contradicts: int
    neutrals: int
    per_cid_mean: dict
    new_chunks: int
    precision_proxy: float               # A9: entails / judged, judged = entails+contradicts
    contradiction_rate: float            # A9: contradicts / judged
    yield_ratio: float = 0.0             # A17(a): entails / n_chunks -- the informative SNR;
                                          # precision_proxy degenerates to 1.00 when contradicts
                                          # is zero, since the denominator becomes the numerator
    action: str | None = None
    stop_reason: str | None = None
    gold_recall_evidence: float | None = None   # A9/A10: only when gold_terms given to run()
    gold_recall_answer: float | None = None     # A9/A10: filled by the answer stage (T62)
    excluded: str | None = None                 # A12: the proposer's original action, when
                                                 # it was substituted for a diluted one


def stats(bundle, rr, cid_of=None) -> dict:
    """PURE, no DB, no network. A8 score-distribution stats plus A9 derived
    metrics -- both computed from data the walk/judge already produced."""
    scores = [bundle.scores.get(o, 0.0) for o in bundle.sampled]
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    sdev = statistics.pstdev(scores) if n > 1 else 0.0

    sampled_set = set(bundle.sampled)
    entails = contradicts = 0
    if rr and rr.get("verdicts"):
        for v in rr["verdicts"]:
            if v.get("ord") not in sampled_set:
                continue
            verdict = v.get("verdict")
            if verdict == "entails":
                entails += 1
            elif verdict == "contradicts":
                contradicts += 1
    elif rr:
        entails = len([o for o in (rr.get("entailed") or []) if o in sampled_set])
        contradicts = len([o for o in (rr.get("contradicts") or []) if o in sampled_set])
    neutrals = max(n - entails - contradicts, 0)

    per_cid_mean: dict = {}
    if cid_of:
        by_cid: dict = {}
        for o in bundle.sampled:
            c = cid_of.get(o)
            if c is None:
                continue
            by_cid.setdefault(c, []).append(bundle.scores.get(o, 0.0))
        per_cid_mean = {c: round(sum(vs) / len(vs), 4) for c, vs in by_cid.items()}

    judged = entails + contradicts               # A9: definitively judged, excludes neutral
    precision_proxy = round(entails / judged, 4) if judged else 0.0
    contradiction_rate = round(contradicts / judged, 4) if judged else 0.0
    yield_ratio = round(entails / n, 4) if n else 0.0   # A17(a): entails / n_chunks

    return {"n_chunks": n, "mean_score": round(mean, 4), "sdev_score": round(sdev, 4),
            "entails": entails, "contradicts": contradicts, "neutrals": neutrals,
            "per_cid_mean": per_cid_mean,
            "precision_proxy": precision_proxy, "contradiction_rate": contradiction_rate,
            "yield_ratio": yield_ratio}


def gold_recall(texts, gold_terms) -> float:
    """A9/A10 pure helper: fraction of gold_terms present in `texts` (a string
    or an iterable of strings), case-insensitive and underscore/space-
    insensitive ("kurt_cobain" matches "Kurt Cobain"). 0.0 when gold_terms is
    empty/None -- an unscored claim is not a perfect one."""
    if not gold_terms:
        return 0.0
    if isinstance(texts, str):
        blob = texts
    else:
        blob = " ".join(t for t in texts if t)
    norm_blob = " ".join(blob.lower().replace("_", " ").split())
    hits = 0
    for term in gold_terms:
        norm_term = " ".join(str(term).lower().replace("_", " ").split())
        if norm_term and norm_term in norm_blob:
            hits += 1
    return round(hits / len(gold_terms), 4)


def render_history(history) -> str:
    """PURE (A8/A9: numbers, not narrative). Fixed-width table, one row per
    IterationRecord, header first, plus a trailing per-community line."""
    header = ("it action     ef  k_anch ring   alias  n   mean   sdev  ent con neu "
              "prec  yield cr   query")
    lines = [header]
    for r in history:
        action = r.action if r.action else "base"
        if r.excluded:                             # A12: substitution note
            action = f"{r.excluded}>{action}"
        ring = f"{r.params.get('ring_top', 0)}/{r.params.get('ring_per', 0)}"
        alias = 1 if r.params.get("expand_aliases") else 0
        lines.append(
            f"{r.i:<2} {action:<9} {r.params.get('ef', 0):<3} "
            f"{r.params.get('k_anchor', 0):<6} {ring:<6} {alias:<5} "
            f"{r.n_chunks:<3} {r.mean_score:<6.4f} {r.sdev_score:<6.4f} "
            f"{r.entails:<3} {r.contradicts:<3} {r.neutrals:<3} "
            f"{r.precision_proxy:<5.2f} {r.yield_ratio:<5.2f} {r.contradiction_rate:<4.2f} {r.query}")
    for r in history:
        if not r.per_cid_mean:
            continue
        top = sorted(r.per_cid_mean.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        cids = " ".join(f"{c}={m:.2f}" for c, m in top)
        lines.append(f"it{r.i} cids: {cids}")
    out = "\n".join(lines)
    return out[:1200]


SUFFICIENCY_SYSTEM = """You judge whether the assembled evidence answers the prompt.

You will be given the prompt, a history table of (parameters -> walk score
distribution -> judge verdicts) for every iteration tried so far, and the latest
evidence digest. Read the history as a signal-to-noise problem, numbers first:

- yield (entails / n_chunks) IS the signal-to-noise ratio to move. Judge any
  action by whether it RAISED yield, not by whether n_chunks grew.
- prec (precision_proxy, entails / judged) is kept for continuity but can be
  DEGENERATE: when the judge returns zero contradicts, judged == entails and
  prec reads 1.00 on every row regardless of how much of the walk actually
  entails -- prec cannot tell a 1/88 walk from a 7/88 walk if both have zero
  contradicts. yield is what distinguishes them.
- An action already marked diluted in the history (shown as
  "PROPOSED>SUBSTITUTED" in the action column, or an action whose log2 mean
  score dropped below the history's band while n_chunks rose) must NOT be
  proposed again -- it is excluded and will be substituted regardless.
- Prefer narrowing (REANCHOR with terms absent from the prompt) over
  amplification (WIDEN) when a prior WIDEN raised n but dropped mean score:
  that is dilution, admitting noise rather than signal.

Decide:

1. sufficient: true only if the evidence, taken together, actually entails an
   answer to the prompt (not merely "some chunks are on-topic").
2. If not sufficient, propose exactly ONE next action from:
   - WIDEN: raise ef and ring budgets (broaden the same walk).
   - REANCHOR: replace or add query terms to re-anchor the walk (e.g. missing
     "grunge nirvana" for a 1990s-music prompt). Put the EXTRA TERMS ONLY in
     query_add (never restate the whole prompt). CRITICAL: query_add and
     `missing` must name CANDIDATE ENTITIES OR TOPICS that are ABSENT from
     the digest -- specific names a good answer would involve -- and must
     NEVER repeat words already in the prompt; repeating the prompt's own
     words retrieves the same chunks again and wastes the iteration.
   - PIVOT: walk from the strongest adjacent community.
   - DEEPEN: raise hop/expansion depth (m, bridge_pairs, ef).
   You may not invent new parameters or actions; these four are the only moves.

Respond with ONLY a JSON object, no prose, matching exactly:
{"sufficient": bool, "missing": [string, ...], "action": "WIDEN"|"REANCHOR"|"PIVOT"|"DEEPEN",
 "query_add": string, "why": string}
`query_add` is extra query terms only, used by REANCHOR; empty string otherwise."""


def propose(query, digest, history, *, missing_hint=None, timeout=PROPOSE_TIMEOUT,
            call=None) -> dict:
    """ONE model call. Never raises (mirrors interpret's I4 posture): any
    transport/parse/schema failure falls back to the deterministic LADDER."""
    call = call or interpret._call
    table = render_history(history)
    digest_text = interpret._clip(digest or "", 6000)
    user = f"PROMPT: {query}\n\n{table}\n\nDIGEST:\n{digest_text}"
    if missing_hint:
        user += f"\n\nMISSING SO FAR: {', '.join(missing_hint)}"

    try:
        text, _backend = call(SUFFICIENCY_SYSTEM, user, timeout, max_tokens=PROPOSE_MAX_TOKENS)
        obj = interpret._json_obj(text)
        if not isinstance(obj, dict):
            raise ValueError("no JSON object in proposer response")
        action = str(obj.get("action", "")).upper()
        if action not in ACTIONS:
            raise ValueError(f"action {action!r} not in {ACTIONS}")
        sufficient = bool(obj.get("sufficient", False))
        missing = [str(m) for m in (obj.get("missing") or [])][:5]
        query_add = str(obj.get("query_add", ""))[:120]
        why = str(obj.get("why", ""))
        return {"sufficient": sufficient, "missing": missing, "action": action,
                "query_add": query_add, "why": why, "source": "model"}
    except Exception as e:                                              # noqa: BLE001
        fallback_action = LADDER[(len(history) - 1) % 3]
        return {"sufficient": False, "missing": [], "action": fallback_action,
                "query_add": "", "why": "", "source": "fallback",
                "error": f"{type(e).__name__}: {e}"}


def apply_action(params, action, *, query, query_add="", pivot_terms="") -> tuple[dict, str]:
    """PURE, no DB. Returns a NEW params dict (input never mutated) plus the
    possibly-extended query string. Unknown action -> inputs unchanged."""
    p = dict(params)
    q = query

    def _append(base_query, terms):
        # A17(b): dedup PER TOKEN against the accumulated query, case-
        # insensitive -- a whole-string containment check let a partially-
        # overlapping proposal slip through (live receipt: iteration 3
        # re-appended "kurt cobain dave grohl eddie vedder grunge nirvana
        # pearl jam soundgarden alice in chains", every token already
        # present). Order preserved; only genuinely new tokens are added.
        base_toks = base_query.split()
        seen = {t.lower() for t in base_toks}
        new_toks = []
        for t in str(terms).split():
            low = t.lower()
            if low not in seen:
                seen.add(low)
                new_toks.append(t)
        if not new_toks:
            return base_query
        return f"{base_query} {' '.join(new_toks)}".strip()

    if action == "WIDEN":
        p["ef"] = p.get("ef", 0) * 2
        p["ring_top"] = p.get("ring_top", 0) + 2
        p["ring_per"] = p.get("ring_per", 0) + 4
    elif action == "REANCHOR":
        p["k_anchor"] = p.get("k_anchor", 0) + 2
        p["expand_aliases"] = True
        q = _append(q, query_add)
    elif action == "PIVOT":
        q = _append(q, pivot_terms)
        p["k_anchor"] = p.get("k_anchor", 0) + 1
    elif action == "DEEPEN":
        p["m"] = p.get("m", 0) + 1
        p["bridge_pairs"] = p.get("bridge_pairs", 0) + 2
        p["ef"] = p.get("ef", 0) + 32
    else:
        return dict(params), query

    for key, (lo, hi) in BOUNDS.items():
        if key in p:
            p[key] = min(max(p[key], lo), hi)
    return p, q


def filter_query_add(query_add: str, prompt: str) -> str:
    """A3(b): drop every proposed term whose lowercased token already appears
    in the PROMPT -- an echoed term re-anchors on the same chunks (live G1
    receipt: query_add='1990s famous musician' against a 1990s-musician
    prompt walked the identical 88 chunks three times). Pure.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A3 amendment
    Task: playbook.md T63
    """
    def norm(t):
        return t.strip("?.,!\"()").replace("'", "").lower()
    prompt_toks = {norm(t) for t in str(prompt).split()}
    kept = [t for t in str(query_add).split() if norm(t) not in prompt_toks]
    return " ".join(kept)


def cap_bundle(bundle, n: int = 100):
    """A5 amendment: cap an accumulated bundle to its top-n chunks by walk
    score before ANY downstream model call (live receipt: 121K chars
    truncated the answer at max_tokens). Returns the bundle unchanged when
    already within n. Pure -- dataclasses.replace, input never mutated.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A5 amendment
    Task: playbook.md T63
    """
    sampled = list(bundle.sampled)
    if len(sampled) <= n:
        return bundle
    keep = sorted(sampled, key=lambda o: -float(bundle.scores.get(o, 0.0)))[:n]
    keep_set = set(keep)
    return dataclasses.replace(
        bundle, sampled=keep,
        scores={o: s for o, s in bundle.scores.items() if o in keep_set},
        origin={o: v for o, v in bundle.origin.items() if o in keep_set})


def entails_first_bundle(bundle, verdict_of, n: int = 50):
    """A13: like cap_bundle, but entailing ords come first (by walk score,
    descending), non-entailing fill the rest by score; the cap still binds --
    total kept is at most n, entails trimmed to their top-n share if there
    are more entails than n. Pure -- dataclasses.replace, input never
    mutated.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A13
    Task: playbook.md T66
    """
    def is_entail(o):
        v = verdict_of.get(o)
        return bool(v) and v.get("verdict") == "entails"

    sampled = list(bundle.sampled)
    entails = sorted((o for o in sampled if is_entail(o)),
                      key=lambda o: -float(bundle.scores.get(o, 0.0)))
    rest = sorted((o for o in sampled if not is_entail(o)),
                  key=lambda o: -float(bundle.scores.get(o, 0.0)))
    keep = (entails + rest)[:n]
    keep_set = set(keep)
    return dataclasses.replace(
        bundle, sampled=keep,
        scores={o: s for o, s in bundle.scores.items() if o in keep_set},
        origin={o: v for o, v in bundle.origin.items() if o in keep_set})


def dilution_detected(history) -> str | None:
    """A12: the house ruler for detecting a dilating action. Over every
    IterationRecord with mean_score > 0, take log2(mean_score) (guards
    zero/negative means, which have no log). The band's lower bound is the
    LOWER of two estimators computed over ALL those log2 values: mean - sdev
    (population sdev) and median - 1.4826*MAD (MAD = median absolute
    deviation from the median). Any iteration whose own log2 mean falls
    below that bound WHILE its n_chunks >= the previous (usable) iteration's
    n_chunks is dilution -- more chunks bought a lower signal. Returns the
    ACTION that produced the first such iteration, or None. Needs >= 3
    usable iterations to fire; a 2-point band is noise. Pure.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A12
    Task: playbook.md T66
    """
    usable = [(r, math.log2(r.mean_score)) for r in history if r.mean_score > 0]
    if len(usable) < 3:
        return None

    logs = [lv for _, lv in usable]
    mean = sum(logs) / len(logs)
    sdev = statistics.pstdev(logs)
    med = statistics.median(logs)
    mad = statistics.median([abs(lv - med) for lv in logs])
    bound = min(mean - sdev, med - 1.4826 * mad)

    for idx in range(1, len(usable)):
        rec, log_mean = usable[idx]
        prev_rec, _ = usable[idx - 1]
        if rec.action and log_mean < bound and rec.n_chunks >= prev_rec.n_chunks:
            return rec.action
    return None


def pivot_terms(conn, run_, cids) -> str:
    """LIVE, one small helper: strongest adjacent community by gt.quotient.
    A proposal aid, not a critical path -- any failure degrades to '' (PIVOT
    then falls back to WIDEN in run())."""
    try:
        import graph_tools as gt
        cid_set = set(cids or [])
        if not cid_set:
            return ""
        for row in gt.quotient(conn, run_, limit=60):
            a, b = row.get("cid_a"), row.get("cid_b")
            in_a, in_b = a in cid_set, b in cid_set
            if in_a == in_b:
                continue
            outside = b if in_a else a
            kw = gt.community(conn, run_, outside).get("keywords") or []
            terms = " ".join(kw[:4])
            if terms:
                return terms
        return ""
    except Exception:                                                    # noqa: BLE001
        return ""


def run(conn, run_, query, *, embed=None, judge_fn=None, walk_fn=None,
        propose_fn=None, max_iters=MAX_ITERS, seed=0, gold_terms=None) -> dict:
    """A1-A9 loop: base walk + up to max_iters proposer-directed re-walks.

    walk_fn(query, params) -> (bundle, ev); default = sampler.ef_evidence(...)
    then evidence.assemble(...).
    judge_fn(bundle, ev) -> rr; default = interpret.reason(..., judge=True).
    propose_fn(query, digest, history, missing_hint=...) -> dict; default = propose.

    Never raises on a judge/proposer failure (both already failure-tolerant);
    DOES let a walk_fn exception propagate -- a broken walk is an upstream
    defect, not a result (Article VI)."""

    def _default_walk(q, params):
        bnd, _tele = sampler.ef_evidence(conn, run_, q, seed=seed, **params)
        ev = evidence.assemble(conn, run_, bnd, embed=embed)
        return bnd, ev

    def _default_judge(bnd, ev):
        return interpret.reason(conn, run_, bnd, ev.terms, ev.concept, embed=embed,
                                 judge=True, pw=ev.pathways, digest=ev.digest)

    walk_fn = walk_fn or _default_walk
    judge_fn = judge_fn or _default_judge
    propose_fn = propose_fn or propose

    params = dict(BASE_PARAMS)
    cur_query = query
    action = None

    verdict_of: dict = {}
    found_at: dict = {}
    scores_all: dict = {}
    origin_all: dict = {}
    missing_acc: list = []
    evidence_texts: list = []
    excluded_actions: set = set()
    excluded_note = None          # A12: the proposer's action when substituted

    history: list = []
    last_bundle = last_ev = last_rr = None
    stop_reason = None
    prev_ent_total = None         # A17(c): cumulative entails as of the prior iteration

    i = 0
    while True:
        bundle, ev = walk_fn(cur_query, params)
        last_bundle, last_ev = bundle, ev

        new = [o for o in bundle.sampled if o not in verdict_of]
        if new:
            new_set = set(new)
            sub = dataclasses.replace(
                bundle, sampled=new,
                scores={o: bundle.scores[o] for o in new if o in bundle.scores},
                origin={o: v for o, v in bundle.origin.items() if o in new_set})
            rr = judge_fn(sub, ev)
            last_rr = rr if rr is not None else last_rr
            for v in (rr.get("verdicts") or []) if rr else []:
                o = v.get("ord")
                if o is not None and o not in verdict_of:
                    verdict_of[o] = {"verdict": v.get("verdict"), "why": v.get("why", "")}
        # new empty -> judged=0, no judge call spent

        for o in bundle.sampled:
            found_at.setdefault(o, i)
            scores_all.setdefault(o, bundle.scores.get(o, 0.0))
            origin_all.setdefault(o, bundle.origin.get(o, "walk"))

        merged_view = {"verdicts": [
            {"ord": o, "verdict": verdict_of[o]["verdict"]}
            for o in bundle.sampled if o in verdict_of]}
        st = stats(bundle, merged_view, getattr(ev, "cid_of", None))

        if gold_terms:
            digest_text = getattr(ev, "digest", None)
            if not digest_text:
                digest_text = " ".join(
                    " ".join(v) for v in (getattr(ev, "concept", {}) or {}).values())
            evidence_texts.append(digest_text or "")
            gr_evidence = gold_recall(evidence_texts, gold_terms)
        else:
            gr_evidence = None

        record = IterationRecord(
            i=i, params=dict(params), query=cur_query,
            n_chunks=st["n_chunks"], mean_score=st["mean_score"],
            sdev_score=st["sdev_score"], entails=st["entails"],
            contradicts=st["contradicts"], neutrals=st["neutrals"],
            per_cid_mean=st["per_cid_mean"], new_chunks=len(new),
            precision_proxy=st["precision_proxy"],
            contradiction_rate=st["contradiction_rate"],
            yield_ratio=st["yield_ratio"],
            action=action, stop_reason=None,
            gold_recall_evidence=gr_evidence, gold_recall_answer=None,
            excluded=excluded_note)
        history.append(record)
        excluded_note = None

        diluted = dilution_detected(history)          # A12
        if diluted:
            excluded_actions.add(diluted)

        ent_total = sum(1 for v in verdict_of.values() if v["verdict"] == "entails")

        # A17(c) no-movement stop, LOOSENED (operator decision, 2026-09-07):
        # the original three-way conjunction (mean within 1%, per-cid mass
        # within 0.02, AND no new entails) never fired in practice -- mean
        # and per-cid mass keep wiggling even when the loop is adding
        # nothing useful, so the conjunction almost never held even across
        # genuinely wasted iterations. The ONE condition that actually
        # matters is whether the iteration added any new entailing chunks:
        # if the accumulated entail count did not increase versus the
        # previous iteration, the loop is spending budget for nothing and
        # SHALL stop. Needs a previous record (i >= 2), and still requires
        # ent_total > 0: a walk that has found NOTHING yet is already routed
        # by A2's forced-insufficient path, and "flat at zero" is not the
        # stagnation this guards against -- it is the ordinary zero-entail
        # case, which keeps spending budget on purpose (A6) rather than
        # giving up early.
        if i >= 2 and prev_ent_total is not None and ent_total > 0:
            no_new_entails = ent_total <= prev_ent_total
            if no_new_entails:
                stop_reason = "no-movement"
                break
        prev_ent_total = ent_total

        if ent_total == 0 and i == max_iters:
            stop_reason = "budget"
            break
        if i == max_iters:
            # still call the proposer only if needed for reporting; budget bites
            stop_reason = "budget"
            break

        prop = propose_fn(cur_query, getattr(ev, "digest", None), history,
                           missing_hint=missing_acc)
        sufficient = bool(prop.get("sufficient", False))
        if ent_total == 0:
            sufficient = False        # A2: forced, regardless of the model
        if sufficient:
            stop_reason = "sufficient"
            break

        for m in prop.get("missing") or []:
            if m not in missing_acc:
                missing_acc.append(m)
        missing_acc = missing_acc[:8]

        act = prop.get("action")
        pterms = ""
        if act == "PIVOT":
            if conn is None:
                act = "WIDEN"
            else:
                pterms = pivot_terms(conn, run_, getattr(ev, "cids", None))
                if not pterms:
                    act = "WIDEN"

        # A3(b): REANCHOR terms filtered against the PROMPT's own tokens --
        # the live G1 run showed the judge echoing the prompt back, which
        # re-anchors on the same chunks. Empty after filtering -> escalate.
        query_add = filter_query_add(prop.get("query_add", ""), query)
        if act == "REANCHOR" and not query_add:
            act = "DEEPEN"

        # A12: the proposer may repeat an action already marked dilution --
        # enforce the exclusion algorithmically rather than trusting the
        # model to honour the history table. Substitute the first
        # non-excluded action from the deterministic ladder.
        substituted_from = None
        if act in excluded_actions:
            substituted_from = act
            act = next((a for a in LADDER if a not in excluded_actions), act)

        new_params, new_query = apply_action(
            params, act, query=cur_query, query_add=query_add,
            pivot_terms=pterms)

        # A3(c): never re-walk a (query, params) pair already walked. One
        # escalation attempt along the ladder; still a repeat -> fixed point.
        seen = {(r.query, tuple(sorted(r.params.items()))) for r in history}
        if (new_query, tuple(sorted(new_params.items()))) in seen:
            act = LADDER[(LADDER.index(act) + 1) % len(LADDER)] if act in LADDER else "WIDEN"
            new_params, new_query = apply_action(
                params, act, query=cur_query, query_add=query_add,
                pivot_terms=pterms)
            if (new_query, tuple(sorted(new_params.items()))) in seen:
                stop_reason = "fixed-point"
                break

        if new_params == params and new_query == cur_query:
            stop_reason = "no-op action"
            break

        params, cur_query, action = new_params, new_query, act
        excluded_note = substituted_from
        i += 1

    if stop_reason is None:
        stop_reason = "budget"
    history[-1] = dataclasses.replace(history[-1], stop_reason=stop_reason)

    ords = list(found_at.keys())
    entails = [o for o, v in verdict_of.items() if v["verdict"] == "entails"]
    contradicts = [o for o, v in verdict_of.items() if v["verdict"] == "contradicts"]

    # A5/A13: the accumulated-union bundle across every iteration -- template
    # fields (query, run_id, anchors, communities, ...) come from the last
    # walk, but sampled/scores/origin are the full union so the answer stage
    # sees everything the loop found, not just the final walk.
    union_bundle = None
    if last_bundle is not None:
        union_bundle = dataclasses.replace(
            last_bundle, sampled=ords,
            scores={o: scores_all.get(o, 0.0) for o in ords},
            origin={o: origin_all.get(o, "walk") for o in ords})
    answer_bundle = (entails_first_bundle(union_bundle, verdict_of, 50)
                      if union_bundle is not None else None)

    return {"ords": ords, "found_at": found_at, "verdicts": verdict_of,
            "entails": entails, "contradicts": contradicts,
            "iterations": history, "history_table": render_history(history),
            "stop_reason": stop_reason, "n_iters": len(history) - 1,
            "missing": missing_acc,
            "bundle": union_bundle, "ev": last_ev, "rr": last_rr,
            "answer_bundle": answer_bundle}
