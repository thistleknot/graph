"""diag_rerun.py -- re-run the FROZEN diagnostic prompt set against the live
serve path (sampler.ef_evidence) and print per-row measurements.

Spec: .spec/specs/graph-explorer/diagnostic-prompts.md (FROZEN set, sections A-E)
Task: playbook.md T7

No reimplementation of the walk: this file calls gt.connect() / gt.get_run() /
sampler.ef_evidence() only. It contains no BM25, no anchor allocation, no walk
logic of its own -- those live in sampler.py / graph_tools.py and are observed
here through the return values (bundle, tele).

Usage: PYTHONPATH=. python src/diag_rerun.py <run-label>
"""
import json
import sys
import time
from pathlib import Path

import config
import evidence
import graph_tools as gt
import sampler

# Term lexicons -- the evaluator's operationalization of the doc's prose
# clauses (diagnostic-prompts.md sections A3/A4/E2/E3). Frozen constants,
# dated 2026-09-03; not an edit to the doc's expectations.
STORM = {"storm", "hurricane", "cyclone", "tropical", "flood"}
BOAT = {"boat", "race", "oxford", "cambridge", "rowing", "crew"}
GAME = {"game", "mario", "zelda", "nintendo", "player", "console", "video"}
WAR = {"battle", "war", "army", "navy", "troops", "military"}
POLIT = {"senator", "senate", "congress", "tax", "bill", "legislature",
         "governor", "election"}


def _source_counts(conn, run, bundle):
    """One batch query: doc_id + source for every sampled ordinal."""
    ords = bundle.sampled
    if not ords:
        return {}, {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ord, doc_id, attrs->>'source' AS source FROM node "
            "WHERE run_id=%s AND ord = ANY(%s)", (run.run_id, ords))
        rows = cur.fetchall()
    by_ord = {r["ord"]: r for r in rows}
    counts = {}
    for o in ords:
        src = by_ord.get(o, {}).get("source")
        counts[src] = counts.get(src, 0) + 1
    return counts, by_ord


def _anchor_info(conn, run, bundle):
    """Anchors get their OWN batch query -- an anchor need not end up in
    bundle.sampled (ef_search may not walk into it), so resolving doc_id via
    the sampled-only lookup would silently drop anchors."""
    ords = bundle.anchors
    if not ords:
        return [], []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ord, doc_id, attrs->>'source' AS source FROM node "
            "WHERE run_id=%s AND ord = ANY(%s)", (run.run_id, ords))
        rows = {r["ord"]: r for r in cur.fetchall()}
    doc_ids = [rows[o]["doc_id"] for o in ords if o in rows]
    sources = [rows[o]["source"] for o in ords if o in rows]
    return doc_ids, sources


def _touched_keywords(communities):
    kws = set()
    for c in communities:
        for k in (c.get("keywords") or []):
            kws.add(k.lower())
    return kws


def _kw_hits(communities, lexicon):
    """True if any touched community's keywords intersect the lexicon.
    Community keywords are compound tokens (e.g. 'tropical_cyclone'); match
    if any lexicon term is a substring of any keyword token."""
    for c in communities:
        for kw in (c.get("keywords") or []):
            kwl = kw.lower()
            if any(term in kwl for term in lexicon):
                return True
    return False


def _chunks_by_community_kw(conn, run, bundle, by_ord, lexicon):
    """Count sampled chunks whose OWN community's keywords hit the lexicon."""
    # map ord -> cid via community membership (reuse communities_touched rows,
    # which are grouped by cid, not per-ord) -- do a direct per-ord lookup.
    if not bundle.sampled:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT v.ord, c.keywords FROM unnest(%s::int[]) AS v(ord) "
            "LEFT JOIN community c ON c.run_id=%s AND c.members @> ARRAY[v.ord]",
            (bundle.sampled, run.run_id))
        rows = cur.fetchall()
    n = 0
    for r in rows:
        kws = r.get("keywords") or []
        if any(any(term in kw.lower() for term in lexicon) for kw in kws):
            n += 1
    return n


def _xprov(conn, run, bundle):
    ords = bundle.sampled
    if not ords:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.provenance, count(*) FROM edge_sym s "
            "JOIN node na ON na.run_id=s.run_id AND na.ord=s.a "
            "JOIN node nb ON nb.run_id=s.run_id AND nb.ord=s.b "
            "WHERE s.run_id=%s AND s.valid_to IS NULL AND s.a < s.b "
            "AND s.a = ANY(%s) AND s.b = ANY(%s) "
            "AND na.attrs->>'source' IS DISTINCT FROM nb.attrs->>'source' "
            "GROUP BY 1", (run.run_id, ords, ords))
        return {r["provenance"]: r["count"] for r in cur.fetchall()}


def _share(counts, total, key):
    return counts.get(key, 0) / total if total else 0.0


# -- row definitions: (id, prompt, knobs) -- assertion logic is inline in run_row.
WALK_DEFAULT = {}
EF24 = {"ef": 24, "T": 0.0, "bridge_pairs": 0}

ROWS = [
    ("A1", "how did aircraft carriers decide the battle of midway", WALK_DEFAULT),
    ("A2", "what happened at the battle of the coral sea", WALK_DEFAULT),
    ("A3", "how does wikipedia describe hurricane damage", WALK_DEFAULT),
    ("A4", "history of the boat race between oxford and cambridge", WALK_DEFAULT),
    ("A5", "how did the battleship yamato sink", WALK_DEFAULT),
    ("B1", "a quote about courage", EF24),
    ("B2", "a quote about love and loss", EF24),
    ("B3", "a quote about books and reading", EF24),
    ("B4", "a quote about being broken", EF24),
    ("B5", "a quote about dreams and hope", EF24),
    ("C1", "jury trial grand jury investigation", WALK_DEFAULT),
    ("C2", "school children teacher education", WALK_DEFAULT),
    ("C3", "church congregation sunday sermon", WALK_DEFAULT),
    ("C4", "city council tax revenue budget hearing", WALK_DEFAULT),
    ("D1", "once people are broken they cannot be fixed", WALK_DEFAULT),
    ("D2", "what the average reader is always on the lookout for", WALK_DEFAULT),
    ("D3", "how a city rebuilds after a disaster", WALK_DEFAULT),
    ("E1", "what courage means in a losing battle", WALK_DEFAULT),
    ("E2", "the boss battle at the end of the game", WALK_DEFAULT),
    ("E3", "the senator fought a losing battle over the tax bill", WALK_DEFAULT),
]


def run_row(conn, run, rid, prompt, knobs):
    t0 = time.time()
    bundle, tele = sampler.ef_evidence(conn, run, prompt, seed=0, **knobs)
    counts, by_ord = _source_counts(conn, run, bundle)
    total = len(bundle.sampled)
    doc_ids, sources = _anchor_info(conn, run, bundle)
    dt = time.time() - t0

    rec = {
        "id": rid, "prompt": prompt, "knobs": knobs, "n": total,
        "mix": counts, "anchor_doc_ids": doc_ids, "anchor_sources": sources,
        "communities": [{"cid": c["cid"], "keywords": c.get("keywords") or [],
                          "hits": c.get("hits")} for c in bundle.communities],
        "elapsed_s": round(dt, 2),
    }

    if rid == "A1":
        ok = _share(counts, total, "wiki") >= 0.70 and "wiki/28410" in doc_ids
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "A2":
        ok = _share(counts, total, "wiki") > 0.50 and "wiki/16278" in doc_ids
        rec["verdict"] = "CHANGED (now passes)" if ok else "KNOWN-FAIL (expected)"
    elif rid == "A3":
        ok = (_share(counts, total, "wiki") > 0.50
              and _kw_hits(bundle.communities, STORM))
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "A4":
        ok = (_share(counts, total, "wiki") > 0.50
              and _kw_hits(bundle.communities, BOAT))
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "A5":
        ok = _share(counts, total, "wiki") > 0.50
        rec["verdict"] = "PASS" if ok else "FAIL"
        rec["note"] = "title clause recorded not asserted (pre-title run)"
    elif rid.startswith("B"):
        quotes_all = _share(counts, total, "quotes")
        walk_ords = [o for o in bundle.sampled if bundle.origin.get(o) == "walk"]
        walk_total = len(walk_ords)
        walk_quotes = sum(1 for o in walk_ords if by_ord.get(o, {}).get("source") == "quotes")
        quotes_walk = walk_quotes / walk_total if walk_total else 0.0
        ring_ords = [o for o in bundle.sampled if bundle.origin.get(o) == "ring"]
        ring_n = len(ring_ords)
        ring_quotes = sum(1 for o in ring_ords if by_ord.get(o, {}).get("source") == "quotes")
        rec["quotes_all"] = quotes_all
        rec["quotes_walk"] = quotes_walk
        rec["ring_n"] = ring_n
        rec["ring_quotes"] = ring_quotes
        ok = quotes_all >= 0.40
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "C1":
        ok = _share(counts, total, "brown") >= 0.20
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "C2":
        ok = _share(counts, total, "brown") >= 0.10 and "brown" in sources
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "C3":
        ok = _share(counts, total, "brown") >= 0.20
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "C4":
        ok = "brown" in sources and _share(counts, total, "brown") >= 0.05
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "D1":
        xprov = _xprov(conn, run, bundle)
        n_sources = len({s for s in counts if s is not None})
        cross = xprov.get("dense", 0) + xprov.get("both", 0)
        rec["xprov"] = xprov
        ok = n_sources >= 2 and cross >= 1
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "D2":
        n_sources = len({s for s in counts if s is not None})
        ok = "brown" in sources and n_sources >= 2
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "D3":
        ok = _share(counts, total, "wiki") > 0.50
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "E1":
        ok = _share(counts, total, "wiki") >= 0.50 and counts.get("quotes", 0) >= 2
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "E2":
        game_chunks = _chunks_by_community_kw(conn, run, bundle, by_ord, GAME)
        war_chunks = _chunks_by_community_kw(conn, run, bundle, by_ord, WAR)
        rec["game_chunks"] = game_chunks
        rec["war_chunks"] = war_chunks
        ok = game_chunks > war_chunks and (war_chunks / total if total else 0) < 0.20
        rec["verdict"] = "PASS" if ok else "FAIL"
    elif rid == "E3":
        game_chunks = _chunks_by_community_kw(conn, run, bundle, by_ord, GAME)
        polit_wiki = _chunks_by_community_kw(conn, run, bundle, by_ord, POLIT)
        rec["game_chunks"] = game_chunks
        rec["polit_chunks"] = polit_wiki
        combined = (counts.get("brown", 0) + polit_wiki) / total if total else 0
        rec["combined_share"] = combined
        ok = combined >= 0.50 and (game_chunks / total if total else 0) < 0.10
        rec["verdict"] = "PASS" if ok else "FAIL"
    else:
        rec["verdict"] = "ERROR"

    return rec, bundle


def _write_pin_digest(pin_dir, rid, conn, run, bundle, embed):
    """T35 opt-in: write ev.digest (or the byte-pin sentinel) to <pin_dir>/<rid>.txt.
    Never touches rec, verdicts, printing, or diag_rerun_last.json -- see
    .playbook/T35.subplan.md sec.4. Same header convention as .tmp/pin_digest.py."""
    Path(pin_dir).mkdir(parents=True, exist_ok=True)
    n_sampled = len(bundle.sampled)
    kept_n = "?"
    try:
        if not bundle.sampled:
            body = "__NO_SAMPLE__\n"
            kept_n = 0
        else:
            ev = evidence.assemble(conn, run, bundle, embed=embed)
            if ev.dendrite is None:
                _, kept = gt.subgraph_embeddings(conn, run, bundle.sampled)
                kept_n = len(kept)
                body = f"__NO_DIGEST__ kept={kept_n}\n"
            else:
                kept_n = len(ev.dendrite["kept"])
                body = ev.digest
    except Exception as e:                              # noqa: BLE001
        body = f"__ERROR__ {e!r}\n"
    header = (f"# rid={rid} run_id={run.run_id} embed={'on' if embed else 'off'} "
              f"n_sampled={n_sampled} kept={kept_n}\n")
    (Path(pin_dir) / f"{rid}.txt").write_text(header + body, encoding="utf-8")


def main():
    argv = sys.argv[1:]
    expand_aliases = "--expand-aliases" in argv
    pin_dir = None
    if "--pin-digest" in argv:
        i = argv.index("--pin-digest")
        if i + 1 < len(argv):
            pin_dir = argv[i + 1]
    positional = [a for a in argv if a not in ("--expand-aliases", "--pin-digest", pin_dir)]
    if len(positional) != 1 or ("--pin-digest" in argv and not pin_dir):
        print("usage: PYTHONPATH=. python src/diag_rerun.py <run-label> "
              "[--expand-aliases] [--pin-digest <dir>]", file=sys.stderr)
        sys.exit(2)
    label = positional[0]
    # T21: applied at call time, not to the frozen ROWS constants themselves --
    # merged into each row's knobs dict just before ef_evidence runs.

    exit_code = 0
    try:
        conn = gt.connect()
        run = gt.get_run(conn, label)
    except Exception as e:
        print(f"ERROR: could not resolve run {label!r}: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"run_id={run.run_id} label={run.label} n_chunks={run.n_chunks}")

    embed = evidence.load_embed(config.MODEL_DIR) if pin_dir else None

    results = []
    fail_ids, known_fail_ids = [], []
    n_pass = 0
    for rid, prompt, knobs in ROWS:
        row_knobs = {**knobs, "expand_aliases": True} if expand_aliases else knobs
        try:
            rec, bundle = run_row(conn, run, rid, prompt, row_knobs)
        except Exception as e:
            print(f"{rid}: ERROR {e!r}")
            results.append({"id": rid, "verdict": "ERROR", "error": repr(e)})
            exit_code = 2
            continue

        if pin_dir:
            _write_pin_digest(pin_dir, rid, conn, run, bundle, embed)

        v = rec["verdict"]
        if v in ("PASS", "CHANGED (now passes)"):
            n_pass += 1
        elif v.startswith("KNOWN-FAIL"):
            known_fail_ids.append(rid)
        elif v == "FAIL":
            fail_ids.append(rid)

        extra = ""
        if rid.startswith("B"):
            extra = (f" quotes_all={rec['quotes_all']:.0%} "
                      f"quotes_walk={rec['quotes_walk']:.0%} "
                      f"ring={rec['ring_quotes']}/{rec['ring_n']}")
        print(f"{rid}: {v} n={rec['n']} mix={rec['mix']}{extra} "
              f"({rec['elapsed_s']}s)")
        results.append(rec)

    b_shares = [r["quotes_all"] for r in results if r["id"].startswith("B")]
    print("---")
    print(f"PASS {n_pass}/20 | FAIL {fail_ids} | KNOWN-FAIL {known_fail_ids}")
    if b_shares:
        print(f"B quotes shares: {[f'{s:.0%}' for s in b_shares]} "
              f"(min {min(b_shares):.0%}, max {max(b_shares):.0%})")
    print(f"run_id={run.run_id}")

    with open(".tmp/diag_rerun_last.json", "w") as f:
        json.dump({"run_id": run.run_id, "label": label, "results": results},
                   f, indent=2, default=str)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
