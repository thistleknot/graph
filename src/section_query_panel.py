"""section_query_panel.py -- the three-question strip at the bottom of the section map graphic: per question, the map with the subgraph lit and the GLOBAL answer beside the SUBGRAPH list.

Spec: operator 2026-10-07 ("where are the query examples we used to have at the bottom of the graphic? We should be showing the global community summary along with the subgraph per
query (we had 3 queries before)"; the earlier graphic is .tmp/three_questions.png, "One corpus map - three questions - the region that answers lights up"). Task: playbook.md T172.
No other governing spec.

    python -u src\\section_query_panel.py [--tag xpa] [--base .tmp/sections_xpa_labelled_community_map.png] [--from-json]

Runs section_graphrag.answer on the three questions (or reads .tmp/sections_<tag>_queries.json with --from-json, no LLM call), draws the strip, and stacks it under the base graphic into
.tmp/sections_<tag>_queries_community_map.png (a NEW name: the viewer may hold the base file open, and the base is never rewritten).

P1  Each question's column carries: the question; the map (every section grey, the communities that answered the GLOBAL view tinted, the retrieved sections as red stars, their graph
    neighbours as orange dots); the GLOBAL summary in full with the communities that produced it (cluster tag, title, size), labelled as an LLM summary; the ANSWER, the final LLM
    call over the question + the global summary + the evidence sections, labelled as such, in full; the SUBGRAPH list (retrieved first, then neighbours with their edge weight, each section the
    LLM was given marked ▸) and the typical section of each top community (◆). Nothing in the GLOBAL summary or the ANSWER is cut; a section title longer than TITLE_CHARS is shortened
    with an ellipsis and its key stays whole.
P2  The strip is as wide as the base graphic and as tall as its longest column needs.
P4  (operator 2026-10-08: "we need to get the paper name if we're going to show 'conclusion' 'introduction'") WHEN the paper's title is known (section_store.paper_titles) every section line of the
    SUBGRAPH, HOP and TYPICAL lists ends with it after the heading, the heading cut to SECTION_WITH_PAPER characters and the title to PAPER_CHARS; a book or a paper without a title shows the heading
    alone, as before.
P3  (operator 2026-10-08: "update the bottom queries based on these hops and show the # of hops and the size of the total subgraph selected for and its mini composition for entities";
    2026-10-09: hops and agent queries are parameters, each query a masked subgraph of 13) WHEN a result carries a ReAct traversal (`traversal`, `nodes`) the column gains a TRAVERSAL block after
    the ANSWER: the hops each query walked, the sections seen, how many each hop holds, the sections read, why it stopped, the communities shown to the agent, one line per query (how many of the
    13 it holds, its pool, the already-shown sections it met and skipped), and the entities that most characterise the subgraph (the sections mentioning each); the SUBGRAPH list gains one group
    per query (the question is query 1) with each section's hop, and the map lights each agent query in its own colour. A result without a traversal renders as before.
P5  (operator 2026-10-09: "all sections in subgraph should be seen") WHEN a result's nodes carry their query, the SUBGRAPH list is every section the traversal saw, one group per query, query 1
    (the question) first, each line its hop and ▸ when the LLM was given it; the count in the heading is the traversal's, not the first retrieval's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

QUESTIONS = ["how does speculative decoding speed up inference",
             "what are the main approaches to long-term memory for LLM agents",
             "how do researchers evaluate the safety of language models"]
TITLE_CHARS, WRAP, LINE_IN, PAD_IN = 46, 92, 0.165, 1.0
SECTION_WITH_PAPER, PAPER_CHARS = 24, 40                       # P4: a section heading and its paper's title share one line
TINTS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
QUERY_COLOURS = {2: "#e6c200", 3: "#7fbf3f", 4: "#3f9fbf"}                         # P3: query 1 is the red stars and orange neighbours; each search of the agent has a colour
STOP_TEXT = {"answered": "the agent chose ANSWER", "done": "every search and round was used"}


def shorten(s: str, n: int = TITLE_CHARS) -> str:
    """P1. Guarantee: s whole when it fits n characters, else cut to n - 1 and ended with an ellipsis."""
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


def column_lines(res: dict, rows: dict, comm: dict, titles: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """P1. Require: res = one section_graphrag.answer result with `partials` ((cid, text) pairs), `global_answer`, `seeds`, `subgraph`, `candidates`; rows = {ord: {doc_id, section_idx,
    section_title, community}}; comm = {cid: {title, size}}. Guarantee: [(style, line)] -- style 'h' heading, 'b' body, 'k' key line -- the column's text: the GLOBAL view whole, then the SUBGRAPH list."""
    def wrapped(text: str) -> list[tuple[str, str]]:
        return [("b", ln) for para in text.split("\n") for ln in (textwrap.wrap(para, WRAP) or [""])]

    out = []
    who = [c for c, _ in res["partials"]]
    out.append(("h", "GLOBAL  - LLM summary of the %d of %d candidate communities that answered" % (len(who), len(res["candidates"]))))
    for c in who:
        out.append(("k", "c%d  %s  (%d sections)" % (c, shorten(comm[c]["title"], 52), comm[c]["size"])))
    out += wrapped(res["global_answer"].strip() or "(no community summary bore on the question)")
    ev = res.get("evidence") or []
    shown = {e["ord"] for e in ev}
    out.append(("h", "ANSWER  - LLM over question + global summary + ▸ ◆ sections"))
    out += wrapped((res.get("final") or "").strip() or "(no answer was generated for this question)")
    tr = res.get("traversal")
    if tr:
        out.append(("h", "TRAVERSAL  - ReAct agent: %d hops asked of each query, %d sections seen, %d read" % (tr["hops"], tr["size"], tr["read"])))
        out.append(("k", "sections per hop  " + " · ".join("h%d %d" % (h, n) for h, n in enumerate(tr["per_hop"]))))
        out.append(("k", "stopped: %s; %d %s run; %d communities shown to it" % (STOP_TEXT.get(tr["stop"], tr["stop"]), tr.get("queries", 1), "query" if tr.get("queries", 1) == 1 else "queries",
                                                                                  res.get("shown_communities", 0))))
        for w in tr.get("walks", []):
            out.append(("k", "query %d  %d of %d shown  pool %d  %d already-shown met" % (w["query"], w["held"], tr.get("per_query", 13), w["pool"], w["masked"])))
        out += wrapped("entities of the subgraph (sections mentioning each): " + (" · ".join("%s %d" % (s, n) for s, n in tr["entities"]) or "none"))
        if tr.get("by_community"):
            out.append(("h", "ENTITIES BY COMMUNITY  (sections of the subgraph mentioning each)"))
            for bc in tr["by_community"]:
                out.append(("k", "c%d  %s  %d of %d sections" % (bc["cid"], shorten(comm.get(bc["cid"], {}).get("title", "(no summary)"), 40), bc["n"], tr["size"])))
                out += [("k", "    " + ln) for ln in textwrap.wrap(" · ".join("%s %d" % (s, n) for s, n in bc["entities"]) or "no entity", WRAP - 6)]
    key = lambda o: "%s#%s" % (rows[o]["doc_id"].replace("arxiv/", ""), rows[o]["section_idx"])
    mark = lambda o: "▸" if o in shown else " "

    def named(o: int, n: int) -> str:
        """P4. The section's heading cut to n characters; WHEN the paper's title is known, the heading is cut to SECTION_WITH_PAPER and the title follows after ' | ' (a bare 'Conclusion' or
        'Introduction' says nothing without its paper)."""
        t = (titles or {}).get(rows[o]["doc_id"])
        return shorten(rows[o]["section_title"], SECTION_WITH_PAPER if t else n) + (" | " + shorten(t, PAPER_CHARS) if t else "")

    nodes = res.get("nodes") or []
    if any("query" in n for n in nodes):                                    # P5: every section of every query, query 1 (the question) included
        queries = sorted({n.get("query", 1) for n in nodes})
        out.append(("h", "SUBGRAPH  - %d sections in %d %s, %d communities  (▸ = given to the LLM)" % (
            len(nodes), len(queries), "query" if len(queries) == 1 else "queries", len({rows[n["ord"]]["community"] for n in nodes}))))
        for q in queries:
            here = [n for n in nodes if n.get("query", 1) == q]
            out.append(("h", 'QUERY %d  "%s"  - %d sections' % (q, shorten(res["question"] if q == 1 else here[0].get("words", ""), 60), len(here))))
            for n in here:
                o = n["ord"]
                out.append(("k", "%s%s h%d %s  c%d  %s" % (mark(o), "★" if n["hop"] == 0 else "·", n["hop"], key(o), rows[o]["community"], named(o, TITLE_CHARS))))
    else:
        out.append(("h", "SUBGRAPH  - %d retrieved, %d neighbours in %d communities  (▸ = given to the LLM)" % (
            len(res["seeds"]), len(res["subgraph"]), len({rows[o]["community"] for o in res["seeds"] + [s["ord"] for s in res["subgraph"]]}))))
        for o in res["seeds"]:
            out.append(("k", "%s★ %s  c%d  %s" % (mark(o), key(o), rows[o]["community"], named(o, TITLE_CHARS))))
        for s in res["subgraph"]:
            out.append(("k", "%s· %s  c%d  w%.1f  %s" % (mark(s["ord"]), key(s["ord"]), rows[s["ord"]]["community"], s["weight"], named(s["ord"], 36))))
    typical = [e for e in ev if e["role"].startswith("typical")]
    if typical:
        out.append(("h", "TYPICAL SECTION of each top community (◆ = given to the LLM)"))
        for e in typical:
            o = e["ord"]
            out.append(("k", "◆ %s  c%d  %s" % (key(o), rows[o]["community"], named(o, TITLE_CHARS))))
    return out


def strip_height_in(columns: list[list[tuple[str, str]]]) -> float:
    """P2. Guarantee: inches the strip needs: the longest column's lines plus the map and padding."""
    return PAD_IN + 4.2 + LINE_IN * max(len(c) for c in columns)


def draw_strip(results: list[dict], rows_all: list[dict], comm_all: list[dict], XY: np.ndarray, lab: np.ndarray, width_in: float, dpi: int = 100, titles: dict[str, str] | None = None):
    """P1, P2, P4. Guarantee: a matplotlib figure `width_in` wide, one column per result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [column_lines(r, rows_all[i], comm_all[i], titles) for i, r in enumerate(results)]
    h = strip_height_in(cols)
    fig = plt.figure(figsize=(width_in, h), dpi=dpi)
    n = len(results)
    w = 0.94 / n
    map_h = 4.0 / h
    for i, r in enumerate(results):
        x0 = 0.03 + i * w
        ax = fig.add_axes([x0, 1 - (0.55 / h) - map_h, w - 0.015, map_h])
        ax.scatter(XY[:, 0], XY[:, 1], s=0.25, c="#dcdcdc", linewidths=0, rasterized=True)
        for t, (c, _) in enumerate(r["partials"]):
            m = lab == c
            ax.scatter(XY[m, 0], XY[m, 1], s=1.5, c=TINTS[t % len(TINTS)], alpha=0.55, linewidths=0, rasterized=True)
        nb = [n["ord"] for n in r["nodes"] if n.get("query", 1) == 1 and n["hop"] >= 1] if any("query" in n for n in r.get("nodes", [])) else [s["ord"] for s in r["subgraph"]]
        ax.scatter(XY[nb, 0], XY[nb, 1], s=34, c="#f28e00", edgecolors="black", linewidths=0.6, zorder=3)
        for q, colour in QUERY_COLOURS.items():                                                # P3: each search of the agent in its own colour
            far = [n["ord"] for n in r.get("nodes", []) if n.get("query", 1) == q]
            if far:
                ax.scatter(XY[far, 0], XY[far, 1], s=34, c=colour, edgecolors="black", linewidths=0.6, zorder=3)
        ax.scatter(XY[r["seeds"], 0], XY[r["seeds"], 1], s=260, marker="*", c="#d62728", edgecolors="black", linewidths=0.8, zorder=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999999")
        fig.text(x0, 1 - 0.18 / h, '"%s"' % r["question"], fontsize=13, fontweight="bold", va="top")
        y = 1 - (0.55 / h) - map_h - 0.12 / h
        for style, line in cols[i]:
            fig.text(x0, y, line, fontsize=10.5 if style == "h" else 9, fontweight="bold" if style == "h" else "normal",
                     family="monospace" if style == "k" else "sans-serif", va="top", color="#8b0000" if style == "h" else "black")
            y -= (LINE_IN * (1.35 if style == "h" else 1.0)) / h
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xpa")
    ap.add_argument("--base", default=None)
    ap.add_argument("--from-json", action="store_true")
    ap.add_argument("--agent-model", default="anthropic/claude-haiku-5.5", help="the ReAct agent's model; the pipeline's own model stays the default for every other call (flash-lite did not traverse, 2026-10-08)")
    ap.add_argument("--hops", type=int, default=None, choices=(1, 2, 3), help="hops each query walks out from its seeds (default section_graphrag.HOPS)")
    ap.add_argument("--queries", type=int, default=None, choices=(1, 2, 3), help="searches the agent may make (default section_graphrag.AGENT_QUERIES)")
    a = ap.parse_args()
    os.chdir(ROOT)
    import section_graphrag as g
    import section_store as st
    import sparsevec_store as ss
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    conn = ss.connect()
    build, _ = st.live_build(conn, a.tag)
    jpath = ".tmp/sections_%s_queries.json" % a.tag
    if a.from_json:
        results = json.load(open(jpath, encoding="utf-8"))
    else:
        sg = g.SectionGraph(conn, a.tag)
        results = []
        for q in QUESTIONS:
            r = g.agent_answer(sg, q, k=3, agent_model=a.agent_model, hops=a.hops or g.HOPS, queries=a.queries or g.AGENT_QUERIES)
            results.append({k: r[k] for k in ("question", "seeds", "subgraph", "candidates", "partials", "global_answer", "final", "evidence", "traversal")}
                           | {"nodes": [{"ord": n["ord"], "key": k, "hop": n["hop"], "query": n.get("query", 1), "words": n.get("query_text", "")} for k, n in r["agent"]["nodes"].items()],
                              "shown_communities": sum(len(b) for b in r["agent"]["shown"]),
                              "trace": r["agent"]["trace"]})
        json.dump(results, open(jpath, "w", encoding="utf-8"))
    need = lambda r: r["seeds"] + [s["ord"] for s in r["subgraph"]] + [e["ord"] for e in r.get("evidence", [])] + [n["ord"] for n in r.get("nodes", [])]
    secs = st.sections(conn, build, sorted({o for r in results for o in need(r)}))
    rows_all = [{o: secs[o] for o in need(r)} for r in results]
    comm_all = []
    for r in results:
        cids = sorted({c for c, _ in r["partials"]} | {bc["cid"] for bc in r.get("traversal", {}).get("by_community", [])})
        comm_all.append({c: {"title": v["title"], "size": v["size"]} for c, v in st.summaries(conn, build, cids).items()})
    xy = np.array(conn.execute("SELECT x, y, community FROM sect_node WHERE build_id = %s ORDER BY ord", (build,)).fetchall(), np.float64)
    base_path = a.base or ".tmp/sections_%s_labelled_community_map.png" % a.tag
    base = Image.open(base_path).convert("RGB")
    titles = st.paper_titles(conn, sorted({s["doc_id"] for s in secs.values()}))                  # P4: the paper's title beside each section heading (section_store.paper_titles, filled by src/arxiv_titles.py)
    print("paper titles for %d of %d papers shown" % (len(titles), len({s["doc_id"] for s in secs.values()})))
    fig = draw_strip(results, rows_all, comm_all, xy[:, :2], xy[:, 2].astype(int), base.width / 100, titles=titles)
    strip_path = ".tmp/sections_%s_queries_strip.png" % a.tag
    fig.savefig(strip_path, dpi=100)
    strip = Image.open(strip_path).convert("RGB")
    out = Image.new("RGB", (base.width, base.height + strip.height), "white")
    out.paste(base, (0, 0))
    out.paste(strip, (0, base.height))
    out_path = ".tmp/sections_%s_queries_community_map.png" % a.tag
    try:
        out.save(out_path)
    except OSError:                                          # a viewer holds the file open (Windows Errno 22): write a timestamped sibling and say so
        import time
        out_path = out_path.replace(".png", ".%s.png" % time.strftime("%Y%m%d-%H%M%S"))
        out.save(out_path)
        print("the plain name is held open by a viewer; wrote the timestamped sibling")
    print("wrote %s  (%dx%d: base %dx%d + strip %dx%d)" % (os.path.abspath(out_path), out.width, out.height, base.width, base.height, strip.width, strip.height))


if __name__ == "__main__":
    main()
