"""arxiv_titles.py -- the title of every paper in the section map, prefilled from the local CSVs, completed from the arXiv API, and stored once in `paper_title`.

Spec: operator 2026-10-08 ("we need to get the paper name if we're going to show 'conclusion' 'introduction'... pull all those arxiv id titles ... through the api beforehand"; "I believe we have
a csv that already has most of these titles, we just need to reconnect to that and have these prefilled so we don't have to do it fresh every run"). Task: playbook.md T175. No other governing spec.

    python -u tools\\arxiv_titles.py [--tag xpa] [--csv-dir C:/Users/user/arxiv_id_lists]

AT1  A paper whose title `paper_title` already holds is never looked up again. For the rest the CSVs are read in CSV_FILES order and the first title wins (section_store.save_titles): measured
     2026-10-08, they hold 1,948 of the 2,217 arXiv papers of build 32 (87.9%), the enriched file alone 1,957 rows with an abstract and a `thesis`.
AT2  Only the papers no CSV holds go to the arXiv API, BATCH ids a request, one request every PAUSE seconds (the API's terms are one request in 3 seconds; the first call of 2026-10-08 was
     answered 429 "Rate exceeded"). A 429 or 5xx is retried after 20 s, 40 s, ... up to TRIES attempts; a batch that still fails stops the run with that batch named, never a silent skip.
AT3  A title is the API's title with its line breaks and runs of spaces collapsed; a paper the API does not return keeps no row and is shown by its key alone, as a book is.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import section_store as st

CSV_DIR = "C:/Users/user/arxiv_id_lists"
CSV_FILES = ("_arxiv_data_with_enriched_bm25.csv", "_missing_papers.csv", "_reextract_targets.csv")
API = "https://export.arxiv.org/api/query"
BATCH, PAUSE, TRIES = 100, 3.5, 4
_ATOM = "{http://www.w3.org/2005/Atom}"


def tidy(text: str) -> str:
    """AT3. Guarantee: `text` with every run of whitespace, line breaks included, as one space."""
    return " ".join((text or "").split())


def load_csv(path: str) -> list[tuple[str, str, str | None, str]]:
    """AT1. Guarantee: (arxiv_id, title, thesis or None, file name) for every row of the CSV with an id and a title, in file order. The CSVs carry `arxiv_id` and `title`; `thesis` is read when the file has it."""
    csv.field_size_limit(10 ** 9)
    out = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            aid, title = (row.get("arxiv_id") or "").strip(), tidy(row.get("title") or "")
            if aid and title:
                out.append((aid, title, tidy(row.get("thesis") or "") or None, os.path.basename(path)))
    return out


def parse_atom(xml_text: str) -> dict[str, str]:
    """AT3. Guarantee: {arxiv_id: title} for every entry of an arXiv API Atom reply, the id without its version ('http://arxiv.org/abs/2504.07891v2' -> '2504.07891')."""
    out = {}
    for entry in ET.fromstring(xml_text).iter(_ATOM + "entry"):
        m = re.search(r"/abs/(.+?)(?:v\d+)?$", (entry.findtext(_ATOM + "id") or "").strip())
        title = tidy(entry.findtext(_ATOM + "title") or "")
        if m and title:
            out[m.group(1)] = title
    return out


def http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "section-map-titles/1.0 (research use)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def fetch_titles(ids: list[str], get=http_get, batch: int = BATCH, pause: float = PAUSE, tries: int = TRIES, sleep=time.sleep) -> dict[str, str]:
    """AT2. Guarantee: {arxiv_id: title} the API returned for `ids`, BATCH ids a request with `pause` seconds between requests; a 429 or 5xx is retried after 20 s x the attempt number, up to
    `tries` attempts, and a batch that still fails raises RuntimeError naming it. `get(url) -> reply text` is injected so the retry rule is testable without a network."""
    out: dict[str, str] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        url = "%s?id_list=%s&max_results=%d" % (API, ",".join(chunk), len(chunk))
        for attempt in range(1, tries + 1):
            try:
                out.update(parse_atom(get(url)))
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == tries:
                    raise RuntimeError("arXiv API batch %d-%d failed after %d attempt(s): HTTP %s" % (i, i + len(chunk), attempt, e.code)) from e
                sleep(20 * attempt)
        if i + batch < len(ids):
            sleep(pause)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xpa")
    ap.add_argument("--csv-dir", default=CSV_DIR)
    a = ap.parse_args()
    import sparsevec_store as ss
    conn = ss.connect()
    st.ensure_schema(conn)
    build, _ = st.live_build(conn, a.tag)
    docs = [r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM sect_node WHERE build_id = %s", (build,)).fetchall()]
    ids = sorted({i for i in map(st.arxiv_id_of, docs) if i})
    held = {r[0] for r in conn.execute("SELECT arxiv_id FROM paper_title").fetchall()}
    todo = [i for i in ids if i not in held]
    print("build %d: %d documents, %d arXiv papers, %d already hold a title, %d to find" % (build, len(docs), len(ids), len(ids) - len(todo), len(todo)), flush=True)
    wanted, from_csv = set(todo), 0
    for name in CSV_FILES:
        path = os.path.join(a.csv_dir, name)
        new = st.save_titles(conn, [r for r in load_csv(path) if r[0] in wanted])
        from_csv += new
        wanted -= {r[0] for r in conn.execute("SELECT arxiv_id FROM paper_title WHERE arxiv_id = ANY(%s)", (list(wanted),)).fetchall()}
        print("  %-40s %d new titles, %d papers still without" % (name, new, len(wanted)), flush=True)
    api = fetch_titles(sorted(wanted)) if wanted else {}
    from_api = st.save_titles(conn, [(i, t, None, "arxiv api") for i, t in api.items()])
    left = sorted(wanted - set(api))
    print("titles: %d from the CSVs, %d from the arXiv API; %d papers without one%s" % (from_csv, from_api, len(left), (": " + ", ".join(left[:8])) if left else ""))
    print("paper_title now holds %d rows" % conn.execute("SELECT count(*) FROM paper_title").fetchone()[0])


if __name__ == "__main__":
    main()
