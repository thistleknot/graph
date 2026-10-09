"""Pins tools/arxiv_titles.py guards AT1-AT3 on hand-built CSVs and Atom replies. No network, no database: the HTTP call is injected.

Run:  pytest tests/test_arxiv_titles.py -v
"""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import arxiv_titles as at

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2504.07891v2</id><title>Speculative Decoding:
      A Survey of   Draft-and-Verify</title></entry>
  <entry><id>http://arxiv.org/abs/hep-th/9901001v1</id><title>An Old Paper</title></entry>
  <entry><id>http://arxiv.org/abs/2602.06036</id><title>   </title></entry>
</feed>"""


def test_tidy_collapses_line_breaks_and_runs_of_spaces():
    assert at.tidy("A\n   B \t C") == "A B C" and at.tidy(None) == ""


def test_parse_atom_reads_ids_without_versions_collapses_titles_and_skips_an_empty_one():
    assert at.parse_atom(ATOM) == {"2504.07891": "Speculative Decoding: A Survey of Draft-and-Verify", "hep-th/9901001": "An Old Paper"}


def test_load_csv_reads_id_title_and_thesis_skips_rows_without_either_and_tolerates_a_file_without_thesis(tmp_path):
    p = tmp_path / "enriched.csv"
    p.write_text('arxiv_id,title,abstract,thesis\n1504.04788,"Compressing Neural Networks\nwith the Hashing Trick",abs,The paper presents HashedNets.\n,No id,abs,t\n2505.1,,abs,t\n2506.2,Has no thesis,abs,\n',
                 encoding="utf-8")
    assert at.load_csv(str(p)) == [("1504.04788", "Compressing Neural Networks with the Hashing Trick", "The paper presents HashedNets.", "enriched.csv"),
                                   ("2506.2", "Has no thesis", None, "enriched.csv")]
    q = tmp_path / "missing.csv"
    q.write_text("arxiv_id,title,abstract\n2310.02025,DeepZero,abs\n", encoding="utf-8")
    assert at.load_csv(str(q)) == [("2310.02025", "DeepZero", None, "missing.csv")]


def reply(ids):
    return '<feed xmlns="http://www.w3.org/2005/Atom">' + "".join("<entry><id>http://arxiv.org/abs/%sv1</id><title>T %s</title></entry>" % (i, i) for i in ids) + "</feed>"


def test_fetch_titles_batches_ids_and_pauses_between_requests_only():
    urls, naps = [], []

    def get(url):
        urls.append(url)
        return reply(url.split("id_list=")[1].split("&")[0].split(","))

    got = at.fetch_titles(["a.1", "a.2", "a.3", "a.4", "a.5"], get=get, batch=2, pause=3.5, sleep=naps.append)
    assert [u.split("id_list=")[1] for u in urls] == ["a.1,a.2&max_results=2", "a.3,a.4&max_results=2", "a.5&max_results=1"]
    assert got == {"a.%d" % i: "T a.%d" % i for i in range(1, 6)} and naps == [3.5, 3.5]                  # two pauses for three requests, none after the last


def test_fetch_titles_retries_a_429_after_20_seconds_times_the_attempt_and_then_succeeds():
    calls, naps = [], []

    def get(url):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 429, "Rate exceeded", {}, None)
        return reply(["a.1"])

    assert at.fetch_titles(["a.1"], get=get, sleep=naps.append) == {"a.1": "T a.1"} and len(calls) == 3 and naps == [20, 40]


def test_fetch_titles_stops_naming_the_batch_when_every_attempt_fails_and_does_not_retry_a_404():
    naps = []

    def always_429(url):
        raise urllib.error.HTTPError(url, 429, "Rate exceeded", {}, None)

    with pytest.raises(RuntimeError, match=r"batch 2-3 failed after 4 attempt\(s\): HTTP 429"):
        at.fetch_titles(["a.1", "a.2", "a.3"], get=lambda u: reply(["a.1", "a.2"]) if "a.1" in u else always_429(u), batch=2, sleep=naps.append)
    assert naps == [3.5, 20, 40, 60]                                                                      # the pause after batch 1, then three backoffs before the fourth try fails

    attempts = []

    def not_found(url):
        attempts.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    with pytest.raises(RuntimeError, match=r"after 1 attempt\(s\): HTTP 404"):
        at.fetch_titles(["a.1"], get=not_found, sleep=naps.append)
    assert len(attempts) == 1
