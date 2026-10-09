"""Pins domain_corpora.py's D2-D5 guards, with conservation (D4) as the load-bearing one.

The packer is the only stage that can silently lose or duplicate text, and a
duplicated paragraph inflates every term's df in it -- which is the exact
quantity the term-analysis sidecar measures. So conservation is asserted as an
identity over the joined text, not as a length check.

No PDF, no database and no network: the PDF path is exercised in the module's own
diagnostic against the live corpus, because a fixture PDF would prove nothing
about real producer variation (D2).

Run:  pytest tests/test_domain_corpora.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain_corpora import (derive_pack_target, md_sections, md_units, pack,
                            units_for)


def test_pack_conserves_every_unit_exactly_once():
    """D4: concatenation round-trips. This is the guard that protects df."""
    units = ["a" * 100, "b" * 100, "c" * 100, "d" * 50]
    chunks = pack(units, target=250)
    assert "\n\n".join(chunks) == "\n\n".join(units)


def test_pack_never_splits_a_unit():
    """D4: a unit larger than the target still emerges whole."""
    big = "x" * 9000
    chunks = pack(["short", big, "tail"], target=1000)
    assert big in chunks[1]
    assert chunks[1] == big
    assert "\n\n".join(chunks) == "\n\n".join(["short", big, "tail"])


def test_pack_fills_to_target_rather_than_one_unit_per_chunk():
    """Three 100-char units under a 250 target pack 2 then 1, not 1/1/1."""
    chunks = pack(["a" * 100, "b" * 100, "c" * 100], target=250)
    assert len(chunks) == 2
    assert len(chunks[0]) == 202          # 100 + 2 (separator) + 100
    assert len(chunks[1]) == 100


def test_pack_empty_input_gives_no_chunks():
    assert pack([], target=500) == []


def test_md_units_drops_headings_and_image_placeholders():
    text = "## A Heading\n\nbody one\n\n<image 3>\n\n## Another\n\nbody two"
    assert md_units(text) == ["body one", "body two"]


def test_md_units_joins_wrapped_lines_into_one_unit():
    assert md_units("first line\nsecond line\n\nnext") == [
        "first line second line", "next"]


def test_md_sections_excludes_zero_paragraph_sections():
    """23 of Lewy's 53 headings are back-to-back; the skill excludes them."""
    text = "## One\n\n## Two\n\npara a\n\npara b\n\n## Three\n\npara c"
    secs = md_sections(text)
    assert secs == [["para a", "para b"], ["para c"]]


def test_derive_pack_target_is_the_median_chars_per_section():
    """D3: the target is fitted, and on a symmetric input it is the median."""
    doc = "".join("## S%d\n\n%s\n\n" % (i, "w" * n)
                  for i, n in enumerate([100, 200, 300, 400, 500,
                                         600, 700, 800, 900]))
    got = derive_pack_target([doc])
    assert got["n_sections"] == 9
    assert got["p50"] == 500.0
    assert got["target"] == pytest.approx(500.0, rel=0.05)


def test_derive_pack_target_raises_without_sections():
    """A constant fallback here would silently license a made-up ruler (D3)."""
    with pytest.raises(ValueError, match="no ATX sections"):
        derive_pack_target(["just prose\n\nmore prose"])


def test_units_for_pdf_text_splits_on_blank_lines_not_markdown():
    """load_neop pre-joins PDF blocks with a blank line; units_for must recover them."""
    joined = "block one\n\nblock two\n\nblock three"
    assert units_for(joined, "neop_article") == [
        "block one", "block two", "block three"]


def test_units_for_arxiv_uses_the_markdown_path():
    text = "## Title\n\nabstract text\n\n## Intro\n\nintro text"
    assert units_for(text, "arxiv") == ["abstract text", "intro text"]


def test_heading_sections_keeps_headings_aligned_with_their_own_paragraphs():
    """The gold-set bug: pairing two independently filtered lists misaligned every row.

    An earlier version zipped a length-filtered heading list against the section
    list, which drops zero-paragraph sections. Every heading was then matched to
    the WRONG section and the retrieval gold scored hit@1 = 0.000 -- reported at
    the time as a defect in my eval, not as sparse retrieval failing on books.
    This pins the pairing, which is the only thing that made it meaningless.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from diag_domain_recall import heading_sections

    text = ("## Empty One\n\n"            # no paragraphs -> dropped entirely
            "## Real Two\n\npara for two\n\n"
            "## Empty Three\n\n"          # also dropped
            "## Real Four\n\npara for four\n\nsecond para for four")
    got = heading_sections(text)
    assert [h for h, _ in got] == ["Real Two", "Real Four"]
    assert got[0][1] == ["para for two"]
    assert got[1][1] == ["para for four", "second para for four"]


# ---- arxiv section chunker (C1-C8) ----------------------------------------------
def _doc(n_para: int, tag: str) -> str:
    return "\n\n".join("%s sentence number %d about a topic. " % (tag, i) * 3 for i in range(n_para))


def _corpus():
    """Sections of 1..4 paragraphs plus ONE section far over the cap."""
    secs = ["## Small %d\n\n%s" % (i, _doc(1 + i % 4, "s%d" % i)) for i in range(40)]
    secs.insert(20, "## Giant\n\n" + _doc(400, "giant"))
    return "\n\n".join(secs)


def test_clean_md_drops_base64_image_lines_only():
    from domain_corpora import clean_md
    t = "## A\n\n![Image](data:image/png;base64,AAAA)\n\nreal text\n<image 1>"
    assert clean_md(t) == "## A\n\n\nreal text"


def test_fit_is_log_median_plus_2_robust_sigma_with_zeros_dropped():
    from domain_corpora import fit_arxiv
    fit = fit_arxiv([_corpus()])
    assert fit["hi"] > fit["m"] >= 1
    assert fit["n_zero"] == 10          # the 10 one-paragraph bodies have no newline: size 0, excluded
    assert 0 < fit["hi_frac"] < 0.1     # only the Giant exceeds the cap


def test_every_chunk_starts_with_a_header_and_text_is_conserved():
    import re
    from domain_corpora import chunk_arxiv_doc, fit_arxiv, header_sections
    text = _corpus()
    fit = fit_arxiv([text])
    chunks, st = chunk_arxiv_doc("d", text, fit)
    chunks = [c[0] for c in chunks]
    assert st["split"] >= 1
    assert all(re.match(r"#{1,6}\s", c) for c in chunks)
    nonws = lambda s: re.sub(r"\s+", "", s)
    expect = nonws("".join(h + b for h, b in header_sections(text, "d")))
    got = nonws("".join(chunks))
    # a continuation chunk re-states its header, so output is expect plus only those repeats
    extra = len(got) - len(expect)
    assert extra >= 0 and extra % len(nonws("## Giant")) == 0
    # no body text is duplicated: every giant sentence appears exactly 3x per paragraph source
    assert got.count(nonws("giant sentence number 399 about a topic.")) == 3


def test_splitter_tries_a_header_boundary_before_the_library_defaults():
    from domain_corpora import make_splitter
    sp = make_splitter({"hi": 50, "m": 5, "lcap": 100})
    assert sp._separators == ["\n#", "\n\n", "\n", " ", ""]
    assert (sp._chunk_size, sp._chunk_overlap) == (50, 5)


def test_tile_removes_the_real_overlap_exactly_once():
    from domain_corpora import make_size, tile_pieces
    body = "\n".join("line %d" % i for i in range(60))
    a = "\n".join("line %d" % i for i in range(0, 30))
    b = "\n".join("line %d" % i for i in range(25, 60))
    out, k = tile_pieces(body, [a, b], 10, make_size(100))
    assert k == len("\n".join("line %d" % i for i in range(25, 30)))
    assert "".join(out).replace("\n", "") == body.replace("\n", "")


def test_tile_does_not_trim_two_pieces_that_do_not_overlap():
    """Regression: 1403_7426 References. Both pieces touch on a '|'; they share no span."""
    from domain_corpora import make_size, tile_pieces
    body = "| a |\n| b |\n| c |\n| d |"
    out, k = tile_pieces(body, ["| a |\n| b |", "| c |\n| d |"], 10, make_size(100))
    assert k == 0 and out == ["| a |\n| b |", "| c |\n| d |"]


def test_tile_rejects_an_earlier_identical_block_that_would_imply_too_much_overlap():
    """Regression: 2311_04257 figure garbage. The same short block occurs inside the
    previous piece; the earlier hit implies 3 newlines of overlap, more than m=1 allows."""
    from domain_corpora import make_size, tile_pieces
    body = "a\nb\nc\nd\n" * 3 + "z"
    p0 = ("a\nb\nc\nd\n" * 2).strip()
    p1 = "a\nb\nc\nd"                       # occurs at lines 0, 4 (inside p0) and 8 (the real one)
    out, k = tile_pieces(body, [p0, p1], 1, make_size(100))
    assert k == 0 and out == [p0, p1]


def test_tile_finds_the_true_overlap_when_the_text_repeats():
    """Regression: 2212_01762 repeated-row table. Position, not text, decides."""
    from domain_corpora import make_size, tile_pieces
    row = "RAFT-it\n\nRAFT-S-AF\n\nModel\n\n"
    body = "head\n\n" + row * 6 + "x1\n\nx2\n\nTAIL A\n\nTAIL B\n\nnew1\n\n" + row * 6 + "new2"
    cut1 = body.index("TAIL B") + len("TAIL B")
    cut2 = body.index("TAIL A")
    out, k = tile_pieces(body, [body[:cut1], body[cut2:]], 10, make_size(200))
    assert k == cut1 - cut2
    assert out[0] + out[1] == body[:cut1] + body[cut1:]


def test_degenerate_repetition_falls_back_to_no_overlap_and_still_conserves():
    """Regression: 2408_03402 and Pragmatic. ', of-' repeated: every shift is the same text."""
    import re
    from domain_corpora import chunk_arxiv_doc, fit_arxiv
    junk = "## Junk\n\n" + ", of-\n" * 4000
    text = _corpus() + "\n\n" + junk
    fit = fit_arxiv([text])
    chunks, st = chunk_arxiv_doc("d", text, fit)       # must not raise
    assert st["split"] >= 2
    assert all(re.match(r"#{1,6}\s", c[0]) for c in chunks)
    nonws = lambda s: re.sub(r"\s+", "", s)
    assert nonws("".join(c[0] for c in chunks if "of-" in c[0])).count("of-") >= 4000


def test_business_key_is_unique_and_names_the_first_unit():
    from domain_corpora import chunk_arxiv
    recs, fit, st = chunk_arxiv(["a"], [_corpus()])
    keys = [(r["doc_id"], r["section_idx"], r["chunk_idx"]) for r in recs]
    assert len(set(keys)) == len(keys) == len(recs)
    assert all(k[0] == "a" for k in keys)
    assert any(ci > 0 for _, _, ci in keys)        # the Giant section was split


def test_is_reference_matches_titles_that_are_a_references_phrase_only():
    from domain_corpora import header_title, is_reference
    yes = ["## References", "## 9. REFERENCES", "# Bibliography", "## Works Cited",
           "## Literature cited", "## Citations", "## Appendix A. References", "## References and Notes"]
    no = ["## Citation Intent Classification", "## Related Work", "## Reference Frames in Vision",
          "## 3.2 Cross-references between sections", "## Methods"]
    assert all(is_reference(h) for h in yes), [h for h in yes if not is_reference(h)]
    assert not any(is_reference(h) for h in no), [h for h in no if is_reference(h)]
    assert header_title("## 3.2.2 Fine-tuning Strategies") == "3.2.2 Fine-tuning Strategies"


def test_references_carry_title_and_flag_and_never_merge_with_the_section_before():
    from domain_corpora import chunk_arxiv
    doc = "## Conclusion\n\nWe conclude.\n\n## References\n\n[1] A. Author. A paper. 2020."
    recs, fit, st = chunk_arxiv(["a"], [doc + "\n\n" + _corpus()])
    ref = [r for r in recs if r["is_reference"]]
    assert ref and ref[0]["section_title"] == "References"
    assert "[1] A. Author" in ref[0]["text"] and "We conclude" not in ref[0]["text"]
    concl = next(r for r in recs if r["section_title"] == "Conclusion")
    assert concl["is_reference"] is False and "[1] A. Author" not in concl["text"]


def test_headerless_text_gets_a_synthetic_doc_header():
    from domain_corpora import header_sections
    assert header_sections("just prose\n\nmore prose", "arxiv/1234")[0][0] == "# arxiv/1234"
    assert header_sections("\n\n## Real\n\nbody", "x")[0][0] == "## Real"


def test_identical_chunks_across_documents_are_kept_once():
    from domain_corpora import chunk_arxiv
    d = "## Same\n\n" + _doc(3, "dup")
    recs, fit, st = chunk_arxiv(["a", "b"], [d, d])
    assert st["duplicates"] == len(recs) and {r["doc_id"] for r in recs} == {"a"}


def test_load_arxiv_skips_stored_documents_without_reading_them_and_ignores_backups_and_empties(tmp_path):
    from domain_corpora import load_arxiv
    for stem, body in (("2601_0001", "## A\n\nfirst"), ("2601_0002", "## B\n\nsecond"), ("2601_0003", "   \n"), ("2601_0004", "## D\n\nfourth")):
        (tmp_path / (stem + ".md")).write_text(body, encoding="utf-8")
    (tmp_path / "2601_0001.md.fbak").write_text("backup", encoding="utf-8")
    ids, docs, srcs = load_arxiv(root=str(tmp_path))
    assert ids == ["arxiv/2601_0001", "arxiv/2601_0002", "arxiv/2601_0004"]                       # the empty file is dropped
    ids, docs, srcs = load_arxiv(root=str(tmp_path), skip={"arxiv/2601_0001", "arxiv/2601_0004"})
    assert ids == ["arxiv/2601_0002"] and docs == ["## B\n\nsecond"] and srcs == ["arxiv"]
    assert load_arxiv(root=str(tmp_path), skip={"arxiv/2601_0001", "arxiv/2601_0002", "arxiv/2601_0004"})[0] == []


def test_the_paper_key_is_the_arxiv_id_without_its_version_and_the_version_defaults_to_one():
    from domain_corpora import arxiv_stem_key as k
    assert k("2404_08634") == ("2404_08634", 1)                       # no version: v1
    assert k("2403_19889v1") == ("2403_19889", 1)
    assert k("2404_08634v3") == ("2404_08634", 3)
    assert k("2404_08634v12") == ("2404_08634", 12)
    assert k("1301_3781") == ("1301_3781", 1)                          # the older four-digit style
    assert k("hep-th_9901001v2") == ("hep-th_9901001", 2)              # pre-2007 ids
    assert k("2606_17276_methods") == ("2606_17276_methods", 1)        # a methods extract is a part, not a version
    assert k("2606_17276v2_methods") == ("2606_17276_methods", 2)
    assert k("Machine-Learning-Systems") == ("Machine-Learning-Systems", 1)    # a book is its own key
    assert k("ISLP") == ("ISLP", 1) and k("draft_v2") == ("draft_v2", 1)       # a trailing v2 on a non-arXiv name is not a version


def _touch(root, **files):
    for name, body in files.items():
        (root / (name + ".md")).write_text(body, encoding="utf-8")


def test_arxiv_files_keeps_the_highest_version_of_each_paper_and_a_methods_extract_separately(tmp_path):
    from domain_corpora import arxiv_files
    _touch(tmp_path, **{"2601_0001": "a", "2601_0001v2": "b", "2601_0001v3": "c", "2601_0002v1": "d",
                        "2601_0002_methods": "e", "2601_0003": "f", "2601_0003v2": "g"})
    (tmp_path / "2601_0009v7.md.fbak").write_text("backup", encoding="utf-8")
    got = [(k, v, Path(p).name) for k, v, p in arxiv_files(str(tmp_path))]
    assert got == [("2601_0001", 3, "2601_0001v3.md"), ("2601_0002", 1, "2601_0002v1.md"),
                   ("2601_0002_methods", 1, "2601_0002_methods.md"), ("2601_0003", 2, "2601_0003v2.md")]


def test_a_stored_paper_is_replaced_only_by_a_higher_version_and_skipped_files_are_not_read(tmp_path, monkeypatch):
    import domain_corpora as dc
    _touch(tmp_path, **{"2601_0001v2": "two", "2601_0002": "one", "2601_0003v3": "three", "2601_0004": "four",
                        "2601_0005v1": "   \n"})
    reads = []
    real = dc._read
    monkeypatch.setattr(dc, "_read", lambda p: (reads.append(Path(p).name), real(p))[1])
    stored = {"arxiv/2601_0001": 1, "arxiv/2601_0002": 1, "arxiv/2601_0003": 5}            # held v1, v1, v5
    got = dc.load_arxiv_papers(root=str(tmp_path), stored=stored)
    assert [(p["doc_id"], p["version"], p["upgrade"], p["text"]) for p in got] == [
        ("arxiv/2601_0001", 2, True, "two"),                          # v2 over held v1: replaces
        ("arxiv/2601_0004", 1, False, "four")]                        # absent: new
    assert "2601_0002.md" not in reads and "2601_0003v3.md" not in reads     # same version / older: never opened
    assert not [p for p in got if p["doc_id"] == "arxiv/2601_0005"]           # an empty file is dropped
    assert [p["doc_id"] for p in dc.load_arxiv_papers(root=str(tmp_path), stored={})] == [
        "arxiv/2601_0001", "arxiv/2601_0002", "arxiv/2601_0003", "arxiv/2601_0004"]


def test_load_arxiv_returns_version_stripped_ids_and_one_text_per_paper(tmp_path):
    from domain_corpora import load_arxiv
    _touch(tmp_path, **{"2601_0001": "old", "2601_0001v2": "new", "2601_0002_methods": "m"})
    ids, docs, srcs = load_arxiv(root=str(tmp_path))
    assert ids == ["arxiv/2601_0001", "arxiv/2601_0002_methods"] and docs == ["new", "m"] and srcs == ["arxiv", "arxiv"]


def test_reaggregate_drops_the_prepended_header_by_chunk_idx_not_by_comparing_text():
    from domain_corpora import reaggregate_section as r
    assert r([(0, "## Results\n\nfirst"), (1, "## Results\n\nsecond"), (2, "## Results\n\nthird")]) == "## Results\n\nfirst\n\nsecond\n\nthird"
    # a continuation whose header differs from chunk 0's (chunk 0 began an earlier merged section) is still stripped
    assert r([(0, "## Intro\n\nshort\n\n## Results\n\nfirst"), (1, "## Results\n\nsecond")]) == "## Intro\n\nshort\n\n## Results\n\nfirst\n\nsecond"
    # chunk_idx 0 keeps its first line even when a later chunk repeats it: that line is content, not a prepended header
    assert r([(0, "## A\n\nx"), (0, "## A\n\ny")]) == "## A\n\nx\n\n## A\n\ny"
    assert r([(0, "## Only\n\nbody")]) == "## Only\n\nbody" and r([]) == ""
    assert r([(3, "## H\n\nbody")]) == "body" and r([(1, "## H")]) == ""            # a header-only continuation leaves nothing


def _nonws(s: str) -> str:
    import re
    return re.sub(r"\s+", "", s)


def _reassemble(recs, doc_id):
    from domain_corpora import reaggregate_section
    by_section = {}
    for r in recs:
        if r["doc_id"] == doc_id:
            by_section.setdefault(r["section_idx"], []).append((r["chunk_idx"], r["text"]))
    return "\n\n".join(reaggregate_section(sorted(by_section[s])) for s in sorted(by_section))


def test_a_documents_sections_reassemble_to_its_markdown_exactly_without_trimming_any_overlap():
    """The chunker de-overlaps by position (C5), so the stored chunks are disjoint and joining them is exact. Three
    varied documents, including one whose source repeats an identical line thousands of times -- the very text a
    reduce_overlaps pass would delete."""
    from domain_corpora import chunk_arxiv, clean_md
    live_fit = {"lcap": 1992, "m": 10, "hi": 151}
    long_doc = "## Intro\n\n" + _doc(6, "intro") + "\n\n## Long results\n\n" + "\n\n".join("paragraph %d about graphs and edges and nodes" % i for i in range(500))
    repeat_doc = "## Marks\n\n" + "\n\n".join("<!-- formula-not-decoded -->\n\nthe distinct sentence number %d follows" % i for i in range(300)) + "\n\n## After\n\n" + _doc(5, "after")
    table_doc = "## Table\n\n" + "\n".join("| row %d | %s | %d |" % (i, "x" * 30, i) for i in range(600)) + "\n\n## Tail\n\n" + _doc(4, "tail")
    for name, doc in (("long", long_doc), ("repeat", repeat_doc), ("table", table_doc)):
        recs, fit, st = chunk_arxiv(["arxiv/%s" % name], [doc], fit=live_fit)
        assert max(r["chunk_idx"] for r in recs) >= 1 or name != "long"          # the long document really was split
        assert _nonws(_reassemble(recs, "arxiv/%s" % name)) == _nonws(clean_md(doc)), name


def test_a_section_made_only_of_identical_pieces_loses_the_repeats_to_the_keep_identical_chunks_once_rule():
    """Known limit, pinned: C8 keeps identical chunk texts once, so a source that is the SAME line thousands of times
    splits into identical chunks and all but one are dropped when chunking. The re-aggregation is not the cause
    and cannot restore them."""
    from domain_corpora import chunk_arxiv, clean_md
    doc = "## Marks\n\n" + "\n\n".join(["<!-- formula-not-decoded -->"] * 600) + "\n\n## After\n\n" + _doc(5, "after")
    recs, fit, st = chunk_arxiv(["arxiv/marks"], [doc], fit={"lcap": 1992, "m": 10, "hi": 151})
    assert st["duplicates"] > 0
    assert 0 < len(_nonws(_reassemble(recs, "arxiv/marks"))) < len(_nonws(clean_md(doc)))
    assert _nonws(_reassemble(recs, "arxiv/marks")).count("<!--formula-not-decoded-->") >= 1


def test_header_less_text_reassembles_to_itself_plus_the_synthetic_doc_header():
    from domain_corpora import chunk_arxiv, clean_md
    doc = _doc(8, "plain")                                              # no markdown header at all
    recs, fit, st = chunk_arxiv(["arxiv/plain"], [doc], fit={"lcap": 1992, "m": 10, "hi": 151})
    assert _nonws(_reassemble(recs, "arxiv/plain")) == _nonws("# arxiv/plain" + clean_md(doc))      # C7: `# <doc_id>`


def _soup(n: int) -> str:
    return "## Phase\n\n" + "\n\n".join("xyzwvutsrq"[i % 10] for i in range(n))


def test_is_junk_flags_the_image_ocr_marker_empty_chunks_and_letter_soup():
    from domain_corpora import is_junk
    assert is_junk("## 3.2. Convergence\n\n\n\n&lt;latexi sh\n\n1\\_b\n\n64=\"DEUQ\n\n5Kk9")        # c148's marker form
    assert is_junk("#                                      ")                                       # a header with no content
    assert is_junk(_soup(40))                                                                        # letter per line
    assert is_junk("## Plot\n\n" + "\n\n".join(str(6 * i) for i in range(12)))                      # axis tick labels


def test_is_junk_leaves_prose_tables_pseudocode_and_proofs_alone():
    from domain_corpora import is_junk
    prose = "## 4 Compressor\n\nWe formulate prompt compression as a binary token classification problem."
    table = "## Results\n\n" + "\n".join("| model %d | %s | 0.%d |" % (i, " " * 200, i) for i in range(30))
    fence = "## Code\n\n```\n" + "\n".join("    " for _ in range(40)) + "\n```\n\nReturns the sorted list."
    pseudo = "## Algorithm 1\n\nInput: x\n" + "\n".join("for i in range(n): update(theta)" for _ in range(9)) + "\nx\ny\nz"
    proof = "## B.2 Proof\n\nLet x be a random variable.\n\nx\n\n=\n\ny\n\nThen the claim follows from Lemma 3."
    for t in (prose, table, fence, pseudo, proof):
        assert not is_junk(t), t[:60]


def test_is_junk_thresholds_are_inclusive_at_the_stated_cut():
    from domain_corpora import JUNK_LONE_SHARE, JUNK_MIN_LINES, is_junk
    long_line = "a full sentence of ordinary words that is not short"
    at_min_lines = "\n".join(["x"] * JUNK_MIN_LINES)                          # share 1.0, exactly the minimum line count
    one_short_of_lines = "\n".join(["x"] * (JUNK_MIN_LINES - 1))
    below_share = "\n".join(["x"] * 17 + [long_line] * 3)                     # 0.85 of 20 lettered lines
    on_share = "\n".join(["x"] * 18 + [long_line] * 2)                        # exactly 0.90 of 20
    assert JUNK_LONE_SHARE == 0.9 and JUNK_MIN_LINES == 8
    assert is_junk(at_min_lines) and is_junk(on_share)
    assert not is_junk(one_short_of_lines) and not is_junk(below_share)


def test_chunk_arxiv_sets_the_flag_and_retrievable_drops_references_and_junk_but_not_old_records():
    from domain_corpora import chunk_arxiv, retrievable
    doc = _soup(400) + "\n\n## Real\n\n" + _doc(3, "real") + "\n\n## References\n\n[1] A. Author. A paper. 2020."
    live_fit = {"lcap": 1992, "m": 10, "hi": 151}                  # the arxiv fit; the toy corpus fits hi=13, too small to hold 8 lines
    recs, fit, st = chunk_arxiv(["a"], [doc + "\n\n" + _corpus()], fit=live_fit)
    assert any(r["is_junk"] for r in recs) and any(not r["is_junk"] for r in recs)
    assert all(r["is_junk"] for r in recs if r["section_title"] == "Phase")
    kept = retrievable(recs)
    assert kept and not any(r["is_reference"] or r["is_junk"] for r in kept)
    old = [{k: v for k, v in r.items() if k != "is_junk"} for r in recs]     # a cache written before the flag existed
    assert [r["text"] for r in retrievable(old)] == [r["text"] for r in kept]


def test_the_live_cache_has_the_259_junk_chunks_the_measurement_found():
    """258 by marker or line profile (measured), plus the one chunk with no letter or digit."""
    import pickle
    cache = Path(__file__).resolve().parents[1] / ".tmp" / "arxiv_section_chunks.pkl"
    if not cache.exists():
        pytest.skip("no live chunk cache")
    from domain_corpora import is_junk
    recs = [r for r in pickle.load(open(cache, "rb"))["records"] if not r["is_reference"]]
    assert sum(is_junk(r["text"]) for r in recs) == 259


# ---------------------------------------------------------------------------------------------------------- C13 (section nodes)
def _section_doc():
    bodies = ["word %d " % i * (10 + i) for i in range(12)]              # twelve ordinary sections of 70 to about 170 characters
    md = "Preamble before any header.\n\n" + "\n\n".join("## Section %d\n\n%s" % (i, b) for i, b in enumerate(bodies))
    md += "\n\n## Empty one\n\n## Short\n\nhi\n\n## References\n\n[1] a cited work with enough characters to pass any length floor\n"
    md += "\n## Debris\n\n" + "\n".join("a" for _ in range(40))
    md += "\n\n## Fonts\n\n" + "glyph&lt;c=18,font=/FZXHHB+TeXGyrePagellaX-Bold&gt;" * 4 + " and a few real words here"
    md += "\n\n## Mostly prose\n\n" + "A real paragraph of ordinary words about graphs. " * 4 + "glyph&lt;c=3,font=/AAAAAB&gt;"
    return md


def test_section_records_keeps_every_header_block_in_order_with_header_and_body_C13():
    from domain_corpora import section_records
    recs = section_records(["arxiv/d"], [_section_doc()])
    titles = [r["section_title"] for r in recs]
    assert titles[0] == "arxiv/d" and titles[1:4] == ["Section 0", "Section 1", "Section 2"]      # the preamble gets the synthetic header
    assert [r["section_idx"] for r in recs] == list(range(len(recs))) and all(r["chunk_idx"] == 0 for r in recs)
    empty = next(r for r in recs if r["section_title"] == "Empty one")
    assert empty["body_chars"] == 0 and empty["text"] == "## Empty one"
    s3 = next(r for r in recs if r["section_title"] == "Section 3")
    assert s3["text"] == "## Section 3\n\n" + "word 3 " * 13 and s3["body_chars"] == len("word 3 " * 13)
    assert next(r for r in recs if r["section_title"] == "References")["is_reference"] is True


def test_usable_sections_drops_empty_reference_short_and_junk_in_that_order_and_counts_each_C13():
    import numpy as np
    from scipy.stats import median_abs_deviation
    from domain_corpora import section_records, usable_sections
    recs = section_records(["arxiv/d"], [_section_doc()])
    kept, st = usable_sections(recs)
    assert st["all"] == len(recs) and st["empty"] == 1 and st["reference"] == 1 and st["junk"] == 1 and st["glyph"] == 1
    assert st["short"] >= 1 and "Short" not in [r["section_title"] for r in kept] and "Debris" not in [r["section_title"] for r in kept]
    titles = [r["section_title"] for r in kept]
    assert "Fonts" not in titles and "Mostly prose" in titles                              # glyph debris goes; a section with one stray code stays
    assert st["kept"] == len(kept) == st["all"] - st["empty"] - st["reference"] - st["short"] - st["junk"] - st["glyph"]
    x = np.log(np.array([r["body_chars"] for r in recs if r["body_chars"] > 0 and not r["is_reference"]], float))
    assert st["floor"] == pytest.approx(float(np.exp(np.median(x) - 2 * 1.4826 * median_abs_deviation(x))))
    assert all(r["body_chars"] >= st["floor"] and not r["is_reference"] and not r["is_junk"] for r in kept)


def test_glyph_share_is_the_share_of_non_space_characters_inside_glyph_codes_C13():
    from domain_corpora import glyph_share
    assert glyph_share("") == 0.0 and glyph_share("plain words") == 0.0
    code = "glyph&lt;c=18,font=/ABC&gt;"
    assert glyph_share(code) == 1.0 and glyph_share(code + " " + code) == 1.0
    assert glyph_share(code + "x" * len(code)) == pytest.approx(0.5)
    assert glyph_share("glyph<c=3,font=/A> and glyph[7] text") > 0                         # the bracket and angle forms count too
