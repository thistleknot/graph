"""Check the section-cache CLI inputs through its reusable corpus builder."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import section_corpus as corpus


def test_build_reads_markdown_from_explicit_root_and_preserves_section_keys(tmp_path):
    root = tmp_path / "papers" / "post_processed"
    root.mkdir(parents=True)
    (root / "2401_00001.md").write_text("# Abstract\n\nA stable abstract body with enough words to retain.\n\n# Method\n\nA method section with enough words to retain.", encoding="utf-8")
    (root / "2401_00002.md").write_text("# Abstract\n\nAnother paper body with enough words to retain.", encoding="utf-8")
    (root / "2401_00003.md").write_text("# Abstract\n\nA third paper body with enough words to retain.", encoding="utf-8")
    output = tmp_path / "sections.pkl"

    result = corpus.build(str(output), str(root))

    assert output.exists()
    assert result["papers"] == 3
    keys = {(row["doc_id"], row["section_idx"]) for row in result["records"]}
    assert {("arxiv/2401_00001", 0), ("arxiv/2401_00001", 1),
            ("arxiv/2401_00002", 0), ("arxiv/2401_00003", 0)} <= keys
