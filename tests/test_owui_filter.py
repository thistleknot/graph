"""Offline checks for the imported Open WebUI Filter Function.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| TOGGLE | Verify the filter advertises per-chat user control. | [mandatory] | T1 |
| REQUEST | Mock `/agent` and verify inlet augmentation and selected-model forwarding. | [mandatory] | T1 |
| STATUS | Verify progress, evidence counts, and visible errors before generation. | [mandatory] | T4 |
| BYPASS | Verify task prompts and duplicate inlet calls do not invoke the agent twice. | [mandatory] | T2 |
| RENDER | Verify literal source text precedes commentary in legacy content and client output. | [mandatory] | T3, T5, T6 |

T1 The fixture supplies the `arxiv_rag_api.AgentRequest` report shape with passage text on evidence rows.
T2 The inlet marker is request-local; outlet never calls the agent endpoint.
T3 Labels and TOON cells may contain quotes, brackets, hashes, commas, colons, and empty values.
T4 Filter R5/R6 require pending/terminal events, a 600-second default, and errors with or without an emitter.
T5 Filter R10 preserves source citations/HTML/code fences, displays context with empty model output, and never duplicates it.
T6 Filter R11 synchronizes rendered content into the client-visible output_text while preserving reasoning/tool records.
T7 Filter R12 groups canonical papers without discarding variants; R13 retains comparison guidance and active-term diagnostics only.
T8 Filter R14 displays report.final rather than another generated interpretation; the selected model remains forwarded to the pipeline.

Preconditions: Python, pytest, httpx, and pydantic are installed; no network or database is used.
Failure modes: any contract drift between the filter and the repository search response fails here.
"""

import asyncio
import copy
import re
from unittest.mock import AsyncMock, patch

import pytest

import owui_filter
import owui_filter_age


def agent_report():
    return {
        "question": "question",
        "build": 42,
        "final": "Grounded answer [arxiv/2401_00001#4] [c7].",
        "sections": {
            "42": {
                "doc_id": "arxiv/2401_00001",
                "section_idx": 4,
                "section_title": "Graph methods",
                "paper_title": "Paper title",
                "community": 7,
                "url": "https://arxiv.org/abs/2401.00001",
                "text": "Evidence text",
            }
        },
        "communities": {"7": {"title": "Graph retrieval", "size": 9}},
        "evidence": [{"key": "arxiv/2401_00001#4", "ord": 42, "role": "read by the agent", "text": "Evidence text"}],
        "traversal": {
            "hops": 1,
            "size": 1,
            "entities": [["graph", 1]],
            "by_community": [{"cid": 7, "n": 1, "entities": [["graph", 1]]}],
        },
        "agent": {
            "nodes": {"arxiv/2401_00001#4": {"ord": 42, "key": "arxiv/2401_00001#4", "topic": "Graph methods"}},
            "trace": [{"round": 1, "action": "READ", "arg": "arxiv/2401_00001#4", "why": "Inspect evidence", "result": "read"}],
            "stop": "answered",
        },
        "section_scores": [{"key": "arxiv/2401_00001#4", "community": 7, "score": 0.5}],
        "section_threshold": 0.0,
        "partials": [[7, "partial"]],
        "partials_rejected": [],
        "question_rejected": "",
    }


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    calls = []
    timeouts = []
    payload = agent_report()

    def __init__(self, *args, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payload)


def test_filter_inlet_uses_latest_user_turn_and_preserves_the_selected_model():
    FakeClient.calls = []
    FakeClient.payload = agent_report()
    FakeClient.payload["sections"]["42"].pop("text")
    pipeline = owui_filter.Filter()
    assert pipeline.toggle is True
    body = {
        "model": "openrouter/selected-model",
        "metadata": {"chat_id": "chat-1"},
        "messages": [
            {"role": "system", "content": "Existing instruction"},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Latest question"},
        ],
    }

    with patch.object(owui_filter.httpx, "AsyncClient", FakeClient):
        request_metadata = {}
        asyncio.run(pipeline.inlet(body, __metadata__=request_metadata))
        asyncio.run(pipeline.inlet(body, __metadata__=request_metadata))

    assert body["model"] == "openrouter/selected-model"
    assert len(FakeClient.calls) == 1
    assert FakeClient.calls[0][0].endswith("/agent")
    assert FakeClient.calls[0][1]["json"]["question"] == "Latest question"
    assert FakeClient.calls[0][1]["json"]["model"] == "openrouter/selected-model"
    assert FakeClient.calls[0][1]["json"]["history"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    assert any("Return exactly: Retrieval complete." in m.get("content", "") for m in body["messages"])
    assert request_metadata["_arxiv_retrieval_report"]["evidence"][0]["text"] == "Evidence text"


def test_age_filter_requests_and_requires_the_age_backend():
    FakeClient.calls = []
    FakeClient.payload = agent_report() | {"graph_backend": "age"}
    pipeline = owui_filter_age.Filter()
    body = {"model": "selected-model", "messages": [{"role": "user", "content": "Latest question"}]}

    with patch.object(owui_filter_age.httpx, "AsyncClient", FakeClient):
        asyncio.run(pipeline.inlet(body, __metadata__={}))

    assert FakeClient.calls[0][1]["json"]["graph_backend"] == "age"
    assert FakeClient.calls[0][1]["json"]["answer_kind"] == "auto"

    FakeClient.calls = []
    FakeClient.payload = agent_report() | {"graph_backend": "sql"}
    with patch.object(owui_filter_age.httpx, "AsyncClient", FakeClient):
        with pytest.raises(RuntimeError, match="AGE backend confirmation"):
            asyncio.run(pipeline.inlet({"messages": [{"role": "user", "content": "Another question"}]}, __metadata__={}))


def test_age_filter_polls_global_tally_without_claiming_age_traversal():
    class AggregateClient(FakeClient):
        async def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if url.endswith("/agent"):
                return FakeResponse({"aggregate_job": {"job_id": "job-1"}}, status_code=202)
            return FakeResponse({"status": "completed"})

        async def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return FakeResponse({
                "status": "completed",
                "calls_completed": 3,
                "manifest": {"source_build_id": 8},
                "result": {
                    "estimate": 4.5,
                    "interval": [2.0, 7.0],
                    "converged": True,
                    "calls": 3,
                    "audit": [{"N": 10}],
                },
            })

    AggregateClient.calls = []
    body = {"model": "selected-model", "messages": [{"role": "user", "content": "How many papers discuss graph attention?"}]}
    metadata = {}
    with patch.object(owui_filter_age.httpx, "AsyncClient", AggregateClient):
        asyncio.run(owui_filter_age.Filter().inlet(body, __metadata__=metadata))
    report = metadata["_arxiv_retrieval_report"]
    assert report["answer_kind"] == "semantic_aggregate"
    assert "4.5 of 10" in report["final"]
    assert "sampling uncertainty" in report["final"]
    assert "graph_backend" not in report


@pytest.mark.parametrize("identity,expected", [("url", 1), ("doc", 1), ("distinct", 2)])
def test_source_groups_and_status_count_canonical_papers_without_dropping_variants(identity, expected):
    """R12. Guarantee: display identity merges variants while passage keys and text survive."""
    report = agent_report()
    method = copy.deepcopy(report["sections"]["42"])
    method.update(doc_id="arxiv/2401_00001_methods", section_idx=8, text="Distinct methods passage")
    if identity == "doc":
        report["sections"]["42"].pop("url")
        method.pop("url")
    elif identity == "distinct":
        method.update(doc_id="arxiv/2502_00002_methods", url="https://arxiv.org/abs/2502.00002")
    report["sections"]["43"] = method
    report["evidence"].append({"key": method["doc_id"] + "#8", "ord": 43, "text": method["text"]})
    visible = owui_filter.Filter._render_visible_retrieval(report)
    assert f"{expected} papers; 2 passages" in visible
    assert visible.count("<details>\n<summary>") == expected
    assert visible.count("#### Passage:") == 2
    assert "not independent corroboration" in visible
    assert "Evidence text" in visible and method["text"] in visible
    assert method["doc_id"] + "#8" in visible
    FakeClient.payload = report
    emitter = AsyncMock()
    body = {"model": "selected/model", "messages": [{"role": "user", "content": "Compare methods"}]}
    with patch.object(owui_filter.httpx, "AsyncClient", FakeClient):
        asyncio.run(owui_filter.Filter().inlet(body, __event_emitter__=emitter))
    assert f"from {expected} arXiv papers" in emitter.await_args_list[-1].args[0]["data"]["description"]
    instructions = "\n".join(message.get("content", "") for message in body["messages"])
    assert "Return exactly: Retrieval complete." in instructions


@pytest.mark.parametrize("selected,ignored,show", [([], [], False), (["graph"], [], True), ([], ["unknown"], False)])
def test_seed_reweighting_table_requires_applied_terms(selected, ignored, show):
    """R13. Guarantee: unchanged rankings do not occupy the evidence display."""
    report = agent_report()
    report.update(selected_terms=selected, ignored_terms=ignored, seed_scores=[{"ord": 42, "community": 7}])
    visible = owui_filter.Filter._render_visible_retrieval(report)
    assert ("seed_scores[" in visible) is show
    assert ("Ignored (not in the live community inventory)" in visible) is bool(ignored)


@pytest.mark.parametrize("passage", ["A [literal#2] citation.", "<script>literal source</script>", "```python\nx = 1\n```"])
def test_paper_disclosures_use_client_block_syntax_and_preserve_literal_source(passage):
    """R12. Guarantee: the installed newline tokenizer recognizes every source disclosure."""
    report = agent_report()
    report["evidence"][0]["text"] = passage
    visible = owui_filter.Filter._render_visible_retrieval(report)
    assert re.search(r"<details>\n<summary>[^\n]+</summary>\n\n", visible)
    sources = re.findall(r"(?m)^(`{3,})text\n([\s\S]*?)\n\1$", visible)
    assert [text for _, text in sources] == [passage]
    assert visible.endswith("</details>")


def test_user_valves_load_community_dunning_terms_for_the_multiselect():
    response = FakeResponse({"terms": [
        {"term": "retrieval augmented generation", "label": "retrieval augmented generation — c7, c12"},
        {"term": "vector database", "label": "vector database — c4"},
    ]})
    with patch.object(owui_filter.httpx, "get", return_value=response) as get:
        options = owui_filter.Filter.UserValves.get_term_options()
    assert get.call_args.args[0] == f"{owui_filter.DEFAULT_RAG_API_URL}/community-terms"
    assert options == [
        {"value": "retrieval augmented generation", "label": "retrieval augmented generation — c7, c12"},
        {"value": "vector database", "label": "vector database — c4"},
    ]


def test_task_prompts_bypass_retrieval():
    FakeClient.calls = []
    pipeline = owui_filter.Filter()
    body = {"messages": [{"role": "user", "content": "### Task: generate tags"}]}
    with patch.object(owui_filter.httpx, "AsyncClient", FakeClient):
        asyncio.run(pipeline.inlet(body))
    assert FakeClient.calls == []
    assert "_arxiv_retrieval_added" not in body.get("metadata", {})


@pytest.mark.parametrize("evidence_count", [0, 1, 3])
def test_status_counts_only_evidence_and_emits_once(evidence_count):
    report = agent_report()
    report["evidence"] = []
    for index in range(evidence_count):
        ordinal = str(100 + index)
        doc_id = f"arxiv/2501_{index:05}"
        report["sections"][ordinal] = {
            "doc_id": doc_id, "section_idx": index, "text": "Retrieved section",
        }
        report["evidence"].append({"ord": int(ordinal), "key": f"{doc_id}#{index}"})
    FakeClient.payload = report
    FakeClient.calls = []
    emitter = AsyncMock()
    pipeline = owui_filter.Filter()
    body = {"model": "qwen/research-model", "messages": [{"role": "user", "content": "Research question"}]}
    with patch.object(owui_filter.httpx, "AsyncClient", FakeClient):
        asyncio.run(pipeline.inlet(body, __event_emitter__=emitter))
        asyncio.run(pipeline.inlet(body, __event_emitter__=emitter))
    events = [call.args[0] for call in emitter.await_args_list]
    assert len(FakeClient.calls) == 1
    assert FakeClient.timeouts[-1] == 600.0
    assert events[0] == {"type": "status", "data": {"description": "Searching arXiv graph…", "done": False}}
    assert len(events) == 2
    assert events[-1]["data"]["done"] is True
    assert events[-1]["data"]["description"].startswith(
        f"Retrieved {evidence_count} evidence sections from {evidence_count} arXiv papers"
    )
    assert body["model"] == "qwen/research-model"


@pytest.mark.parametrize("failure", ["timeout", "http", "schema"])
@pytest.mark.parametrize("with_emitter", [False, True])
def test_retrieval_failure_stops_generation(failure, with_emitter):
    class FailingClient(FakeClient):
        async def post(self, url, **kwargs):
            if failure == "timeout":
                raise owui_filter.httpx.ReadTimeout("upstream did not respond")
            if failure == "http":
                response = owui_filter.httpx.Response(
                    502, text="provider unavailable", request=owui_filter.httpx.Request("POST", url)
                )
                response.raise_for_status()
            return FakeResponse({"final": "Incomplete report"})

    emitter = AsyncMock() if with_emitter else None
    body = {"model": "chosen/model", "messages": [{"role": "user", "content": "Research question"}]}
    original_messages = list(body["messages"])
    expected = {
        "timeout": "timed out after 600 seconds",
        "http": "HTTP 502: provider unavailable",
        "schema": "missing final answer, sections, or evidence",
    }[failure]
    with patch.object(owui_filter.httpx, "AsyncClient", FailingClient):
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(owui_filter.Filter().inlet(body, __event_emitter__=emitter))
    assert expected in str(caught.value)
    assert "http://192.168.3.122:8780/agent" in str(caught.value)
    assert body["messages"] == original_messages
    assert "_arxiv_retrieval_report" not in body["metadata"]
    if emitter:
        events = [call.args[0] for call in emitter.await_args_list]
        assert len(events) == 2
        assert events[-1]["data"] == {
            "description": str(caught.value), "done": True, "status": "error",
        }


@pytest.mark.parametrize("messages", [[], [{"role": "assistant", "content": "Answer"}],
                                      [{"role": "user", "content": "### Task: generate title"}]])
def test_bypasses_emit_no_progress(messages):
    emitter = AsyncMock()
    with patch.object(owui_filter.httpx, "AsyncClient") as client:
        asyncio.run(owui_filter.Filter().inlet({"messages": messages}, __event_emitter__=emitter))
    client.assert_not_called()
    emitter.assert_not_awaited()


@pytest.mark.parametrize(
    "label,expected",
    [
        ('quotes " and [brackets] #tag & <html>', ('#quot;', '#91;brackets#93;', '#35;tag', '&lt;html&gt;')),
        ("commas, and colons:", ("commas, and colons:",)),
        ("plain query", ("plain query",)),
    ],
)
def test_graph_cards_escape_labels_and_visible_retrieval(label, expected):
    report = agent_report()
    report["question"] = label
    card = owui_filter.Filter._render_graph_cards(report)
    visible = owui_filter.Filter._render_visible_retrieval(report)
    diagram = card.split("```mermaid\n", 1)[1].split("\n```", 1)[0]
    assert all(fragment in diagram for fragment in expected)
    assert "### Retrieved graph evidence" in visible
    assert "```toon\nentities[1]{entity,sections}:\n  graph,1\n```" in visible
    assert "```toon\ncommunities[1]{community,sections,entities}:\n  Graph retrieval,1,graph (1)\n```" in visible
    assert "Paper title: Graph methods (arxiv/2401_00001#4)" in visible
    assert "#### Passage: Paper title: Graph methods (arxiv/2401_00001#4)" in visible
    assert "```text\nEvidence text\n```\n" in visible


def test_outlet_rewrites_known_citations_and_adds_card_only_once():
    pipeline = owui_filter.Filter()
    report = agent_report()
    request_metadata = {"_arxiv_retrieval_report": report}
    body = {
        "messages": [
            {"role": "assistant", "content": "See [arxiv/2401_00001#4] and [c7]."}
        ],
    }
    asyncio.run(pipeline.outlet(body, __metadata__=request_metadata))
    asyncio.run(pipeline.outlet(body, __metadata__=request_metadata))
    answer = body["messages"][-1]["content"]
    assert "[Paper title: Graph methods (section 4)](https://arxiv.org/abs/2401.00001)" in answer
    assert "Graph retrieval" in answer
    assert answer.count("### Retrieved graph evidence") == 1
    assert "```text\nEvidence text\n```" in answer
    assert answer.count("### How this was answered") == 1


def test_graph_cards_cover_subgraph_trace_entities_sources_and_gate_scores():
    report = agent_report()
    report["sections"]["42"].pop("text")
    card = owui_filter.Filter._render_graph_cards(report)
    assert "### Subgraph" in card
    assert "<summary>Agent trace</summary>" in card
    assert "<summary>Gate scores</summary>" in card
    assert 'n42["Paper title: Graph methods"]' in card
    assert "gate_scores[1]{rank,section,community,score,passed}:" in card
    visible = owui_filter.Filter._render_visible_retrieval(report)
    assert "entities[1]{entity,sections}:" in visible
    assert "Graph methods (arxiv/2401_00001#4)" in visible
    assert "```text\nEvidence text\n```" in visible


@pytest.mark.parametrize(
    "results",
    [
        [
            {"rank": 1, "score": 0.125, "doc_id": "arxiv/2401_00001", "section_idx": 2, "url": "https://arxiv.org/abs/2401.00001", "community": {"cid": 3}},
            {"rank": 2, "score": 0.05, "doc_id": "arxiv/2502_01234", "section_idx": 5, "url": "", "community": {"cid": None}},
            {"rank": 3, "score": 0.005, "doc_id": "paper, with colon:", "section_idx": 8, "url": "https://arxiv.org/abs/2603.54321", "community": {"cid": 9}},
        ],
        [
            {"rank": 1, "score": 1.0, "doc_id": "  padded  ", "section_idx": 0, "url": "", "community": {"cid": 0}},
            {"rank": 2, "score": 2.5, "doc_id": "quote \"x\"", "section_idx": 1, "url": "https://example.test/a,b", "community": {"cid": 1}},
            {"rank": 3, "score": 3.25, "doc_id": "literal #hash", "section_idx": 2, "url": "", "community": {"cid": 2}},
        ],
        [
            {"rank": 1, "score": 0.0, "doc_id": "comma, colon: value", "section_idx": 3, "url": "", "community": {"cid": 4}},
            {"rank": 2, "score": 0.75, "doc_id": "", "section_idx": 4, "url": "https://example.test/", "community": {"cid": 5}},
            {"rank": 3, "score": 4.125, "doc_id": "back\\slash", "section_idx": 5, "url": "", "community": {"cid": 6}},
        ],
    ],
)
def test_toon_source_rows_match_installed_encoder(results):
    toon_python = pytest.importorskip("toon_python")
    report = {"query": "q", "results": results}
    fields = ["rank", "paper", "section", "community", "score", "url"]
    rows = [
        {
            "rank": result["rank"],
            "paper": result.get("doc_id") or "unknown paper",
            "section": result.get("section_idx", ""),
            "community": (result.get("community") or {}).get("cid")
            if (result.get("community") or {}).get("cid") is not None
            else "",
            "score": result.get("score", ""),
            "url": result.get("url") or "",
        }
        for result in results
    ]
    actual = owui_filter.Filter._toon_table("sources", fields, [list(row.values()) for row in rows])
    assert actual == toon_python.encode({"sources": rows})


@pytest.mark.parametrize(
    "answer,expected",
    [
        (
            "Both [arxiv/2401_00001#4, arxiv/2401_00001#9].",
            "Both [Paper title: Graph methods (section 4)](https://arxiv.org/abs/2401.00001), [arxiv/2401_00001#9].",
        ),
        (
            "Semicolon [arxiv/2401_00001#4;arxiv/2401_00001#4] then [c7].",
            "Semicolon [Paper title: Graph methods (section 4)](https://arxiv.org/abs/2401.00001), "
            "[Paper title: Graph methods (section 4)](https://arxiv.org/abs/2401.00001) then Graph retrieval.",
        ),
        ("Unknown only [arxiv/9999_00001#1].", "Unknown only [arxiv/9999_00001#1]."),
    ],
)
def test_multi_key_citation_brackets_link_each_known_key(answer, expected):
    assert owui_filter.Filter._rewrite_citations(answer, agent_report()) == expected


def test_rejected_question_is_answered_with_the_gate_notice_not_model_memory():
    report = agent_report()
    report["question_rejected"] = "Question rejected: no section passed the cross-encoder relevance check (best -1.75)."
    report["evidence"] = []
    FakeClient.calls = []
    FakeClient.payload = report
    pipeline = owui_filter.Filter()
    body = {"model": "m", "messages": [{"role": "user", "content": "Who won the 2018 World Cup?"}]}
    metadata = {}
    with patch.object(owui_filter.httpx, "AsyncClient", FakeClient):
        asyncio.run(pipeline.inlet(body, __metadata__=metadata))
    assert body["messages"][0]["role"] == "system"
    assert "do not answer from your own knowledge" in body["messages"][0]["content"]

    body["messages"].append({"role": "assistant", "content": "France won 4-2."})
    asyncio.run(pipeline.outlet(body, __metadata__=metadata))
    asyncio.run(pipeline.outlet(body, __metadata__=metadata))
    answer = body["messages"][-1]["content"]
    assert answer.startswith("The arXiv section graph found no section relevant to this question")
    assert "France" not in answer
    assert "> Question rejected: no section passed the cross-encoder relevance check (best -1.75)." in answer
    assert answer.count("### Retrieved graph evidence") == 1
    assert answer.count("### How this was answered") == 1

    multimodal = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "France won 4-2."}]}]}
    asyncio.run(pipeline.outlet(multimodal, __metadata__=metadata))
    text = multimodal["messages"][0]["content"][0]["text"]
    assert text.startswith("The arXiv section graph found no section relevant") and "France" not in text


def test_accepted_question_displays_pipeline_answer_without_a_second_interpretation():
    FakeClient.calls = []
    FakeClient.payload = agent_report()
    pipeline = owui_filter.Filter()
    body = {"model": "m", "messages": [{"role": "user", "content": "Graph question"}]}
    metadata = {}
    with patch.object(owui_filter.httpx, "AsyncClient", FakeClient):
        asyncio.run(pipeline.inlet(body, __metadata__=metadata))
    assert "do not answer from your own knowledge" not in body["messages"][0]["content"]
    body["messages"].append({"role": "assistant", "content": "Model prose."})
    asyncio.run(pipeline.outlet(body, __metadata__=metadata))
    rendered = body["messages"][-1]["content"]
    assert rendered.index("```text\nEvidence text\n```") < rendered.index("### Model commentary") < rendered.index("Grounded answer")
    assert "Model prose." not in rendered


@pytest.mark.parametrize("passage", [
    "Literal source citation [arxiv/2401_00001#4] and community [c7].",
    '<script>alert("source")</script> & a < b\nA second line.',
    "## Source heading\n\n```python\nprint('source')\n```\n\nEnd of passage.",
])
def test_outlet_preserves_exact_source_text_across_repeated_calls(passage):
    """R10. Guarantee: decoded passages equal the API source after two outlet calls."""
    report = agent_report()
    report["evidence"][0]["text"] = passage
    report["evidence"][0]["truncated"] = True
    body = {"messages": [{"role": "assistant", "content": "Discuss [arxiv/2401_00001#4]."}]}
    pipeline = owui_filter.Filter()
    metadata = {"_arxiv_retrieval_report": report}
    asyncio.run(pipeline.outlet(body, __metadata__=metadata))
    first = body["messages"][0]["content"]
    asyncio.run(pipeline.outlet(body, __metadata__=metadata))
    rendered = body["messages"][0]["content"]
    assert rendered == first
    source = re.search(r"(?m)^(`{3,})text\n([\s\S]*?)\n\1$", rendered).group(2)
    assert source == passage
    assert rendered.count("#### Passage:") == 1
    assert "Passage text is truncated" in rendered
    assert rendered.index("text\n") < rendered.index("### Model commentary")
    assert "[Paper title: Graph methods (section 4)](https://arxiv.org/abs/2401.00001)" in rendered.split("### Model commentary", 1)[1]


@pytest.mark.parametrize("content", [
    "", None,
    [{"type": "text", "text": "First observation."}, {"type": "text", "text": "Second observation."}],
    [{"type": "image_url", "image_url": {"url": "https://example.test/figure.png"}}],
])
def test_outlet_displays_source_without_depending_on_generated_text(content):
    """R10. Guarantee: every supported content shape displays source before commentary."""
    body = {"messages": [{"role": "assistant", "content": content}]}
    pipeline = owui_filter.Filter()
    metadata = {"_arxiv_retrieval_report": agent_report()}
    asyncio.run(pipeline.outlet(body, __metadata__=metadata))
    rendered = body["messages"][0]["content"]
    text = pipeline._text(rendered)
    assert text.count("```text\nEvidence text\n```") == 1
    assert text.count("### Model commentary") == 1
    assert text.index("```text\nEvidence text\n```") < text.index("### Model commentary")
    if isinstance(content, list):
        assert rendered[0]["type"] == "text"
        if content[0]["type"] == "image_url":
            assert rendered[1] == content[0]
        else:
            assert "First observation." not in text and "Second observation." not in text
    asyncio.run(pipeline.outlet(body, __metadata__=metadata))
    assert pipeline._text(body["messages"][0]["content"]) == text


@pytest.mark.parametrize("output", [
    [{"type": "reasoning", "content": [{"type": "output_text", "text": "Private reasoning"}]},
     {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Commentary"}]}],
    [{"type": "function_call", "call_id": "tool-1", "name": "search"},
     {"type": "message", "content": [{"type": "output_text", "text": "Commentary"}, {"type": "image", "url": "figure.png"}]}],
    [{"type": "reasoning", "content": []}],
])
def test_outlet_synchronizes_client_visible_structured_output(output):
    """R11. Guarantee: client output includes exact context and retains nontext records."""
    preserved = copy.deepcopy(output[0])
    body = {"messages": [{"role": "assistant", "content": "Commentary", "output": output}]}
    metadata = {"_arxiv_retrieval_report": agent_report()}
    pipeline = owui_filter.Filter()
    asyncio.run(pipeline.outlet(body, __metadata__=metadata))
    assistant = body["messages"][0]
    final_message = assistant["output"][-1]
    assert final_message["content"][0] == {"type": "output_text", "text": assistant["content"]}
    assert assistant["output"][0] == preserved
    assert "#### Passage:" in final_message["content"][0]["text"]
    once = copy.deepcopy(assistant)
    asyncio.run(pipeline.outlet(body, __metadata__=metadata))
    assert assistant == once

