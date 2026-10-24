"""| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| ENABLE | Install as an active, model-specific Filter and expose its per-chat toggle. | [contract] | R1 |
| SELECT | UserValves presents the live community Dunning terms as a dynamic multiselect. | [optional] | R9 |
| RETRIEVE | Inlet announces retrieval and sends question, selected terms, and model to `/agent` once. | [mandatory] | R2, R5, R9 |
| CHECK | Report evidence counts or stop the chat with the retrieval error. | [mandatory] | R5, R6 |
| ACKNOWLEDGE | Request only an acknowledgement; outlet displays the selected pipeline model's answer directly. | [mandatory] | R3, R10, R14 |
| RENDER | Group exact passages by canonical paper in expandable Markdown and synchronize client output. | [mandatory] | R4, R7, R10-R13 |
| GATE | A question the relevance gate rejected is answered with the gate's notice, never from the model's memory. | [mandatory] | R8 |

title: arXiv Section Retrieval Filter
author: local
version: 0.9.0
description: Displays retrieved source passages before separately labeled model commentary.

R1 Set `self.toggle = True`; keep the Filter active but non-global, then attach it to the intended
model(s). Open WebUI runs it only in chats where the user selects its filter chip or Integrations switch.
The selected model remains `body['model']` and is never replaced.
R2 Only `inlet` calls `POST /agent`; `outlet` formats the saved result without retrieval or model
calls. A request metadata marker prevents duplicate augmentation.
R3 `rag_api_url` must point to the arXiv API origin on the graph host, default
`http://192.168.3.122:8780`. The API must expose the `AgentRequest` contract in
`src/arxiv_rag_api.py` and be reachable from the Open WebUI server on `192.168.3.18`.
R4 The API's `final` answer and each evidence row's `text` are separate inputs; visible retrieval details are generated from the API report's traversal, evidence, section,
and community fields; unavailable data is stated explicitly. Entity and community metadata MUST remain
visible as TOON tables (`entities`, `communities`) and sources as paper links; each evidence passage
MUST be preserved verbatim inside its own visible Markdown `text` block for direct inspection and copying.
R5 When Open WebUI supplies `__event_emitter__`, inlet MUST emit a pending status before retrieval
and a completed status with evidence/paper counts or an error after retrieval; bypassed and duplicate
calls MUST emit no status. Given a successful report, Then the status MUST count its evidence only.
R6 If retrieval times out, returns an HTTP error, or returns an invalid report, inlet MUST raise a
descriptive error before model generation; it MUST NOT inject a fallback instruction. `timeout_seconds`
varies by deployment and defaults to 600 seconds. Given a timeout, Then the chat MUST show its limit
and endpoint. Given no event emitter, Then the same retrieval/error contract MUST hold.
R7 Given a successful report, Then outlet MUST prepend the retrieved entity/community breakdown,
section sources, and each passage's returned text before separately labeled model commentary, independent of generated prose.
Given a citation bracket holding several section keys (`[a#1, b#2]`), Then each known key MUST
become its own paper link and each unknown key MUST stay as literal `[key]` text.
R8 Given a report whose `question_rejected` is non-empty, Then inlet MUST instruct the model to say the
corpus has no relevant section and not answer from its own knowledge, and outlet MUST replace the
assistant prose with that notice plus the gate's reason before appending the evidence and cards.
R9 Selected terms come from the user's multiselect and pass unchanged to `/agent`; the API validates them against its live Dunning inventory. The UI option list loads from `ARXIV_RAG_API_URL` or the default graph API origin when the UserValves dialog opens. If the option endpoint is unavailable, the control has no options; retrieval without a selection remains available.
R10 WHEN rendering context, source text SHALL come directly from API evidence inside a Markdown fence longer than any source backtick run, without HTML escaping. Citation rewriting SHALL apply only to model commentary. A generated-block marker SHALL prevent duplicate rendering or source rewriting on repeated outlet calls. Empty model text and multipart responses without text SHALL still display context. The relevance-gate notice retains precedence for rejected questions.
R11 WHEN structured `output` is present, outlet SHALL synchronize its final assistant message's `output_text` with rendered `content` and preserve reasoning/tool records; if absent, add that assistant output message. Open WebUI's client renders `output`, so content-only persistence does not satisfy visible-context delivery.
R12 Main and `_methods` source records sharing the API's canonical arXiv URL SHALL count as one paper and share one expandable source group; retain all source keys and passages, and label methods records explicitly. Without a URL, use doc_id without its `_methods` suffix. Disclosure opening and summary tags SHALL occupy separate lines, matching the installed OWUI tokenizer.
R13 WHEN sources give comparison criteria, the model SHALL use them as criteria and actual methods as alternatives; unsupported table cells say not reported. Preserve distinct change targets, including parameter-changing extensions, and attribute recommendations/proposals. Overview/subgraph Mermaid blocks SHALL be visible; unused seed-reweighting tables SHALL be omitted.
R14 Accepted Model commentary SHALL use report.final directly, with citation-link rewriting only; discard the framework's generated acknowledgement instead of introducing another interpretation. The selected model remains forwarded to the pipeline, which owns synthesis and its canonical-paper coverage inventory.

CLOSED: This is an in-process Filter Function, not a legacy Pipelines server class or Pipe.
Keep it model-specific for per-chat opt-in; marking it global forces it on across models.
The Filter lifecycle calls `inlet` and `outlet`; retrieval is restricted to `inlet`.
The legacy `Valves.pipelines` model list is not used by global Filter Functions.
The Open WebUI `/api/v1/tasks/queries/completions` endpoint is not used for arXiv retrieval.
The claim that returned passages are entailment-verified is rejected: `keep_grounded_sentences`
checks citation keys against the evidence set; the cross-encoder supplies relevance scores.
Silent continuation after retrieval failure is rejected: an authenticated hosted test exceeded the
300-second inlet limit while the graph agent was still awaiting OpenRouter (2026-10-17).

Preconditions: the Open WebUI server process can reach the arXiv API; the API is available without auth
or the configured bearer key is accepted. The user's selected model remains available in OWUI.
Failure modes: retrieval failures stop the chat with an explicit error. A missing outlet metadata
field suppresses only the visual card.
"""

import html
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
DEFAULT_RAG_API_URL = os.environ.get("ARXIV_RAG_API_URL", "http://192.168.3.122:8780")


class Filter:
    """Add arXiv section evidence to an Open WebUI request and render source metadata.

    Attributes:
        valves: API endpoint and bounded retrieval settings.
        toggle: Enables per-chat selection in Open WebUI.

    Preconditions:
        Open WebUI supplies a chat request and the configured API is reachable.

    Called by:
        Open WebUI's filter lifecycle.

    Depends on:
        The graph API `/agent` response contract.

    Failure modes:
        Retrieval failures stop generation; malformed reports raise a visible error.
    """

    class UserValves(BaseModel):
        selected_terms: list[str] = Field(
            default_factory=list,
            description="Optional community terms to add to lexical search and community reweighting.",
            json_schema_extra={
                "input": {"type": "multiselect", "options": "get_term_options"}
            },
        )

        @classmethod
        def get_term_options(cls, __user__=None) -> list[dict[str, str]]:
            """Load current choices when the user opens this filter's settings.

            Require: the Open WebUI host can reach the graph API.
            Guarantee: term values and readable labels, or an empty list on lookup failure.
            Maintain: no search request or selection is changed.
            """
            try:
                response = httpx.get(f"{DEFAULT_RAG_API_URL}/community-terms", timeout=4.0)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    return []
                return [
                    {"value": row["term"], "label": row["label"]}
                    for row in payload.get("terms", [])
                    if isinstance(row, dict) and row.get("term") and row.get("label")
                ]
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                log.warning("Could not load community term options: %s", exc)
                return []

    class Valves(BaseModel):
        priority: int = Field(default=0, description="Filter execution priority.")
        rag_api_url: str = Field(
            default=DEFAULT_RAG_API_URL,
            description="ArXiv graph API origin on the graph host, reachable from Open WebUI.",
        )
        api_key: str = Field(
            default="",
            description="Optional bearer key for the retrieval API.",
        )
        k: int = Field(default=3, ge=1, le=5, description="Initial section seeds for graph retrieval.")
        hops: int = Field(default=3, ge=0, le=5, description="Maximum section-graph traversal depth.")
        queries: int = Field(default=3, ge=1, le=5, description="Additional searches the agent may make.")
        per_query: int = Field(default=18, ge=1, le=50, description="Subgraph sections exposed per agent search.")
        max_chars: int = Field(
            default=6000,
            ge=500,
            le=20000,
            description="Maximum evidence characters returned per section.",
        )
        timeout_seconds: float = Field(
            default=600.0,
            gt=0,
            description="Retrieval request timeout.",
        )

    def __init__(self) -> None:
        """Require: no arguments. Guarantee: valves are initialized and the UI toggle is enabled; maintain no request state."""
        self.valves = self.Valves()
        self.toggle = True

    @staticmethod
    def _text(content: Any) -> str:
        """Require: OpenAI message content. Guarantee: readable text or empty string; maintain input unchanged."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""

    @staticmethod
    def _toon_value(value: Any) -> str:
        """Require: one scalar. Guarantee: a TOON-safe scalar; maintain its value and type semantics."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        text = "" if value is None else str(value)
        needs_quotes = (
            text == ""
            or text != text.strip()
            or text in {"true", "false", "null"}
            or re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", text) is not None
            or re.search(r'[:,"\\\[\]{}\x00-\x1f]', text) is not None
            or text.startswith(("-", "#"))
        )
        return json.dumps(text, ensure_ascii=False) if needs_quotes else text

    @staticmethod
    def _mermaid_label(value: Any) -> str:
        """Require: any label value. Guarantee: escaped text capped at 180 characters; maintain stable caller ids."""
        text = str(value or "").replace("\r", " ").replace("\n", " ")
        text = text.replace("#", "#35;").replace("&", "&amp;")
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace('"', "#quot;").replace("[", "#91;").replace("]", "#93;")
        return text[:180]

    @staticmethod
    def _rewrite_citations(answer: str, report: dict[str, Any]) -> str:
        """Require: assistant text and an API report. Guarantee: known citations link to papers; maintain unknown tags."""
        citations: dict[str, str] = {}
        community_titles = {
            str(cid): str(community.get("title"))
            for cid, community in (report.get("communities") or {}).items()
            if community.get("title")
        }
        for section in (report.get("sections") or {}).values():
            doc_id = section.get("doc_id") or ""
            section_idx = section.get("section_idx")
            title = section.get("section_title") or doc_id
            paper_title = section.get("paper_title") or doc_id
            url = section.get("url") or ""
            key = f"{doc_id}#{section_idx}"
            if doc_id and section_idx is not None and url:
                citations[key] = f"[{paper_title}: {title} (section {section_idx})]({url})"

        def link_keys(match: re.Match[str]) -> str:
            keys = re.split(r"\s*[,;]\s*", match.group(1))
            return ", ".join(citations.get(key, f"[{key}]") for key in keys)

        answer = re.sub(
            r"\[(arxiv/[^\]\s,;]+#\d+(?:\s*[,;]\s*arxiv/[^\]\s,;]+#\d+)*)\]",
            link_keys,
            answer,
        )
        return re.sub(
            r"\[c(\d+)\]",
            lambda match: community_titles.get(match.group(1), match.group(0)),
            answer,
        )

    @classmethod
    def _render_graph_cards(cls, report: dict[str, Any]) -> str:
        """Render visible diagrams and expandable diagnostics without model transcription.

        Args:
            report: The `/agent` report with graph, traversal, trace, and gate metadata.

        Returns:
            Native Mermaid overview/subgraph blocks and TOON diagnostic disclosures;
            graph diagrams retain at most 40 nodes and perform no I/O.
        """
        sections = report.get("sections") or {}
        communities = report.get("communities") or {}
        traversal = report.get("traversal") or {}
        agent = report.get("agent") or {}
        scores = report.get("section_scores") or []
        threshold = report.get("section_threshold", 0)
        question = cls._mermaid_label(report.get("question", ""))
        cards = []

        if traversal or scores or agent:
            best = max((float(row.get("score", 0)) for row in scores), default=0)
            passed = sum(float(row.get("score", 0)) >= float(threshold) for row in scores)
            accepted = not bool(report.get("question_rejected"))
            yes_cids = [str(row[0]) for row in report.get("partials") or []]
            no_cids = [
                str(row.get("community", "")) if isinstance(row, dict) else str(row)
                for row in report.get("partials_rejected") or []
            ]
            community_label = "Communities YES: " + ", ".join(
                (communities.get(cid) or {}).get("title") or f"c{cid}" for cid in yes_cids
            ) + "; NO: " + ", ".join(
                (communities.get(cid) or {}).get("title") or f"c{cid}" for cid in no_cids
            )
            summary = [
                "flowchart TD",
                f'  q["Question: {question}"]',
                f'  gate["Cross-encoder gate: best {best:.3f}; passed {passed} of {len(scores)}"]',
                f'  comm["{cls._mermaid_label(community_label)[:180]}"]',
                f'  walk["Subgraph: {traversal.get("size", len(report.get("subgraph") or []))} sections; {traversal.get("hops", 0)} hops"]',
                f'  agent["Agent: {len(agent.get("trace") or report.get("trace") or [])} rounds; {cls._mermaid_label(agent.get("stop", "unknown"))}"]',
                f'  answer["Answer: {"accepted" if accepted else "rejected"}; {cls._mermaid_label(report.get("final", ""))[:120]}"]',
                "  q --> gate --> comm --> walk --> agent --> answer",
            ]
            cards.append("\n\n### How this was answered\n\n```mermaid\n" + "\n".join(summary) + "\n```")

        nodes_by_community: dict[str, list[dict[str, Any]]] = {}
        nodes = list((agent.get("nodes") or {}).values())
        if not nodes:
            nodes = report.get("nodes") or []
        ordered_nodes = []
        seen_ords = set()
        for node in nodes:
            ord_ = node.get("ord")
            if ord_ is not None and ord_ not in seen_ords:
                seen_ords.add(ord_)
                ordered_nodes.append(node)
        if not ordered_nodes:
            ordered_nodes = [{"ord": ord_} for ord_ in report.get("seeds", [])]
        diagram = ["flowchart TD"]
        for node in ordered_nodes:
            ord_ = node.get("ord")
            section = sections.get(str(ord_), {})
            community = str(section.get("community", "unassigned"))
            nodes_by_community.setdefault(community, []).append(node)
        shown = ordered_nodes[:40]
        shown_ord_list = [node.get("ord") for node in shown]
        shown_ords = set(shown_ord_list)
        for index, (cid, members) in enumerate(nodes_by_community.items()):
            label = (communities.get(cid) or {}).get("title") or f"Community {cid}"
            diagram.append(f'  subgraph community_{index}["{cls._mermaid_label(label)}"]')
            for node in members:
                ord_ = node.get("ord")
                if ord_ not in shown_ords:
                    continue
                section = sections.get(str(ord_), {})
                key = node.get("key") or f"section {ord_}"
                paper = section.get("paper_title") or key
                title = section.get("section_title") or node.get("topic") or "Section"
                diagram.append(f'    n{ord_}["{cls._mermaid_label(paper)}: {cls._mermaid_label(title)}"]')
            diagram.append("  end")
        omitted = max(0, len(ordered_nodes) - len(shown))
        if omitted:
            diagram.append(f'  more["+{omitted} more sections"]')
        for ord_ in shown_ord_list:
            if ord_ is not None:
                diagram.append(f"  q --> n{ord_}")
        if ordered_nodes:
            cards.append("\n\n### Subgraph\n\n```mermaid\n" + "\n".join(diagram) + "\n```")

        trace = agent.get("trace") or report.get("trace") or []
        if trace:
            lines = ["flowchart TD"]
            previous = None
            for index, step in enumerate(trace):
                node_id = f"trace_{index}"
                action = cls._mermaid_label(step.get("action") or "REPLY")
                arg = cls._mermaid_label(step.get("arg") or "")
                why = cls._mermaid_label(step.get("why") or "")
                result = cls._mermaid_label(step.get("result") or "")
                lines.append(f'  {node_id}["{action}: {arg}<br/>{why}<br/>{result}"]')
                if previous:
                    lines.append(f"  {previous} --> {node_id}")
                previous = node_id
            cards.append(cls._details("Agent trace", "mermaid", "\n".join(lines)))

        if scores:
            rows = []
            for index, row in enumerate(scores, 1):
                score = float(row.get("score", 0))
                rows.append([index, row.get("key", ""), row.get("community", ""), score,
                             "YES" if score >= float(threshold) else "NO"])
            cards.append(cls._details(
                "Gate scores", "toon",
                cls._toon_table("gate_scores", ["rank", "section", "community", "score", "passed"], rows),
            ))
        return "".join(cards)

    @classmethod
    def _render_visible_retrieval(cls, report: dict[str, Any]) -> str:
        """Render retrieval metadata and source passages from an `/agent` report.

        Args:
            report: Successful `/agent` report containing traversal, evidence, sections, and communities.

        Returns:
            Expandable groups of exact passages, one per canonical paper, independent of model prose.

        Raises:
            KeyError: Not raised; missing report fields are rendered as unavailable.
        """
        sections = report.get("sections") or {}
        traversal = report.get("traversal") or {}
        communities = report.get("communities") or {}
        evidence_rows = report.get("evidence") or []
        lines = ["\n\n<!-- arxiv-retrieved-context -->", "### Retrieved graph evidence"]

        selected_terms = report.get("selected_terms") or []
        ignored_terms = report.get("ignored_terms") or []
        if selected_terms or ignored_terms:
            lines.append("\n**Selected lexical terms**")
            lines.append("- Applied: " + (", ".join(f"`{html.escape(str(term))}`" for term in selected_terms) or "none"))
            if ignored_terms:
                lines.append("- Ignored (not in the live community inventory): " + ", ".join(f"`{html.escape(str(term))}`" for term in ignored_terms))
        seed_scores = report.get("seed_scores") or []
        if seed_scores and selected_terms:
            rows = []
            for row in seed_scores:
                cid = str(row.get("community", ""))
                title = (communities.get(cid) or {}).get("title") or cid
                rows.append([row.get("ord"), title, row.get("rrf_score", 0), row.get("affinity", 0), row.get("adjusted_score", 0)])
            lines.extend(["\n**Fused seed ranking after community reweighting**", "", "```toon",
                          cls._toon_table("seed_scores", ["ord", "community", "rrf", "term_affinity", "adjusted"], rows), "```"])

        entities = traversal.get("entities") or []
        lines.append("\n**Entities**")
        if entities:
            lines.extend(["", "```toon", cls._toon_table("entities", ["entity", "sections"], entities), "```"])
        else:
            lines.append("- No entity labels were returned for this subgraph.")

        community_rows = traversal.get("by_community") or []
        lines.append("\n**Communities**")
        if community_rows:
            rows = []
            for row in community_rows:
                cid = str(row.get("cid", ""))
                title = (communities.get(cid) or {}).get("title") or f"Community {cid}"
                entity_text = "; ".join(
                    f"{name} ({count})" for name, count in row.get("entities", [])
                ) or "no entity labels"
                rows.append([title, row.get("n", 0), entity_text])
            lines.extend(["", "```toon", cls._toon_table("communities", ["community", "sections", "entities"], rows), "```"])
        else:
            lines.append("- No community breakdown was returned.")

        lines.append("\n**Retrieved sections**")
        if evidence_rows:
            papers = {}
            for evidence in evidence_rows:
                key = evidence.get("key", "")
                section = next(
                    (item for item in sections.values()
                     if f"{item.get('doc_id')}#{item.get('section_idx')}" == key),
                    {},
                )
                doc_id = section.get("doc_id") or key.split("#", 1)[0] or "Unknown paper"
                paper_id = section.get("url") or doc_id.removesuffix("_methods")
                title = section.get("paper_title") or doc_id.removesuffix("_methods")
                group = papers.setdefault(paper_id, {"title": title, "passages": []})
                section_title = section.get("section_title") or "Untitled section"
                citation = f"{section.get('doc_id', '')}#{section.get('section_idx', '')}"
                url = section.get("url")
                label = f"{title}: {section_title} ({citation})"
                source = f"[{label}]({url})" if url else label
                role = evidence.get("role")
                passage_lines = [f"- {source}" + (f" — {role}" if role else "")]
                if doc_id.endswith("_methods"):
                    passage_lines.append("\nMethods record from the same paper; not independent corroboration.")
                passage = evidence.get("text") or section.get("text")
                if passage:
                    passage_title = html.escape(label)
                    passage_role = html.escape(str(role or "retrieved evidence"))
                    # Preserve literal source Markdown inside the disclosure; a longer
                    # fence keeps embedded code from closing the source block early.
                    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", passage)), default=0))
                    passage_lines.extend([
                        f"\n#### Passage: {passage_title} — {passage_role}",
                        "",
                        f"{fence}text",
                        passage,
                        fence,
                        "",
                    ])
                    if evidence.get("truncated") or section.get("truncated"):
                        passage_lines.append("\nPassage text is truncated at the configured `max_chars` limit.")
                else:
                    passage_lines.append("  - Passage text was not included in the API report.")
                group["passages"].append("\n".join(passage_lines))
            lines.append(f"\n{len(papers)} papers; {len(evidence_rows)} passages. Expand a paper to inspect its source text.")
            for group in papers.values():
                count = len(group["passages"])
                title = html.escape(str(group["title"])) + f" — {count} passage" + ("s" if count != 1 else "")
                lines.append(cls._details(title, None, "\n\n".join(group["passages"])))
        else:
            lines.append("- No evidence sections were returned.")
        return "\n".join(lines)

    @staticmethod
    def _rejection_notice(report: dict[str, Any]) -> str:
        """Require: an API report. Guarantee: the gate's notice when it rejected the question, else an empty string."""
        reason = str(report.get("question_rejected") or "").strip()
        if not reason:
            return ""
        return f"The arXiv section graph found no section relevant to this question, so it is not answered from the corpus.\n\n> {reason}"

    @classmethod
    def _details(cls, title: str, language: str | None, content: str) -> str:
        """Serialize a disclosure in the installed client's block-tokenizer syntax.

        Args:
            title: Escaped summary label.
            language: Fence language, or None for an already formatted Markdown body.
            content: Ordered Markdown or diagnostic source text.

        Returns:
            A disclosure with opening and summary tags on separate lines; source text remains intact.
        """
        body = f"```{language}\n{content}\n```" if language else content
        return f"\n\n<details>\n<summary>{title}</summary>\n\n{body}\n\n</details>"

    @classmethod
    def _toon_table(cls, name: str, fields: list[str], rows: list[Any]) -> str:
        """Require: each row matches `fields`. Guarantee: tabular TOON; assert column counts and maintain row order."""
        lines = [f"{name}[{len(rows)}]{{{','.join(fields)}}}:"]
        for row in rows:
            values = [row[field] for field in fields] if isinstance(row, dict) else list(row)
            assert len(values) == len(fields)
            lines.append("  " + ",".join(cls._toon_value(value) for value in values))
        return "\n".join(lines)

    async def inlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Require: Open WebUI body and optional metadata/emitter. Guarantee: augment once or raise a visible error.

        Maintain: body model is unchanged; outlet owns cards. Assert: bypasses emit nothing; retrieval has terminal status.
        """
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body
        metadata = __metadata__ if isinstance(__metadata__, dict) else body.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            body["metadata"] = metadata
        if metadata.get("_arxiv_retrieval_added"):
            return body

        latest_user = next(
            (message for message in reversed(messages)
             if isinstance(message, dict) and message.get("role") == "user"),
            None,
        )
        if latest_user is None:
            return body
        question = self._text(latest_user.get("content")).strip()
        if not question or question.startswith("### Task:"):
            return body

        history = []
        latest_user_index = next(
            index for index in range(len(messages) - 1, -1, -1)
            if messages[index] is latest_user
        )
        for message in messages[:latest_user_index]:
            if isinstance(message, dict) and message.get("role") in {"user", "assistant"}:
                content = self._text(message.get("content")).strip()
                if content:
                    history.append({"role": message["role"], "content": content})

        user_valves = __user__.get("valves") if isinstance(__user__, dict) else None
        if isinstance(user_valves, dict):
            selected_terms = user_valves.get("selected_terms") or []
        else:
            selected_terms = getattr(user_valves, "selected_terms", []) or []
        if not isinstance(selected_terms, list):
            selected_terms = []
        selected_terms = [term for term in selected_terms if isinstance(term, str)]

        metadata["_arxiv_retrieval_added"] = True
        headers = {"Accept": "application/json"}
        if self.valves.api_key:
            headers["Authorization"] = f"Bearer {self.valves.api_key}"
        api_url = self.valves.rag_api_url.rstrip("/")
        started = monotonic()
        log.info("ArXiv graph retrieval started: %s/agent", api_url)
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "Searching arXiv graph…", "done": False},
            })
        try:
            async with httpx.AsyncClient(timeout=self.valves.timeout_seconds) as client:
                response = await client.post(
                    f"{api_url}/agent",
                    json={
                        "question": question,
                        "model": body.get("model"),
                        "history": history[-8:],
                        "k": self.valves.k,
                        "hops": self.valves.hops,
                        "queries": self.valves.queries,
                        "per_query": self.valves.per_query,
                        "max_chars": self.valves.max_chars,
                        "selected_terms": selected_terms,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                report = response.json()
                if (
                    not isinstance(report, dict)
                    or not isinstance(report.get("final"), str)
                    or not isinstance(report.get("sections"), dict)
                    or not isinstance(report.get("evidence"), list)
                ):
                    raise ValueError("agent API response is missing final answer, sections, or evidence")
        except (httpx.HTTPError, ValueError) as exc:
            if isinstance(exc, httpx.TimeoutException):
                detail = f"timed out after {self.valves.timeout_seconds:g} seconds"
            elif isinstance(exc, httpx.HTTPStatusError):
                detail = f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
            else:
                detail = f"{type(exc).__name__}: {exc}"
            error = f"ArXiv graph retrieval failed ({api_url}/agent): {detail}"
            log.error("%s (%.1fs)", error, monotonic() - started)
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error, "done": True, "status": "error"},
                })
            raise RuntimeError(error) from exc

        sections = report.get("sections") or {}
        evidence_rows = report.get("evidence") or []
        paper_ids = set()
        for evidence in evidence_rows:
            section_data = next(
                (item for item in sections.values()
                 if f"{item.get('doc_id')}#{item.get('section_idx')}" == evidence.get("key")),
                {},
            )
            doc_id = section_data.get("doc_id") or evidence.get("key", "").split("#", 1)[0]
            if doc_id:
                paper_ids.add(section_data.get("url") or doc_id.removesuffix("_methods"))
        context = (
            "The selected model has already answered this question inside the graph pipeline. "
            "The application will display that pipeline answer and its exact retrieved passages directly. "
            "Return exactly: Retrieval complete. Do not add another interpretation or repeat the answer."
        )
        if report.get("question_rejected"):
            context = (
                "The arXiv section-graph relevance gate rejected the user's question: "
                f"{report['question_rejected']} Reply in one short sentence that the arXiv corpus has no "
                "relevant section for it, and do not answer from your own knowledge."
            )
        insert_at = 0
        while (
            insert_at < len(messages)
            and isinstance(messages[insert_at], dict)
            and messages[insert_at].get("role") in {"system", "developer"}
        ):
            insert_at += 1
        body["messages"] = [
            *messages[:insert_at],
            {"role": "system", "content": context},
            *messages[insert_at:],
        ]
        metadata["_arxiv_retrieval_report"] = {
            **report,
            "query": question,
        }
        description = (
            f"Retrieved {len(evidence_rows)} evidence sections from {len(paper_ids)} arXiv papers "
            f"in {monotonic() - started:.1f}s."
        )
        log.info(description)
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": description, "done": True},
            })
        return body

    async def outlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Display exact retrieved context before the model's interpretation.

        Args:
            body: Open WebUI conversation with the assistant response.
            __user__: Optional Open WebUI user record.
            __metadata__: Request metadata shared with `inlet`.

        Returns:
            The response with literal source blocks, model commentary, and graph cards.

        Maintain:
            Only commentary citations are rewritten; repeated calls preserve source text exactly.

        Raises:
            No errors are raised for missing metadata or assistant messages.
        """
        metadata = __metadata__ if isinstance(__metadata__, dict) else body.get("metadata") or {}
        report = metadata.get("_arxiv_retrieval_report") if isinstance(metadata, dict) else None
        messages = body.get("messages")
        if not report or not isinstance(messages, list):
            return body
        assistant = next(
            (message for message in reversed(messages)
             if isinstance(message, dict) and message.get("role") == "assistant"),
            None,
        )
        if assistant is None:
            return body
        content = assistant.get("content")
        text = self._text(content)
        # A second outlet must not rewrite citation-like strings inside source passages.
        if "<!-- arxiv-retrieved-context -->" not in text:
            target, field = assistant, "content"
            if isinstance(content, list):
                # One text block keeps source/context ordering stable for multipart output;
                # retain images and other nontext blocks without model-dependent transcription.
                target = next((part for part in content if isinstance(part, dict) and part.get("type") == "text"),
                              {"type": "text", "text": ""})
                assistant["content"] = [target, *(part for part in content
                                                 if not isinstance(part, dict) or part.get("type") != "text")]
                field = "text"
            notice = self._rejection_notice(report)
            visible = self._render_visible_retrieval(report)
            if notice:
                rendered = notice + visible
            else:
                commentary = self._rewrite_citations(str(report.get("final") or ""), report)
                rendered = visible + "\n\n### Model commentary\n\n" + commentary
            target[field] = rendered + self._render_graph_cards(report)

        output = assistant.get("output")
        if isinstance(output, list):
            # The current OWUI client renders output_text, not the legacy content string.
            # Its outlet middleware persists both, so update the same final answer in each.
            rendered = self._text(assistant.get("content"))
            final_message = next((item for item in reversed(output) if isinstance(item, dict)
                                  and item.get("type") == "message" and item.get("role", "assistant") == "assistant"), None)
            if final_message is None:
                final_message = {"type": "message", "role": "assistant", "status": "completed", "content": []}
                output.append(final_message)
            parts = final_message.get("content") or []
            final_message["content"] = [{"type": "output_text", "text": rendered},
                                        *(part for part in parts if not isinstance(part, dict)
                                          or part.get("type") not in {"output_text", "text"})]
        return body
