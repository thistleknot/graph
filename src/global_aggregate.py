"""Global aggregate QA lexical foundation.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| SOURCE | Stream one pinned section build and group main/method rows as canonical papers. | [mandatory] | R1 |
| TOKENIZE | Union t3 folded tokens per paper without crossing section boundaries. | [mandatory] | R2 |
| BUILD | Count exact paper document frequency per primary community; score routing terms separately. | [mandatory] | R3-R5 |
| PERSIST | Write build-scoped rows transactionally and publish the complete marker last. | [mandatory] | R6 |
| TALLY | Sum full term DF only after source-build and tokenizer validation. | [mandatory] | R7 |
| ROUTE | Rank lexical communities from signatures and widen near-cutoff ties. | [mandatory] | R11 |
| SAMPLE | Pilot every scoped stratum, then draw bounded uniform samples without replacement until stop. | [mandatory] | R9 R12 R15 R16 R22 |
| ALLOCATE | Use matched-eval Neyman reallocation with defensive floor; caller may disable as control. | [opt, promoted on] | R27 |
| ESTIMATE | Bound yes/no/unsure prevalence by stratum and expand each endpoint by finite-population error. | [mandatory] | R9-R10 |
| CLASSIFY | Sample without replacement and validate one fixed JSON yes/no/unsure response per paper. | [mandatory] | R12-R14 |
| JOB CALL | Execute classifier requests through an optional durable logical-call ledger. | [opt] | R25 |
| EVAL | Verify deterministic paper hashes, assignments, lexical gold counts, and SQLite checkpoints. | [contract] | R8 R21 |

EARS guards
R1 A canonical paper ID removes only terminal `_methods`; its primary community is the mode of its section communities, ties to the lowest ID.
R2 Tokens come from entity_derive.tokenize_text t3; each canonical paper contributes a token at most once, and source section boundaries are preserved.
R3 Full per-community DF contains every observed token; routing signatures are separately truncated and never establish exact term presence.
R4 A missing DF row is zero only after a complete index matching source build and tokenizer version is found.
R5 Routing keyness is weighted log-odds over document presence with an informative corpus prior; it never changes exact tallies.
R6 A failed build rolls back all QA rows; complete becomes true only after persisted counts and paper assignments reconcile.
R7 A single-token lexical tally is exact only for the matching complete index; unknown tokenizer tokens and mismatched builds fail explicitly.
R8 The eval verifier reproduces the artifact's deterministic paper assignment, source-text hash, and ten lexical gold totals.
R9 The semantic estimator reports a lower total from yes/n and an upper total from (yes+unsure)/n, weighted by stratum population.
R10 Each endpoint interval uses its own finite-population variance; non-census 0/1 samples use a smoothed proportion so variance is nonzero.
R11 Lexical route scores rank communities only; tie widening never narrows the population of a global tally.
R12 Pilot and tally samples use uniform random sampling without replacement within each community.
R13 Predicate replies are JSON objects with exactly one label in {yes,no,unsure}; malformed replies fail closed.
R14 Truncated paper text is marked; without positive evidence, an incomplete input must classify as unsure, not no.
R21 Eval checkpoints key exact counts by source build, t3 version, and eval-artifact digest; source IDs, hashes, and primary communities are rechecked on cache hits.
R22 Semantic labels are checkpointed by source, predicate, exact classifier input, truncation state, and classifier version; cache hits do not consume model-call budget.
R23 Reported semantic bounds and stopping use hypergeometric PPR confidence sequences with alpha split across communities and both binary endpoints; normal intervals are fixed-size diagnostics only.
R24 A validated label commits before the next call; frozen sample budgets and per-community draw orders are independent of cache warmth.
R26 Each tally result reports semantic-label cache hits separately from classifier requests; API job results report provider attempts and durable response reuses separately.
R27 Neyman reallocation MUST default on because the matched 10-question, 100-seed eval lowered mean absolute error at equal observed interval coverage and call count; callers MAY disable it for a matched control.
R25 When a durable job call executor is supplied, each selected paper uses its stable ID and exact input digest as the call key; completed responses are reused and uncertain responses fail visibly.

CLOSED
- PostgreSQL owns these count tables because the section source rows already live there; no performance advantage is claimed over a graph database.
- English tsvector and existing JSONB payloads are not lexical QA sources because eval gold uses entity_derive t3.
- Top-k routing rows are not reused as the exact count store.
- OpenRouter calls reuse interpret._via_openrouter with one attempt; no second HTTP client or retry path is added.

Preconditions: psycopg 3 can connect to the graph database; the requested section build exists; the t3 tokenizer is importable.
Failure modes: missing build, incomplete marker, tokenization mismatch, unsorted streamed paper rows, count invariant violation, or database error aborts the operation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
from typing import Callable, Iterable, Iterator, Mapping

from entity_derive import TOKENIZER_VERSION, tokenize_text
import interpret

ROOT = Path(__file__).resolve().parents[1]
DDL_PATH = ROOT / "sql" / "003_global_aggregate.sql"
METHODS_SUFFIX = "_methods"
DEFAULT_SIGNATURE_TOP_K = 100
DEFAULT_PRIOR_STRENGTH = 1.0
EVAL_CACHE_PATH = ROOT / ".tmp" / "global_aggregate_eval.sqlite"
PREDICATE_PROMPT_VERSION = "predicate-json-v1"

EVAL_CACHE_DDL = (
    "CREATE TABLE IF NOT EXISTS eval_cache_meta (cache_key TEXT PRIMARY KEY, source_build_id INTEGER NOT NULL, "
    "tokenizer_version TEXT NOT NULL, artifact_sha256 TEXT NOT NULL, complete INTEGER NOT NULL DEFAULT 0, "
    "n_docs INTEGER NOT NULL, n_communities INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS eval_cache_community (cache_key TEXT NOT NULL, community_id INTEGER NOT NULL, "
    "n_docs INTEGER NOT NULL, PRIMARY KEY (cache_key, community_id))",
    "CREATE TABLE IF NOT EXISTS eval_cache_df (cache_key TEXT NOT NULL, community_id INTEGER NOT NULL, "
    "term TEXT NOT NULL, df INTEGER NOT NULL, PRIMARY KEY (cache_key, community_id, term))",
    "CREATE TABLE IF NOT EXISTS eval_cache_global_df (cache_key TEXT NOT NULL, term TEXT NOT NULL, df INTEGER NOT NULL, "
    "PRIMARY KEY (cache_key, term))",
    "CREATE TABLE IF NOT EXISTS eval_cache_paper (cache_key TEXT NOT NULL, paper_id TEXT NOT NULL, "
    "community_id INTEGER NOT NULL, text_sha256 TEXT NOT NULL, PRIMARY KEY (cache_key, paper_id))",
)

DEFAULT_TALLY_CACHE_PATH = ROOT / ".tmp" / "global_aggregate_tally.sqlite"
TALLY_CACHE_DDL = "CREATE TABLE IF NOT EXISTS semantic_label_cache (source_build_id INTEGER NOT NULL, " \
    "paper_id TEXT NOT NULL, predicate_sha256 TEXT NOT NULL, text_sha256 TEXT NOT NULL, " \
    "classifier_version TEXT NOT NULL, truncated INTEGER NOT NULL, label TEXT NOT NULL, " \
    "PRIMARY KEY (source_build_id,paper_id,predicate_sha256,text_sha256,classifier_version,truncated))"

SOURCE_QUERY = """
WITH community_counts AS (
    SELECT CASE WHEN right(doc_id, 8) = '_methods' THEN left(doc_id, length(doc_id) - 8) ELSE doc_id END AS paper_id,
           community AS community_id, count(*) AS section_count
    FROM sect_node
    WHERE build_id = %s
    GROUP BY 1, 2
), primary_community AS (
    SELECT DISTINCT ON (paper_id) paper_id, community_id
    FROM community_counts
    ORDER BY paper_id, section_count DESC, community_id ASC
)
SELECT p.paper_id, p.community_id, n.doc_id, n.section_idx, n.text
FROM primary_community p
JOIN sect_node n ON n.build_id = %s
    AND (CASE WHEN right(n.doc_id, 8) = '_methods' THEN left(n.doc_id, length(n.doc_id) - 8) ELSE n.doc_id END) = p.paper_id
WHERE (%s::text[] IS NULL OR p.paper_id = ANY(%s::text[]))
ORDER BY p.paper_id, n.doc_id, n.section_idx
"""


@dataclass(frozen=True)
class CanonicalPaper:
    """One canonical paper's deterministic community, token set, and source digest.

    Require: rows for this paper are complete and ordered by source doc ID and section index.
    Guarantee: paper identity, modal community, t3 token set, and digest match the source rows.
    Maintain: source labels are not rewritten; only the aggregate QA assignment is selected.
    Assert: at least one section row contributes to the paper.
    """

    paper_id: str
    community_id: int
    tokens: frozenset[str]
    text_sha256: str


@dataclass(frozen=True)
class CommunityTermIndex:
    """Exact document-frequency data plus the separate routing signature.

    Require: input papers are canonical and each belongs to one primary community.
    Guarantee: df counts papers, not section or token occurrences.
    Maintain: signature truncation never changes the full df mappings.
    Assert: community sizes sum to the number of indexed canonical papers.
    """

    n_docs: Mapping[int, int]
    df: Mapping[int, Mapping[str, int]]
    global_df: Mapping[str, int]
    signature: Mapping[int, tuple[tuple[str, float, int], ...]]
    paper_community: Mapping[str, int]
    paper_sha256: Mapping[str, str]


@dataclass(frozen=True)
class StratifiedEstimate:
    """Conservative semantic aggregate and per-stratum audit values.

    Require: strata passed validation by stratified_estimate.
    Guarantee: interval covers sampling uncertainty for both possible-label endpoints.
    Maintain: unsure labels remain in the upper endpoint and are never counted as no.
    Assert: point estimate is the midpoint of raw lower and upper totals.
    """

    estimate: float
    interval: tuple[float, float]
    variance_lower: float
    variance_upper: float
    audit: tuple[dict, ...]


@dataclass(frozen=True)
class SemanticTallyConfig:
    """Bounded sampling and checkpoint settings for one semantic tally.

    Require: budgets and batch sizes are positive and target_halfwidth is relative.
    Guarantee: Sampling and classification stay within the configured request budget.
    Maintain: R27-promoted Neyman reallocation is enabled by default; callers can disable it for matched control.
    Assert: Cache keys include the caller's classifier version.
    """

    budget: int = 2000
    pilot_budget: int = 1000
    pilot_per_stratum_cap: int = 30
    batch_size: int = 200
    target_halfwidth: float = 0.02
    z: float = 1.96
    use_reallocation: bool = True
    defensive_floor: float = 0.10
    seed: int = 0
    max_chars: int = 60000
    classifier_version: str = interpret.OPENROUTER_MODEL + ":" + PREDICATE_PROMPT_VERSION
    cache_path: str | Path | None = DEFAULT_TALLY_CACHE_PATH
    sample_budget: int | None = None
    alpha: float = 0.05


@dataclass(frozen=True)
class SemanticTallyResult:
    """Auditable estimate and uncertainty for one semantic predicate.

    Require: Every included paper was selected by SRS without replacement.
    Guarantee: The result records uncertainty, call count, seed, and stratum outcomes.
    Maintain: Unsure labels remain separate from yes and no counts.
    Assert: mode is estimated and calls do not exceed the requested budget.
    """

    estimate: float
    interval: tuple[float, float]
    converged: bool
    calls: int
    seed: int
    audit: tuple[dict, ...]
    mode: str = "estimated"
    interval_method: str = "hypergeometric_ppr"
    confidence_level: float = 0.95
    stop_reason: str = "budget"
    cache_hits: int = 0


def canonical_paper_id(doc_id: str) -> str:
    """Return the source ID with only a terminal methods suffix removed.

    Args:
        doc_id: Source record ID from sect_node.
    Returns:
        Canonical paper ID.
    Raises:
        ValueError: If the source ID is empty.
    Require: doc_id is a stored source identifier.
    Guarantee: no prefix or non-terminal substring is rewritten.
    Maintain: IDs not ending in `_methods` are unchanged.
    Assert: result is non-empty.
    """
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("doc_id must be a non-empty string")
    return doc_id[:-len(METHODS_SUFFIX)] if doc_id.endswith(METHODS_SUFFIX) else doc_id


def iter_canonical_papers(rows: Iterable[Mapping]) -> Iterator[CanonicalPaper]:
    """Collapse ordered section rows into canonical paper records.

    Args:
        rows: Mapping rows with paper_id, community_id, doc_id, section_idx, and text.
    Yields:
        One paper with one modal community and a unique t3 token set.
    Raises:
        ValueError: If rows are out of paper/source order or a field is invalid.
    Require: Rows are ordered by paper_id, doc_id, section_idx as SOURCE_QUERY emits them.
    Guarantee: Methods sections join the main paper before assignment and token counting.
    Maintain: A token repeated in multiple sections contributes once to that paper's DF.
    Assert: Every yielded paper has at least one source section.
    """
    current_id = None
    prior_key = None
    community_counts: Counter[int] = Counter()
    tokens: set[str] = set()
    digest = None

    def emit():
        if current_id is None:
            return None
        if not community_counts:
            raise ValueError("canonical paper has no community rows")
        primary = min(community_counts, key=lambda cid: (-community_counts[cid], cid))
        return CanonicalPaper(current_id, primary, frozenset(tokens), digest.hexdigest())

    for row in rows:
        paper_id = row["paper_id"]
        doc_id = row["doc_id"]
        section_idx = int(row["section_idx"])
        key = (paper_id, doc_id, section_idx)
        if prior_key is not None and key < prior_key:
            raise ValueError("section rows must be ordered by paper_id, doc_id, section_idx")
        prior_key = key
        if paper_id != current_id:
            completed = emit()
            if completed is not None:
                yield completed
            current_id = paper_id
            community_counts = Counter()
            tokens = set()
            digest = hashlib.sha256()
        if canonical_paper_id(doc_id) != paper_id:
            raise ValueError("source doc_id does not match canonical paper_id")
        community_counts[int(row["community_id"])] += 1
        text = row["text"]
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(section_idx).encode("ascii"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        for _kind, segment_tokens in tokenize_text(text):
            tokens.update(folded for _surface, folded, _shape, _initial in segment_tokens)
    completed = emit()
    if completed is not None:
        yield completed


def _keyness_scores(n_docs: Mapping[int, int], df: Mapping[int, Mapping[str, int]], global_df: Mapping[str, int],
                    top_k: int, prior_strength: float) -> dict[int, tuple[tuple[str, float, int], ...]]:
    """Rank community-enriched terms by informative-prior weighted log-odds.

    Args:
        n_docs: Canonical paper count per community.
        df: Exact paper-presence counts per community and term.
        global_df: Corpus-wide paper-presence counts per term.
        top_k: Maximum routing terms retained per community.
        prior_strength: Positive mass for the informative Dirichlet prior.
    Returns:
        Positive `(term, z_score, df)` rows, sorted by score then term.
    Raises:
        ValueError: If top_k or prior_strength is invalid.
    Require: DF values are bounded by their community and corpus sizes.
    Guarantee: Signature rows rank over-represented terms and do not alter exact DF.
    Maintain: Every term uses its corpus-wide DF as the informative prior base.
    Assert: No returned keyness score is non-finite or non-positive.
    """
    if top_k < 1 or not math.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("top_k and prior_strength must be positive")
    total_docs = sum(n_docs.values())
    total_presence = sum(global_df.values())
    vocab_size = len(global_df)
    if total_docs <= 0 or total_presence <= 0 or vocab_size <= 0:
        raise ValueError("cannot score an empty index")
    prior_denominator = total_presence + 0.5 * (vocab_size + 1)
    scored: dict[int, tuple[tuple[str, float, int], ...]] = {}
    for community_id, community_df in df.items():
        community_n = n_docs[community_id]
        other_n = total_docs - community_n
        terms: list[tuple[str, float, int]] = []
        for term, count in community_df.items():
            corpus_count = global_df[term]
            other_count = corpus_count - count
            alpha_term = prior_strength * (corpus_count + 0.5) / prior_denominator
            alpha_other = prior_strength - alpha_term
            delta = math.log((count + alpha_term) / (community_n - count + alpha_other))
            if other_n:
                delta -= math.log((other_count + alpha_term) / (other_n - other_count + alpha_other))
                variance = (1.0 / (count + alpha_term)
                            + 1.0 / (community_n - count + alpha_other)
                            + 1.0 / (other_count + alpha_term)
                            + 1.0 / (other_n - other_count + alpha_other))
                score = delta / math.sqrt(variance)
            else:
                score = 0.0
            if math.isfinite(score) and score > 0:
                terms.append((term, score, count))
        terms.sort(key=lambda item: (-item[1], item[0]))
        scored[community_id] = tuple(terms[:top_k])
    return scored


def signature_variant(index: CommunityTermIndex, top_k: int,
                      prior_strength: float = DEFAULT_PRIOR_STRENGTH) -> CommunityTermIndex:
    """Rebuild only the routing signature from an unchanged exact DF index.

    Args:
        index: Complete exact community DF index.
        top_k: Maximum positive keyness terms retained per community.
        prior_strength: Informative-prior mass used by the keyness score.
    Returns:
        Index with identical source counts and a signature truncated to top_k.
    Raises:
        ValueError: If top_k or prior_strength is invalid.
    Require: Exact per-community and global DF are complete.
    Guarantee: Signature-width comparisons reuse identical source data and labels.
    Maintain: Paper assignments and exact tallies are unchanged.
    Assert: All non-signature mappings are the same objects as the input index.
    """
    signature = _keyness_scores(index.n_docs, index.df, index.global_df, top_k, prior_strength)
    return CommunityTermIndex(index.n_docs, index.df, index.global_df, signature,
                              index.paper_community, index.paper_sha256)


def eval_cache_identity(eval_path: str | Path, source_build_id: int) -> tuple[str, str]:
    """Derive the exact SQLite checkpoint key for an eval artifact.

    Args:
        eval_path: JSONL artifact whose labels and sample membership are frozen.
        source_build_id: Source section build represented by the artifact.
    Returns:
        `(cache_key, artifact_sha256)`.
    Raises:
        OSError: If the eval artifact cannot be read.
    Require: source_build_id is the manifest's pinned build.
    Guarantee: Any source build, tokenizer, or artifact-byte change produces a different key.
    Maintain: Cache identity includes the current t3 tokenizer version.
    Assert: Key derives only from stable identifiers and artifact bytes.
    """
    artifact_sha256 = hashlib.sha256(Path(eval_path).read_bytes()).hexdigest()
    payload = "%d\0%s\0%s" % (source_build_id, TOKENIZER_VERSION, artifact_sha256)
    return hashlib.sha256(payload.encode("ascii")).hexdigest(), artifact_sha256


def save_eval_index_cache(cache_path: str | Path, cache_key: str, source_build_id: int,
                          artifact_sha256: str, index: CommunityTermIndex) -> None:
    """Checkpoint verified exact eval counts in SQLite without storing source text.

    Args:
        cache_path: SQLite checkpoint path.
        cache_key: Key from eval_cache_identity.
        source_build_id: Pinned source section build.
        artifact_sha256: Eval artifact digest from eval_cache_identity.
        index: Verified in-memory eval index.
    Returns:
        None.
    Raises:
        sqlite3.Error: Propagates write failures; no incomplete checkpoint is committed.
    Require: index has matching source assignments, hashes, and exact DF.
    Guarantee: All tables and the complete marker commit atomically.
    Maintain: Checkpoint contains IDs, hashes, counts, and no paper text.
    Assert: Stored community sizes sum to n_docs.
    """
    if sum(index.n_docs.values()) != len(index.paper_community):
        raise ValueError("cannot checkpoint inconsistent paper/community counts")
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        for statement in EVAL_CACHE_DDL:
            db.execute(statement)
        db.execute("BEGIN IMMEDIATE")
        db.execute("INSERT INTO eval_cache_meta VALUES (?,?,?,?,0,?,?)",
                   (cache_key, source_build_id, TOKENIZER_VERSION, artifact_sha256,
                    len(index.paper_community), len(index.n_docs)))
        db.executemany("INSERT INTO eval_cache_community VALUES (?,?,?)",
                       [(cache_key, cid, n_docs) for cid, n_docs in sorted(index.n_docs.items())])
        db.executemany("INSERT INTO eval_cache_df VALUES (?,?,?,?)",
                       [(cache_key, cid, term, df) for cid, terms in sorted(index.df.items())
                        for term, df in sorted(terms.items())])
        db.executemany("INSERT INTO eval_cache_global_df VALUES (?,?,?)",
                       [(cache_key, term, df) for term, df in sorted(index.global_df.items())])
        db.executemany("INSERT INTO eval_cache_paper VALUES (?,?,?,?)",
                       [(cache_key, paper_id, index.paper_community[paper_id], index.paper_sha256[paper_id])
                        for paper_id in sorted(index.paper_community)])
        db.execute("UPDATE eval_cache_meta SET complete=1 WHERE cache_key=?", (cache_key,))


def load_eval_index_cache(cache_path: str | Path, cache_key: str,
                          source_build_id: int, artifact_sha256: str) -> CommunityTermIndex | None:
    """Load one exact eval checkpoint only when all identity fields match.

    Args:
        cache_path: SQLite checkpoint path.
        cache_key: Key from eval_cache_identity.
        source_build_id: Expected section build.
        artifact_sha256: Expected JSONL artifact digest.
    Returns:
        Complete CommunityTermIndex or None when no matching checkpoint exists.
    Raises:
        RuntimeError: If a matching key exists but is incomplete or inconsistent.
        sqlite3.Error: Propagates corruption and I/O failures.
    Require: The checkpoint was written by save_eval_index_cache.
    Guarantee: A mismatched key is a cache miss, never a partial result.
    Maintain: Returned index includes every exact DF row and paper digest.
    Assert: Cached community DF sums reconcile to cached global DF.
    """
    path = Path(cache_path)
    if not path.exists():
        return None
    with sqlite3.connect(path) as db:
        meta = db.execute("SELECT source_build_id,tokenizer_version,artifact_sha256,complete,n_docs,n_communities "
                          "FROM eval_cache_meta WHERE cache_key=?", (cache_key,)).fetchone()
        if meta is None:
            return None
        if meta[:3] != (source_build_id, TOKENIZER_VERSION, artifact_sha256) or not meta[3]:
            raise RuntimeError("matching eval checkpoint metadata is incomplete or inconsistent")
        n_docs_rows = db.execute("SELECT community_id,n_docs FROM eval_cache_community WHERE cache_key=? ORDER BY community_id",
                                 (cache_key,)).fetchall()
        df_rows = db.execute("SELECT community_id,term,df FROM eval_cache_df WHERE cache_key=? ORDER BY community_id,term",
                             (cache_key,)).fetchall()
        global_rows = db.execute("SELECT term,df FROM eval_cache_global_df WHERE cache_key=? ORDER BY term",
                                 (cache_key,)).fetchall()
        paper_rows = db.execute("SELECT paper_id,community_id,text_sha256 FROM eval_cache_paper WHERE cache_key=? ORDER BY paper_id",
                                (cache_key,)).fetchall()
    n_by_community = dict(n_docs_rows)
    if len(paper_rows) != meta[4] or len(n_by_community) != meta[5]:
        raise RuntimeError("eval checkpoint cardinality does not match its complete marker")
    per_community: dict[int, dict[str, int]] = {cid: {} for cid in n_by_community}
    for cid, term, df in df_rows:
        if cid not in per_community:
            raise RuntimeError("eval checkpoint DF references an unknown community")
        per_community[cid][term] = df
    global_df = dict(global_rows)
    for term, count in global_df.items():
        if sum(terms.get(term, 0) for terms in per_community.values()) != count:
            raise RuntimeError("eval checkpoint community DF does not match global DF")
    paper_community = {paper_id: cid for paper_id, cid, _digest in paper_rows}
    paper_sha256 = {paper_id: digest for paper_id, _cid, digest in paper_rows}
    signature = _keyness_scores(n_by_community, per_community, global_df, DEFAULT_SIGNATURE_TOP_K,
                                DEFAULT_PRIOR_STRENGTH)
    return CommunityTermIndex(n_by_community, per_community, global_df, signature,
                              paper_community, paper_sha256)


def verify_eval_source_identity(conn, source_build_id: int, expected: Mapping[str, Mapping]) -> None:
    """Verify cached eval identities against current source rows without tokenization.

    Args:
        conn: Readable psycopg connection to the pinned source build.
        source_build_id: Section build represented by the evaluation artifact.
        expected: Evaluation paper rows keyed by canonical paper ID.
    Returns:
        None.
    Raises:
        RuntimeError: If source IDs, modal communities, or section-text digests differ.
    Require: The evaluation artifact has unique canonical paper IDs.
    Guarantee: A cache hit is reused only for the same source identity and text.
    Maintain: This verification does not tokenize text or alter source rows.
    Assert: Every expected paper is observed exactly in its expected community.
    """
    found: dict[str, tuple[int, str]] = {}
    for row in iter_source_rows(conn, source_build_id, set(expected)):
        paper_id = row["paper_id"]
        if paper_id not in found:
            found[paper_id] = (row["community_id"], hashlib.sha256())
        community_id, digest = found[paper_id]
        if community_id != row["community_id"]:
            raise RuntimeError("source primary community changed within " + paper_id)
        digest.update(row["doc_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(row["section_idx"])).encode("ascii"))
        digest.update(b"\0")
        digest.update(row["text"].encode("utf-8"))
        digest.update(b"\0")
    if set(found) != set(expected):
        raise RuntimeError("eval sample paper IDs differ from source build")
    for paper_id, (community_id, digest) in found.items():
        manifest_row = expected[paper_id]
        if community_id != manifest_row["primary_community"]:
            raise RuntimeError("eval primary community mismatch for " + paper_id)
        if digest.hexdigest() != manifest_row["text_sha256"]:
            raise RuntimeError("eval source text hash mismatch for " + paper_id)


def build_index(rows: Iterable[Mapping], top_k: int = DEFAULT_SIGNATURE_TOP_K,
                prior_strength: float = DEFAULT_PRIOR_STRENGTH) -> CommunityTermIndex:
    """Build exact per-community DF and a separate routing signature.

    Args:
        rows: Ordered section rows from one source build.
        top_k: Maximum routing signature terms per community.
        prior_strength: Positive informative-prior mass for keyness scoring.
    Returns:
        CommunityTermIndex with exact full DF and top-k signature.
    Raises:
        ValueError: If grouping, tokenization, or count invariants fail.
    Require: Rows use one source build and SOURCE_QUERY ordering.
    Guarantee: Every canonical paper is counted once in one community.
    Maintain: Full DF includes all observed tokens regardless of top_k.
    Assert: Sum of community sizes equals canonical papers consumed.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    n_docs: Counter[int] = Counter()
    per_community: dict[int, Counter[str]] = defaultdict(Counter)
    corpus_df: Counter[str] = Counter()
    paper_community: dict[str, int] = {}
    paper_sha256: dict[str, str] = {}
    for paper in iter_canonical_papers(rows):
        if paper.paper_id in paper_community:
            raise ValueError("canonical paper appeared in disjoint input runs")
        paper_community[paper.paper_id] = paper.community_id
        paper_sha256[paper.paper_id] = paper.text_sha256
        n_docs[paper.community_id] += 1
        per_community[paper.community_id].update(paper.tokens)
        corpus_df.update(paper.tokens)
    if not paper_community or sum(n_docs.values()) != len(paper_community):
        raise ValueError("index has no papers or community sizes do not reconcile")
    for community_id, counts in per_community.items():
        if any(count <= 0 or count > n_docs[community_id] for count in counts.values()):
            raise ValueError("term DF is outside its community population")
    for term, corpus_count in corpus_df.items():
        if sum(per_community[cid].get(term, 0) for cid in per_community) != corpus_count:
            raise ValueError("community DF does not reconcile to corpus DF")
    signatures = _keyness_scores(n_docs, per_community, corpus_df, top_k, prior_strength)
    return CommunityTermIndex(dict(n_docs), {cid: dict(counts) for cid, counts in per_community.items()},
                              dict(corpus_df), signatures, paper_community, paper_sha256)


def normalize_single_term(term: str) -> str:
    """Normalize one lexical query using the eval tokenizer.

    Args:
        term: User-supplied lexical term.
    Returns:
        One t3 case-folded token.
    Raises:
        ValueError: If the term produces zero or multiple tokens.
    Require: Caller intends an exact single-token predicate.
    Guarantee: Query normalization is the same t3 operation used by the index.
    Maintain: Digit-bearing identifiers are preserved by t3.
    Assert: Exactly one normalized token is returned.
    """
    normalized = [folded for _kind, tokens in tokenize_text(term)
                  for _surface, folded, _shape, _initial in tokens]
    if len(normalized) != 1:
        raise ValueError("exact lexical tally requires one t3 token")
    return normalized[0]


def exact_tally(index: CommunityTermIndex, term: str) -> int:
    """Return the exact corpus document count for one lexical token.

    Args:
        index: Complete in-memory index for one source build.
        term: One lexical term.
    Returns:
        Exact number of canonical papers containing the token.
    Raises:
        ValueError: If the predicate is not a single t3 token.
    Require: Index construction completed successfully.
    Guarantee: Count is the sum of full per-community DF, never signature DF.
    Maintain: An absent token returns zero only for a complete index instance.
    Assert: Tally is bounded by indexed paper count.
    """
    token = normalize_single_term(term)
    total = sum(index.df[cid].get(token, 0) for cid in index.n_docs)
    if not 0 <= total <= sum(index.n_docs.values()):
        raise ValueError("exact tally exceeds the indexed population")
    return total


def stratified_estimate(strata: Mapping[int, Mapping[str, int]], z: float = 1.96) -> StratifiedEstimate:
    """Estimate a corpus total with separate lower and upper semantic labels.

    Args:
        strata: Mapping of community ID to `N`, `n`, `yes`, and `unsure` counts.
        z: Normal critical value for the requested two-sided interval.
    Returns:
        Midpoint estimate, conservative interval, endpoint variances, and per-stratum audit.
    Raises:
        ValueError: If a stratum is empty, malformed, or has an invalid sample size.
    Require: Each stratum is an SRS without replacement and labels are yes/no/unsure.
    Guarantee: The interval expands weighted yes and yes-or-unsure totals by finite-population sampling error.
    Maintain: Unsure is visible and contributes only to the upper endpoint.
    Assert: Endpoint variances are positive for non-census all-zero/all-one samples.
    """
    if not strata or not math.isfinite(z) or z <= 0:
        raise ValueError("strata and positive finite z are required")
    lower_total = 0.0
    upper_total = 0.0
    variance_lower = 0.0
    variance_upper = 0.0
    population = 0
    audit = []
    for community_id, sample in strata.items():
        try:
            N, n, yes, unsure = (int(sample[key]) for key in ("N", "n", "yes", "unsure"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each stratum needs integer N, n, yes, and unsure") from exc
        if N < 1 or n < 1 or n > N or yes < 0 or unsure < 0 or yes + unsure > n:
            raise ValueError("stratum counts are outside their valid bounds")
        p_lower = yes / n
        p_upper = (yes + unsure) / n
        fpc = 1.0 - n / N
        finite_sample = n / (n - 1) if n > 1 else 1.0
        smooth_lower = (yes + 0.5) / (n + 1.0)
        smooth_upper = (yes + unsure + 0.5) / (n + 1.0)
        variance_lower += N * N * smooth_lower * (1.0 - smooth_lower) * finite_sample / n * fpc
        variance_upper += N * N * smooth_upper * (1.0 - smooth_upper) * finite_sample / n * fpc
        lower_total += N * p_lower
        upper_total += N * p_upper
        population += N
        audit.append({"community_id": community_id, "N": N, "n": n, "p_yes": p_lower,
                      "p_possible": p_upper, "unsure": unsure})
    point = (lower_total + upper_total) / 2.0
    interval = (max(0.0, lower_total - z * math.sqrt(variance_lower)),
                min(float(population), upper_total + z * math.sqrt(variance_upper)))
    return StratifiedEstimate(point, interval, variance_lower, variance_upper, tuple(audit))


def hypergeometric_bounds(N: int, n: int, successes: int, alpha: float) -> tuple[int, int]:
    """Bound a finite binary population with a hypergeometric PPR confidence sequence.

    Args:
        N: Fixed population size.
        n: Number observed uniformly without replacement.
        successes: Observed positive outcomes.
        alpha: Error probability for this endpoint and stratum.
    Returns:
        Smallest and largest surviving integer population totals.
    Raises:
        ValueError: If the observation counts or error allowance are invalid.
        RuntimeError: If numerical computation yields an empty confidence set.
    Require: Outcomes are fixed and draw order is uniform without replacement.
    Guarantee: Bounds remain valid at arbitrary stopping times via the PPR martingale.
    Maintain: The working prior is BetaBinomial(N,1,1), not a claim about data generation.
    Assert: A census collapses to the observed total.
    """
    if N < 1 or not 0 <= successes <= n <= N or not 0 < alpha < 1:
        raise ValueError("invalid hypergeometric observation or alpha")
    if n == N:
        return successes, successes
    log_mixture = math.lgamma(successes + 1) + math.lgamma(n - successes + 1) - math.lgamma(n + 2)
    threshold = -math.log(alpha)
    survivors = []
    for total in range(successes, N - n + successes + 1):
        log_likelihood = (math.lgamma(total + 1) - math.lgamma(total - successes + 1)
                          + math.lgamma(N - total + 1) - math.lgamma(N - total - n + successes + 1)
                          - math.lgamma(N + 1) + math.lgamma(N - n + 1))
        if log_mixture - log_likelihood < threshold:
            survivors.append(total)
    if not survivors:
        raise RuntimeError("empty hypergeometric PPR confidence set")
    return survivors[0], survivors[-1]


def sequential_estimate(strata: Mapping[int, Mapping[str, int]], alpha: float = 0.05,
                        prior_bounds: dict | None = None) -> StratifiedEstimate:
    """Combine simultaneous finite-population bounds for yes and possible outcomes.

    Args:
        strata: Community N/n/yes/unsure counts.
        alpha: Total error allowance across both endpoints and all communities.
        prior_bounds: Optional mutable running intersections from preceding looks.
    Returns:
        Existing midpoint estimate and audit with sequentially valid count bounds.
    Raises:
        ValueError: If counts or alpha are invalid.
        RuntimeError: If running confidence sets cease to intersect.
    Require: Each stratum follows an independent uniform permutation of its population.
    Guarantee: Union-bound allocation covers every stratum endpoint at every look.
    Maintain: Unsure widens the possible-label endpoint; classifier errors are not sampling error.
    Assert: Census bounds retain exactly the uncertainty due to unsure labels.
    """
    baseline = stratified_estimate(strata)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    bounds = prior_bounds if prior_bounds is not None else {}
    lower, upper = 0, 0
    endpoint_alpha = alpha / (2 * len(strata))
    for cid, sample in strata.items():
        for endpoint, successes in (("yes", sample["yes"]), ("possible", sample["yes"] + sample["unsure"])):
            current = hypergeometric_bounds(sample["N"], sample["n"], successes, endpoint_alpha)
            old = bounds.get((cid, endpoint), (0, sample["N"]))
            current = max(old[0], current[0]), min(old[1], current[1])
            if current[0] > current[1]:
                raise RuntimeError("sequential confidence sets have empty running intersection")
            bounds[(cid, endpoint)] = current
        lower += bounds[(cid, "yes")][0]
        upper += bounds[(cid, "possible")][1]
    return StratifiedEstimate(baseline.estimate, (float(lower), float(upper)),
                              baseline.variance_lower, baseline.variance_upper, baseline.audit)


def route_lexical(index: CommunityTermIndex, query: str, top_k: int = 5,
                  tie_margin: float = 0.1) -> dict:
    """Rank communities by matching signature keyness and include near-cutoff ties.

    Args:
        index: Exact community index with separate routing signatures.
        query: Lexical query text normalized using t3.
        top_k: Strict number of ranked communities before tie expansion.
        tie_margin: Maximum relative gap treated as a cutoff tie.
    Returns:
        Dict with all ranked scores, strict top-k, tie-expanded communities, and matched terms.
    Raises:
        ValueError: If top_k or tie_margin is outside its valid range.
    Require: Index signature scores are positive weighted log-odds keyness values.
    Guarantee: Communities sort by descending score then ascending community ID.
    Maintain: Ranking is informational and does not alter a tally's global population.
    Assert: A score tie at the cutoff is expanded when tie_margin is positive.
    """
    if top_k < 1 or not math.isfinite(tie_margin) or tie_margin < 0:
        raise ValueError("top_k must be positive and tie_margin non-negative")
    terms = {folded for _kind, tokens in tokenize_text(query)
             for _surface, folded, _shape, _initial in tokens}
    scores = {community_id: 0.0 for community_id in index.n_docs}
    matched = set()
    for community_id, signature in index.signature.items():
        by_term = {term: keyness for term, keyness, _df in signature}
        for term in terms.intersection(by_term):
            scores[community_id] += by_term[term]
            matched.add(term)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    strict = ranked[:min(top_k, len(ranked))]
    selected = list(strict)
    tie_expanded = False
    if top_k < len(ranked):
        cutoff = ranked[top_k - 1][1]
        next_score = ranked[top_k][1]
        relative_gap = (cutoff - next_score) / max(abs(cutoff), 1e-12)
        if relative_gap < tie_margin:
            floor = cutoff - tie_margin * max(abs(cutoff), 1e-12)
            selected = [row for row in ranked if row[1] >= floor]
            tie_expanded = len(selected) > len(strict)
    return {"ranked": ranked, "top_k": strict, "selected": selected,
            "tie_expanded": tie_expanded, "matched_terms": sorted(matched)}


def evaluate_lexical_routes(index: CommunityTermIndex, questions: Iterable[Mapping],
                            top_k: int = 5, tie_margin: float = 0.1) -> dict:
    """Measure lexical community routing against known positive eval strata.

    Args:
        index: Community index built from the eval slice.
        questions: Eval manifest questions; only lexical questions are measured.
        top_k: Strict routing width.
        tie_margin: Relative score gap that widens the selected set.
    Returns:
        Aggregate hit, positive-community recall, tie-frequency, and per-question measurements.
    Raises:
        ValueError: If a lexical question has no gold-positive community.
    Require: Each lexical question has one exact t3 alias and a known gold count.
    Guarantee: Metrics distinguish strict top-k from tie-expanded coverage.
    Maintain: This is an evaluation only; no tally scope is pruned.
    Assert: Every lexical question contributes exactly one measurement.
    """
    results = []
    for question in questions:
        if question.get("kind") != "lexical":
            continue
        term = normalize_single_term(question["aliases"][0])
        positives = {cid for cid in index.n_docs if index.df[cid].get(term, 0) > 0}
        if not positives:
            raise ValueError("lexical eval question has no positive community: " + question["id"])
        route = route_lexical(index, term, top_k, tie_margin)
        strict_communities = {cid for cid, _score in route["top_k"]}
        selected_communities = {cid for cid, _score in route["selected"]}
        results.append({"question_id": question["id"], "term": term,
                        "positive_communities": len(positives),
                        "top_k_hit": bool(strict_communities & positives),
                        "top_k_positive_recall": len(strict_communities & positives) / len(positives),
                        "selected_positive_recall": len(selected_communities & positives) / len(positives),
                        "tie_expanded": route["tie_expanded"],
                        "n_selected": len(route["selected"]),
                        "n_matched_terms": len(route["matched_terms"])})
    if not results:
        raise ValueError("evaluation set contains no lexical questions")
    n = len(results)
    return {"n_questions": n,
            "top_k_hit_rate": sum(row["top_k_hit"] for row in results) / n,
            "top_k_positive_community_recall": sum(row["top_k_positive_recall"] for row in results) / n,
            "selected_positive_community_recall": sum(row["selected_positive_recall"] for row in results) / n,
            "tie_frequency": sum(row["tie_expanded"] for row in results) / n,
            "questions": results}


def sample_paper_ids(paper_community: Mapping[str, int], community_id: int,
                     sample_n: int, seed: int) -> list[str]:
    """Draw a reproducible uniform sample without replacement inside one community.

    Args:
        paper_community: Canonical paper ID to primary community mapping.
        community_id: Stratum to sample.
        sample_n: Requested number of papers.
        seed: Local random seed for reproducibility.
    Returns:
        Sampled canonical IDs in reproducible draw order.
    Raises:
        ValueError: If sample_n is not positive or the community is empty.
    Require: paper_community is the complete population for the requested scope.
    Guarantee: Every selected paper has equal inclusion probability and appears once.
    Maintain: Sampling does not use query relevance, MMR, or labels.
    Assert: Result length is min(sample_n, stratum population).
    """
    if sample_n < 1:
        raise ValueError("sample_n must be positive")
    population = sorted(paper_id for paper_id, cid in paper_community.items() if cid == community_id)
    if not population:
        raise ValueError("cannot sample an empty community")
    return random.Random(seed).sample(population, min(sample_n, len(population)))


def allocate_uniform(capacity: Mapping[int, int], amount: int) -> dict[int, int]:
    """Spread an integer draw budget evenly across eligible strata.

    Args:
        capacity: Remaining selectable papers per community.
        amount: Maximum number of papers to allocate.
    Returns:
        Per-community allocations summing to no more than amount or total capacity.
    Raises:
        ValueError: If amount or any capacity is negative.
    Require: Community IDs and capacities describe the current unsampled pool.
    Guarantee: Lower IDs receive deterministic remainder units after equal rounds.
    Maintain: No community allocation exceeds its remaining population.
    Assert: Sum of allocations is bounded by amount.
    """
    if amount < 0 or any(value < 0 for value in capacity.values()):
        raise ValueError("allocation amount and capacities must be non-negative")
    allocation = {community_id: 0 for community_id in capacity}
    remaining = min(amount, sum(capacity.values()))
    while remaining:
        eligible = [cid for cid in sorted(capacity) if allocation[cid] < capacity[cid]]
        if not eligible:
            break
        rounds = max(1, remaining // len(eligible))
        progressed = 0
        for cid in eligible:
            take = min(rounds, capacity[cid] - allocation[cid], remaining)
            allocation[cid] += take
            remaining -= take
            progressed += take
            if not remaining:
                break
        if not progressed:
            break
    return allocation


def allocate_neyman(capacity: Mapping[int, int], weights: Mapping[int, float], amount: int,
                    defensive_floor: float) -> dict[int, int]:
    """Allocate a bounded draw batch by Neyman weights with an even defensive floor.

    Args:
        capacity: Remaining selectable papers per community.
        weights: Non-negative `N * smoothed_sd` allocation weights.
        amount: Maximum number of papers to allocate.
        defensive_floor: Fraction of amount reserved for even allocation.
    Returns:
        Per-community allocations no larger than capacity and amount.
    Raises:
        ValueError: If weights or defensive_floor are invalid.
    Require: Every capacity and weight is non-negative and community keys match.
    Guarantee: Every community gets an even floor before weighted allocation.
    Maintain: The result cannot exceed the batch budget or a stratum's unsampled pool.
    Assert: All allocations are integers and non-negative.
    """
    if set(capacity) != set(weights) or not 0.0 <= defensive_floor <= 1.0:
        raise ValueError("allocation keys must match and defensive_floor must be in [0,1]")
    if any(weight < 0 or not math.isfinite(weight) for weight in weights.values()):
        raise ValueError("Neyman weights must be finite and non-negative")
    total = min(amount, sum(capacity.values()))
    floor = allocate_uniform(capacity, int(total * defensive_floor))
    result = dict(floor)
    remaining = total - sum(result.values())
    while remaining:
        eligible = [cid for cid in sorted(capacity) if result[cid] < capacity[cid]]
        if not eligible:
            break
        active_weight = {cid: weights[cid] for cid in eligible}
        weight_total = sum(active_weight.values())
        if weight_total <= 0:
            extra = allocate_uniform({cid: capacity[cid] - result[cid] for cid in eligible}, remaining)
            for cid, value in extra.items():
                result[cid] += value
            break
        quotas = {cid: remaining * active_weight[cid] / weight_total for cid in eligible}
        assigned = 0
        for cid in eligible:
            take = min(capacity[cid] - result[cid], int(quotas[cid]))
            result[cid] += take
            assigned += take
        remaining -= assigned
        if not remaining:
            break
        # Largest-remainder assignment makes the integer budget exact and deterministic.
        remainder_order = sorted(eligible, key=lambda cid: (-(quotas[cid] % 1.0), cid))
        progressed = 0
        for cid in remainder_order:
            if result[cid] < capacity[cid] and remaining:
                result[cid] += 1
                remaining -= 1
                progressed += 1
        if not progressed:
            break
    return result


def _semantic_label_digest(paper: Mapping) -> str:
    """Hash the exact classifier input and truncation state for label-cache identity.

    Args:
        paper: Text-loader row containing text and truncated fields.
    Returns:
        SHA-256 hex digest.
    Raises:
        KeyError: If text or truncation metadata is missing.
    Require: paper is the exact payload passed to the classifier.
    Guarantee: Different text or truncation states use distinct cache keys.
    Maintain: Raw source text is hashed but never stored in SQLite.
    Assert: Digest is stable for identical classifier inputs.
    """
    payload = paper["text"].encode("utf-8") + b"\0" + (b"1" if paper["truncated"] else b"0")
    return hashlib.sha256(payload).hexdigest()


def semantic_tally(paper_community: Mapping[str, int], predicate: str,
                   text_loader: Callable[[set[str], int], Mapping[str, Mapping]],
                   classifier: Callable[[str, str, bool], Mapping],
                   config: SemanticTallyConfig, source_build_id: int,
                   scope_communities: Iterable[int] | None = None,
                   call_executor: Callable[[str, Callable[[], Mapping]], Mapping] | None = None) -> SemanticTallyResult:
    """Estimate one semantic predicate through a bounded, auditable stratified sample.

    Args:
        paper_community: Complete canonical-paper to primary-community population.
        predicate: Predicate supplied to the one-paper classifier.
        text_loader: Loads selected IDs with the requested per-paper character cap.
        classifier: Returns a validated mapping containing `label` for one paper.
        config: Budgets, confidence settings, seed, and optional allocation behavior.
        source_build_id: Immutable source build used to key cached model labels.
        scope_communities: Optional strata subset; global callers pass every community.
        call_executor: Optional durable job-call wrapper keyed by stable paper/input digest.
    Returns:
        SemanticTallyResult with estimate, interval, per-stratum audit, calls, and convergence.
    Raises:
        ValueError: If the population, scope, predicate, or configuration is invalid.
        RuntimeError: If a selected document is missing or a classifier label is invalid.
        Database errors: Propagate from the SQLite label checkpoint.
    Require: The population has one row per canonical paper and IDs are stable for source_build_id.
    Guarantee: Samples are uniform without replacement; actual classifier calls never exceed budget.
    Maintain: Global scope includes every community; cached labels match predicate, text, truncation, and classifier version.
    Assert: Every stratum is piloted before a result is estimated and every audit count reconciles.
    """
    if not predicate.strip() or not paper_community:
        raise ValueError("predicate and canonical-paper population must be non-empty")
    if (config.budget < 1 or config.pilot_budget < 1 or config.batch_size < 1 or
            config.pilot_per_stratum_cap < 1 or config.target_halfwidth <= 0 or
            not math.isfinite(config.target_halfwidth) or config.z <= 0 or
            not math.isfinite(config.z) or config.max_chars < 1 or
            not config.classifier_version.strip() or config.defensive_floor < 0 or
            config.defensive_floor > 1 or not 0 < config.alpha < 1):
        raise ValueError("semantic tally configuration is invalid")
    communities = sorted(set(paper_community.values()) if scope_communities is None else set(scope_communities))
    if not communities or any(cid not in paper_community.values() for cid in communities):
        raise ValueError("semantic tally scope must contain non-empty population strata")
    populations = {cid: sorted(pid for pid, community in paper_community.items() if community == cid)
                   for cid in communities}
    sample_budget = config.budget if config.sample_budget is None else config.sample_budget
    if sample_budget < len(communities):
        raise ValueError("sample budget must cover every stratum")
    if config.pilot_budget > config.budget:
        raise ValueError("pilot budget cannot exceed total classifier-call budget")
    if config.pilot_budget < len(communities):
        raise ValueError("pilot budget must cover at least one paper in every stratum")

    # Freeze one permutation per stratum so batching, caching and resume do not change draws.
    for cid, population in populations.items():
        random.Random(f"{config.seed}:{cid}").shuffle(population)
    capacity = {cid: len(populations[cid]) for cid in communities}
    pilot_cap = {cid: min(capacity[cid], config.pilot_per_stratum_cap) for cid in communities}
    pilot_size = allocate_uniform(pilot_cap, min(config.pilot_budget, sample_budget, sum(pilot_cap.values())))
    strata = {cid: {"N": capacity[cid], "n": 0, "yes": 0, "unsure": 0, "seen": set()}
              for cid in communities}
    predicate_sha256 = hashlib.sha256(predicate.strip().encode("utf-8")).hexdigest()
    calls = 0
    cache_hits = 0

    def classify_batch(draws: Mapping[int, list[str]]) -> None:
        """Load cached labels or classify and checkpoint one bounded sample batch.

        Require: draws contain IDs within the frozen sample order and loader returns exact inputs.
        Guarantee: each valid result is checkpointed before another classifier call begins.
        Maintain: optional job ledger keys bind paper identity to exact text and truncation digest.
        Assert: every returned label belongs to the yes/no/unsure set.
        """
        nonlocal calls, cache_hits
        ids = {paper_id for selected in draws.values() for paper_id in selected}
        documents = text_loader(ids, config.max_chars)
        if set(documents) != ids:
            raise RuntimeError("text loader did not return every selected canonical paper")
        labels: dict[str, str] = {}
        cache_path = config.cache_path
        cached: dict[str, str] = {}
        digests = {paper_id: _semantic_label_digest(documents[paper_id]) for paper_id in ids}
        if cache_path is not None:
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as db:
                db.execute(TALLY_CACHE_DDL)
                for paper_id in sorted(ids):
                    row = db.execute(
                        "SELECT label FROM semantic_label_cache WHERE source_build_id=? AND "
                        "predicate_sha256=? AND text_sha256=? AND classifier_version=? AND truncated=? AND paper_id=?",
                        (source_build_id, predicate_sha256, digests[paper_id], config.classifier_version,
                         int(bool(documents[paper_id]["truncated"])), paper_id),
                    ).fetchone()
                    if row is not None:
                        cached[paper_id] = row[0]
        for paper_id in sorted(ids):
            if paper_id in cached:
                cache_hits += 1
                if cached[paper_id] not in {"yes", "no", "unsure"}:
                    raise RuntimeError("semantic label checkpoint contains an invalid label")
                labels[paper_id] = cached[paper_id]
                continue
            if calls >= config.budget:
                raise RuntimeError("semantic tally exhausted its classifier budget within a batch")
            paper = documents[paper_id]
            calls += 1
            invoke = lambda: classifier(predicate, paper["text"], bool(paper["truncated"]))
            result = (call_executor(f"{paper_id}:{digests[paper_id]}", invoke)
                      if call_executor is not None else invoke())
            label = result.get("label")
            if label not in {"yes", "no", "unsure"}:
                raise RuntimeError("classifier returned an invalid semantic label")
            if paper["truncated"] and label == "no":
                label = "unsure"
            labels[paper_id] = label
            # Commit paid work immediately; a later model failure cannot erase this result.
            if cache_path is not None:
                with sqlite3.connect(cache_path) as db:
                    db.execute(TALLY_CACHE_DDL)
                    db.execute(
                        "INSERT INTO semantic_label_cache VALUES (?,?,?,?,?,?,?) "
                        "ON CONFLICT(source_build_id,paper_id,predicate_sha256,text_sha256,classifier_version,truncated) "
                        "DO UPDATE SET label=excluded.label",
                        (source_build_id, paper_id, predicate_sha256, digests[paper_id], config.classifier_version,
                         int(bool(documents[paper_id]["truncated"])), label),
                    )
        for community_id, selected in draws.items():
            stratum = strata[community_id]
            for paper_id in selected:
                label = labels[paper_id]
                stratum["n"] += 1
                stratum["yes"] += label == "yes"
                stratum["unsure"] += label == "unsure"
                stratum["seen"].add(paper_id)

    def draw_batch(allocations: Mapping[int, int]) -> dict[int, list[str]]:
        """Draw deterministic without-replacement IDs from each remaining stratum."""
        draws = {}
        for community_id, amount in allocations.items():
            remaining_ids = populations[community_id][strata[community_id]["n"]:]
            take = min(amount, len(remaining_ids))
            draws[community_id] = remaining_ids[:take]
        return {community_id: selected for community_id, selected in draws.items() if selected}

    classify_batch(draw_batch(pilot_size))
    confidence_bounds = {}
    stop_reason = "budget"
    while True:
        estimate = sequential_estimate(strata, config.alpha, confidence_bounds)
        halfwidth = (estimate.interval[1] - estimate.interval[0]) / 2.0
        if halfwidth <= config.target_halfwidth * max(estimate.estimate, 1.0):
            converged = True
            stop_reason = "target"
            break
        remaining_capacity = {cid: capacity[cid] - strata[cid]["n"] for cid in communities}
        sampled = sum(sample["n"] for sample in strata.values())
        if calls >= config.budget or sampled >= sample_budget or sum(remaining_capacity.values()) == 0:
            converged = False
            stop_reason = "census" if sum(remaining_capacity.values()) == 0 else "budget"
            break
        next_amount = min(config.batch_size, config.budget - calls, sample_budget - sampled,
                          sum(remaining_capacity.values()))
        if config.use_reallocation:
            weights = {}
            for cid, sample in strata.items():
                n = sample["n"]
                lower = (sample["yes"] + 0.5) / (n + 1.0)
                upper = (sample["yes"] + sample["unsure"] + 0.5) / (n + 1.0)
                deviation = max(math.sqrt(lower * (1.0 - lower)), math.sqrt(upper * (1.0 - upper)))
                weights[cid] = sample["N"] * deviation
            allocations = allocate_neyman(remaining_capacity, weights, next_amount, config.defensive_floor)
        else:
            allocations = allocate_uniform(remaining_capacity, next_amount)
        draws = draw_batch(allocations)
        if not draws:
            converged = False
            break
        classify_batch(draws)

    final = sequential_estimate(strata, config.alpha, confidence_bounds)
    audit = []
    for row in final.audit:
        values = strata[row["community_id"]]
        audit.append({**row, "yes": values["yes"], "no": values["n"] - values["yes"] - values["unsure"],
                      "n_sampled": values["n"], "p_hat": (row["p_yes"] + row["p_possible"]) / 2.0,
                      "paper_ids": sorted(values["seen"])})
    if calls > config.budget or any(row["n_sampled"] > row["N"] for row in audit):
        raise RuntimeError("semantic tally violated its budget or population invariant")
    return SemanticTallyResult(final.estimate, final.interval, converged, calls, config.seed, tuple(audit),
                               confidence_level=1 - config.alpha, stop_reason=stop_reason,
                               cache_hits=cache_hits)


def load_paper_texts(conn, source_build_id: int, paper_ids: set[str], max_chars: int = 60000) -> dict[str, dict]:
    """Read stable section text for selected papers with explicit truncation metadata.

    Args:
        conn: psycopg connection to the source database.
        source_build_id: Section build containing the selected papers.
        paper_ids: Canonical paper IDs to read.
        max_chars: Maximum paper body characters retained for one model call.
    Returns:
        Mapping to section-preserving text, truncation flag, and source-section count.
    Raises:
        ValueError: If max_chars is not positive.
        Database errors: Propagate from the source cursor.
    Require: paper_ids belong to the pinned source build.
    Guarantee: Sections remain separated by source ID and section index.
    Maintain: Truncation is explicit and callers must send it to the classifier.
    Assert: Every requested paper has at least one source section.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not paper_ids:
        return {}
    out: dict[str, dict] = {}
    for row in iter_source_rows(conn, source_build_id, paper_ids):
        paper = out.setdefault(row["paper_id"], {"text": "", "truncated": False, "n_source_sections": 0})
        paper["n_source_sections"] += 1
        header = "[%s section %s]\n" % (row["doc_id"], row["section_idx"])
        block = header + row["text"] + "\n\n"
        room = max_chars - len(paper["text"])
        if room > 0:
            paper["text"] += block[:room]
        if len(block) > max(room, 0):
            paper["truncated"] = True
    missing = paper_ids - out.keys()
    if missing:
        raise ValueError("selected canonical paper missing from source build: " + min(missing))
    for paper in out.values():
        if paper["truncated"]:
            paper["text"] += "\n[TEXT TRUNCATED AT %d CHARACTERS]" % max_chars
    return out


def parse_predicate_label(reply: str) -> str:
    """Validate the model's fixed JSON yes/no/unsure label.

    Args:
        reply: Raw model response content.
    Returns:
        One validated label: yes, no, or unsure.
    Raises:
        ValueError: If JSON shape or label is outside the fixed schema.
    Require: The model was asked for the single-property JSON schema.
    Guarantee: No prose, extra fields, aliases, or malformed output is silently accepted.
    Maintain: `unsure` remains distinct from `no`.
    Assert: Returned value belongs to the three-label domain.
    """
    try:
        parsed = json.loads(reply)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("predicate response is not valid JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"label"}:
        raise ValueError("predicate response must contain exactly the label property")
    label = parsed["label"]
    if label not in {"yes", "no", "unsure"}:
        raise ValueError("predicate label must be yes, no, or unsure")
    return label


def predicate_prompt(predicate: str, paper_text: str, truncated: bool) -> tuple[str, str]:
    """Build the fixed classification prompt with explicit evidence sufficiency rules.

    Args:
        predicate: One semantic claim to classify for the paper.
        paper_text: Section-preserving source text.
        truncated: Whether source text was cut to fit the call budget.
    Returns:
        System and user message strings for JSON-mode classification.
    Raises:
        ValueError: If predicate or paper_text is empty.
    Require: The caller has selected one paper and one stable predicate.
    Guarantee: Prompt distinguishes explicit positive, supported negative, and insufficient evidence.
    Maintain: Truncated input without direct positive evidence maps to unsure.
    Assert: User message identifies truncation state and preserves section headers.
    """
    if not predicate.strip() or not paper_text.strip():
        raise ValueError("predicate and paper text must be non-empty")
    system = (
        "Classify whether the supplied paper discusses the stated predicate. "
        "Use only the supplied paper text. Return one JSON object with exactly one property: "
        ' {"label":"yes"}, {"label":"no"}, or {"label":"unsure"}. '
        "Use yes only when the paper substantively discusses the predicate. Use no only when the supplied "
        "paper is sufficiently complete to support absence. Use unsure when evidence is insufficient, ambiguous, "
        "or the text was truncated before relevant evidence could be ruled out. Do not add prose or fields."
    )
    status = "TRUNCATED: absence cannot be inferred from this text." if truncated else "COMPLETE: supplied stored sections."
    user = "Predicate: %s\nText status: %s\nPaper sections:\n%s" % (predicate.strip(), status, paper_text)
    return system, user


def classify_predicate(predicate: str, paper_text: str, truncated: bool = False,
                       llm_call: Callable[[str, str], tuple[str, str]] | None = None,
                       model: str | None = None) -> dict:
    """Classify one paper through the existing JSON-mode OpenRouter transport.

    Args:
        predicate: Semantic predicate being counted.
        paper_text: Section-preserving paper text.
        truncated: Whether the text cap removed source content.
        llm_call: Optional test transport returning `(content, provider)`.
        model: Optional OpenRouter model override pinned by a batch job.
    Returns:
        Validated label, provider, and actual request-attempt count.
    Raises:
        RuntimeError: If OpenRouter credentials or the upstream call are unavailable.
        ValueError: If response JSON does not match the fixed schema.
    Require: One model request is budgeted by the caller.
    Guarantee: Exactly one OpenRouter attempt is made when using the default transport.
    Maintain: Invalid output is never folded into a negative label.
    Assert: label belongs to {yes,no,unsure}.
    """
    system, user = predicate_prompt(predicate, paper_text, truncated)
    if llm_call is None:
        content, provider = interpret._via_openrouter(system, user, timeout=60, retries=1, max_tokens=64,
                                                      model=model)
        attempts = 1
    else:
        content, provider = llm_call(system, user)
        attempts = 1
    model_label = parse_predicate_label(content)
    truncation_override = truncated and model_label == "no"
    label = "unsure" if truncation_override else model_label
    return {"label": label, "model_label": model_label, "truncation_override": truncation_override,
            "provider": provider, "attempts": attempts}


def ensure_schema(conn) -> None:
    """Apply the idempotent global QA DDL for an existing database.

    Args:
        conn: psycopg connection to the graph database.
    Returns:
        None.
    Raises:
        Database errors: Propagate so callers do not proceed without the schema.
    Require: sect_build exists and the caller can create tables/indexes.
    Guarantee: All SQL objects in the checked-in migration exist.
    Maintain: DDL runs transactionally and does not rewrite section source data.
    Assert: Every statement commits or the transaction rolls back.
    """
    statements = [statement.strip() for statement in DDL_PATH.read_text(encoding="utf-8").split(";")
                  if statement.strip()]
    with conn.transaction():
        for statement in statements:
            conn.execute(statement)


def iter_source_rows(conn, source_build_id: int, paper_ids: set[str] | None = None) -> Iterator[Mapping]:
    """Stream source sections in canonical paper order.

    Args:
        conn: psycopg connection.
        source_build_id: Existing sect_build identifier.
        paper_ids: Optional canonical IDs to restrict the stream.
    Yields:
        Mapping rows consumed by build_index.
    Raises:
        Database errors: Propagate if the source query fails.
    Require: The caller pins source_build_id and does not mutate that build.
    Guarantee: Primary-community ties resolve by lowest community ID.
    Maintain: Server-side cursor bounds client memory to cursor batches.
    Assert: Returned rows are sorted for iter_canonical_papers.
    """
    from psycopg.rows import dict_row

    with conn.transaction():
        with conn.cursor(name="global_qa_source", row_factory=dict_row) as cursor:
            cursor.execute(SOURCE_QUERY, (source_build_id, source_build_id,
                                          sorted(paper_ids) if paper_ids is not None else None,
                                          sorted(paper_ids) if paper_ids is not None else None))
            for row in cursor:
                yield row


def persist_index(conn, source_build_id: int, index: CommunityTermIndex,
                  signature_top_k: int = DEFAULT_SIGNATURE_TOP_K) -> int:
    """Persist and atomically publish a complete index snapshot.

    Args:
        conn: psycopg connection to the source database.
        source_build_id: Source section build represented by index.
        index: Fully validated in-memory index.
        signature_top_k: Signature truncation used to build index.signature.
    Returns:
        New global QA build identifier.
    Raises:
        RuntimeError: If the source build or persisted invariants fail.
        Database errors: Propagate and roll back the complete transaction.
    Require: ensure_schema ran; index paper assignments and counts reconcile.
    Guarantee: Complete marker is set only after every child row and invariant commits.
    Maintain: Prior complete indexes remain available as history.
    Assert: Persisted paper count and community totals equal the in-memory index.
    """
    expected_docs = len(index.paper_community)
    if expected_docs != sum(index.n_docs.values()):
        raise RuntimeError("paper/community counts do not reconcile")
    with conn.transaction():
        if conn.execute("SELECT 1 FROM sect_build WHERE build_id=%s", (source_build_id,)).fetchone() is None:
            raise RuntimeError("source section build does not exist")
        source_docs = conn.execute(
            "SELECT count(DISTINCT CASE WHEN right(doc_id,8)='_methods' "
            "THEN left(doc_id,length(doc_id)-8) ELSE doc_id END) "
            "FROM sect_node WHERE build_id=%s", (source_build_id,),
        ).fetchone()[0]
        if source_docs != expected_docs:
            raise RuntimeError("refusing to publish a partial canonical-paper index")
        qa_build_id = conn.execute(
            "INSERT INTO global_qa_build (source_build_id, tokenizer_version, signature_top_k, n_docs) "
            "VALUES (%s,%s,%s,%s) RETURNING qa_build_id",
            (source_build_id, TOKENIZER_VERSION, signature_top_k, expected_docs),
        ).fetchone()[0]
        with conn.cursor().copy("COPY global_qa_doc (qa_build_id,paper_id,community_id,text_sha256) FROM STDIN") as copy:
            for paper_id in sorted(index.paper_community):
                copy.write_row((qa_build_id, paper_id, index.paper_community[paper_id], index.paper_sha256[paper_id]))
        with conn.cursor().copy("COPY global_qa_community (qa_build_id,community_id,n_docs) FROM STDIN") as copy:
            for community_id, n_docs in sorted(index.n_docs.items()):
                copy.write_row((qa_build_id, community_id, n_docs))
        with conn.cursor().copy("COPY global_qa_term_df (qa_build_id,community_id,term,df) FROM STDIN") as copy:
            for community_id, terms in sorted(index.df.items()):
                for term, count in sorted(terms.items()):
                    copy.write_row((qa_build_id, community_id, term, count))
        with conn.cursor().copy("COPY global_qa_signature (qa_build_id,community_id,term,keyness,df) FROM STDIN") as copy:
            for community_id, rows in sorted(index.signature.items()):
                for term, keyness, count in rows:
                    copy.write_row((qa_build_id, community_id, term, keyness, count))
        persisted_docs = conn.execute(
            "SELECT count(*), count(DISTINCT paper_id) FROM global_qa_doc WHERE qa_build_id=%s", (qa_build_id,)
        ).fetchone()
        persisted_community_total = conn.execute(
            "SELECT coalesce(sum(n_docs),0) FROM global_qa_community WHERE qa_build_id=%s", (qa_build_id,)
        ).fetchone()[0]
        if persisted_docs != (expected_docs, expected_docs) or persisted_community_total != expected_docs:
            raise RuntimeError("persisted paper and community counts do not reconcile")
        missing_source_docs = conn.execute(
            "WITH source_papers AS (SELECT DISTINCT CASE WHEN right(doc_id,8)='_methods' "
            "THEN left(doc_id,length(doc_id)-8) ELSE doc_id END AS paper_id "
            "FROM sect_node WHERE build_id=%s) "
            "SELECT count(*) FROM source_papers s LEFT JOIN global_qa_doc d "
            "ON d.qa_build_id=%s AND d.paper_id=s.paper_id WHERE d.paper_id IS NULL",
            (source_build_id, qa_build_id),
        ).fetchone()[0]
        if missing_source_docs:
            raise RuntimeError("persisted index omits source canonical papers")
        db_df = conn.execute(
            "SELECT term, sum(df)::bigint FROM global_qa_term_df WHERE qa_build_id=%s GROUP BY term ORDER BY term",
            (qa_build_id,),
        ).fetchall()
        if dict(db_df) != dict(index.global_df):
            raise RuntimeError("persisted community DF does not match corpus DF")
        conn.execute("UPDATE global_qa_build SET complete=true WHERE qa_build_id=%s", (qa_build_id,))
    return qa_build_id


def build_from_database(conn, source_build_id: int, signature_top_k: int = DEFAULT_SIGNATURE_TOP_K) -> int:
    """Build and persist exact DF for one immutable source build.

    Args:
        conn: psycopg connection.
        source_build_id: Existing section build ID.
        signature_top_k: Maximum routing signature size per community.
    Returns:
        Published global QA build ID.
    Raises:
        RuntimeError: If persistence invariants fail.
        Database errors: Propagate; no partial complete build is retained.
    Require: PostgreSQL source build and tokenizer version match the requested index contract.
    Guarantee: A complete marker exists only for fully persisted exact counts.
    Maintain: Source section rows remain unchanged.
    Assert: Every source canonical paper receives one primary community.
    """
    ensure_schema(conn)
    index = build_index(iter_source_rows(conn, source_build_id), top_k=signature_top_k)
    return persist_index(conn, source_build_id, index, signature_top_k)


def lexical_tally(conn, source_build_id: int, term: str,
                  tokenizer_version: str = TOKENIZER_VERSION) -> dict:
    """Read one exact lexical tally from the matching complete database index.

    Args:
        conn: psycopg connection.
        source_build_id: Source build requested by the caller.
        term: One exact lexical token.
        tokenizer_version: Tokenizer version expected by the caller.
    Returns:
        Exact result containing count, mode, source build, and QA build ID.
    Raises:
        RuntimeError: If no matching complete index exists.
        ValueError: If term is not one t3 token or tokenizer version differs.
    Require: A complete index matches both source_build_id and tokenizer_version.
    Guarantee: Count sums the full DF table and is labeled exact.
    Maintain: Missing term rows mean zero only after marker validation.
    Assert: Count is bounded by the indexed population.
    """
    if tokenizer_version != TOKENIZER_VERSION:
        raise ValueError("requested tokenizer version does not match this implementation")
    token = normalize_single_term(term)
    build = conn.execute(
        "SELECT qa_build_id,n_docs FROM global_qa_build WHERE source_build_id=%s "
        "AND tokenizer_version=%s AND complete ORDER BY qa_build_id DESC LIMIT 1",
        (source_build_id, tokenizer_version),
    ).fetchone()
    if build is None:
        raise RuntimeError("no complete global QA index matches source build and tokenizer")
    qa_build_id, n_docs = build
    count = conn.execute(
        "SELECT coalesce(sum(df),0)::bigint FROM global_qa_term_df WHERE qa_build_id=%s AND term=%s",
        (qa_build_id, token),
    ).fetchone()[0]
    if not 0 <= count <= n_docs:
        raise RuntimeError("exact lexical count is outside the indexed population")
    return {"count": int(count), "mode": "exact", "source_build_id": source_build_id,
            "qa_build_id": qa_build_id, "term": token, "n_docs": n_docs}


def verify_eval_slice(conn, eval_path: str | Path, source_build_id: int = 1163) -> dict:
    """Reconstruct the labeled slice and check hashes, primary strata, and lexical gold totals.

    Args:
        conn: Readable psycopg connection to the pinned source build.
        eval_path: JSONL artifact with manifest and canonical paper rows.
        source_build_id: Section build named by the manifest.
    Returns:
        Audit summary with verified papers, terms, communities, and counts.
    Raises:
        RuntimeError: If source records, assignments, hashes, or lexical totals differ.
    Require: Eval artifact is the deterministic canonical-paper artifact for source_build_id.
    Guarantee: The ten lexical manifest gold counts match direct t3 token presence.
    Maintain: Exact counts are checkpointed by source, tokenizer, and artifact identity.
    Assert: Every sampled paper and lexical question is audited.
    """
    records = [json.loads(line) for line in Path(eval_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = next((row for row in records if row.get("record_type") == "manifest"), None)
    papers = [row for row in records if row.get("record_type") != "manifest"]
    if manifest is None or manifest["build_id"] != source_build_id:
        raise RuntimeError("eval manifest does not match requested source build")
    expected = {row["paper_id"]: row for row in papers}
    if len(expected) != len(papers):
        raise RuntimeError("eval artifact contains duplicate canonical paper IDs")
    cache_key, artifact_sha256 = eval_cache_identity(eval_path, source_build_id)
    indexed = load_eval_index_cache(EVAL_CACHE_PATH, cache_key, source_build_id, artifact_sha256)
    cache_hit = indexed is not None
    if cache_hit:
        verify_eval_source_identity(conn, source_build_id, expected)
        if (indexed.paper_community != {paper_id: row["primary_community"] for paper_id, row in expected.items()} or
                indexed.paper_sha256 != {paper_id: row["text_sha256"] for paper_id, row in expected.items()}):
            raise RuntimeError("eval checkpoint paper identity differs from artifact")
    else:
        indexed = build_index(iter_source_rows(conn, source_build_id, set(expected)), top_k=500)
        if set(indexed.paper_community) != set(expected):
            raise RuntimeError("eval sample paper IDs differ from source build")
        for paper_id, row in expected.items():
            if indexed.paper_community[paper_id] != row["primary_community"]:
                raise RuntimeError("eval primary community mismatch for " + paper_id)
            if indexed.paper_sha256[paper_id] != row["text_sha256"]:
                raise RuntimeError("eval source text hash mismatch for " + paper_id)
        save_eval_index_cache(EVAL_CACHE_PATH, cache_key, source_build_id, artifact_sha256, indexed)
    lexical = [q for q in manifest["questions"] if q["kind"] == "lexical"]
    results = {}
    for question in lexical:
        token = normalize_single_term(question["aliases"][0])
        count = sum(indexed.df[cid].get(token, 0) for cid in indexed.n_docs)
        if count != question["gold_count"]:
            raise RuntimeError("lexical gold mismatch for %s: got %d expected %d" %
                               (question["id"], count, question["gold_count"]))
        results[question["id"]] = count
    route_metrics = {
        "signature_top_k_100": evaluate_lexical_routes(signature_variant(indexed, 100), manifest["questions"]),
        "signature_top_k_500": evaluate_lexical_routes(signature_variant(indexed, 500), manifest["questions"]),
    }
    return {"source_build_id": source_build_id, "n_papers": len(indexed.paper_community),
            "n_communities": len(indexed.n_docs), "lexical_counts": results,
            "routing_metrics": route_metrics, "cache_hit": cache_hit}
