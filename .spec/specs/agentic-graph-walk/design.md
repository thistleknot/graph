# Agentic Graph Walk — Design

**Spec layer:** Design. Commits to structure and behavior for
`requirements.md`. Task breakdown deferred to `tasks.md`.

**This design supersedes the traversal mechanism the requirements assume.**
Requirements describe an LLM tool-caller choosing the route. That was built
(`walk_agent.py`) and measured, and it does not work. The replacement removes
the model from retrieval altogether. §6 lists every requirement whose text now
needs amending, so the two documents do not silently disagree.

---

## 1. Why the router was replaced

`walk_agent.py` ran against `brown-50` on an uncontended GPU, 14-step budget,
model `qwen3.5-oc:4b`:

```
steps      : 14        visited: 26 chunks
saturation : 3,3,5,5,5,5,5,7,7,7,7,7,7,7
complete   : False     evidence: []     answer: None
```

Three findings, none of them about model size:

- **It never terminated.** `finish` was never called; the budget ran out. R6.3
  reported the walk incomplete, which is correct behaviour and a useless answer.
- **It thrashed.** `search` was called at steps 1, 3, 4 and 11; `#972` was
  expanded at both 5 and 12. Roughly half the budget bought nothing.
- **The stopping signal was already there and unused.** Community saturation
  went flat at 7 from step 8. Six further steps added no new region. The
  evidence had converged and only the model's judgment disagreed.

The conclusion is not "use a bigger model." It is that **route selection is a
sampling problem, and sampling is what we should have written.** An LLM asked
to decide "have I seen enough" has no calibrated notion of enough; a sample
count does.

## 2. The mechanism

    anchors -> induced subgraph -> Boltzmann sample n chunks
            -> group by stored cid -> rank communities by hit count
            -> top communities' text -> LLM summarises -> answer

**Nothing before the last step involves a model.** The graph was built
deterministically at ingest; the walk is now deterministic too, given a seed.
The LLM sees a fixed bundle of evidence and writes prose about it. That places
this design fully inside the steering ban rather than merely beside it.

The intuition is the operator's and it is the right one: **sampling the graph is
looking up a word in the index; the communities you land in are the chapters.**
You do not read the whole book, and you do not read one paragraph — you read
the chapters the index kept pointing at.

### 2.1 Boltzmann sampling

Over the candidate set `C` (the induced subgraph, §2.2), each chunk `v` carries
a score `s(v)` and is drawn with

    P(v) = exp(s(v) / T) / SUM_u exp(s(u) / T)

`T` is the one exploration knob and it has clean limits worth stating, because
they are how the parameter gets tuned:

| `T` | behaviour |
|---|---|
| `T -> 0` | argmax — collapses to top-`n` by score, no exploration |
| `T = 1` | scores used as-is |
| `T -> inf` | uniform over `C` — score ignored, pure structure |

Sampling is **without replacement** (draw, renormalise, repeat). With
replacement, a single dominant chunk consumes the budget and the community
histogram degenerates toward one bar.

**Scores must be normalised before exponentiation.** BM25 is unbounded and
corpus-scaled; `exp` of a raw BM25 score saturates to a one-hot distribution
and `T` stops meaning anything. Scores are z-scored over `C` first, which makes
`T` comparable across queries and corpora.

### 2.2 What is sampled from

The candidate set is the H-hop induced neighbourhood of the validated anchors,
built by `graph_tools.neighbors` — indexed `src`/`dst` traversal, R4 preserved.

`s(v)` is the strength-decayed path score already computed by the expansion,
optionally combined with the anchor BM25 score for chunks that also match
lexically. Anchors themselves are included in `C`; they are usually the highest
scoring members and should be able to win draws on merit rather than by fiat.

### 2.3 Community aggregation

Every sampled chunk is mapped to its **stored** `cid` (`communities_touched`),
never a recomputed one. Communities are ranked by how many sampled chunks they
contain. The top `k_comm` communities become the evidence bundle.

This is the step that makes the design worth building. A chunk is a paragraph
and carries almost no context; a community is a coherent region with keywords,
a medoid, and a stored draft label. Handing the model twelve scattered
paragraphs and handing it "these three regions, and here is what they are
about" are different prompts, and only the second one can be checked by a human
against the community table.

### 2.4 Where the LLM enters

One call, at the end, over a fixed bundle: the top communities' keywords,
medoid text, and the sampled chunks belonging to them. It writes the answer and
cites chunk ordinals. It cannot request more evidence and it cannot influence
which evidence it got.

## 3. Determinism and replay

The sampler takes an explicit `seed`. Given `(run_id, query, anchors, n, T,
seed)` the sampled set is **exactly reproducible** — not merely replayable.

This is strictly stronger than R5.4, which only asked that a recorded walk
reproduce its observations. The router could not do better than that, because
its route came from model sampling nobody controls. Here the whole retrieval
is a pure function of stored data and a seed, so a disputed answer can be
re-derived rather than taken on trust.

The evidence bundle is what gets recorded, and it is small: `n` ordinals plus
`k_comm` cids.

## 4. The tuning problem

`n` is the parameter the operator asked to tune, and it is measurable in a way
the router's step budget never was.

**The evaluation quantity is community stability, not chunk overlap.** Two
draws at the same `n` will rarely pick the same chunks — that is sampling, not
error. They should pick the same *communities*. So:

    run the sampler R times at a given (n, T) with different seeds
    -> Jaccard of the top-k_comm community sets across runs
    -> raise n until that stability crosses a threshold and plateaus

That plateau is "sufficient evidence" made operational, and it needs no gold
labels — it is an internal consistency measure. A gold set is still required
for *answer* quality, but not for sizing `n`.

Both knobs interact and must be swept together: low `T` makes the sample
concentrate and stabilise at small `n`; high `T` needs a larger `n` for the
same stability. The measured 2-hop reach on `brown-50` (p50 20, p99 111) bounds
where `n` can usefully sit — sampling 100 from a 20-node neighbourhood is
enumeration wearing a costume.

## 5. Structure

```
explorer/sampler.py      boltzmann_sample(), community_histogram()  -- no LLM
explorer/summarize.py    one model call over a frozen evidence bundle
walk_agent.py            REPLACED -- kept only until sampler.py lands
```

`graph_tools.py` is unchanged: `search`, `neighbors`, `communities_touched` and
`node` are exactly the primitives the sampler needs. The tool surface was built
for a model to call and turns out to serve a sampler equally well, which is
what a well-factored surface should do.

**GUARDS (EARS)** — to carry into `explorer/sampler.py`'s module docstring:

```
S1  Sampling SHALL be a pure function of (candidate set, scores, n, T, seed).
    No model call occurs before the evidence bundle is frozen.
S2  Scores SHALL be normalised over the candidate set before exponentiation;
    raw unbounded scores make T meaningless.
S3  Sampling SHALL be without replacement.
S4  WHERE n >= |C|, the sampler SHALL return C and report that it enumerated
    rather than sampled.
S5  Sampled chunks SHALL be grouped by their STORED cid; no partitioning
    occurs at query time.
S6  The evidence bundle handed to the model SHALL be frozen: the model cannot
    request further retrieval.
S7  Every answer SHALL cite chunk ordinals, and every cited ordinal SHALL be
    a member of the bundle.
```

## 6. Requirements this design changes

`requirements.md` was written around an LLM router. These clauses no longer
describe the system and need amending rather than quietly diverging:

| Requirement | Status |
|---|---|
| R1.1–1.4 tool surface closed | **unchanged** — the sampler uses the same primitives |
| R2 read-only enforced | **unchanged** |
| R3 stored communities | **unchanged, now central** — community ranking is the output, not a side panel |
| R5.1 ordered step record | **weaker and better** — there is no route to record; the bundle plus seed replaces it |
| R5.4 replayable observations | **strengthened** — exactly reproducible, not merely replayable |
| R6.1 explicit budget | **satisfied by construction** — `n` is the budget |
| R6.2 saturation as stopping signal | **repurposed** — no longer a live signal to a router; it becomes the offline criterion for choosing `n` (§4) |
| R6.3 report incomplete walks | **obsolete** — a fixed-`n` sample cannot run out; drop it |
| "LLM tool-caller chooses the route" (Context) | **false** — the model chooses nothing |

## 7. Open questions

1. **What exactly is `s(v)`?** Path-decayed strength alone, or blended with
   anchor BM25 for lexically-matching chunks? The blend needs a weight nobody
   has calibrated, and rank fusion is the fallback if score-space refuses to
   behave.
2. **How many communities?** `k_comm` is a second cardinality knob. Fixed, or
   cut where the community histogram drops off?
3. **Does the histogram have a shoulder?** The design assumes sampled chunks
   concentrate in a few communities. On `brown-50` a 2-hop neighbourhood
   touched 25 of 40 communities raw and 13 with the strength floor — so the
   distribution has a long tail, and whether the head is clean enough to cut is
   unmeasured.
4. **What happens with one anchor?** A tight neighbourhood may span a single
   community, making the histogram degenerate. Falls back to plain retrieval,
   but that path is unwritten.
