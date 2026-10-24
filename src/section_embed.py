"""Pool model2vec token embeddings with full-text coverage.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| TOKENIZE | Batch tokenize with padding and truncation disabled. | [mandatory] | R1 |
| POOL | Weighted mean token rows in bounded token blocks. | [mandatory] | R2 |
| NORMALIZE | L2-normalize nonzero outputs when requested by the model. | [contract] | R3 |

R1 The model tokenizer supports no-padding, no-truncation, and batch tokenization.
R2 Token IDs without embedding rows contribute zero; long text is not truncated.
R3 Empty token streams return zero vectors; the tokenizer call uses no version-specific keyword arguments.

Preconditions: model embedding and tokenizer assets are loaded. Failure modes: incompatible model API and allocation errors propagate.
"""
from __future__ import annotations

import itertools

import numpy as np


BLOCK = 262_144                  # tokens gathered at a time: one 16.7M-character section is 25M tokens, 24 GB if gathered whole (2026-10-05)


def pool(model, texts: list[str], step: int = 500) -> np.ndarray:
    """Return weighted mean embeddings for all input texts.

    Require:
        `model` exposes embedding, weights, normalize, tokenizer, and tokenize attributes; `step` is positive.
    Guarantee:
        Returns one float32 vector per text without truncation or multiprocessing.
    Maintain:
        Accumulates at most BLOCK token rows at a time and keeps input order.
    Assert:
        Output shape is `(len(texts), model.embedding.shape[1])`.

    Args:
        model: Loaded model2vec static model.
        texts: Text strings to encode.
        step: Number of texts tokenized together.

    Returns:
        Float32 matrix of normalized pooled vectors.

    Raises:
        ValueError: If step is not positive.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    emb, w = model.embedding, model.weights
    rows, dim = emb.shape
    model.tokenizer.no_padding()                # jina's tokenizer pads every text of a batch to the longest one with id 128004 (past the kept
    model.tokenizer.no_truncation()             # rows): 2,000 sections of 5.0M characters came out as 28.9M tokens. No pads, no cut.
    out = np.zeros((len(texts), dim), np.float32)
    for s in range(0, len(texts), step):
        ids = model.tokenize(texts[s:s + step])                             # tokenizer.no_truncation above preserves the full text on current and prior APIs
        lens = np.fromiter((len(x) for x in ids), np.int64, len(ids))
        total = int(lens.sum())
        if total == 0:
            continue
        flat = np.fromiter(itertools.chain.from_iterable(ids), np.int64, total)
        tid = np.repeat(np.arange(len(ids)), lens)                          # which text each token belongs to
        acc = np.zeros((len(ids), dim), np.float64)
        for a in range(0, total, BLOCK):                                    # a text may span blocks: its partial sums add into acc
            f = flat[a:a + BLOCK]
            keep = f < rows                                                 # an id with no row (jina's pad ids) adds nothing
            v = np.zeros((f.size, dim), np.float32)
            v[keep] = emb[f[keep]]
            if w is not None:
                v[keep] *= w[f[keep]][:, None]
            seg = tid[a:a + BLOCK]
            starts = np.concatenate([[0], np.flatnonzero(np.diff(seg)) + 1])
            acc[seg[starts]] += np.add.reduceat(v, starts, axis=0)
        nz = lens > 0
        out[s:s + len(ids)][nz] = (acc[nz] / lens[nz][:, None]).astype(np.float32)
    if model.normalize:
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-32
    assert out.shape == (len(texts), dim)
    return out


def center(E: np.ndarray, mu: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Guarantee: (rows minus the SAMPLE mean vector, then L2-normalised; the mean used). The inner product of two such rows is the cosine of
    their deviations from the corpus mean (repo guard R9's "correlation"): it removes the component every embedding shares, which otherwise makes
    all cosines near 1. It is NOT np.corrcoef between rows, which removes each row's own mean over dimensions; a zero row stays zero."""
    mu = E.mean(0) if mu is None else mu
    X = E - mu
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, 1e-12)).astype(np.float32), mu
