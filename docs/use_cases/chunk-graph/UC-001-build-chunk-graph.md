# Use Case: Build Chunk Graph

## Overview

**Use Case ID:** UC-001  
**Use Case Name:** Build Chunk Graph  
**Primary Actor:** Operator  
**Goal:** A labelled, reproducible graph of the corpus's text chunks is stored, so chunks can be explored by their connections and by topic community, with every connection traceable to the documents it came from.  
**Trigger:** The operator starts a build for a corpus under a run label.  
**Status:** Draft  

## Preconditions

- The corpus is available as a set of text documents, each with an identifier.
- The database is writable and has the vector extension installed.

## Main Success Scenario

1. Operator gives the run a label and selects the corpus (a sample of the Brown corpus, a mixed set of sources, or the operator's own documents).
2. System splits each document into chunks at paragraph boundaries, never inside a word, and keeps each document's identifier on every chunk.
3. System scores every pair of chunks by shared wording. When a local embedding model is configured, it also scores every pair by meaning.
4. System keeps the pairs whose similarity is significant, and keeps each chunk's nearest neighbours in each kind of similarity whatever their significance.
5. System merges the kept pairs into one graph and records, for each connection, which kind of similarity produced it.
6. System groups the chunks into communities and gives each community its most distinctive terms and a representative chunk.
7. System stores the run, its chunks, connections, communities and embeddings together, and marks the run as live under its label.
8. System marks the earlier live run with the same label as superseded.

## Alternative Flows

### A1: No embedding model is configured

**Trigger:** No embedding model is configured for the run (step 3).  
**Flow:**

1. System scores pairs by shared wording only.
2. System records every connection as found by shared wording.
3. Use case continues at step 4.

### A2: Too few chunks

**Trigger:** The corpus yields fewer than ten chunks (step 2).  
**Flow:**

1. System refuses the run and names the chunk count.
2. Use case ends.

### A3: The strength scale cannot be fitted

**Trigger:** The similarity values of one kind are too degenerate to fit a scale (step 4).  
**Flow:**

1. System ranks the connections of each kind and uses the average rank as their strength.
2. Use case continues at step 5.

### A4: Storing the run fails

**Trigger:** The store rejects the run or the write is interrupted (step 7).  
**Flow:**

1. System stores nothing from this run.
2. The earlier live run with the same label stays live.
3. Use case ends.

## Postconditions

### Success Postconditions

- The new run is live under its label and holds all its chunks, connections, communities and embeddings.
- The earlier live run with the same label is superseded and still stored.

### Failure Postconditions

- No partial run is stored.
- The live run under the label is unchanged.

## Business Rules

### BR-001: Every chunk names its source document

Each chunk keeps the identifier of the document it came from, so every connection can name the documents it derives from.

### BR-002: No language model shapes the graph

Chunking, connections and communities are computed without a language model. Two runs over the same corpus and settings give the same graph.

### BR-003: A run needs at least ten chunks

A run with fewer than ten chunks is refused.

### BR-004: Superseded runs are kept

A new live run under a label supersedes the earlier live run; the earlier run is kept, never deleted.

### BR-005: Chunks never split a word

A chunk boundary never falls inside a word.

### BR-006: Every chunk stays connected

Each chunk keeps its nearest neighbours in each kind of similarity, whatever their significance, so no chunk is left without a connection.
