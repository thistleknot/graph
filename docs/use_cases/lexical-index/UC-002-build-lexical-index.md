# Use Case: Build Lexical Index

## Overview

**Use Case ID:** UC-002  
**Use Case Name:** Build Lexical Index  
**Primary Actor:** Operator  
**Goal:** The chunks of the corpus are indexed by their distinctive words, so that keyword search over the corpus can rank them, and the index is published as the one the retrieval service serves.  
**Trigger:** The operator starts a full build of the keyword index from the live graph.  
**Status:** Draft  

## Preconditions

- A live graph of the corpus exists.
- The database is writable and has the vector extension installed.

## Main Success Scenario

1. Operator chooses the index label and the size of the vocabulary.
2. System reads the chunks of the live graph and keeps every chunk, flagging reference and junk chunks rather than dropping them.
3. System learns the vocabulary and the term weights from the corpus and freezes them for this build.
4. System encodes each chunk as a weighted list of its vocabulary terms and stores it with its document identifier and section.
5. System groups the chunks into communities and records which community each chunk falls in.
6. System records the build as complete and live.
7. System publishes the graph generation and the keyword generation together as the pair the retrieval service serves.

## Alternative Flows

### A1: A chunk is already stored

**Trigger:** A chunk's text is already in the index from an earlier build (step 4).  
**Flow:**

1. System skips the chunk.
2. Use case continues at step 5.

### A2: The build does not complete

**Trigger:** A step of the build fails before the build is complete (step 6).  
**Flow:**

1. System leaves the new build not live.
2. The previously live index stays live.
3. Use case ends.

## Postconditions

### Success Postconditions

- The new keyword index is live and complete.
- The graph and keyword generations served together are the new pair.

### Failure Postconditions

- No incomplete build is served.
- The previously live index and its pair are unchanged.

## Business Rules

### BR-001: The vocabulary and weights are frozen per build

A build fixes its vocabulary, term weights and the centre of its embeddings. Later additions do not change them.

### BR-002: A chunk is indexed once

A chunk whose text is already stored is not stored again.

### BR-003: Reference and junk chunks are kept

Reference and junk chunks are stored with a flag. They are never dropped.

### BR-004: A generation is served only when complete

A keyword generation becomes live only when all its chunk records and vocabulary are stored. A failed publication leaves the earlier generation live.

### BR-005: Unseen terms carry no weight until the next full build

A term the frozen vocabulary has never seen carries no weight until the next full build.
