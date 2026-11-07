# Use Case: Keep Corpus Current

## Overview

**Use Case ID:** UC-003  
**Use Case Name:** Keep Corpus Current  
**Primary Actor:** Scheduler  
**Goal:** Newly extracted papers become searchable without a manual step, and the index is fully rebuilt once enough new text has arrived.  
**Trigger:** A cycle of the schedule, every thirty minutes at a fixed minute of the hour.  
**Status:** Draft  

## Preconditions

- A live keyword index exists.
- Newly extracted paper files are placed in the corpus folder.

## Main Success Scenario

1. Scheduler starts a cycle at its fixed minute.
2. System finds the extracted paper files that are not yet in the index.
3. System chunks each new paper with the frozen settings of the live index, so its chunks match what a full build would give.
4. System computes each new chunk's keyword and dense vectors under the frozen statistics.
5. System places each new paper in the community of the nearest centre.
6. System stores the new chunks, vectors, placements and entity mentions together.
7. When the chunks added since the build reach the rebuild fraction of the build's size, system runs the full pipeline.

## Alternative Flows

### A1: The previous cycle is still running

**Trigger:** The previous cycle has not finished (step 1).  
**Flow:**

1. System skips this cycle.
2. Use case ends.

### A2: Resources are busy

**Trigger:** The corpus file is held by another process, or the graphics memory is more than half in use (step 1).  
**Flow:**

1. System skips this cycle.
2. Use case ends.

### A3: A paper is already indexed

**Trigger:** A paper's identifier is already in the index (step 2).  
**Flow:**

1. System skips the paper.
2. Use case continues at step 6 for the remaining papers.

### A4: The cycle fails part way

**Trigger:** A step fails before the store step (step 6).  
**Flow:**

1. System stores nothing from this cycle.
2. Use case ends.

## Postconditions

### Success Postconditions

- Each new paper is indexed with the same records a full build would give.
- Chunks added since the last full build are counted towards the rebuild.

### Failure Postconditions

- The stored index is unchanged by a cycle that did not complete.

## Business Rules

### BR-001: A paper is ingested once

A paper whose identifier is already indexed is skipped. A chunk whose text is already stored is skipped in every run.

### BR-002: New chunks match a full build

New chunks are produced with the frozen settings of the live index, so each new chunk equals the chunk a full build would produce for the same document.

### BR-003: Papers are keyed without their version

A paper is keyed by its arXiv identifier without the version. The version is kept as a second key.

### BR-004: Unseen terms wait for a full build

A term outside the frozen vocabulary carries no weight until the next full build.

### BR-005: A full rebuild runs at the rebuild fraction

When the chunks added since the build reach the rebuild fraction of the build's size, a full pipeline run starts.

### BR-006: Reference and junk chunks are kept

New reference and junk chunks are stored with a flag, as in the full build.
