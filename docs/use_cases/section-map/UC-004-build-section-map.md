# Use Case: Build Section Map

## Overview

**Use Case ID:** UC-004  
**Use Case Name:** Build Section Map  
**Primary Actor:** Operator  
**Goal:** The sections of the corpus's papers are grouped into topic communities with a two-dimensional layout, stored under a tag so that the map can be reviewed and served.  
**Trigger:** The operator starts a section build from a chosen folder of paper Markdown files.  
**Status:** Draft  

## Preconditions

- The folder holds paper Markdown files named with arXiv identifiers.
- The output location is writable.

## Main Success Scenario

1. Operator selects the folder of paper Markdown files and a tag for the build.
2. System keeps the latest version of each paper and cuts each paper into sections, keeping the whole text of each usable section.
3. System describes each section two ways: by its meaning and by its shared wording.
4. System keeps, in each way, the pairs of sections whose similarity is significant and that are among a section's closest, then merges both ways into one set of weighted connections.
5. System keeps each section's two strongest connections in each way, whatever their significance.
6. System groups the sections into communities and computes a two-dimensional layout.
7. System builds an exemplar section for each community and draws the map.
8. System stores the build with its sections, connections, communities and layout under the tag, and marks it live.

## Alternative Flows

### A1: A paper file is malformed

**Trigger:** A paper file cannot be read or cut into sections (step 2).  
**Flow:**

1. System stops the build and names the file.
2. The previously saved section cache is left in place.
3. Use case ends.

### A2: The store step fails

**Trigger:** The store rejects the build or the write is interrupted (step 8).  
**Flow:**

1. System stores nothing from this build.
2. The earlier live build under the tag stays live.
3. Use case ends.

## Postconditions

### Success Postconditions

- The new build is live under its tag and holds its sections, connections, communities and layout.
- The earlier builds under the tag are superseded and kept.

### Failure Postconditions

- No partial build is stored.
- The live build under the tag is unchanged.

## Business Rules

### BR-001: Each section keeps its paper and its whole text

A section keeps the identifier of its paper and its whole text. Nothing is cut from it.

### BR-002: Connections need significance and a close neighbour

Two sections are connected in one way only when their similarity is significant (at least two standard deviations above the corpus) and the other section is among the first fifteen nearest neighbours of this one.

### BR-003: Every section keeps its two strongest connections per way

Each section keeps its two strongest connections in each way, regardless of significance, so no section is left isolated.

### BR-004: A new build supersedes the earlier builds of its tag

A new live build under a tag supersedes the earlier builds of that tag. Nothing is deleted.

### BR-005: A failed build leaves the live build unchanged

A build that is not stored completely does not change the live build under its tag.
