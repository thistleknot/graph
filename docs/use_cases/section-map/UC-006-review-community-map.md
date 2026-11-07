# Use Case: Review Community Map

## Overview

**Use Case ID:** UC-006  
**Use Case Name:** Review Community Map  
**Primary Actor:** Operator  
**Goal:** The operator reads a labelled picture of the corpus's communities, with their terms, entities, paper titles and exemplar sections, to judge whether the communities mean something.  
**Trigger:** The operator asks for the labelled map of the live section build.  
**Status:** Draft  

## Preconditions

- A live section build exists with its layout.

## Main Success Scenario

1. Operator asks for the labelled map of the live section build.
2. System draws one card per community, largest first, with the community's distinctive terms, its top entities and its exemplar sections.
3. System names the paper each exemplar section came from.
4. System writes a markdown listing of every exemplar section with a link to its paper.
5. Operator reads the picture and the listing.

## Alternative Flows

### A1: Summaries are not drafted yet

**Trigger:** No community summary is stored for the build (step 2).  
**Flow:**

1. System draws each card without a summary.
2. Use case continues at step 3.

## Postconditions

### Success Postconditions

- A picture and a markdown listing of the live build are written.

### Failure Postconditions

- No picture or listing is written for a build that is not live.

## Business Rules

### BR-001: Cards are ordered by size

Community cards are shown largest first.

### BR-002: Each card shows distinctive terms

Each card shows the terms most over-represented in its community compared with the whole corpus.

### BR-003: Each exemplar names its paper

Each exemplar section is shown with the title of its paper.
