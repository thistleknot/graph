# Use Case: Enrich Section Map

## Overview

**Use Case ID:** UC-005  
**Use Case Name:** Enrich Section Map  
**Primary Actor:** Operator  
**Secondary Actors:** LLM Provider  
**Goal:** Each community carries the entities its sections mention, the titles of its papers and a drafted summary, so that the community can be recognised without reading its sections.  
**Trigger:** The operator runs the entity, title and summary steps on the live section build.  
**Status:** Draft  

## Preconditions

- A live section build exists.
- For the summary step, the LLM Provider is reachable with the operator's credentials.

## Main Success Scenario

1. Operator asks for the frozen entity inventory to be matched onto the live section build.
2. System records, for each section, which entities it mentions and how often.
3. System keeps the entity links whose entities co-occur in enough sections and more often than chance.
4. Operator asks for paper titles to be filled in.
5. System fills in the titles from the local listings first and looks up the rest in the arXiv service.
6. Operator asks for community summaries to be drafted.
7. System drafts one summary per community from the community's exemplar sections, skipping communities already drafted.
8. System stores the entities per community, the titles and the summaries with the live build.

## Alternative Flows

### A1: A title lookup fails repeatedly

**Trigger:** The arXiv service keeps failing for one batch of papers after retries (step 5).  
**Flow:**

1. System stops and names the failing batch.
2. Titles already filled in are kept.
3. Use case ends.

### A2: A summary comes back empty or unfinished

**Trigger:** The LLM Provider returns an empty or unfinished reply for a community (step 7).  
**Flow:**

1. System reports the failure for that community.
2. The community keeps no draft.
3. Use case continues at step 8 for the communities already drafted.

## Postconditions

### Success Postconditions

- Each community lists its mentioned entities with counts.
- Each paper with a found title has its title stored.
- Each community with a complete reply has a drafted summary.

### Failure Postconditions

- No partial summary is stored.
- Titles and entities already stored are kept.

## Business Rules

### BR-001: A stored title is never looked up again

A paper whose title is already stored is never looked up again. Where several local listings give a title, the first listing wins.

### BR-002: Entity links need shared sections and are not hubs

An entity link is kept only when the two entities occur together in at least five sections, each entity occurs in at most 2,000 sections, and the pair occurs together more often than chance.

### BR-003: Summaries are drafted from full exemplar text

A community summary is drafted from the full text of its exemplar sections, never from a shortened version.

### BR-004: An empty or unfinished reply is not a draft

A reply that is empty or ends before it is complete is reported as a failure and is not stored as a summary.

### BR-005: Summaries are labelled as machine drafts

A stored summary is shown as a machine draft wherever it appears.
