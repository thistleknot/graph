# Use Case: Ask Question

## Overview

**Use Case ID:** UC-007  
**Use Case Name:** Ask Question  
**Primary Actor:** Chat User, Operator  
**Secondary Actors:** Open WebUI, LLM Provider  
**Goal:** The asker receives an answer grounded in the sections of the corpus most relevant to the question, with those sections shown as evidence and kept apart from the model's own commentary.  
**Trigger:** A chat user sends a question in a chat where the retrieval filter is switched on, or the operator asks one of the example questions from the query panel.  
**Status:** Draft  

## Preconditions

- A live section build and its keyword index exist.
- The retrieval service is reachable from Open WebUI.
- The chat model and the synthesis model are available through the LLM Provider.

## Main Success Scenario

1. Chat User writes a question and may choose the community terms and the graph backend to use.
2. Open WebUI sends the question with those choices to the retrieval service once.
3. System finds candidate sections by meaning and by keyword and combines the two rankings.
4. System reorders the candidates by how well their communities match the chosen terms.
5. System expands the strongest sections through their neighbours in the map.
6. System scores the candidates against the question and keeps the communities that hold a passing section.
7. System writes a partial answer for each kept community and combines them into a global view.
8. System runs a bounded research loop in which a model reads sections, searches again, or stops.
9. System writes the final answer from the retained evidence, tagging each claim with the community it came from.
10. Open WebUI shows the source passages grouped by paper first, then the model's answer, labelled as the model's commentary.

## Alternative Flows

### A1: No section passes the relevance gate

**Trigger:** No candidate section scores at or above the pass threshold (step 6).  
**Flow:**

1. System rejects the question.
2. Open WebUI shows the relevance notice in place of an answer.
3. Use case ends.

### A2: The selected graph backend is unavailable

**Trigger:** The chosen traversal backend has no matching graph for the live build (step 5).  
**Flow:**

1. System returns an error naming the missing graph.
2. System does not switch to the other backend.
3. Use case ends.

### A3: The retrieval call fails

**Trigger:** The retrieval service returns an error or a reply Open WebUI cannot validate (step 2).  
**Flow:**

1. Open WebUI stops with the retrieval error.
2. No answer is generated.
3. Use case ends.

### A4: The synthesis reply is truncated

**Trigger:** The synthesis model stops before finishing its answer (step 9).  
**Flow:**

1. System asks one other provider for the same answer.
2. If that reply is complete, use case continues at step 10.
3. Otherwise System fails the request. No partial answer is returned.
4. Use case ends.

### A5: The operator's query panel

**Trigger:** The operator runs the query panel (step 1).  
**Flow:**

1. System runs the same steps for the three fixed example questions.
2. System draws the panel with each question's sections, communities and answer.
3. Use case ends.

## Postconditions

### Success Postconditions

- The asker sees the evidence passages before the model's answer.
- Each claim in the answer names the community it came from.

### Failure Postconditions

- No answer is shown that was not built from retrieved evidence.
- No partial answer is returned.

## Business Rules

### BR-001: A rejected question gets no model memory answer

When the relevance gate rejects a question, no answer is written from the model's own knowledge.

### BR-002: Evidence is shown as retrieved

The source passages are shown as retrieved, grouped by canonical paper, and are never edited by the model.

### BR-003: The live build only

Retrieval reads the live section build and its keyword index. If that build is missing or inconsistent, the request fails.

### BR-004: Bounded research loop

The research loop makes at most three extra searches, and each search is kept apart from the earlier ones.

### BR-005: Single-section communities are excluded

A community holding a single section is never offered to the model as a candidate or as context.

### BR-006: Graph expansion is bounded

Graph expansion walks at most three steps from each seed section.

### BR-007: Relevance gate threshold

A section passes the relevance gate when its score is at or above zero.

### BR-008: Synthesis is never returned truncated

A synthesis reply that stops before it is complete is retried once with another provider, and otherwise fails.
