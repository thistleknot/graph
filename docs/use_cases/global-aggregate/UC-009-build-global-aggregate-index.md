# Use Case: Build Global Aggregate Index

## Overview

**Use Case ID:** UC-009  
**Use Case Name:** Build Global Aggregate Index  
**Primary Actor:** Operator  
**Goal:** Every term of the corpus's papers is counted per community, once per paper, so that corpus-wide counting questions can be answered from a complete index.  
**Trigger:** The operator builds the index for a pinned section build and a tokenizer version.  
**Status:** Draft  

## Preconditions

- The pinned section build exists.
- The tokenizer version named by the operator is available.

## Main Success Scenario

1. Operator pins the section build and the tokenizer version.
2. System reads the build's sections and groups them into papers; each paper belongs to the community most of its sections belong to.
3. System counts, per community, the papers in which each term occurs, counting each paper once per term.
4. System stores the counts and the paper assignments for the build.
5. System marks the index complete once every count is stored.

## Alternative Flows

### A1: The pinned build or tokenizer does not match

**Trigger:** The pinned section build is not available or the tokenizer version differs from the one the index was built with (step 1).  
**Flow:**

1. System refuses the build and names the mismatch.
2. Use case ends.

### A2: The build does not complete

**Trigger:** A step fails before the index is marked complete (step 5).  
**Flow:**

1. System leaves the index unmarked.
2. An unmarked index is never used to answer a question.
3. Use case ends.

## Postconditions

### Success Postconditions

- The index for the pinned build is complete and can be used to answer counts.

### Failure Postconditions

- No index is marked complete.

## Business Rules

### BR-001: A paper counts once per term

Each paper adds at most one count to each term in its community.

### BR-002: Method extracts belong to their paper

A paper's method extract is counted with the paper it came from.

### BR-003: A missing count is zero only for a complete index

A term with no stored count is treated as zero only when the index for the matching build is complete.

### BR-004: Routing summaries never prove a term

Routing summaries are shortened and never establish that a term is present in a community.

### BR-005: The index must match its build and tokenizer

An index is used only with the build and tokenizer version it was built from.
