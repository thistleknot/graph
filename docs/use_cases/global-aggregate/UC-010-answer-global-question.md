# Use Case: Answer Global Question

## Overview

**Use Case ID:** UC-010  
**Use Case Name:** Answer Global Question  
**Primary Actor:** Chat User  
**Secondary Actors:** Open WebUI, LLM Provider  
**Goal:** A corpus-wide count or yes/no prevalence question is answered with an estimate and its error margin, from a sample of papers checked by the model and recorded as a durable job.  
**Trigger:** A chat question is classified as a corpus-wide count.  
**Status:** Draft  

## Preconditions

- A complete global aggregate index exists for the live section build.
- The LLM Provider is reachable.

## Main Success Scenario

1. Chat User asks a corpus-wide count question in a chat.
2. System classifies the question as a count and selects the communities it covers.
3. System creates a job with a fixed plan: a pilot in every stratum, then bounded draws of papers without replacement.
4. System asks the LLM Provider whether each sampled paper answers yes, no or unsure, and records each answer as it arrives.
5. Open WebUI polls the job and resumes it when it was interrupted.
6. System computes the prevalence for each stratum and its error margin.
7. Open WebUI shows the estimate with the job's status.

## Alternative Flows

### A1: A provider call is uncertain

**Trigger:** The process stops or the LLM Provider fails during a call (step 4).  
**Flow:**

1. System marks the call as uncertain and does not repeat it silently.
2. System marks the job incomplete.
3. The job can be resumed by its caller, which repeats only the calls that did not complete.
4. Use case continues at step 4.

### A2: A reply is not one of the three answers

**Trigger:** The model's reply is not yes, no or unsure (step 4).  
**Flow:**

1. System does not count that paper.
2. Use case continues at step 4.

### A3: The question is not a count

**Trigger:** The question is not classified as a corpus count (step 2).  
**Flow:**

1. System passes the question to the normal question flow (UC-007).
2. Use case ends.

## Postconditions

### Success Postconditions

- The job is complete and each stratum has an estimate with its error margin.

### Failure Postconditions

- The job is marked incomplete.
- No partial estimate is shown as final.

## Business Rules

### BR-001: A job's plan is fixed at creation

A job records its model, prompt, seed, draw order and budgets when it is created, and they do not change.

### BR-002: A completed call is never repeated

A provider call that completed is returned from the job's record without calling the provider again.

### BR-003: Estimates carry their margins

The prevalence of each stratum is bounded, and each bound is widened by the finite-population error of the sample.

### BR-004: Only three answers are counted

A paper counts only when the model's answer is yes, no or unsure.

### BR-005: Provider concurrency is bounded

The number of provider calls running at once is bounded.
