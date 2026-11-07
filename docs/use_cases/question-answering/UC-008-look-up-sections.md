# Use Case: Look Up Sections

## Overview

**Use Case ID:** UC-008  
**Use Case Name:** Look Up Sections  
**Primary Actor:** API Client  
**Goal:** A calling program retrieves sections of the live corpus, a section by its key, the list of communities, or a community's terms, without asking a question of its own.  
**Trigger:** The API Client requests sections matching words, one section by key, the community list, or the terms of a community.  
**Status:** Draft  

## Preconditions

- The retrieval service is reachable.
- A live section build exists.

## Main Success Scenario

1. API Client asks for the sections that best match its words.
2. System returns each matching section with its whole text, its paper and its community.
3. API Client asks for one section by its key.
4. System returns that section with its paper and its community.
5. API Client asks for the community list.
6. System returns each community of the live build with its title.
7. API Client asks for the terms of a community.
8. System returns the community's distinctive terms.

## Alternative Flows

### A1: A request is out of range or names an unknown section

**Trigger:** A parameter is out of range, or the section key is unknown (step 1).  
**Flow:**

1. System refuses the request and names the problem.
2. System returns no partial result.
3. Use case ends.

### A2: No live build exists

**Trigger:** The service has no live section build (step 1).  
**Flow:**

1. System reports that it has no live build.
2. System returns no results.
3. Use case ends.

## Postconditions

### Success Postconditions

- The caller receives the requested sections, communities or terms of the live build.

### Failure Postconditions

- Nothing is stored or changed by the request.

## Business Rules

### BR-001: Whole sections are returned

Sections are returned with their whole text, never a fragment.

### BR-002: References and junk are left out

Reference and junk chunks are left out of section results unless the caller asks for them.

### BR-003: The live build only

Every answer comes from the build the service has published as live.
