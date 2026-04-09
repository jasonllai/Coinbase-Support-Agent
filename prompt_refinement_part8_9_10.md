# Prompt Refinement for Parts 8, 9, and 10

## Scope

This document covers prompt refinement for:

- Part 8: Repo Quality and Documentation
- Part 9: Implementation Preferences
- Part 10: Bonus Polish / Going Beyond Minimum

## Deliverables in This File

This file contains:

- what was refined
- why those refinements were made
- the final refined prompt text

## What I Refined

### 1. Converted vague quality language into concrete deliverables

The original prompt used strong but broad language such as "clean," "professional," "beautiful," and "polished." Those phrases communicate intent, but they do not tell the model exactly what to produce.

Refinement made:

- converted broad quality goals into explicit deliverables
- added required sections and content expectations for each documentation file
- added quality rules that are specific enough to verify

Why this improves the prompt:

- reduces ambiguity
- makes outputs easier to evaluate
- increases the chance that the model produces usable project artifacts instead of generic prose

### 2. Turned Part 8 from a checklist into a documentation specification

The original Part 8 mostly listed which documents should exist. That is useful, but incomplete. A model can satisfy that by producing shallow or repetitive documents.

Refinement made:

- defined the role of each document
- specified the audience and purpose of each document
- added content requirements and writing quality expectations
- added synchronization rules so documents must match the implemented repository

Why this improves the prompt:

- pushes the model toward useful documentation instead of placeholder documentation
- better supports grading, demo preparation, and teammate handoff
- reduces the risk of documentation describing features that do not actually exist

### 3. Reframed Part 9 as implementation constraints, not just preferences

The original Part 9 included good technical preferences, but they were loosely grouped and partly aspirational.

Refinement made:

- organized the section into technical baseline, code quality requirements, model integration requirements, engineering priorities, implementation guardrails, and module expectations
- made architectural expectations more explicit
- emphasized validation, fallback behavior, and explainability

Why this improves the prompt:

- gives the model clearer engineering boundaries
- supports maintainability and demo reliability
- helps prevent overengineered or hard-to-explain solutions

### 4. Added explicit anti-overengineering guidance

The course project context matters. A system can be technically impressive but still be a poor fit if it is hard to explain in a live presentation.

Refinement made:

- added direct instructions not to overengineer infrastructure
- added direct instructions not to hide critical logic behind overly abstract layers
- prioritized simple, robust, explainable design choices

Why this improves the prompt:

- aligns the implementation with course goals
- reduces unnecessary complexity
- improves the team's ability to answer Q&A during demo

### 5. Structured Part 10 into priority tiers

The original bonus section was a flat list of optional ideas. That makes it easy for a model or team to chase low-value extras before core requirements are stable.

Refinement made:

- added a selection rule that bonus work only happens after core functionality is stable
- grouped bonus features into Tier 1, Tier 2, and Tier 3
- prioritized bonus items by demo value, user trust, and implementation risk

Why this improves the prompt:

- prevents feature sprawl
- encourages high-impact polish first
- keeps the project focused on stability and presentation value

### 6. Added documentation truthfulness and repo alignment rules

One common failure mode in LLM-generated documentation is that it describes intended features as if they were already implemented.

Refinement made:

- explicitly required documentation to reflect the actual repository state
- explicitly prohibited describing unimplemented features as completed
- required docs to stay synchronized with code and evaluation artifacts

Why this improves the prompt:

- improves credibility
- reduces inconsistency between code and docs
- makes the repo safer to present to graders and teammates

### 7. Improved the prompt for evaluation and explainability, not just output volume

The original sections leaned heavily on completeness. I refined them so they also optimize for explainability.

Refinement made:

- emphasized concise, reliable, presentation-ready documentation
- prioritized outputs that support demo, grading, and teammate onboarding
- added rules that prefer clarity over verbose filler

Why this improves the prompt:

- makes the results more usable in an actual meeting or demo
- reduces low-signal documentation
- better fits a course project setting

## Refinement Summary

In short, the refinement did not change the high-level goals of Parts 8, 9, and 10. It changed how those goals are expressed.

The refined version:

- is more specific
- is easier to evaluate
- is better aligned with course demo constraints
- is more likely to generate realistic, synchronized, explainable outputs

## Final Refined Prompt

```text
==================================================
PART 8 — REPO QUALITY AND DOCUMENTATION
==================================================

The repository must be presentation-ready, easy to navigate, and easy for every teammate to explain during demo and Q&A.

Documentation requirements are not optional. Each document must be concise, concrete, and aligned with the implemented system. Avoid vague marketing language. Prefer direct descriptions of what the system does, how it is structured, and how to run it.

Create and maintain the following documentation assets:

1. README.md
The README must function as the primary entry point for a new reviewer, teammate, or grader. It should allow someone to understand the project quickly and run it reliably.

Required sections:
- project overview
- key features
- architecture summary
- setup and installation
- environment variables
- data ingestion process
- indexing / retrieval build process
- backend and frontend startup
- evaluation workflow
- deployment workflow
- limitations
- attribution of major dependencies and external resources

README quality requirements:
- clearly state that this is an educational Coinbase-support demo, not an official Coinbase product
- clearly distinguish between real knowledge grounding and mock operational actions
- include a short architecture diagram or architecture summary
- include exact commands for the most important setup and run steps
- keep instructions short, reliable, and copy-pasteable
- describe generated artifacts such as corpus, FAISS index, eval outputs, and persistent SQLite data
- document the expected demo path
- avoid stale placeholder sections or TODO language

2. docs/architecture.md
This document must explain the system in a way that supports both technical understanding and live presentation.

Required content:
- end-to-end data flow
- ingestion and indexing pipeline
- retrieval design
- orchestration / graph design
- storage and persistence design
- safety / guardrail design
- major trade-offs and limitations

Quality requirements:
- explain why each major component exists
- explain how LangGraph routing works at a high level
- explain how citations are produced
- explain how persistence supports memory, tickets, and recovery flows
- explain important design trade-offs in plain language
- avoid excessive implementation trivia unless it improves explainability

3. docs/demo_script.md
This document must support a smooth 5-minute live demo and a fallback demo path.

Required content:
- recommended live demo flow
- backup demo flow
- strongest prompts to use
- expected outputs to highlight
- likely professor / TA questions
- short, strong answers for those questions

Quality requirements:
- the script should reflect the actual implemented system
- prompts should cover at least one KB question, one action, one multi-turn flow, and one guardrail / refusal
- include a fallback plan if the live system fails
- optimize for clarity and timing, not completeness

4. docs/presentation_notes.md
This document must help the team build a clean executive-style deck.

Required content:
- suggested 5-slide story
- what metrics to show
- what product / engineering polish to emphasize
- key innovation points
- how to discuss limitations honestly and defensibly

Quality requirements:
- support a concise presentation narrative
- map evaluation results to course requirements
- highlight explainable engineering choices rather than vague claims
- include one or two clear statements about what makes the system feel more polished than a baseline course project

5. docs/team_handoff.md
This document must help every teammate answer questions outside their own implementation area.

Required content:
- simple explanation of each major module
- what each part is responsible for
- how data flows through the system
- what the main actions do
- what the guardrails do
- what to say if asked about limitations or trade-offs

Quality requirements:
- explain modules simply, without assuming deep code familiarity
- optimize for fast teammate onboarding and demo Q&A readiness
- avoid restating source code mechanically; explain purpose and behavior

Documentation execution rules:
- every document must reflect the actual repository state
- do not describe features that are not implemented
- keep documents synchronized with final code and evaluation outputs
- prefer fewer, clearer documents over verbose but low-signal prose
- write for graders, teammates, and demo observers, not just developers


==================================================
PART 9 — IMPLEMENTATION PREFERENCES
==================================================

Use the following implementation constraints to keep the project maintainable, explainable, and demo-reliable.

Technical baseline:
- Python 3.11+
- FastAPI backend
- Streamlit frontend
- LangGraph for orchestration
- FAISS for vector retrieval
- SQLite for persistent session and mock action state unless a simpler justified alternative is clearly better
- Pydantic models for typed schemas and structured outputs

Code quality requirements:
- use modular service boundaries; avoid large monolithic files
- keep routing, retrieval, actions, storage, and UI concerns clearly separated
- use strong typing where practical
- use environment-based configuration
- use structured logging for important runtime events
- add comments only when they materially improve readability
- prefer explicit control flow over hidden framework magic when explainability matters

Model integration requirements:
- integrate the provided Qwen endpoint through a dedicated client abstraction
- keep prompts centralized and easy to inspect
- prefer structured JSON outputs for router, guardrails, and action parameter extraction
- validate model outputs before use
- handle malformed structured outputs gracefully
- implement retries, fallbacks, or safe degradation where appropriate

Engineering decision priorities:
1. demo reliability
2. requirement coverage
3. explainability in presentation and Q&A
4. maintainability
5. polish

Implementation guardrails:
- do not overengineer infrastructure that does not improve the demo
- do not introduce unnecessary abstractions that make the system harder for students to explain
- do not hide critical logic inside overly clever helper layers
- do not rely on fragile prompt behavior without validation or fallback handling
- prefer simple, robust components over ambitious but brittle designs

File and module expectations:
- keep prompt definitions centralized
- keep schemas explicit and version-stable where possible
- keep storage access isolated behind a simple interface
- keep retrieval pipeline reproducible
- keep evaluation code separate from runtime serving code
- keep deployment configuration minimal and transparent

The final implementation should look like a clean student-built product with professional engineering discipline, not an overcomplicated research prototype.


==================================================
PART 10 — BONUS POLISH / GOING BEYOND MINIMUM
==================================================

If time permits after all required functionality is complete and stable, implement a small number of high-value polish features that materially improve demo quality.

Selection rule:
- prioritize bonus features only after core requirements are working end to end
- choose features that improve product clarity, user trust, or demo impact
- avoid bonus features that add large complexity with little presentation value

Recommended bonus features, in priority order:

Tier 1 — High-value, low-risk polish
- source preview drawer or expandable source panel
- confidence / evidence labels for KB answers
- suggested follow-up questions
- improved refusal UX with safer redirection
- session resume
- admin / debug view showing route, retrieved docs, and tool trace

Tier 2 — Useful if stable
- lightweight retrieval caching
- reranking
- conversation export
- synthetic mock transaction dataset with more realistic fields
- simple analytics view for evaluation results

Tier 3 — Only if already stable and easy to explain
- multiple demo personas or scenario presets
- tasteful animations or transitions in Streamlit
- additional UI affordances that improve clarity without increasing fragility

Bonus feature quality requirements:
- each bonus feature must be visible, demoable, and easy to explain
- each bonus feature must improve trust, usability, or evaluation clarity
- do not add hidden complexity that increases failure risk during live demo
- do not sacrifice reliability for cosmetic polish

If bonus features are implemented, document:
- what was added
- why it improves the product
- how it should be shown during the demo
- any trade-offs or limitations introduced

The project should exceed the minimum requirements through focused polish, not feature sprawl.
```

## Short Meeting Summary

If you need a short explanation during the meeting, you can say:

"I refined Parts 8, 9, and 10 by making them more specific, more measurable, and more aligned with demo needs. For Part 8, I turned the documentation checklist into a document-by-document specification with content and quality requirements. For Part 9, I converted technical preferences into explicit engineering constraints and anti-overengineering rules. For Part 10, I prioritized bonus features into tiers so the team focuses on high-impact polish without creating feature sprawl."
