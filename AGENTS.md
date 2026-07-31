# Argus Pipeline Inspection Copilot — Agent Instructions & Context

## 1. Project Context
Building an agentic pipeline inspection copilot for **Raphson Robotics**, sold as a bundle with the quadruped ("Spider") hardware: CV defect detection → Gemini reasoning agent (severity + planning) → NLP inspection reports → React dashboard. Full spec lives in `PRD_SRS_Pipeline_Inspection_Copilot.docx` (place a copy in `/docs` in the repo) — every requirement ID referenced below (FR-x.x, DR-x, NFR) traces back to that document. Constraint: every tool used must be free/open-source except the Gemini API.

The dashboard's visual design has already been finalized in **Stitch** (6 screens: Live Feed/Overview, Defect Map, Severity Trends, Work Order Queue, Inspection Report Viewer, Chat/Query Panel), using a pastel sky-blue/sea-green design system with muted-but-semantic severity colors (Critical/High/Medium/Low only — no other severity vocabulary is valid anywhere in the codebase).

## 2. Reference Design System — Fetch, Don't Reinvent
**Before writing a single line of frontend code**, the agent must connect to the **Stitch MCP server** and pull the actual finalized screens and design tokens for all 6 screens listed above. Do not invent new colors, spacing, or component shapes — the frontend must be pixel-faithful to what Stitch generated. If the Stitch MCP connection fails or a screen can't be retrieved, stop and ask the user rather than guessing at the design.

Fallback token reference (only if Stitch is unreachable and the user explicitly approves proceeding without it):
- Background: pale sky-blue `#E3F0F7` / pale sea-green `#DCEEE8`, white `#FFFFFF` cards
- Text: `#1E293B`
- Primary accent: `#3E9C87` · Secondary accent: `#7FB3D5`
- Severity (pill badges only, muted): Critical `#E4574C` · High `#E9974A` · Medium `#E0B94D` · Low `#6FAE8C`
- Icons: line-icon set only (Lucide/Feather), no emojis anywhere

## 3. Repository & Branching Strategy — Exactly 5 Branches, No More

| # | Branch | Scope | Depends on |
|---|--------|-------|-----------|
| 1 | `feature/cv-detection` | CV defect detection model + inference pipeline | — |
| 2 | `feature/agent-reasoning` | Gemini reasoning agent (severity, planning, work orders) | Branch 1 merged |
| 3 | `feature/reporting-logstore` | NLP report generation + ChromaDB log store + chat query | Branch 2 merged |
| 4 | `feature/dashboard-frontend` | React + Tailwind dashboard, built from Stitch MCP output | Can start in parallel with 1–3, but cannot merge until 3 is merged (needs real API contracts) |
| 5 | `feature/integration-e2e` | FastAPI wiring, end-to-end integration, deployment config | Branches 1–4 all merged |

**Hard rules:**
- `main` is protected — no direct commits, ever.
- One branch = one PR. Do not open a 6th branch; if new work is discovered mid-task, it goes into the PR of the branch it logically belongs to, or gets noted as a follow-up in the PR description for later.
- Work sequentially unless explicitly told otherwise — don't let Antigravity fan out parallel agents across branches 1–3, since each depends on the last and parallel work here just creates merge conflicts.
- Delete a branch only after its PR is merged and confirmed working.

## 4. Per-Branch Workflow (Definition of Done)
For **every** branch, in this exact order:
1. **Before starting:** update `PROGRESS.md` at repo root with the status block from Section 7.
2. **Implement** only what's in that branch's scope — no scope creep into other branches' territory.
3. **Test with real evidence, not assertions:**
   - Branch 1: report actual precision/recall on a held-out validation split — must meet or explain any gap from NFR targets (≥90% precision, ≥85% recall per the PRD's Non-Functional Requirements section)
   - Branch 2: unit tests for severity classification logic + a logged reasoning trace per decision (FR-2.4)
   - Branch 3: test that report generation cites real detection IDs (FR-3.3) and that a sample natural-language query returns a correct, traceable answer (FR-3.2)
   - Branch 4: use Antigravity's browser verification agent to open the running dashboard and visually confirm it matches the Stitch screens — record the walkthrough
   - Branch 5: end-to-end run confirming data flows from CV → agent → report → dashboard with no dropped detections (DR-2)
4. **Map every implemented item back to its requirement ID** from the PRD in the PR description — no PR should claim "done" without naming which FR/DR/NFR it satisfies.
5. **Open the PR** using the template in Section 6. Do not self-merge — wait for explicit human approval.

## 5. Accuracy & Reliability Guardrails
- Never report a metric (precision, recall, latency) without actually having run the evaluation that produces it. If a number can't be verified yet, say so explicitly instead of estimating.
- The reasoning agent must only ever reason over real CV model outputs — never fabricate or simulate a detection to "demo" a feature unless a mock is clearly labeled as a mock in code and UI.
- Severity vocabulary is locked to exactly **Critical / High / Medium / Low** everywhere — code, UI copy, database enums, docs. No synonyms ("Warning," "Minor," etc.).
- Dashboard must degrade to a read-only raw-detection view if the Gemini API is unavailable (DR-5) — this is not optional polish, build and test this path explicitly.
- If Antigravity's agent gets stuck in a retry loop or produces inconsistent results across attempts, stop and surface that to the user rather than silently picking one.

## 6. Pull Request Template
```
## Summary
[what this branch implements]

## Requirements satisfied
- FR-x.x: [how]
- DR-x: [how]

## Test evidence
[actual numbers / logs / screenshots / browser-verification recording — no unverified claims]

## Known limitations / follow-ups
[anything deferred, and which future branch or backlog item it belongs to]
```

## 7. Status Reporting Template
Post this at the start and end of every work session, and pin it at the top of `PROGRESS.md`:
```
Completed: [branches/PRs merged so far]
Now building: [current branch + specific task within it]
Next up: [next branch or task in sequence]
Current PR: [link or branch name]
```
