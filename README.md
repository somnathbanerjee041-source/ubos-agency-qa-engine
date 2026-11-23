# UBOS — Agency Brief-to-Delivery QA & Revision Engine (Phase 1 MVP)

**Homingo Hack Days | Track: Agents | Problem Statement: A2 — The full-stack AI firm**

UBOS is an evidence-backed Quality Assurance and revision engine that helps small digital agencies verify video scripts and content drafts against client briefs and brand guidelines before delivery.

## 1. Who It Is For, What It Does & What "Finished" Looks Like
- **Who It Is For:** Small digital, content, and social media agencies managing creative writers, editors, and client deliveries.
- **One Core Task:** Runs deterministic checks and semantic compliance verification on briefs and time-coded scripts, generating an editor-ready revision checklist with mandatory Human-in-the-Loop approval.
- **What "Finished" Looks Like:** A working browser prototype with testable scenarios (`TEST 1`, `TEST 2`, `TEST 3`), Layer 3 audit trail logging, and human approval or revision controls.

## 2. How to Install
1. Clone or download this public repository (`ubos-agency-qa-engine`).
2. Keep local environment/API secrets out of source control (refer to `.env.example` and `.gitignore`).
3. Obtain a Google Gemini API key from Google AI Studio if running Live Gemini mode (the app also includes internal test modes and browser `sessionStorage` key handling).

## 3. How to Run
1. Open the project/app in a browser (via Google AI Studio preview or local dev/build setup using `index.html` and project files).
2. Enter a Gemini API key in the session input if using Live Gemini mode, or select a test scenario/mode.
3. Run the UBOS QA audit, review the findings and revision checklist, and click the Human-in-the-Loop approval/revision action to record the outcome in the audit trail.

## 4. AI Tools, Models & Libraries Used
- **AI Model:** Google Gemini (`gemini-2.5-flash`).
- **AI Tools & Assistants:** Google AI Studio and Gemini.
- **Tech Stack & Files:** HTML5 (`index.html`), TypeScript/JavaScript (`src`, `tsconfig.json`), Vite (`vite.config.ts`), `package.json`, and `metadata.json`.

## 5. What We Would Add Next
- Direct video/audio timeline inspection for editors and automated multi-channel brief ingestion.
- Backend proxy/authentication and team role permissions for multi-user agency workspaces.
- Webhook/workflow integrations (such as Slack, Notion, or drive sync) with human approval gates.
