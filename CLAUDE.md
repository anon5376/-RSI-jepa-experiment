# Claude environment startup instructions

This repository is the **RSI JEPA Experiment**.

**At the beginning of every Claude/Codex session:** read `AGENTS.md` and `README.md` and follow them. They are the source of truth for the project goal, reproducible JEPA tasks, user ownership and credentials.

## First run in a new Claude environment

- Inspect the repo and report which tools actually exist (`devin`, `pi`, `python`, `git`); do not report imagined installations.
- The top priority is **Devin CLI setup followed by human-controlled login**. See README section 1 for the verified official install command and `devin auth login --force-manual-token-flow` for remote login.
- **Stop and hand the interactive terminal to the owner at authentication.** The owner signs in. Claude must not request, retrieve, display or save secrets.
- If the hosted Claude environment cannot grant the owner direct terminal access, state that limitation explicitly. Provide a safe local-machine alternative; do not log in through an agent.
- Once authentication is confirmed by the owner, check `devin auth status` and `devin models list`, ask the owner to select SWE-2 in Devin's `/model`, then pursue the small JEPA baseline.
- Do not spawn indefinite loops or purchase cloud/model usage. Cloud runtime and model entitlements must be checked before unattended work.

The repo currently provides **instructions**, not an installed harness or implemented JEPA model. Begin with a minimal testable Python implementation rather than a generic agent framework.
