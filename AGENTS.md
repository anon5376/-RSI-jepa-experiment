# Instructions for Codex, Devin, Pi, and other coding agents

**Read `README.md` first.** This project is a JEPA research experiment, initially an empty repository. No code, scientific success, login, or persistent worker exists unless verified in the current checkout/environment.

## Immediate priority: get the human onto Devin CLI safely

1. Read the repository and inspect installed tools, OS, shell, and resource limits. Check `git status --short`.
2. Follow `README.md` section "Devin CLI installation and human takeover"; the user explicitly wants setup information so that **they** can personally log in.
3. Use only the documented installer from `https://cli.devin.ai/install.sh` on a suitable user-controlled environment if Devin CLI is absent. If your host does not permit persistent installation, report the obstacle rather than pretending.
4. **Hand control to the user at the authentication prompt**. Suggest `devin auth login --force-manual-token-flow` for remote terminals (or `devin auth login` with a local browser). Stop. Never ask the user to paste a password, token, authentication link containing a token, or 2FA secret into your context. Never perform the human login yourself.
5. Only after the user confirms login, verify `devin auth status` and `devin models list`, then let the user select SWE-2 in `/model` if their entitlement allows it. Never assume model selection or pricing.
6. Proceed to the minimum reproducible JEPA baseline only when the login handoff is resolved or when the user explicitly chooses to skip Devin.

## Research execution rules

- Prefer Python, PyTorch, a small local CLI, and reproducible automated tests; CPU-first and fixed random seeds.
- Keep JEPA (latent representation prediction) distinct from the coding agent (LLM). Do not claim latent vectors by themselves constitute a natural-language assistant.
- Implement the simplest scientifically defensible experiment first, with baselines and collapse checks. Tests and exact configurations accompany claims.
- Commit small logical changes. Record hypotheses, controls, commands, seeds, metrics, failures, and conclusions in `EXPERIMENTS.md`; only include measured numbers.
- Before external model downloads, GPU runs, paid endpoints, large installations or a remote persistent service, check budget, permissions and environment.
- No fabricated citations, measurements, benchmarks or installation status.
- For each completed task: describe files changed, commands run, tests passing/failing, remaining blockers and the next objective.

## Agent/harness boundaries

- Devin CLI is the **verified** interactive SWE-2 path. Current free promotion is limited to qualifying Desktop/CLI usage through 2026-10-16, subject to account eligibility and actual terms. It does not imply a free SWE-2 API or free Devin Cloud compute.
- Pi is a separate optional harness. Install from `pi.dev` and use only a supported authorized provider. Do not invent a SWE-2 Pi adapter/API; demonstrate a legitimate provider connection before integrating.
- Hermes and any custom distributed harness are deferred until the Python experiment has a repeatable interface.
- A 24/7 loop requires an approved persistent runner, meaningful tasks, hard stop conditions, compute/time/usage budgets, checkpointing and logs; don't start an unrestricted unattended loop or bypass rate limits.
- Never commit `.env`, authentication files, model-service credentials, personal identifiers, large downloaded weights, full chat transcripts or sensitive URLs.
- Repository may be public. Treat all content as public until visibility is verified. Avoid posting private infrastructure details.

## Collaboration conventions

- `AGENTS.md` is the instruction source for Codex and compatible tools; `CLAUDE.md` points Claude to the same rules. Keep them in sync conceptually.
- If these rules conflict with a human request in the active session, ask for clarification on security, billing or publishing consequences instead of silently weakening controls.
- Do not modify/delete previous results just to produce a positive experiment outcome.
