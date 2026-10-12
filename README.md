# RSI JEPA Experiment

An independent research sandbox for **Joint-Embedding Predictive Architectures (JEPA)** and small, reproducible representation/world-model experiments.

**Status:** bootstrap instructions only. No JEPA implementation, successful installation, authenticated session, benchmark results, or always-on worker is claimed yet.

## Launch in your own cloud terminal (fastest login route)

**[Open or resume this public repository in GitHub Codespaces](https://codespaces.new/anon5376/-RSI-jepa-experiment?quickstart=1)**

Follow **[CLOUD_SETUP.md](CLOUD_SETUP.md)** to install Devin CLI, personally authenticate inside your own browser-based terminal, and keep private credentials out of Git. This provides direct terminal control in your GitHub account; it does not expose the ChatGPT execution machine. Codespaces has inactivity and 12-hour runtime limits and is not a free guaranteed 24/7 server.

## Start here (Codex, Claude, Devin, Pi agents)

1. Read [AGENTS.md](AGENTS.md) and this README before touching code.
2. Inspect the checkout and report the current state rather than assuming code already exists.
3. **First operational task:** prepare Devin CLI in the user's own terminal/environment and explicitly **hand control to the user for authentication**. Never attempt to sign in for the user.
4. After the user confirms login, work on the smallest reproducible JEPA baseline, not a giant speculative platform.

The intended arrangement is:

~~~text
Human researcher
  | objectives, approvals, sign-in, reviews
  v
Coding agent: Devin CLI (SWE-2) OR Claude/Codex
  | writes and tests Python; records results in Git
  v
Python JEPA experiment (CPU first; optional GPU)
  | embeddings, latent predictions, metrics, checkpoints
  v
Reports/experiments/benchmarks in this repository

Optional: Pi coding harness (separately authenticated supported model)
  | orchestrates reproducible Python runs and tools
  +-- NOT an assumed direct SWE-2 API client
~~~

**Technical boundary:** JEPA is self-supervised latent/representation prediction, not a drop-in language-model chat runtime. The coding LLM may *design, call, train, evaluate, and inspect* a local JEPA implementation; JEPA does not thereby become the coding LLM. Connect components through files/CLI/structured outputs first. No claim of JEPA replacing an LLM without actual evidence.

## 1. Devin CLI installation and human takeover

The user said "Devin SL"; this bootstrap assumes **Devin CLI**, the official terminal agent, not an unrelated product. Official references: [Devin CLI Quickstart](https://docs.devin.ai/cli), [Commands & Flags](https://docs.devin.ai/cli/reference/commands).

**For agents operating in a user-controlled macOS/Linux/WSL terminal:**

- First verify the environment: `pwd`, `uname -a`, `command -v devin`, `git status --short`. Do not assume a hosted/temporary cloud shell will persist.
- If `devin` is missing, the official installation command is:

~~~bash
curl -fsSL https://cli.devin.ai/install.sh | bash
~~~

Prefer reviewing the downloaded official installer before execution when practical. After installation, restart the shell or update PATH as instructed by the installer, and check:

~~~bash
devin --version
devin auth status
~~~

**Stop at the authentication checkpoint.** The agent must give the interactive terminal / user-facing session back to the **human owner**, with this exact next command:

~~~bash
devin auth login --force-manual-token-flow
~~~

This is the documented remote/SSH option; on a desktop the user can instead run `devin auth login`. Any browser link, token, redirect, password, 2FA challenge or other credential must be completed/entered **by the human privately**, never posted in GitHub, an agent conversation, terminal transcript, screenshots or logs. If the agent environment does not permit a real human terminal takeover, **stop and report that blocker**. Do not pretend authentication succeeded or attempt to automate account ownership transfer. The agent may resume only after the user confirms that login was completed.

After the user has authenticated, verify **only non-secret status**:

~~~bash
devin auth status
devin models list
devin
~~~

Within interactive Devin, use `/model` to select **SWE-2**, if the logged-in account actually offers it. Do not invent a CLI model slug or silently fall back to another paid model. Confirm the selected model and cost/free status in the account before launching large workloads.

**Promo checkpoint (verified 12 Oct 2026):** [Devin's official pricing page](https://devin.ai/pricing) advertises free SWE-2 **in Devin Desktop and CLI through 16 October 2026** on qualifying paid plans. This does **not** prove free Devin Cloud sessions, lifetime unlimited service, API access, or availability after that date. Account, rate, quota, concurrency and provider terms remain binding. The agent must not bypass quotas, rotate accounts to evade limits, or activate paid usage without explicit permission.

Devin's `/handoff` can send a local session into Devin Cloud; this is **not** equivalent to keeping the CLI SWE-2 promotion in effect, nor a guarantee of 24/7 operation. Verify cloud pricing, execution lifetime and persistence separately before using `/handoff` or `devin --cloud`.

## 2. Optional Pi harness (preferred for an extensible research loop)

Pi is useful as a small, inspectable coding-agent shell for controlling local Python tools, notebooks/scripts and experiment records. **Do not assume SWE-2 is callable from Pi:** a supported SWE-2 endpoint and authentication mechanism have not been established; Devin CLI is the supported SWE-2 interface in this plan.

Official [Pi Quickstart](https://pi.dev/docs/latest/quickstart) on macOS/Linux:

~~~bash
curl -fsSL https://pi.dev/install.sh | sh
pi
~~~

Alternative if Node.js 22.19+ is available:

~~~bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
pi
~~~

Inside Pi, the **human** should use `/login` for a supported provider and `/model` to choose a model. See [Pi model configuration](https://pi.dev/docs/latest/models). When Pi runs in a remote shell, the human must complete its OAuth/device/browser prompts. Never store API keys in project files.

**Choice:** Use Devin CLI for the temporary SWE-2 experiment now. Use Pi **later or alongside it** for a reproducible, model-agnostic harness with a legitimately supported provider. Hermes is not necessary for the first JEPA experiment. Don't build a custom multi-agent harness before a baseline runs.

## 3. Research objectives and measurable milestones

Phase A — **reproducible minimal JEPA** (small synthetic sequences, CPU only):
- Implement a context encoder, target/teacher encoder (e.g. EMA), predictor, masked target selection, and stop-gradient target branch in PyTorch or a justified equivalent.
- Start with synthetic latent dynamics with known ground truth, fixed seeds, deterministic splits and pinned dependencies. Verify that inputs/targets don't leak across train/test.
- Compare against trivial last-state/mean, linear regression, and comparably sized supervised/MLP predictive baselines.
- Log predicted-target embedding loss, embedding variance/collapse diagnostics, held-out prediction error and downstream linear-probe accuracy/regression error with at least 3 seeds. Report wall time, hardware and memory.
- Run an ablation without predictive training or without meaningful masking to test what JEPA contributes. Stop if results are indistinguishable from trivial baselines.

Phase B — **meaningful small-world model**:
- Move from toy sequence data to a simple controllable dynamical process or a modest public time-series/video representation task.
- Test whether learned embeddings support state estimation and multi-step latent prediction better than baseline methods.
- Maintain a separation between **observation**, **latent state**, **prediction** and **action/control**. Do not claim planning or understanding until measured.

Phase C — **agent integration**:
- Make a local `python -m jepa_lab ...` CLI (arguments and subcommands to be implemented and documented) that executes training/evaluation and writes structured, schema-versioned JSON metrics.
- Have the coding agent propose a hypothesis, choose a bounded compute budget, run experiments, compare outcomes, and append a factual log entry in `EXPERIMENTS.md`.
- Investigate a Pi extension only after these CLI contracts work. Never force JEPA into a text-generation API.

Recommended layout **once code is created**:

~~~text
src/jepa_lab/       # dataset/model/training/evaluation modules
tests/              # deterministic unit and smoke tests
configs/            # versioned small experiments
reports/            # human-readable summaries and metrics
EXPERIMENTS.md      # dated hypothesis, config, seed, result, next step
pyproject.toml      # pinned Python dependencies
~~~

Don't commit datasets, large checkpoints, secrets or raw provider transcripts. Don't invent results. If compute/GPU is unavailable, run a tiny CPU experiment and state limitations.

## 4. Limited autonomous operation

The user's long-term goal is cloud-hosted work, potentially around the clock. Treat this as a **future infrastructure milestone**, not a claim about today's environment.

Before unattended execution, establish: (1) a persistent host independent of ephemeral chat sessions, (2) explicit human approval of the host, credential method, charges and quotas, (3) a cancellable job supervisor with hard time/spend/compute ceilings, (4) durable Git commits and artifacts, (5) monitoring, checkpoints, and clear failure stop conditions. Avoid infinite prompting, permission bypasses and automatic purchases. A time-limited model promotion does not fund continuous hosting, remote GPU compute or Devin Cloud execution.

A reasonable first cycle is **one hypothesis -> one testable code change -> unit/smoke tests -> one small experiment -> report -> commit**, not uncontrolled 24/7 model calls.

## 5. First agent response expected

Report succinctly:
1. Repository, OS, available Python/CPU/GPU and installed `devin`/`pi` binaries.
2. Whether the user can take over the *same* interactive terminal for Devin login; if yes, display `devin auth login --force-manual-token-flow` and **pause** at the checkpoint.
3. Confirmed SWE-2 availability and price *after human login* only.
4. The exact next smallest implementation milestone and test command.
5. Any constraints on cloud lifetime, network, rate limits or billing.

## Official references

- Devin CLI: https://docs.devin.ai/cli
- Devin CLI commands/auth: https://docs.devin.ai/cli/reference/commands
- Devin model/pricing offer: https://devin.ai/pricing
- Pi installation: https://pi.dev/docs/latest/quickstart
- Pi providers/models: https://pi.dev/docs/latest/models
- Meta I-JEPA research: https://ai.meta.com/research/publications/self-supervised-learning-from-images-with-a-joint-embedding-predictive-architecture/
