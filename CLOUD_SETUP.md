# Owner-controlled cloud login: GitHub Codespaces

**Open this repository in your own cloud terminal:** https://codespaces.new/anon5376/-RSI-jepa-experiment?quickstart=1

This is a public research repository. The Codespace is a separate, account-controlled development environment; it does **not** grant the user access to a ChatGPT-hosted execution sandbox. All authentication happens in the owner's Codespace, not in GitHub commits or agent chat.

## Quick setup

1. Sign into your GitHub account. Open the link above. Choose **Create codespace**, or resume your own existing codespace. GitHub may show the machine type and the party responsible for billing. Choose an included/free machine if available and review limits before creating.
2. The VS Code web editor opens with a terminal. Read `README.md` and `AGENTS.md`.
3. In your Codespaces **terminal**, install Devin CLI from its documented official installer (review the script first if desired):

   ~~~bash
   curl -fsSL https://cli.devin.ai/install.sh | bash
   ~~~

   Follow any PATH instructions from the installer. Open a new terminal if necessary.
4. As the **human owner**, run:

   ~~~bash
   devin auth login --force-manual-token-flow
   ~~~

   Complete any instructions and paste any required token **only into your own Codespaces terminal**. Do not post token, login URL, one-time code, or auth logs in chats or the public repository. Never ask a remote assistant to enter your credentials.
5. After successful authentication, in the **same Codespaces terminal**:

   ~~~bash
   devin auth status
   devin
   ~~~

   In Devin, use `/model` to select SWE-2 **if your account currently offers it**. Price and quotas must be checked against your account and the provider's current terms.
6. Your first agent request should be to read `AGENTS.md` and implement the minimal CPU JEPA baseline with measured, reproducible tests. Commit code/reports to your public repository, but never secrets or credential dumps.

## Public experiment, private access

Your GitHub source repository stays **public** for portfolio and reproducibility. Your personal Codespace and authentication are separate. GitHub Codespaces uses account-based access; keep port forwarding private unless you have a specific reason to publish a demo. Use GitHub Codespaces secrets for additional API keys; never commit `.env` files.

## Availability limits and the 24/7 ambition

Codespaces is for **interactive setup and development**, not a guaranteed 24/7 free server. GitHub's documentation states it stops for inactivity (**30 minutes by default**) and has a **12-hour maximum lifetime for an active runtime**, after which it stops. Save work in Git regularly. Confirm current included quota, billing and retention before long sessions.

For real continuous research execution later, use a user-controlled paid VPS or other persistent runner with explicit compute budgets, checked billing, a supervisor, logs, secrets outside the public repo, job timeouts and manual cancellation. Do not infer that SWE-2 CLI promotional pricing covers remote hosting or Devin Cloud compute.

## Sources

- GitHub official deep link pattern: https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/setting-up-your-repository/facilitating-quick-creation-and-resumption-of-codespaces
- GitHub Codespaces creation: https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository
- GitHub Codespaces lifetime: https://docs.github.com/en/codespaces/about-codespaces/understanding-the-codespace-lifecycle
- Devin CLI installation: https://docs.devin.ai/cli
- Devin CLI authentication commands: https://docs.devin.ai/cli/reference/commands
