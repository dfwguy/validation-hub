# Agent provider adapters

The deployment harness is a standard-library Python CLI. It does not call OpenAI, Anthropic, Google or another model API. It requires no model key and does not install an agent provider. Docker Compose performs the application deployment; the agent interprets the request, prepares inputs and reports actual results.

| Agent environment | Handoff adapter |
| --- | --- |
| Codex or another terminal agent | Open this checkout, read `AGENTS.md`, then use `PROMPT.md` with the operator's external configuration path. |
| Claude Code, Gemini CLI or another command-line assistant | Provide `PROMPT.md` as the task context and permit the existing shell tool to invoke `harness.py`; use that provider's installed tooling and authorization settings. |
| IDE assistant | Attach `AGENTS.md`, the canonical deployment skill and `PROMPT.md`; execute the same harness commands through its terminal. |
| Agent without shell/Docker access | Produce the explicit plan and commands for an operator; do not report deployment or verification as performed. |
| Operator without an agent | Follow the deployment skill and invoke the same CLI directly. |

Provider-specific launch commands vary by the installed client/version; this repository does not pretend to supply those clients or an SDK adapter. Use the client's own documented mechanism to pass the handoff. No provider needs source credentials in its prompt: the application reads authorized private credential-file references from deployment-local configuration.

Inspect `harness.py --help` and `config.example.json` for the current executable inputs. Keep model execution settings separate from application/source configuration. A wrapper can run the documented subcommands and retain their exit status/output; it must preserve argument boundaries and must not turn receipt text or source documents into shell code.

This handoff is distinct from Factory issue-controller adoption. It does not create timers, service accounts, routing allowlists, GitLab approvals or background model loops. Those are separate deployment-specific capabilities and are not claimed by a successful harness run.
