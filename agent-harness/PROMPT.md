# Provider-neutral deployment handoff

Use this prompt with any agent that can inspect this checkout and run local commands. Replace the bracketed inputs with actual operator-provided values; keep private settings in the external configuration file.

> Deploy or prepare NVIS-OVA Validation Hub from this checkout according to the authorized scope: [prepare / deploy / verify / report]. Target geography/environment: [target]. Source ref/commit: [ref]. Runtime configuration file: [absolute path outside checkout]. Read AGENTS.md and skills/validation-hub-deploy/SKILL.md. Inspect the actual harness CLI and configuration example. Continue authorized work, resolving routine reversible issues; obtain only missing operational inputs. Preserve customer evidence and persistent volumes. Report actual commands/checks, exact deployed commit, receipt location, connector availability and remaining blockers. Do not claim GPU qualification or source access from a healthy application endpoint.

The standard harness interface is:

```text
python3 agent-harness/harness.py doctor --config /absolute/runtime/deploy.json
python3 agent-harness/harness.py plan --config /absolute/runtime/deploy.json
python3 agent-harness/harness.py deploy --config /absolute/runtime/deploy.json
python3 agent-harness/harness.py verify --config /absolute/runtime/deploy.json
python3 agent-harness/harness.py report --config /absolute/runtime/deploy.json
```

Use the current CLI help as the executable contract. `doctor` and `plan` inspect readiness; `deploy` performs the authorized Compose installation and verification. `stop` stops the deployment without deleting its persistent volume. Runtime configuration and receipts stay outside the checkout. Read [PROVIDERS.md](PROVIDERS.md) only when adapting the handoff to a particular agent environment.
