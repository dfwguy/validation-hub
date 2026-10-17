# Deploy with your own agent

Any authorized local coding agent or operator can run this provider-neutral Python standard-library harness. It does not require a model SDK, API key, Factory controller, global Python install, sudo, scheduler access, or a GPU.

Read [easy deployment](../docs/easy-deploy.md), then give your agent [PROMPT.md](PROMPT.md) and an operator-provisioned configuration outside the checkout. The [example configuration](config.example.json) contains placeholders only.

```bash
python3 agent-harness/harness.py plan --config /absolute/private/deploy.json
python3 agent-harness/harness.py doctor --config /absolute/private/deploy.json
python3 agent-harness/harness.py deploy --config /absolute/private/deploy.json
python3 agent-harness/harness.py verify --config /absolute/private/deploy.json
python3 agent-harness/harness.py report --config /absolute/private/deploy.json
python3 agent-harness/harness.py stop --config /absolute/private/deploy.json
```

`plan` is read-only and requires no Docker. `doctor` checks prerequisites without changing the service. `deploy` renders the existing deployment template into private external state, uses the approved local Docker Engine and Compose plugin, and verifies HTTP health plus summary evidence. `stop` stops only the named matching deployment and retains its persistent volume. No operation executes GPU tests or disables authentication/TLS checks. External access still requires the enterprise authenticated TLS proxy described in the deployment guide.

Results are JSON; exit zero means that command completed, not that other phases ran. `DEPLOYED` requires observed verification. `BLOCKED` or `FAILED` needs the exact missing prerequisite or local container-state review. `report` returns the latest receipt and its original observation time; it never rechecks current health. There is no automatic rollback or volume deletion. Docker deployment could not be executed on the packaging workstation because Docker was unavailable; tests exercise configuration, rendering, orchestration failures, and real local HTTP verification.
