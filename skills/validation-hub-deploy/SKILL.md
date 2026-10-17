---
name: validation-hub-deploy
description: Prepare, deploy, verify and hand over NVIS-OVA Validation Hub in a new geography using explicit runtime settings and the repository deployment harness. Use for Hub application deployment, not for submitting generated GPU benchmarks or changing fabric policy.
---

# Deploy Validation Hub

Use the current checkout's [AGENTS.md](../../AGENTS.md), [deployment guide](../../docs/deployment.md), [operations guide](../../docs/operations.md) and [harness handoff](../../agent-harness/PROMPT.md). Source connectors and their authentication boundaries are documented in [reference integrations](../../docs/reference-integrations.md).

Confirm the requested target, checkout/ref, persistent volume, localhost port and external runtime directory. Use the user's existing authorization. Keep credentials as private files outside the checkout and image; request missing file references instead of tokens in chat. A missing optional connector does not prevent an empty Hub deployment.

Use the harness's current `--help` and `agent-harness/config.example.json` to construct an explicit private configuration. Run `doctor` and `plan` before an authorized `deploy`; inspect their actual results. The harness uses Docker Compose and deployment-local receipts. It is provider-neutral and does not invoke a model or enroll a resident agent controller. If Docker/Compose is unavailable, finish the reviewable plan and report that blocker rather than claiming a running deployment.

After deployment run `verify` and inspect the health response, service/container state and persistent storage. Check the empty application before importing approved test evidence. Source access, reference refresh and hardware/fabric recipe checks require their own actual inputs and evidence; a healthy HTTP endpoint does not prove successful source login, a qualified cluster or a GPU benchmark result.

Use `report` for handover. Include the exact deployed commit, configured data/volume locations, localhost/proxy endpoint, checks actually executed, connector states, and remaining site-specific steps. Keep the detailed receipt private. The `stop` command preserves the named data volume; verify the operator's intended scope before any operation outside that behavior. Do not delete a volume or original upload as routine recovery.

For software delivery, follow [the Factory contract](../../docs/factory.md): run required checks, package the committed tracked source with `python3 tools/package_release.py --output dist`, and verify its manifest/checksum with `tools/verify_release.py`. Record exact commit and pipeline evidence. A source package, passing fixture test or configured CI file does not establish deployment in another geography.
