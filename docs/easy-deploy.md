# Easy deployment with an agent

Use an approved local Docker Engine and Compose plugin on the application server. Your agent runs ordinary commands under its existing deployment identity. The harness does not install Docker, change group membership, use sudo, contact compute nodes, execute GPU tests, disable authentication, or configure a public endpoint.

## Get the framework

```bash
git clone --branch framework-enhancements --single-branch https://gitlab-master.nvidia.com/nvis-ova/validation-hub.git
cd validation-hub
install -d -m 700 "$HOME/.config/validation-hub"
cp agent-harness/config.example.json "$HOME/.config/validation-hub/deploy.json"
chmod 600 "$HOME/.config/validation-hub/deploy.json"
```

Edit that JSON with an absolute private `runtime_dir`, a unique geography-local project/volume name and an available localhost port. Keep `environment_file` and `secrets_dir` null for the first empty startup. Use the same absolute configuration path in every command below. GitLab clone access uses your organization's normal credential mechanism; never embed an access token in the URL. For a release rollout, use the reviewed immutable commit or verified source archive rather than assuming a moving branch is pinned.

## Prepare the local inputs

Clone the reviewed release and check its source revision. Use Python 3 to run the harness; application dependencies are installed inside the Python 3.14 container image. Docker must already be available to the deployment identity without sudo. Docker's active context must use a local Unix socket, not a remote daemon. Keep the checkout and runtime directory separate.

Copy [config.example.json](../agent-harness/config.example.json) to an operator-managed private directory outside the checkout. Replace the placeholder `runtime_dir` with an absolute directory this identity may create. Choose a unique `project_name` and `persistent_volume` for this geography. Do not reuse an unrelated application's volume. Both names must contain lowercase letters, digits, hyphens or underscores and start with a letter. Choose a free localhost port between 1024 and 65535; the container always listens on 8081. Leave optional source integration settings null for an empty first deployment.

The JSON parser rejects duplicate or unknown keys. Configuration contains paths and deployment settings only, never passwords, cookies, authorization headers or tokens. Existing runtime directories must have mode 0700. Receipt/configuration files are created with mode 0600. Do not put the configuration, evidence, downloaded sources or source credentials into Git.

## Run or delegate the workflow

Give your agent [the portable prompt](../agent-harness/PROMPT.md) and the configuration path. From the reviewed checkout:

```bash
python3 agent-harness/harness.py plan --config /absolute/private/deploy.json
python3 agent-harness/harness.py doctor --config /absolute/private/deploy.json
python3 agent-harness/harness.py deploy --config /absolute/private/deploy.json
python3 agent-harness/harness.py verify --config /absolute/private/deploy.json
python3 agent-harness/harness.py report --config /absolute/private/deploy.json
```

`plan` prints exact argument arrays and source/template/rendered hashes without Docker or filesystem mutation. `doctor` checks strict configuration, private paths, local Docker context, Compose availability and daemon access. Missing prerequisites return `BLOCKED` with exit code 2; resolve them through the platform owner and rerun. No automatic installation or privilege change occurs.

`deploy` derives a private Compose file from [the existing template](../deploy/compose.yaml), retaining one application worker, UID/GID 10001, read-only root filesystem, temporary storage, and capability restrictions. It builds from the current checkout, binds only `127.0.0.1:<host_port>`, and uses the explicitly named persistent volume. No copied production database or customer data is needed. It waits for `GET /api/health` to return HTTP 200 with `ok=true` and for `/api/summary` to return HTTP 200 with a valid record count. Deployment also checks that Compose reports exactly one running application service with the reviewed localhost port mapping. This is lightweight service/readiness evidence, not image-digest attestation or proof of an authenticated proxy. Standalone `verify` also requires the matching rendered deployment, project ownership, local Docker prerequisites and running Compose service/port evidence before HTTP checks. Verification records aggregate counts rather than customer names or full API responses. An empty catalog is valid startup evidence.

Private receipts live under `<runtime_dir>/receipts/`; `<runtime_dir>/latest.json` is the latest command receipt. They include the Git commit and dirty state when available (or source-archive manifest hash), actual source-tree hash, UTC observation time, project, status and observed verification. `report` reads that receipt; use `verify` for fresh owned-service health evidence. Receipts are deployment evidence, not cluster certification or proof that benchmarks ran.

## Optional source access

Set `environment_file` to an absolute mode-0600 file outside the checkout. It accepts only the documented integration URL/site/email/cloud-ID settings and credential **file paths**. It is parsed as literal `KEY=value` configuration, never shell-sourced. Unknown keys, duplicate settings, shell substitutions and inline credential settings are rejected.

For file credentials, set `secrets_dir` to an external private directory and use paths such as `BEACON_HEADERS_FILE=/run/validation-secrets/beacon-headers.json` in the environment file. The secrets directory must be mode 0700 and owned by container UID 10001 even for read-only mounts, so that the container can traverse it. Each credential file must be mode 0600 and owned by container UID 10001. Provision ownership through the approved platform process; the harness never chowns files or elevates privileges. The directory mounts read-only by default. Beacon cookie rotation requires explicitly setting `secrets_read_only=false` and a UID-10001-owned writable directory. Cookie values, bearer headers and tokens remain inside private files and are never copied into receipts. See [reference integration documentation](reference-integrations.md) for authorized source provisioning and supported exports.

## Stop, resume and failures

```bash
python3 agent-harness/harness.py stop --config /absolute/private/deploy.json
```

Stopping checks the named project's rendered configuration and runs Compose `down` without deleting volumes. Data persists. Re-running `deploy` reuses that named volume; it can rebuild/recreate containers and is not a database rollback mechanism. Never delete evidence merely to make a deployment pass.

A build, launch or verification failure is recorded as a failure when deployment state has been created. Inspect the private receipt and actual local Compose state. Resolve the stated prerequisite or redeploy an approved known revision with the same retained volume. No automatic rollback is implemented or claimed, and a failed command does not prove the previous service was restored. Keep a site backup before upgrading an established catalog.

The harness does not provide login or RBAC. Keep localhost binding and place the service behind the organization's authenticated TLS reverse proxy before granting remote access. Apply the site's upload, timeout, storage, backup and retention policy. Container-build execution remains unverified on the original packaging host because Docker was absent; run deployment verification on the authorized target.
