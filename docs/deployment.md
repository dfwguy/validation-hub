# Fresh deployment in another geography

Deploy the framework with empty local storage, then configure site inputs and authorized reference sources. Do not copy a production database, customer archive, cookie file, downloaded wiki corpus or reference snapshot into the repository/image.

## Local service

The packaged runtime and dependency lock target Python 3.14. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
mkdir -p var/validation-hub
export VALIDATION_DATA_ROOT="$PWD/var/validation-hub"
uvicorn backend.app:app --host 127.0.0.1 --port 8081 --workers 1
```

Open the local address and check `GET /api/health`. The first launch creates the writable catalogs as needed; no source credentials or reference dataset is required for the application to start. Run `pytest -q` for framework checks before rollout. Dependency locks, where used by the deployment package, must match the selected runtime; validate installation rather than copying a virtual environment between machines.

## Runtime configuration

| Variable | Purpose / default |
|---|---|
| `VALIDATION_DATA_ROOT` | Persistent data directory; defaults to repository-local `var/validation-hub` |
| `VALIDATION_DB` | Optional primary SQLite catalog override |
| `VALIDATION_UPLOADS` | Optional original/derived upload directory override |
| `BEACON_DB` | Optional training-reference SQLite override |
| `BEACON_NCCL_DB` | Optional NCCL-reference SQLite override |
| `CONFLUENCE_EVIDENCE_DB` | Optional indexed-guidance SQLite override |
| `CONFLUENCE_EVIDENCE_DIR` | Optional downloaded wiki-evidence directory override |
| `BEACON_HEADERS_FILE` / `BEACON_COOKIE_FILE` | Private, runtime-only source authentication files |
| `ATLASSIAN_EMAIL`, `ATLASSIAN_TOKEN_FILE`, `ATLASSIAN_CLOUD_ID` | Authorized read-only Confluence connector credentials/site identity |
| `ATLASSIAN_SITE` | Confluence site URL; use the authorized target site |
| `BEACON_BASE_URL`, `BEACON_API_URL`, `GPUCOMMS_URL`, `NCCL_DASHBOARD_URL` | Approved source locations; defaults are the supported source services |

Set a persistent volume appropriate to the target geography. The service account needs write access to its data directories and read access to the mounted credential files. Use absolute paths for externally managed data/credential locations. Source-origin restrictions and HTTPS validation remain active; an environment override is not a reason to use arbitrary hosts.

For example, keep private files in a directory outside the checkout:

```bash
install -d -m 700 "$HOME/.config/validation-hub"
# Provision authorized credential files through your approved secret mechanism.
# Set each resulting credential file to mode 0600.
export BEACON_COOKIE_FILE="$HOME/.config/validation-hub/beacon.cookie"
export ATLASSIAN_TOKEN_FILE="$HOME/.config/validation-hub/atlassian.token"
```

The example provisions paths only; it does not generate a valid source token. Supply the deployment's authorized email/cloud identity when enabling Confluence. Never commit a credential or include one directly in a command-line URL.

## Container or service deployment

The supplied Compose definition runs as UID/GID 10001 with a persistent named volume, read-only root filesystem and a 2 GiB temporary filesystem. From the repository root:

```bash
cp .env.example .env
# Edit paths and optional source settings for this deployment.
docker compose -f deploy/compose.yaml config
docker compose -f deploy/compose.yaml up --build -d
curl --fail http://127.0.0.1:8081/api/health
```

The image build must be validated in your Docker-capable environment; Docker was unavailable during the initial packaging review. A Compose example is packaging, not evidence of deployment in another geography. The container data root is `/var/lib/validation-hub`; mount a persistent volume there and mount source credentials separately as private files. Tokens/header files can be read-only; the BEACON cookie file may need write access for authorized session rotation. Keep its permissions private. Mounted mode-0600 files must be owned by/readable to container UID/GID 10001; a private host-user file is not automatically readable by that different UID. The cookie directory must also be writable for atomic session rotation. Inspect the rendered configuration before startup. The host-service template is `deploy/validation-hub.service`, with `/opt/nvis-ova-validation-hub` as its checkout, a dedicated `validationhub` user, and optional `/etc/validation-hub/environment`. Provision that user/virtual environment and adapt paths before installing the unit.

Run **one Uvicorn worker**. Background synchronization/reassessment is process-managed, not a distributed job system. Put the application behind your organization's authenticated TLS reverse proxy. The application does not implement its own login, per-customer access control or RBAC. Binding to localhost is appropriate until that access boundary is configured. Apply proxy upload limits/timeouts consistent with the 1 GB upload path and potentially lengthy analysis.

## Site validation before use

1. Confirm persistent storage, available capacity and backup ownership.
2. Open the empty explorer and source library; verify missing references are shown as unavailable.
3. Upload an approved small test log and verify its original receipt, parsed measurements and saved-report link.
4. Configure source access or import approved exports, inspect source receipts, and reassess the test evidence.
5. Review GPU/fabric recipes for the actual topology, HCA ports, interfaces, container/MPI paths and scheduler integration.
6. Generate and review a test script before submitting it under the site's job-change process.

Recipes do not discover or change switch QoS, traffic class, PFC, ECN, congestion control, firmware, routing or service levels. Spectrum-X/InfiniBand policy and GB-system NVLink/IMEX domains require site-specific evidence. Do not guess these values to make a configuration check appear green.
