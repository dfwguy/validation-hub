# Agent instructions

This repository contains the portable NVIS-OVA Validation Hub framework. Customer evidence, reference snapshots, private wiki corpora, databases, credentials and deployment receipts belong in deployment-local storage outside Git and software packages.

For deployment, use the canonical [deployment skill](skills/validation-hub-deploy/SKILL.md) and the provider-neutral [handoff prompt](agent-harness/PROMPT.md). Read only the source integrations needed for the requested site. Continue authorized preparation and deployment with available inputs; obtain only missing target facts. Do not infer live fabric policy, GPU topology, credentials or successful validation from examples.

For repository work, read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Factory contract](docs/factory.md). Keep changes on a feature branch and publish the requested merge request with exact-commit validation evidence. Preserve the retained baseline's documented provenance; do not relabel it as a pre-enhancement original. Follow the user's existing publication and deployment authorization without adding a second approval workflow.

Run the checks appropriate to the change: `pytest -q`, `python3 tools/check_publication.py`, and Python/JavaScript syntax checks. Package only the committed tracked tree with `python3 tools/package_release.py --output dist`; verify it with `tools/verify_release.py` according to its CLI. CI artifacts contain software, file hashes and validation reports only.

Keep measured results separate from theoretical limits, expectations, guidance and job completion. Compare NCCL points at identical actual positive message bytes and matching configuration; display buckets do not change scoring bytes. Missing context remains unknown. Preserve original uploads and their hashes, correctness failures, source versions and comparison provenance.

Use one application worker and persistent external storage. Shared access requires an authenticated TLS proxy; the application does not supply customer isolation/RBAC itself. Never execute generated NCCL scripts, modify a fabric, reset evidence storage or copy production data merely to demonstrate a deployment.
