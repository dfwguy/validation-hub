# CI/CD Factory publication and delivery contract

This framework follows the clean-source, agent handoff and evidence-based publication pattern used by the NVIS OVA Collection Framework and Exemplar deployment kit.

## Repository contract

1. Preserve honest source provenance and keep all customer evidence, private source snapshots, credentials and deployed inventories outside both the source tree and release bundles.
2. Provide an outcome-focused branded README, explicit deployment inputs, an easy deployment path, a discoverable skill and provider-neutral agent instructions.
3. Validate the application, harness behavior, documentation navigation, SVGs, publication boundaries and deployment configuration. Retain exact source commit and test evidence; do not turn a planned check into a claimed result.
4. Publish feature-branch changes for human review and merge. This project's owner requested the merge request later; `framework-enhancements` remains unmerged, with a draft in `docs/merge-request-draft.md`.
5. Observe the pipeline for the exact published commit. CI tests software and packages source; it never deploys the hub or launches cluster jobs.
6. Deliver committed, tracked source with a per-file integrity manifest and archive checksum. Local runtime files, untracked files and ignored files are not release inputs.

## Build and verify a source release

From a clean committed checkout:

```bash
python3 tools/package_release.py --output dist
python3 tools/verify_release.py dist/nvis-ova-validation-hub.tar.gz --sha256 <TRUSTED_ARCHIVE_SHA256>
```

The archive contains `nvis-ova-validation-hub/FILE_MANIFEST.json`, recording the exact source commit and each tracked file's hash, size and mode. The sibling `SHA256SUMS` covers the archive. Packaging automatically verifies its result without extracting it. The receiver must obtain the expected archive checksum from a trusted delivery channel; a hash shipped alongside an untrusted archive does not authenticate its origin.

The source archive is a portable software bundle, not a prebuilt image, offline dependency closure or database backup. A disconnected deployment needs approved Python dependencies or a matching prebuilt image staged separately. Neither a source bundle nor a valid checksum establishes GPU performance acceptance.

## CI and deployment boundaries

CI uses the Factory Linux Docker runner tags `size/medium`, `type/docker`, `os/linux`. Python/application/harness tests and frontend syntax validation precede source packaging. The optional image build runs only on explicitly enabled release tags with a suitable Docker runner.

`agent-harness/` is a deployment execution adapter for an agent already authorized to work on the target application host. It is independent of a model vendor. It does not enroll this repository in the resident Factory issue controller, install an AI model, grant agent permissions, change Factory routing/timers or create source credentials. Those are separate operational integrations.

The optional Factory NVFTP envelope requires the deployed controller's reviewed manifest, source tar, Git bundle, checksums and genuine human classification attestation. This repository's source-package command does not fabricate that attestation or claim envelope enrollment.
