# NVIS OVA Validation Hub

A deployable engineering workspace for GPU validation evidence, performance comparison, troubleshooting and customer engagement tracking. Upload logs, keep the original evidence, inspect measured training and NCCL results, and generate reviewable follow-up tests.

This distribution contains the framework and documentation. Customer logs, database contents, reference snapshots, source-page bodies, credentials and production inventory are deployment data and are not part of the release.

## Capabilities

- **Validation explorer:** search workload, GPU/fabric, execution scale, collective, NCCL version and recorded topology. NCCL rows have compact **16 GiB** and **32 GiB** columns showing your GB/s and percentage over/under the best matching recorded result. Training retains throughput and step-time comparisons.
- **Upload and investigate:** disk-backed uploads up to **1 GB**, archive support, transfer progress and cancellation, retained-original receipts, configuration checks, grouped crash evidence and follow-up tests.
- **Evidence lake:** measured results, saved investigations, reference inventories, source provenance and storage status. Missing measurements and incomplete context remain visible.
- **Customer board:** validations, valid logs, active/planned engagements, best measured results and explicitly designated final/Exemplar results, with links to underlying evidence.
- **Repeatable tests:** recipe-based Slurm NCCL launchers for supported GPUs with InfiniBand or Spectrum-X, correctness checks, warm-up, repeats and optional controlled experiments. Generation does not submit jobs.

## Start here

1. Follow [deployment](docs/deployment.md) to start a fresh instance with empty writable storage.
2. Read the [product workflow](docs/product-workflow.md) and [comparison policy](docs/comparison-policy.md).
3. Configure authorized [reference integrations](docs/reference-integrations.md) independently for the target geography.
4. Use [operations](docs/operations.md) for storage, backup, recovery and known limits.

[Architecture](docs/architecture.md) describes module ownership and data flow. [Release enhancements](docs/release-enhancements.md) describes the framework changes. A [Slack announcement draft](docs/slack-announcement.md) is supplied for review; it has not been posted.

See [published source history](docs/source-history.md) for the retained baseline and enhancement branch, and the [future merge request draft](docs/merge-request-draft.md). No merge request has been created.

[Publication validation](docs/validation.md) records the checks performed and remaining container-build validation.
