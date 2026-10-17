# Merge request draft — not created

This file is a reviewable title/body for a future merge request. No merge request has been opened or posted by this document.

## Suggested title

Expand validation evidence ingestion, troubleshooting, references and engagement workflows

## Suggested description

The framework now takes uploaded GPU-validation evidence through retained-original ingestion, measured-result indexing, configuration/crash investigation and repeatable follow-up tests. Operators can associate investigations with customer engagements and review final results without losing the source evidence behind the numbers.

NCCL explorer rows show compact **16 GiB** and **32 GiB** columns with your GB/s and the percentage over/under the best independent recorded matching result. For example, a synthetic 400 GB/s observation against a 500 GB/s reference displays **20% under**. The detail view has two comparison rows and four improvement options; unmeasured sizes remain blank, and parent-job failures remain diagnostic. Raw curves stay expandable. Training preserves model/configuration-aware throughput and step-time comparisons with explicit statistic/context labels.

### Main changes

- Disk-backed 1 GB uploads, compressed archive support, bounded expansion, transfer progress/cancellation, original receipts and recoverable saved investigations.
- Versioned GPU/fabric checks, crash signatures with file/line context, source-backed follow-up plans and reviewable Slurm NCCL scripts with correctness, warm-up, repeats and controlled experiments.
- Customer/cluster ownership, engagement states, validation and valid-log counts, best measured configurations and explicit final/Exemplar designations.
- BEACON training sync/import, measured NCCL reference libraries, compact/hash-verifiable snapshots and read-only Confluence guidance with versioned provenance; clear missing/contextual reference states and reassessment progress.
- Portable deployment/data-root configuration, private runtime source credentials, container/service examples, synthetic tests and operational documentation.

### Scope and review notes

This branch publishes framework code and documentation. It excludes customer logs, production catalogs, source snapshots, downloaded wiki bodies, credentials and real engagement history. NVBugsPro remains a manual external investigation handoff; no bug-query connector or external Exemplar submission is claimed.

The baseline branch represents the earliest retained framework snapshot available for publication. It is not presented as an invented pre-session origin. Review the actual branch diff for changes already present in that retained baseline.

The service runs one worker and requires an authenticated TLS reverse proxy for shared deployment. Reference access and site GPU/fabric policy must be configured independently in the target geography. Generated scripts are not executed by the application.

### Validation

Before opening the merge request, attach the final branch's test/CI result and deployment smoke-test evidence. Framework tests cover archive handling, parser correctness, matching/eligibility, source receipts, customer linkage and the two-size NCCL summary. JavaScript syntax checks cover the frontend assets. Container build/start must be verified in a Docker-capable environment; packaging alone is not proof of deployment.

### Rollout

Start against an empty persistent data root, verify the health endpoint and a synthetic upload, then enable authorized source integrations. Configure site-specific interfaces, topology, containers and QoS policy before reviewing/submitting generated test scripts. Existing production evidence is not migrated by this publication.
