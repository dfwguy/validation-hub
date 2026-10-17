# Framework enhancements

This release expands the framework across ingestion, investigation, comparison, references and engagement tracking. Runtime customer data and source snapshots are deliberately separate from this release inventory.

## Evidence ingestion and recovery

- Disk-backed evidence uploads up to 1 GB, ZIP/tar/gzip/zstd bundle handling, 32 GiB expanded and 25,000-member bounds.
- Transfer progress, cancellation, original SHA-256 receipts and retained-original recovery visibility.
- Large multi-log investigations with measured-first capped lists, explicit totals and reopenable saved-report links.
- Training throughput/step-time ingestion alongside NCCL measurements; failed and diagnostic artifacts retain their evidence state.

## Investigation and reproducibility

- Versioned GPU/fabric configuration checks, crash grouping with file/line evidence and applicable next tests.
- Investigation plans with source references, expected outcomes and scope/rollback notes.
- Slurm NCCL script generation with explicit site inputs, correctness, warm-up, repeats, registration options and controlled Spectrum-X all-to-all experiments.
- Optional user-declared benchmark context kept distinct from measured log evidence.

## Comparison and source libraries

- Compact NCCL **16 GiB/32 GiB columns** and a two-row best/yours/over-under detail view with four improvement options.
- Missing measurements remain unavailable; actual-byte rounding is disclosed; parent failures remain diagnostic.
- Workload-aware training references with explicit mean/median labels, direction-aware percentages and contextual/unknown states.
- BEACON training sync/import and snapshot receipts; measured NCCL source imports and source-specific/global inventory counts.
- Read-only Confluence indexing with version/hash receipts and reviewed guidance; measured tables/curves require explicit extraction rather than counting document mentions.
- Reference reassessment jobs with progress and partial/failure reporting.

## Engagements and portable deployment

- Customer mappings, planned/active/blocked/completed engagements, validation/valid-log counts and linked investigations.
- Best eligible training/NCCL results by configuration plus explicit final-number and Exemplar designations.
- Framework-only deployment packaging with separate data roots, private runtime credentials and documentation for a fresh geography.

NVBugsPro remains a manual external investigation reference rather than a working bug-query connector. Exemplar-inspired workflows and recipe references do not imply automatic publication to the Exemplar service.
