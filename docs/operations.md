# Operations and known limits

## Storage and backup

The data root holds the run/customer catalog, retained uploads, derived evidence, reference catalogs, source snapshots and indexed guidance. Provision capacity for compressed uploads **and** expanded/derived evidence, SQLite WAL files and backups. The 32 GiB expanded-archive bound is per archive, not an estimate of the total volume requirement. Storage status reports local filesystem usage; it does not add capacity or expand the backing disk.

Keep framework releases and persistent data separate. Back up SQLite with a consistent online-backup mechanism or stop writers before taking a coordinated catalog/filesystem backup. Copying a live `.sqlite` file without its transaction state can lose data. Back up retained originals and source receipts alongside the catalogs. Test restoration into an isolated data root before relying on a backup.

This release does not implement automated retention, customer-data deletion policy, cloud replication or unlimited storage. Operators must define these policies for their geography. Do not delete retained originals merely because a compact report or cached assessment exists.

## Upload recovery

The main upload path streams to disk and issues an original SHA-256 receipt. Received-evidence inventory exposes retention and recovery status. An analysis failure can leave the original safely retained while parsed/indexed results are incomplete. Preserve that receipt and inspect the error before retrying.

Use recovery tools such as `tools/reprocess_received.py` only after reviewing `--help`, the target receipt/path and available disk space. The generic archive extractor and recovery reconciler require explicit cluster/archive or receipt/job inputs. Deployment-specific remote mirroring and acceptance scripts are excluded; inspect each maintenance command’s input paths and scope before using it. Do not run a second recovery on the same archive while the first is still active. A failed/partial investigation is not a completed validation.

## Reference refresh and reassessment

Inspect source status and credentials before refresh. Training and NCCL libraries have independent coverage. A source's page count, snapshot count and measured-record count are different inventories. Successful imports retain receipt hashes; last-import counts must not be substituted for global active counts.

Refresh and reassessment jobs expose queued/running/complete/partial/failed states. Keep the service running while they execute. Startup marks interrupted queued/running jobs failed with a retry message; it does not silently resume them. Avoid concurrent full-catalog reassessments. Inspect partial-job details before retrying so an operator can distinguish source import success from per-analysis failures.

A reference refresh can change comparisons without changing the original measurements. Preserve the snapshot identity and reassessment timestamp when communicating a result. Reassessing a saved investigation is available through its report action.

## Security and source handling

Deploy behind authenticated TLS access; there is no application RBAC or customer isolation boundary. Treat customer ownership as workflow metadata. Source credentials must remain private runtime files, and source exports/wiki bodies may require restricted storage even when the framework repository is shareable internally.

Credential scrubbing is defensive processing, not proof that every arbitrary log contains no sensitive information. Review exports and screenshots before distributing them. The framework does not post to Slack, create NVBugsPro issues or submit results to Exemplar automatically.

## Practical limits

- Main investigation upload: 1,000,000,000 bytes; expanded archive: 32 GiB; archive members: 25,000. The separate single-log comparison endpoint retains a smaller 32 MiB limit.
- Reference imports: 50 MiB compressed, with bounded expansion. Source pagination and authentication are source-specific.
- Large UI lists are capped/paginated with explicit totals. The simplified NCCL comparison contains only 16/32 GiB; other raw measurements remain retained.
- NCCL script generation currently supports maximum 16 GiB, even though imported 32 GiB measurements can be compared.
- Generated launchers require Slurm/Pyxis/Enroot/PMIx and compatible MPI nccl-tests. They are reviewable artifacts, not remote execution.
- Host attribution is capped per analysis (5,000 tracked keys, top 25 shown). The cross-analysis roll-up reads the newest 500 saved analyses by default, and reports saved before it existed have no host data. See [crash evidence](crash-evidence.md).
- Unknown or incomplete context cannot establish a qualified baseline. Crash signatures are leads and do not prove root cause.
- Background jobs are process-managed. Multi-worker/distributed operation requires additional coordination work.
- External source availability and session expiry can limit refresh. Previously imported receipts remain visible; unavailable access must not be presented as fresh coverage.
