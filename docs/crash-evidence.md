# Crash evidence

Two additions to the crash analyzer. Both use only the matching log lines.

## Xid codes

The `cuda` signature matches kernel messages like `NVRM: Xid (PCI:0000:5a:00): 79, ...` as well as bare `Xid 13`, `Xid: 31` and `Xid=48`. Codes are up to three digits, so ids like `xid=12345` don't match.

Each code is listed with its count, first file and line, PCI address and reporting host, plus a meaning and suggested action from NVIDIA's [Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html) (checked 2026-11-04). The table is `XID_CATALOG` in `backend/crash_enrichment.py`. A code that isn't in it is shown without a meaning.

"Likely area" is our reading of the catalog's wording, not a diagnosis. It is one of application, hardware/node, GPU memory (ECC), driver/firmware, or follow-on (Xid 45, which follows another event). Only the node-level areas add guidance to the follow-up test, and only for Xids present in the log.

## Hosts

Each crash group lists the hosts its lines name, and the report adds a roll-up ordered by line count. A host is recorded under one of three roles, which are never merged:

| Role | Source |
| --- | --- |
| Logged the line | `host:pid:tid [dev]` NCCL debug prefix (optionally behind an srun label or mpirun rank prefix), or a syslog/journal prefix. The host that wrote an error may be the victim of a peer's failure. |
| Named by scheduler | `slurmstepd ... STEP/JOB <id> ON <host>` |
| Peer address | `peer <addr><port>` or `Connect to <addr><port>` in NCCL connection errors. Not resolved to a hostname. |

A line with no host adds none. A line that matches several signatures is counted once, under the first. Per analysis, at most 5,000 (host, role, signature) keys are tracked and 100,000 lines per file are remembered; the report sets `truncated` when a cap is hit. The roll-up shows the top 25 hosts.

### Repeat hosts across analyses

`GET /api/host-attribution` aggregates saved analyses. A host counts once per analysis, however many lines it produced.

- `cluster`: limit to one cluster (default: all)
- `min_analyses`: minimum analyses a host must appear in (default 2)
- `limit`: hosts returned (default 50, max 200)
- `max_analyses`: newest analyses considered (default 500, max 5,000)

Reports saved before this feature have no host data and aren't counted; re-upload the evidence to include it. The app has no per-customer isolation, so the roll-up covers every analysis the filter selects.

Recurrence is a reason to run the follow-up checks (port counters, GPU health, a repro on the same node). It isn't a fault finding.
