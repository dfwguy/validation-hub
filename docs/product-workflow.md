# Product workflow

## Upload evidence

Choose the GPU/fabric recipe and target cluster, then upload a log or bundle. Customer and engagement ownership can associate the result with an investigation. An optional benchmark-context JSON field records user-declared information such as model, precision, sequence length or NCCL protocol; a declaration is not a measured observation.

The main investigation path accepts up to **1,000,000,000 bytes per file**. Supported bundles include ZIP, tar, tar.gz/tgz and tar.zst/tzst; supported individual evidence includes text logs and supported compressed logs. The interface also recognizes the legacy `.tar.zg` spelling. Archives are limited to **32 GiB expanded data** and **25,000 members**. The upload note states these limits.

The progress bar measures transfer. Saving, unpacking, parsing and assessment continue afterward. Keep the page open until completion. Received-evidence receipts expose the original SHA-256 and recovery state. If processing fails after the original has been retained, an operator can recover from that receipt without requesting another upload. Cancelling transfer is distinct from deleting an already retained original.

## Investigate

The saved report brings together four tools:

1. **Checks:** compare available configuration evidence against a versioned GPU/fabric recipe. Missing firmware, topology, transport or site-policy evidence stays unknown.
2. **Crash analyzer:** group recognized signatures with source-file and line excerpts. Parent-job failure, scheduler termination and numerical failures remain distinct from individual-file NCCL correctness.
3. **Follow-up tests:** explain what to collect, why it helps and the expected outcome. Source guidance includes applicability and cautions; infrastructure changes are not executed automatically.
4. **NCCL script generator:** produce a downloadable launcher using the chosen recipe and site-supplied paths/interfaces.

Saved investigations can be reopened through `?analysis=<analysis-id>`. Large artifact listings show explicit totals with a capped visible list; measured training and NCCL evidence is prioritized. The explorer retains access to the other indexed artifacts.

## Compare performance

NCCL explorer rows show **16 GiB** and **32 GiB** columns. A cell contains your GB/s and a short over/under percentage; the reference is available through the tooltip and detail view. Details contain two comparison rows and four short improvement options. Raw curves and source/configuration details remain expandable. Unmeasured sizes display a dash.

Training uses model/configuration-aware throughput and step-time comparisons. A run's mean summary and the median used for benchmark comparison are labeled separately. Missing context or mismatched software/context produces a conditional or unknown result rather than acceptance.

## Customer and engagement board

Create a customer and map its cluster identifiers. Track planned, active, blocked and completed engagements, validation counts, valid-log counts, saved investigations, best measured configurations, final numbers and Exemplar designations.

Final/Exemplar labels require explicit user designation and eligible measured evidence. Completed training is grouped by workload/model/configuration; NCCL retains its correctness and configuration requirements. Failed, parent-failed, incomplete or nonfinite evidence cannot establish a qualified final result. A designation is a local evidence label, not a submission to an external Exemplar service.

## Generate and review tests

Recipes cover H100, H200, B200, B300, GB200 and GB300 with InfiniBand or Spectrum-X. The operator supplies the approved container, partition, MPI nccl-tests location, GPU-facing HCA ports and bootstrap interface. The launcher assumes Slurm, Pyxis/Enroot and PMIx.

Defaults are fixed 2 GiB messages, 50 warm-up iterations, 100 measured iterations and three repeats. The generator currently permits sizes up to 16 GiB; the explorer's 32 GiB comparison does not imply that this generator can generate a 32 GiB run. Optional registration modes require a supporting test build. Experimental all-to-all presets are restricted to Spectrum-X and record their overrides. Review generated scripts before submitting them on the target cluster.
