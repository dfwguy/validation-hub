# Published source history

`main` preserves the earliest retained source archive, timestamped 2026-10-16 22:48:02 UTC, with customer labels anonymized. It already contains some validation, engagement and troubleshooting work. No earlier pre-session source was available; this repository does not invent that history.

`SOURCE_PROVENANCE.json` records the original archive hash and original/sanitized hashes for its recovered files. Those hashes describe the baseline commit, not current files changed by the enhancement branch. The archive itself is excluded.

`framework-enhancements` contains the complete current framework, portable packaging, synthetic regression tests and documentation. A later merge request can compare this branch with `main`. The baseline snapshot lacked dependency/deployment files and some tool sources; deploy from the enhancement branch until it is merged.

No customer evidence, runtime catalogs, downloaded reference corpus or credentials are included in either branch. The deployed application and evidence have not been modified by this repository publication.
