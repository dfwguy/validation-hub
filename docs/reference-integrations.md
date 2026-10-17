# Reference integrations and source provenance

Integrations are configured independently at each deployment. Source access available in one geography is not transferred by distributing the framework. An empty deployment has no measured benchmark coverage until it imports or fetches authorized source evidence.

| Source | Implemented behavior | Limits |
|---|---|---|
| BEACON training results | Authenticated API fetch/sync, authorized export import, searchable filters and versioned compact snapshot downloads | Requires valid runtime authentication or an authorized export; a reference record is not an uploaded customer validation |
| NCCL benchmark services and dashboards | Measured record import, supported source indexers, compact curve previews, GPU/fabric/scale/collective filters and reassessment | Coverage is whatever measured rows were actually imported; count-only metadata does not establish measured coverage |
| Confluence | Authorized read-only REST discovery/download, version/hash receipts, searchable indexed pages, reviewed guidance and supported measured-table extraction | Page bodies are guidance, not automatically numeric baselines; chart/table extraction requires an actual supported adapter and receipt |
| Exemplar | Training-log analysis, versioned recipe/source references and local final/Exemplar evidence labels | No general authenticated Exemplar-site crawler, live service submission or automatic certification is implied |
| NVBugsPro | Manual external investigation reference/handoff only | No authenticated bug-query, issue synchronization, automatic bug matching or issue creation connector is implemented |

## Authentication

Keep authentication outside the repository and outside container images. BEACON accepts `BEACON_HEADERS_FILE` containing a JSON object of HTTP header strings or `BEACON_COOKIE_FILE` containing a cookie header. These files must have mode `0600`; the backend rejects group/world-readable credential files. For containers, ensure mounted mode-0600 files are owned by/readable to service UID/GID 10001; matching the host login user is insufficient. The cookie directory must be writable for atomic replacement. The cookie file can be updated by session rotation and may require write access; token/header files can be mounted read-only. Sessions expire and can require renewal through an authorized login. Do not paste credentials into source URLs, investigation notes, screenshots or announcement drafts.

Confluence uses `ATLASSIAN_EMAIL`, `ATLASSIAN_TOKEN_FILE`, `ATLASSIAN_CLOUD_ID` and the configured site. The connector makes read-only source requests; it does not mutate pages. Supply a scoped credential permitted for the deployment's source space. Source authentication is independent of access control for the hub itself.

Use an authorized JSON, CSV, JSONL or supported gzip export when source access is unavailable. Reference imports have a separate **50 MiB compressed-file limit** and bounded expanded parsing; this is not the main evidence upload limit. Do not bypass source origin restrictions to fetch arbitrary URLs.

## Receipts and counts

Measured reference receipts retain source URL, source record/job identifiers, import/snapshot identity, hashes and applicable configuration. Compact downloads have SHA-256 receipts. Full curve data and compact previews serve different purposes; the source library preview is not the entire measurement set.

Global measured-record counts come from the active reference inventory across sources. The most recent import's record count describes that import only. Source-specific snapshot receipts expose their own scope. GPU/fabric coverage and missing coverage are computed from actual imported rows rather than assumed from a document title.

Confluence receipts include page ID, title, version, updated time, body/payload hashes and indexed time. Curated findings retain section paths and table/row context where applicable. A source's applicability, topology, release, precision, registration mode and test protocol must be checked before treating a numeric target as comparable.

## Fresh-source operation

Use Data lake & sources to inspect connection status, available filters, source errors and snapshot provenance. Import or synchronize the source, then reassess saved evidence against the resulting references. Job status distinguishes queued, running, complete, partial and failed. Partial results require inspection; they are not a successful full refresh.

CLI adapters under `tools/index_*` provide operator workflows. Read their `--help` output and configure source authorization before running them. Their output belongs on the deployment data volume, not in the framework publication. No copied production reference database is required to start the application.
