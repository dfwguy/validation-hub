# Publication validation

Validated the sanitized enhancement checkout on 2026-10-17 using Python 3.14:

- Full framework suite: **227 passed**. One upstream Starlette/httpx deprecation warning remains.
- Both frontend JavaScript files parsed successfully in headless Chromium. The new NVIDIA-green SVG banner was rendered and visually inspected.
- All 62 local Markdown/image navigation targets and SVG structure checks passed.
- Harness tests cover Docker-free planning, unsafe configuration rejection, deployment ownership, real local HTTP evidence, failure receipts and volume-preserving stop. Independent handoff checking returned PLANNED and correctly reported BLOCKED for missing Docker without creating a deployment.
- Source-release tests reject archive tampering, member-hash mismatch, unsafe paths and unmanifested files.
- Python compilation and the existing environment dependency consistency check passed.
- Fresh temporary data root without source credentials: `/api/health`, `/api/summary` and `/` returned HTTP 200; catalog contained zero runs.
- Tracked-file artifact/credential check passed. Manual publication review removed site-specific scripts, private identifiers and copied customer measurement fixtures.
- Customer evidence, source corpora and runtime databases are excluded from both published branches.

The application tests use the existing compatible Python environment; this does not establish a fresh dependency installation or a container build. Docker and Compose were unavailable on the packaging host, so image build/start remain to be verified on a Docker-capable runner. CI jobs are supplied for Python tests, JavaScript syntax and an opt-in tagged image build. A configured pipeline is not itself a passing pipeline result.

Shared deployment requires an authenticated TLS reverse proxy and authorized, deployment-local source credentials. No production deployment or evidence migration was performed by publication.

## Crash evidence (Xid decoding, host lists)

Checked 2026-11-04 on Python 3.14.10 with synthetic logs:

- `pytest`: 260 passed (227 existing, 33 new). Same Starlette/httpx warning as before.
- `compileall`, `check_publication.py`, `check_docs.py`, `node --check` on both frontend files and `pip check` pass.
- Uploaded synthetic crash logs to a local instance and opened the Crash analyzer tab in headless Chrome. The Xid table and host list render, and a report saved before this change still renders.
- The Xid table was checked against NVIDIA's Xid catalog.

Not checked: real cluster logs, the container image, very large archives. The `cuda` pattern now accepts Xid codes of up to three digits, so `xid=12345` no longer counts as a crash.
