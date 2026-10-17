# Publication validation

Validated the sanitized enhancement checkout on 2026-10-17 using Python 3.14:

- Full framework suite: **206 passed**. One upstream Starlette/httpx deprecation warning remains.
- Both frontend JavaScript files parsed successfully in headless Chromium.
- Fresh temporary data root without source credentials: `/api/health`, `/api/summary` and `/` returned HTTP 200; catalog contained zero runs.
- Tracked-file artifact/credential check passed. Manual publication review removed site-specific scripts, private identifiers and copied customer measurement fixtures.
- Customer evidence, source corpora and runtime databases are excluded from both published branches.

The application tests use the existing compatible Python environment; this does not establish a fresh dependency installation or a container build. Docker and Compose were unavailable on the packaging host, so image build/start remain to be verified on a Docker-capable runner. CI jobs are supplied for Python tests, JavaScript syntax and an opt-in tagged image build. A configured pipeline is not itself a passing pipeline result.

Shared deployment requires an authenticated TLS reverse proxy and authorized, deployment-local source credentials. No production deployment or evidence migration was performed by publication.
