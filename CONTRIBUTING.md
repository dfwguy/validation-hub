# Contributing

Read [AGENTS.md](AGENTS.md), [the Factory contract](docs/factory.md) and the relevant architecture/integration guide before changing the framework. Use a feature branch and a reviewable merge request when requested by the repository owner; human reviewers approve and merge. Keep customer data and credentials outside Git and release archives.

Run the validation contract from the repository root:

```bash
python -m pytest -q
python -m compileall -q backend tools agent-harness tests
python tools/check_publication.py
python tools/check_docs.py
node --check frontend/app.js
node --check frontend/workflows.js
```

Install the dependency lock in a local Python 3.14 environment for application tests. The deployment harness itself uses the Python standard library. JavaScript syntax can also be checked in a browser environment when Node is unavailable; record which check actually ran.

Use synthetic fixtures. Preserve exact units, message sizes, scope and version matching; do not convert incomplete context into a confident performance verdict. Document new environment settings, integration behavior and migration limits. No CI job may upload evidence, submit benchmark jobs or change a deployment.

After committing and validating, build the clean tracked-tree source package with `python3 tools/package_release.py --output dist`. Verify exact-commit CI status. Docker image/Compose startup must be tested on a Docker-capable host before claiming a verified container deployment.
