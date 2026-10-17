#!/usr/bin/env python3
"""Reject runtime data, secrets and deployment identity in tracked release files."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {'data', 'uploads', 'artifacts', 'var', '.ci-data', '__pycache__', '.venv'}
FORBIDDEN_SUFFIXES = {'.sqlite', '.db', '.parquet', '.arrow', '.csv', '.jsonl', '.log', '.out', '.zip', '.zst', '.tgz', '.pyc', '.pem', '.key', '.token', '.cookie'}
SECRET_PATTERNS = [
    re.compile(r'AT' + r'ATT[A-Za-z0-9_=/+\-]{40,}'),
    re.compile(r'gl' + r'pat-[A-Za-z0-9_\-]{15,}'),
    re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    re.compile(r'eyJ[A-Za-z0-9_\-]{30,}\.[A-Za-z0-9_.\-]{50,}'),
]


def main():
    try:
        names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    except FileNotFoundError:
        # Minimal CI Python images may omit Git after the runner checks out source.
        names = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
                 if p.is_file() and '.git' not in p.relative_to(ROOT).parts]
    failures = []
    for name in filter(None, names):
        path = Path(name)
        if FORBIDDEN_DIRS.intersection(path.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name == '.env':
            failures.append(f'{name}: runtime/credential artifact')
            continue
        content = (ROOT / path).read_bytes()
        if b'\0' in content:
            failures.append(f'{name}: binary artifact')
            continue
        if any(pattern.search(content.decode('utf-8', errors='replace')) for pattern in SECRET_PATTERNS):
            failures.append(f'{name}: possible credential; inspect privately')
    if failures:
        print('\n'.join(failures))
        return 1
    print('Tracked framework files passed publication artifact and credential checks.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
