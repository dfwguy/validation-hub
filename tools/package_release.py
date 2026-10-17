#!/usr/bin/env python3
"""Package one clean, committed framework tree; never include untracked runtime data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'nvis-ova-validation-hub'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    args = parser.parse_args()
    if git('status', '--porcelain', '--untracked-files=normal').strip():
        parser.error('Package only a clean committed checkout; commit or isolate your changes first')
    subprocess.run(['python3', str(ROOT / 'tools/check_publication.py')], cwd=ROOT, check=True)
    commit = git('rev-parse', 'HEAD').decode().strip()
    entries = []
    files = []
    for record in git('ls-tree', '-r', '-z', 'HEAD').split(b'\0'):
        if not record:
            continue
        attrs, raw_path = record.split(b'\t', 1)
        mode, kind, oid = attrs.decode().split()
        name = raw_path.decode()
        if mode not in ('100644', '100755') or kind != 'blob':
            raise ValueError('Only regular tracked source files may be packaged: ' + name)
        payload = git('cat-file', 'blob', oid)
        entries.append(dict(path=name, size=len(payload), sha256=hashlib.sha256(payload).hexdigest(), mode=mode))
        files.append((name, payload, 0o755 if mode == '100755' else 0o644))
    manifest = dict(schema_version=1, project=PREFIX, commit=commit, files=entries)
    files.append(('FILE_MANIFEST.json', (json.dumps(manifest, indent=2) + '\n').encode(), 0o644))
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / (PREFIX + '.tar.gz')
    partial = archive.with_suffix('.partial')
    with tarfile.open(partial, 'w:gz') as tar:
        for name, payload, mode in files:
            info = tarfile.TarInfo(PREFIX + '/' + name)
            info.size = len(payload)
            info.mode = mode
            info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
    partial.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.output / 'SHA256SUMS').write_text(digest + '  ' + archive.name + '\n')
    subprocess.run(['python3', str(ROOT / 'tools/verify_release.py'), str(archive), '--sha256', digest], check=True)
    print(json.dumps(dict(status='PACKAGED', commit=commit, archive=str(archive), sha256=digest, files=len(entries))))


if __name__ == '__main__':
    main()
