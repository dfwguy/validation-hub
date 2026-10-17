#!/usr/bin/env python3
"""Verify a framework release checksum, member safety and every manifest hash without extracting."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

PREFIX = 'nvis-ova-validation-hub'


def verify(archive, expected):
    digest = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    if digest.lower() != expected.lower():
        raise ValueError('Archive SHA-256 mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or len(path.parts) < 2 or path.parts[0] != PREFIX or member.name in names):
                raise ValueError('Unsafe or duplicate release member')
            names.add(member.name)
        manifest_name = PREFIX + '/FILE_MANIFEST.json'
        manifest = json.load(tar.extractfile(manifest_name))
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported release manifest')
        expected_names = {PREFIX + '/' + entry['path'] for entry in manifest['files']}
        if len(expected_names) != len(manifest['files']) or names != expected_names | {manifest_name}:
            raise ValueError('Archive inventory differs from manifest')
        for entry in manifest['files']:
            member = tar.getmember(PREFIX + '/' + entry['path'])
            content = tar.extractfile(member).read()
            expected_mode = 0o755 if entry['mode'] == '100755' else 0o644
            if (len(content) != entry['size'] or member.mode != expected_mode
                    or hashlib.sha256(content).hexdigest() != entry['sha256']):
                raise ValueError('Member hash, mode or size mismatch: ' + entry['path'])
    return dict(status='VERIFIED', commit=manifest['commit'], files=len(manifest['files']), sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--sha256', required=True, help='Expected checksum obtained from the trusted release channel')
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.sha256)))


if __name__ == '__main__':
    main()
