#!/usr/bin/env python3
"""Read retained materialized logs once, auditing signatures without metric parsing."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.compute as pc
from backend.security import scrub
from backend.troubleshooting import SIGNATURES, runtime_failure


def audit(manifest_path, output_path):
    started = time.monotonic()
    manifest = json.loads(Path(manifest_path).read_text())
    patterns = [(key, title, re.compile(pattern, re.I)) for key, title, pattern, _ in SIGNATURES]
    union = '|'.join('(?:' + pattern + ')' for _, _, pattern, _ in SIGNATURES)
    groups = {key: dict(id=key, title=title, count=0, evidence=[]) for key, title, _, _ in SIGNATURES}
    files = []
    bytes_scanned = 0
    original_runtime_count = 0
    for index, member in enumerate(manifest['members'], 1):
        counts = Counter()
        evidence = {}
        raw_runtime = 0
        offset = 0
        with Path(member['path']).open(encoding='utf-8') as stream:
            while lines := list(islice(stream, 16384)):
                array = pa.array(lines)
                mask = pc.match_substring_regex(array, pattern=union, ignore_case=True)
                selected = pc.indices_nonzero(mask).to_pylist()
                for local_index in selected:
                    line = lines[local_index].rstrip('\r\n')
                    line_number = offset + local_index + 1
                    for key, _, regex in patterns:
                        if not regex.search(line):
                            continue
                        if key == 'runtime':
                            raw_runtime += 1
                            if not runtime_failure(line):
                                continue
                        counts[key] += 1
                        group = groups[key]
                        group['count'] += 1
                        if len(evidence.setdefault(key, [])) < 8 or len(group['evidence']) < 8:
                            hit = dict(file=member['name'], line=line_number, excerpt=scrub(line[:700]))
                            if len(evidence[key]) < 8:
                                evidence[key].append(hit)
                            if len(group['evidence']) < 8:
                                group['evidence'].append(hit)
                offset += len(lines)
        files.append(dict(filename=member['name'], sha256=member['sha256'], size_bytes=member['size'],
                          line_count=offset, original_runtime_count=raw_runtime,
                          corrected_counts=dict(counts), evidence=evidence,
                          has_true_crash=bool(counts)))
        original_runtime_count += raw_runtime
        bytes_scanned += member['size']
        if index % 250 == 0 or index == len(manifest['members']):
            print(json.dumps(dict(files=index, total_files=len(manifest['members']), bytes_scanned=bytes_scanned,
                                  elapsed_seconds=round(time.monotonic()-started, 1),
                                  true_crash_files=sum(f['has_true_crash'] for f in files))), flush=True)
    result = dict(manifest=str(manifest_path), created_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(time.monotonic()-started, 2), files_scanned=len(files),
                  bytes_scanned=bytes_scanned, original_runtime_count=original_runtime_count,
                  corrected_runtime_count=groups['runtime']['count'],
                  true_crash_files=sum(f['has_true_crash'] for f in files),
                  crashes=[group for group in groups.values() if group['count']], files=files)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.partial')
    temp.write_text(json.dumps(result, indent=2)+'\n')
    temp.replace(output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'crashes')}), flush=True)
    return result



def repair_generic(manifest_path, output_path):
    """Recheck only files with prior generic hits after a signature correction."""
    started = time.monotonic()
    output = Path(output_path)
    report = json.loads(output.read_text())
    members = {m['name']: m for m in json.loads(Path(manifest_path).read_text())['members']}
    pattern = next(pattern for key, _, pattern, _ in SIGNATURES if key == 'generic')
    global_group = dict(id='generic', title='Job failure', count=0, evidence=[])
    scanned = 0
    prior_count = sum(f['corrected_counts'].get('generic', 0) for f in report['files'])
    for item in report['files']:
        if not item['corrected_counts'].get('generic'):
            continue
        member = members[item['filename']]
        if member['sha256'] != item['sha256'] or member['size'] != item['size_bytes']:
            raise ValueError('Manifest identity changed: '+item['filename'])
        count = 0
        hits = []
        offset = 0
        with Path(member['path']).open(encoding='utf-8') as stream:
            while lines := list(islice(stream, 16384)):
                selected = pc.indices_nonzero(pc.match_substring_regex(pa.array(lines), pattern=pattern, ignore_case=True)).to_pylist()
                count += len(selected)
                for index in selected:
                    if len(hits) < 8:
                        hits.append(dict(file=item['filename'], line=offset+index+1, excerpt=scrub(lines[index].rstrip('\r\n')[:700])))
                offset += len(lines)
        if count:
            item['corrected_counts']['generic'] = count
            item['evidence']['generic'] = hits
            global_group['count'] += count
            global_group['evidence'].extend(hits[:max(0, 8-len(global_group['evidence']))])
        else:
            item['corrected_counts'].pop('generic', None)
            item['evidence'].pop('generic', None)
        item['has_true_crash'] = bool(item['corrected_counts'])
        scanned += 1
    report['crashes'] = [g for g in report['crashes'] if g['id'] != 'generic']
    if global_group['count']:
        report['crashes'].append(global_group)
    report['true_crash_files'] = sum(f['has_true_crash'] for f in report['files'])
    report['generic_correction'] = dict(created_at=datetime.now(timezone.utc).isoformat(),
        reason='Match FATAL as a standalone token; non-fatal diagnostics are not crash evidence.',
        pattern=pattern, files_rescanned=scanned, prior_count=prior_count,
        corrected_count=global_group['count'], elapsed_seconds=round(time.monotonic()-started, 2))
    temp = output.with_suffix('.partial')
    temp.write_text(json.dumps(report, indent=2)+'\n')
    temp.replace(output)
    print(json.dumps(dict(correction=report['generic_correction'], true_crash_files=report['true_crash_files'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--output', required=True,help='Operator-selected audit output JSON path')
    parser.add_argument('--repair-generic', action='store_true')
    args = parser.parse_args()
    if args.repair_generic:
        repair_generic(args.manifest, args.output)
    else:
        audit(args.manifest, args.output)
