"""Repair classifier results in an existing upload without creating another ingestion.

Dry run by default. Run from the repository with PYTHONPATH=. and its virtualenv.
Uses the retained member manifest, scans only the selected job directories, and
commits catalog/report changes together after checking for concurrent changes.
"""
import argparse
import collections
import copy
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path

# Exact patterns used by the initial recovery. These establish a count delta;
# they are deliberately independent of the subsequently corrected classifier.
ORIGINAL_SIGNATURES = {
    'rdma': r'ibv_poll_cq.*(?:error|failed)|Got completion.*(?:error|status)|IB.*(?:retry exceeded|error 12)|transport retry counter exceeded',
    'bootstrap': r'No route to host|Connection refused|Network is unreachable|Bootstrap.*(?:failed|no socket)|socketStartConnect.*failed',
    'memory': r'CUDA out of memory|out of memory|OOM-kill|cudaErrorMemoryAllocation',
    'cuda': r'unhandled cuda error|illegal memory access|Xid\s*[:=]?\s*\d+|CUDA error|GPU has fallen off',
    'timeout': r'Watchdog.*(?:timeout|timed out)|DUE TO TIME LIMIT|collective.*timed out|NCCL (?:WARN|ERROR).*timeout',
    'imex': r'(?:IMEX|MNNVL|NVLS).*(?:failed|error|unavailable)',
    'runtime': r'undefined symbol|error while loading shared libraries|libnccl.*not found|libcuda.*not found',
    'generic': r'NCCL (?:WARN|ERROR).*?(?:error|failed)|FATAL|Traceback \(most recent call last\)|Segmentation fault|TEST_END.*rc=[1-9]',
}



def read_only(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def job_directories(members, jobs):
    result = {}
    for member in members:
        match = re.search(r'/log-exemplar-.*_(\d+)_0\.out$', member['name'])
        if match and match[1] in jobs:
            result[member['name'].rsplit('/', 1)[0] + '/'] = match[1]
    if set(result.values()) != jobs:
        raise ValueError('Manifest does not contain exactly the requested job contexts')
    return result


def reconcile(args):
    from backend import lake, store, troubleshooting, upload_io
    from backend.compare import compare
    from backend.security import scrub

    members = json.loads(args.manifest.read_text())['members']
    crash_audit_path = getattr(args, 'crash_audit', None)
    crash_audit = json.loads(crash_audit_path.read_text()) if crash_audit_path else None
    audit_files = {file['filename']: file for file in crash_audit['files']} if crash_audit else {}
    if crash_audit:
        if len(audit_files) != len(members) or set(audit_files) != {member['name'] for member in members}:
            raise ValueError('Crash audit must cover exactly every retained manifest member')
        for member in members:
            audited = audit_files[member['name']]
            if audited['sha256'] != member['sha256'] or audited['size_bytes'] != member['size']:
                raise ValueError('Crash audit SHA/size provenance mismatch: ' + member['name'])
    prefixes = job_directories(members, set(args.jobs.split(',')))
    selected = [(member, job) for member in members for prefix, job in prefixes.items()
                if member['name'].startswith(prefix)]
    if crash_audit:
        selected = [(member, job) for member, job in selected
                    if re.search(r'/log-exemplar-.*_\d+_0\.out$|/sbatch_[^/]+\.out$', member['name'])]
    with read_only(args.database) as db:
        matches = list(db.execute('SELECT * FROM analyses WHERE sha256=?', (args.sha256,)))
        if args.analysis_id:
            matches = [row for row in matches if row['id'] == args.analysis_id]
        if len(matches) != 1:
            raise ValueError('Expected one completed analysis; specify --analysis-id if several exist')
        analysis = dict(matches[0])
        report = json.loads(analysis['report'])
        originals = {row['id']: dict(row) for row in db.execute(
            "SELECT id,payload FROM runs WHERE cluster=? AND source_kind='upload'", (report['cluster'],))}
    if report.get('classifier_reconciliation'):
        raise ValueError('This analysis was already reconciled; refusing to apply original count deltas twice')
    if not report.get('comparisons') or report.get('ingestion', {}).get('original_sha256') != args.sha256:
        raise ValueError('Completed original recovery report is required')
    @contextmanager
    def planning_connection():
        db = read_only(args.database)
        try:
            db.execute('PRAGMA query_only=ON')
            yield db
        finally:
            db.close()
    entries = []
    old_counts = collections.Counter()
    job_by_name = {}
    class ReconciliationLog(upload_io.DiskLog):
        counted_original = False

        def _index(self):
            lines = super()._index()
            if not self.counted_original and not crash_audit:
                # Mark before matches(), which calls _index() recursively.
                self.counted_original = True
                for key, pattern in ORIGINAL_SIGNATURES.items():
                    old_counts[key] += sum(1 for _ in self.matches(pattern, re.I))
            return lines

    for member, job in selected:
        text = ReconciliationLog(Path(member['path']), member['size'], member['sha256'], '')
        entries.append((member['name'], text))
        job_by_name[member['name']] = job
    fresh = troubleshooting.analyze(entries, analysis['recipe_id'])
    affected_names = set(job_by_name)
    fresh_by_name = {entry['filename']: entry for entry in fresh.pop('runs')}
    if set(fresh_by_name) != affected_names:
        raise ValueError('Classifier did not return every selected member')
    contexts = troubleshooting.job_contexts([member['name'] for member in members])
    global_jobs = {member['name']: troubleshooting.context_job_id(member['name'], contexts)
                   for member in members}
    failed_jobs = set()
    for comparison in report['comparisons']:
        name = comparison['filename']
        parsed = fresh_by_name[name]['parsed'] if name in fresh_by_name else comparison['run']
        actual_failure = audit_files[name]['has_true_crash'] if crash_audit else parsed['status'] == 'failed'
        if actual_failure and global_jobs.get(name):
            failed_jobs.add(global_jobs[name])
    changed_runs = {}
    comparisons = []
    excluded = {member['sha256'] for member in members}
    parent_warning = 'Another artifact for this job reports a failure; clean point measurements cannot establish a passing validation job.'
    status_warning = 'Uploaded log is incomplete, lacks correctness evidence, or reports an error; performance acceptance cannot establish correctness.'
    for old in report['comparisons']:
        name = old['filename']
        if name not in affected_names and not global_jobs.get(name) and not crash_audit:
            comparisons.append(old)
            continue
        entry = fresh_by_name.get(name)
        run_id = old['run_id']
        if run_id not in originals:
            raise ValueError('Catalog run missing: ' + run_id)
        existing = json.loads(originals[run_id]['payload'])
        if entry:
            # One content-addressed catalog run may have multiple archive names
            # within the same job. Keep its original canonical member context.
            canonical_name = existing.get('metadata', {}).get('archive_member')
            entry = fresh_by_name.get(canonical_name, entry)
        run = copy.deepcopy(existing)
        metadata = {**existing.get('metadata', {})}
        metadata.pop('parent_job_failed', None)
        if entry:
            parsed = copy.deepcopy(entry['parsed'])
            metadata.pop('exit_failure', None)
            metadata.update(parsed.pop('metadata'))
            metadata.pop('parent_job_failed', None)
            for key in existing.get('metadata', {}).get('user_supplied_fields', []):
                parsed[key] = existing.get(key)
            run.update(parsed)
        run['job_id'] = global_jobs.get(name) or job_by_name.get(name) or existing.get('job_id')
        metadata['job_context_source'] = 'LLMB experiment directory; associated primary/scheduler filename'
        if run['job_id'] in failed_jobs:
            metadata['parent_job_failed'] = True
        if crash_audit:
            if audit_files[name]['has_true_crash']:
                metadata['exit_failure'] = True
                run['status'] = 'failed'
            else:
                metadata.pop('exit_failure', None)
                if run['status'] == 'failed':
                    iterations = [metric.get('iteration', 0) for metric in run.get('metrics', [])]
                    expected = metadata.get('expected_iterations')
                    run['status'] = ('passed' if expected and max(iterations, default=0) == expected else
                                     'incomplete' if run.get('metrics') else 'unknown')
        run['metadata'] = metadata
        if metadata.get('mixed_split_masks'):
            run['status'] = 'incomplete'
        if run_id in changed_runs and changed_runs[run_id] != run:
            raise ValueError('Identical content has conflicting member contexts: ' + run_id)
        changed_runs[run_id] = run
        if entry and run.get('metrics'):
            with patch.object(store, 'connection', planning_connection):
                outcome = compare(run, report.get('reference_cluster'), old.get('policy', {}).get('tolerance_pct', 10), excluded_hashes=excluded)
        else:
            # Existing point comparisons retain their exact baseline provenance.
            # Association changes only job acceptance, not numeric measurements.
            outcome = {k: copy.deepcopy(v) for k, v in old.items()
                       if k not in ('filename', 'run_id', 'measurement')}
            outcome['run'] = run
            points = outcome.get('comparisons', [])
            scored = [point for point in points if point['verdict'] in ('pass', 'fail')]
            outcome['verdict'] = ('fail' if run['status'] == 'failed' or metadata.get('parent_job_failed') or
                                  any(point['verdict'] == 'fail' for point in points) else
                                  'insufficient_evidence' if run['status'] != 'passed' or not scored or
                                  any(point['verdict'] == 'no_baseline' for point in points) else 'pass')
            warnings = [warning for warning in outcome.get('warnings', [])
                        if warning not in (parent_warning, status_warning)]
            if metadata.get('parent_job_failed'):
                warnings.append(parent_warning)
            if run['status'] != 'passed':
                warnings.append(status_warning)
            outcome['warnings'] = warnings
        comparisons.append(dict(filename=name, run_id=run_id, measurement=lake.metrics_summary(run), **outcome))
    if len({c['filename'] for c in comparisons if c['filename'] in affected_names}) != len(affected_names):
        raise ValueError('Existing report does not contain every selected member')

    # Replace only this subset's contribution. Retain bounded original evidence
    # from other members; add newly classified evidence with original line numbers.
    if crash_audit:
        advice = {key: action for key, _, _, action in troubleshooting.SIGNATURES}
        crashes = [dict(copy.deepcopy(group), next_action=advice[group['id']],
                        confidence='signature match; cause requires verification')
                   for group in crash_audit['crashes']]
    else:
        fresh_groups = {group['id']: group for group in fresh['crashes']}
        old_groups = {group['id']: group for group in report.get('crashes', [])}
        crashes = []
        for key in dict.fromkeys([*old_groups, *fresh_groups]):
            previous, current = old_groups.get(key), fresh_groups.get(key)
            count = (previous or {}).get('count', 0) - old_counts[key] + (current or {}).get('count', 0)
            if count < 0:
                raise ValueError('Original crash count provenance differs; full reanalysis is required: ' + key)
            if not count:
                continue
            group = copy.deepcopy(current or previous)
            evidence = [e for e in (previous or {}).get('evidence', []) if e['file'] not in affected_names]
            evidence += (current or {}).get('evidence', [])
            group.update(count=count, evidence=evidence[:8])
            crashes.append(group)
    report['crashes'] = crashes
    report['comparisons'] = comparisons
    report['baseline_attainment'] = lake.attainment(comparisons)
    runs_by_name = {c['filename']: c['run'] for c in comparisons}
    for file in report.get('files', []):
        if file['filename'] in runs_by_name:
            parsed = runs_by_name[file['filename']]
            file.update(status=parsed['status'], metrics=len(parsed['metrics']), job_id=parsed.get('job_id'),
                        parent_job_failed=bool(parsed.get('metadata', {}).get('parent_job_failed')))
    all_runs = [c['run'] for c in comparisons]
    failed = any(r['status'] == 'failed' or r.get('metadata', {}).get('parent_job_failed') or
                 any(m.get('errors') for m in r.get('metrics', [])) for r in all_runs)
    checks = [c for c in report['checks'] if c['id'] != 'crash']
    for check in checks:
        if check['id'] == 'correctness':
            check.update(status='fail' if failed else 'pass' if all_runs and all(r['status'] == 'passed' for r in all_runs) else 'unknown',
                         observed=[r.get('metadata', {}).get('correctness_errors') for r in all_runs])
    if crashes:
        checks.append(dict(id='crash', title='Crash evidence', status='fail',
                           expected='Resolve observed crash signatures before accepting results', observed=sum(g['count'] for g in crashes)))
    report['checks'] = checks
    report['verdict'] = 'fail' if any(c['status'] == 'fail' for c in checks) or report['baseline_attainment']['verdict'] == 'fail' else 'needs_evidence'
    report['follow_up_tests'] = sorted(lake.recommendations(comparisons) + troubleshooting.followups(report['recipe'], crashes, checks), key=lambda t: t['priority'])
    report['ingestion']['measured_runs'] = sum(bool(c['run'].get('metrics')) for c in comparisons)
    stamp = datetime.now(timezone.utc).isoformat()
    report['classifier_reconciliation'] = dict(at=stamp, jobs=sorted(set(job_by_name.values())), members=len(selected),
        original_report_sha256=hashlib.sha256(analysis['report'].encode()).hexdigest(),
        associated_jobs=sorted(set(global_jobs.values()) - {None}), failed_jobs=sorted(failed_jobs),
        method='Targeted retained-member reanalysis and original signature count deltas; all experiment members associated with evidenced jobs and propagated parent failures without rescanning; run and receipt identities preserved')
    if crash_audit:
        report['classifier_reconciliation']['crash_audit'] = dict(path=str(crash_audit_path),
            sha256=hashlib.sha256(crash_audit_path.read_bytes()).hexdigest(), files_validated=len(audit_files),
            method='Complete retained-manifest crash audit overrides stale optional-symbol failures; primary measurement evidence retained')
    summary = dict(analysis_id=analysis['id'], selected_members=len(selected), updated_runs=len(changed_runs),
                   associated_jobs=len(set(global_jobs.values()) - {None}), failed_jobs=sorted(failed_jobs),
                   changed_statuses=collections.Counter(run['status'] for run in changed_runs.values()),
                   old_crash_counts=dict(old_counts), new_crash_counts={g['id']: g['count'] for g in fresh['crashes']},
                   baseline_attainment=report['baseline_attainment'], applied=False)
    if args.apply:
        backup = args.database.with_name(args.database.name + '.before-reconcile-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        db = sqlite3.connect(args.database, timeout=60)
        try:
            with sqlite3.connect(backup) as dest:
                db.backup(dest)
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT report FROM analyses WHERE id=?', (analysis['id'],)).fetchone()
            if not current or current[0] != analysis['report']:
                raise ValueError('Analysis changed during planning; retry reconciliation')
            for run_id, run in changed_runs.items():
                current = db.execute('SELECT payload FROM runs WHERE id=?', (run_id,)).fetchone()
                if not current or current[0] != originals[run_id]['payload']:
                    raise ValueError('Catalog changed during planning; retry reconciliation')
                run = scrub(run)
                columns = ['workload','collective','nodes','gpus','racks','sus','hardware','transport','nccl_version','status','avg_bus_bw_gbps','peak_bus_bw_gbps','tflops_gpu','iterations','job_id']
                db.execute('UPDATE runs SET ' + ','.join(k + '=?' for k in columns) + ',payload=? WHERE id=?',
                           [run.get(k) for k in columns] + [json.dumps(run, allow_nan=False), run_id])
            db.execute('UPDATE analyses SET report=? WHERE id=?', (json.dumps(scrub(report), allow_nan=False), analysis['id']))
            db.execute('UPDATE upload_batches SET measured_runs=? WHERE id=?', (report['ingestion']['measured_runs'], report['ingestion']['batch_id']))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
        summary.update(applied=True, backup=str(backup))
    return summary


def main():
    from backend import config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=config.data_path('VALIDATION_DB','catalog/validation.sqlite'))
    parser.add_argument('--sha256', required=True,help='SHA-256 of the original retained upload')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--analysis-id')
    parser.add_argument('--crash-audit', type=Path)
    parser.add_argument('--jobs', required=True,help='Comma-separated explicit job IDs to reconcile')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{64}',args.sha256):parser.error('Use a lowercase 64-character SHA-256')
    if not re.fullmatch(r'\d+(?:,\d+)*',args.jobs):parser.error('Use comma-separated numeric job IDs')
    args.manifest = args.manifest or config.uploads_root() / 'received' / (args.sha256 + '.members.json')
    print(json.dumps(reconcile(args), indent=2, default=dict))


if __name__ == '__main__':
    main()
