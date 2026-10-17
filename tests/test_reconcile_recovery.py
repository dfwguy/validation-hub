import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from backend import store, troubleshooting, upload_io
from backend.app import ingest_upload
from tools.reconcile_recovery import ORIGINAL_SIGNATURES, reconcile


@pytest.mark.parametrize('use_audit', [False, True])
def test_targeted_reconciliation_preserves_identity_and_scans_once(tmp_path, monkeypatch, use_audit):
    database = tmp_path / 'catalog.sqlite'
    monkeypatch.setattr(store, 'DB_PATH', str(database))
    monkeypatch.setenv('VALIDATION_UPLOADS', str(tmp_path / 'uploads'))
    optional = 'NCCL INFO ncclOsDlsym(ncclGinPlugin_v12) failed: libnccl-net.so: undefined symbol: ncclGinPlugin_v12\n'
    train = 'iteration 1/1 | elapsed time per iteration (ms): 123.4 | global batch size: 8\nGPU utilization: 800MODEL_TFLOP/s/GPU\n'
    logs = {
        'experiment53/log-exemplar-pretrain_deepseek_gpus512_53_0.out': optional + train,
        'experiment53/sbatch_exemplar-pretrain_53.out': optional,
        'experiment53/log-exemplar-pretrain_deepseek_gpus512_53_1.out': optional + train,
        'experiment37/log-exemplar-pretrain_nemotron_gpus512_37_0.out': 'slurmstepd: error: *** STEP 37.1 CANCELLED DUE TO TASK FAILURE ***\n',
        'unrelated/log-exemplar-pretrain_qwen_gpus512_99_0.out': train,
        'unrelated/nccl-worker99.out': (optional if use_audit else '') + 'NCCL INFO normal worker diagnostics\n',
        'experiment77/log-exemplar-pretrain_nemotron_gpus512_77_0.out': 'Traceback (most recent call last):\n',
        'experiment77/nccl-worker77.out': 'NCCL INFO normal worker diagnostics77\n',
    }
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as tar:
        for name, text in logs.items():
            member = tarfile.TarInfo(name);member.size = len(text.encode())
            tar.addfile(member, io.BytesIO(text.encode()))
    content = archive.getvalue();sha = hashlib.sha256(content).hexdigest()
    with monkeypatch.context() as old:
        old.setattr(troubleshooting, 'SIGNATURES', [s for s in troubleshooting.SIGNATURES if s[0] in ORIGINAL_SIGNATURES])
        old.setattr(troubleshooting, 'runtime_failure', lambda line: True)
        old.setattr(troubleshooting, 'context_job_id', lambda name, contexts: None)
        report = ingest_upload(content, 'fixture.tar', 'b300-spectrum-x', cluster='cluster-fixture')
    members = []
    for ordinal, (name, text) in enumerate(logs.items()):
        path = tmp_path / f'member-{ordinal}.log';path.write_text(text)
        members.append(dict(name=name, path=str(path), size=path.stat().st_size, sha256=hashlib.sha256(text.encode()).hexdigest()))
    manifest = tmp_path / 'members.json';manifest.write_text(json.dumps(dict(members=members)))
    args = SimpleNamespace(database=database, manifest=manifest, sha256=sha, analysis_id=report['id'], jobs='53,37', apply=False)
    if use_audit:
        audited = troubleshooting.analyze(list(logs.items()), 'b300-spectrum-x')
        audit_by_name = {entry['filename']: entry for entry in audited['runs']}
        audit_path = tmp_path / 'crash-audit.json'
        audit_path.write_text(json.dumps(dict(crashes=audited['crashes'], files=[dict(filename=member['name'],
            sha256=member['sha256'], size_bytes=member['size'],
            has_true_crash=audit_by_name[member['name']]['parsed']['status'] == 'failed') for member in members])))
        args.crash_audit = audit_path
    reads = [];original_index = upload_io.DiskLog._index
    def index(self):
        if self._lines is None: reads.append(str(self.path))
        return original_index(self)
    monkeypatch.setattr(upload_io.DiskLog, '_index', index)
    with store.connection() as db:
        original_report = db.execute('SELECT report FROM analyses WHERE id=?', (report['id'],)).fetchone()[0]
        before_counts = tuple(db.execute('SELECT (SELECT COUNT(*) FROM runs),(SELECT COUNT(*) FROM analyses),(SELECT COUNT(*) FROM upload_batches),(SELECT COUNT(*) FROM upload_members)').fetchone())
    dry = reconcile(args)
    expected_reads = 3 if use_audit else 4
    assert not dry['applied'] and dry['selected_members'] == expected_reads
    assert len(reads) == expected_reads and len(set(reads)) == expected_reads
    with store.connection() as db:
        assert db.execute('SELECT report FROM analyses WHERE id=?', (report['id'],)).fetchone()[0] == original_report
    args.apply = True;applied = reconcile(args)
    assert applied['applied'] and Path(applied['backup']).is_file()
    with store.connection() as db:
        assert tuple(db.execute('SELECT (SELECT COUNT(*) FROM runs),(SELECT COUNT(*) FROM analyses),(SELECT COUNT(*) FROM upload_batches),(SELECT COUNT(*) FROM upload_members)').fetchone()) == before_counts
        corrected = json.loads(db.execute('SELECT report FROM analyses WHERE id=?', (report['id'],)).fetchone()[0])
        statuses = {c['filename']: c['run']['status'] for c in corrected['comparisons']}
        assert statuses[next(n for n in logs if '_53_0.out' in n)] == 'passed'
        assert statuses[next(n for n in logs if '_37_0.out' in n)] == 'failed'
        assert 'runtime' not in {g['id'] for g in corrected['crashes']}
        workers = {c['filename']: c for c in corrected['comparisons'] if 'worker' in c['filename']}
        assert workers['unrelated/nccl-worker99.out']['run']['job_id'] == '99'
        assert not workers['unrelated/nccl-worker99.out']['run']['metadata'].get('parent_job_failed')
        if use_audit:
            assert workers['unrelated/nccl-worker99.out']['run']['status'] == 'unknown'
            assert not workers['unrelated/nccl-worker99.out']['run']['metadata'].get('exit_failure')
        assert workers['experiment77/nccl-worker77.out']['run']['job_id'] == '77'
        assert workers['experiment77/nccl-worker77.out']['run']['metadata']['parent_job_failed']
        assert workers['experiment77/nccl-worker77.out']['verdict'] == 'fail'
        assert set(corrected['classifier_reconciliation']['failed_jobs']) == {'37', '77'}
        assert next(g['count'] for g in corrected['crashes'] if g['id'] == 'scheduler') == 1
        for c in corrected['comparisons']:
            payload = json.loads(db.execute('SELECT payload FROM runs WHERE id=?', (c['run_id'],)).fetchone()[0])
            assert payload == c['run']
    with pytest.raises(ValueError, match='already reconciled'): reconcile(args)
