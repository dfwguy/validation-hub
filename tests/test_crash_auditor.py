import hashlib
import json
from tools.audit_retained_crashes import audit


def test_auditor_counts_filters_and_keeps_original_line_numbers(tmp_path):
    log = tmp_path / 'retained.log'
    text = ('ordinary chatter\n' * 3
            + 'NCCL INFO ncclOsDlsym(ncclGinPlugin_v12) failed: undefined symbol: ncclGinPlugin_v12\n'
            + 'python: undefined symbol: actualRequiredSymbol\n'
            + '*** JOB 53 CANCELLED AT 2026-09-25T00:33:32 ***\n')
    log.write_text(text)
    digest = hashlib.sha256(log.read_bytes()).hexdigest()
    manifest = tmp_path / 'archive.members.json'
    manifest.write_text(json.dumps({'members': [{'name': 'nested/rank.out', 'path': str(log),
                                                'size': log.stat().st_size, 'sha256': digest}]}))
    output = tmp_path / 'audit.json'
    result = audit(manifest, output)
    assert result['files_scanned'] == 1
    assert result['original_runtime_count'] == 2
    assert result['corrected_runtime_count'] == 1
    member = result['files'][0]
    assert member['sha256'] == digest
    assert member['line_count'] == 6
    assert member['has_true_crash'] is True
    assert member['corrected_counts'] == {'runtime': 1, 'scheduler': 1}
    assert member['evidence']['runtime'][0]['line'] == 5
    assert member['evidence']['scheduler'][0]['line'] == 6
    assert json.loads(output.read_text()) == result
