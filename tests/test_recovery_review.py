"""Disk-backed analysis must preserve crash signatures and parser metadata."""
from backend.upload_io import DiskLog
from backend.parser import parse_log
from backend.troubleshooting import analyze
import pytest


def disk_log(tmp_path, text):
    path = tmp_path / 'evidence.log'
    path.write_text(text)
    return DiskLog(path, len(text.encode()), 'review-test', '')


@pytest.mark.parametrize('text,signature', [
    ('illegal memory access\n', 'cuda'),
    ('Bootstrap : no socket interface found\n', 'bootstrap'),
])
def test_disk_analysis_preserves_bare_crash_signatures(tmp_path, text, signature):
    report = analyze([('evidence.log', disk_log(tmp_path, text))], 'h100-spectrum-x')
    assert signature in {group['id'] for group in report['crashes']}
    assert report['runs'][0]['parsed']['status'] == 'failed'


def test_disk_parser_preserves_job_and_steps(tmp_path):
    text = 'SLURM_JOB_ID=1234\nsteps=9\n'
    run = parse_log(disk_log(tmp_path, text).parser_lines())
    assert run['job_id'] == '1234'
    assert run['metadata']['steps'] == 9


@pytest.mark.parametrize('text', [
    'NCCL INFO ncclOsDlsym(ncclGinPlugin_v12) failed: libnccl-net.so: undefined symbol: ncclGinPlugin_v12',
    'NCCL INFO ncclOsDlsym(ncclProfiler_v6) failed: libnccl-net.so: undefined symbol: ncclProfiler_v6',
    'NCCL INFO ncclOsDlsym(ncclTunerPlugin_v5) failed: libnccl-net.so: undefined symbol: ncclTunerPlugin_v5',
    'bnccl-net.so: undefined symbol: ncclGinPlugin_v11',
])
def test_optional_plugin_probes_do_not_fail_completed_job(tmp_path, text):
    text += '\niteration 1 / 1 elapsed time per iteration (ms): 100.0\n'
    report = analyze([('llama.log', disk_log(tmp_path, text))], 'h100-spectrum-x')
    assert not report['crashes']
    assert report['runs'][0]['parsed']['status'] == 'passed'


@pytest.mark.parametrize('text', [
    'python: undefined symbol: ncclCommInitRank',
    'undefined symbol: ncclGinPlugin_v12; undefined symbol: requiredRuntimeSymbol',
    'error while loading shared libraries: libnccl.so: cannot open shared object file; undefined symbol: ncclProfiler_v6',
    'undefined symbol: ncclGinPlugin_v12_required',
])
def test_real_runtime_failures_survive_optional_probe_exclusion(tmp_path, text):
    report = analyze([('evidence.log', disk_log(tmp_path, text))], 'h100-spectrum-x')
    assert 'runtime' in {group['id'] for group in report['crashes']}
    assert report['runs'][0]['parsed']['status'] == 'failed'


@pytest.mark.parametrize('text,signature', [
    ('slurmstepd: error: *** JOB 53 ON node001 CANCELLED AT 2026-09-25T00:33:32 ***', 'scheduler'),
    ('*** STEP 53.0 CANCELLED AT 2026-09-25T00:33:32 ***', 'scheduler'),
    ('iteration 8/50 | grad norm: nan | number of nan iterations: 1 |', 'numerical'),
    ('AssertionError: gradient norm is NaN', 'numerical'),
])
def test_scheduler_and_nonfinite_gradients_fail_job(tmp_path, text, signature):
    report = analyze([('evidence.log', disk_log(tmp_path, text))], 'h100-spectrum-x')
    assert signature in {group['id'] for group in report['crashes']}
    assert report['runs'][0]['parsed']['status'] == 'failed'


def test_nan_configuration_and_finite_grad_norm_are_not_failure(tmp_path):
    text = 'check_for_nan_in_loss_and_grad=True\niteration 1 / 1 elapsed time per iteration (ms): 100.0 | grad norm: 3.704 | number of nan iterations: 0 |\n'
    report = analyze([('llama.log', disk_log(tmp_path, text))], 'h100-spectrum-x')
    assert not report['crashes']
    assert report['runs'][0]['parsed']['status'] == 'passed'


def test_nested_diagnostics_inherit_primary_job_and_failure():
    entries = [
        ('bundle/experimentA/diagnostics/nested/nccl-all_reduce.out', '# Collective test starting: all_reduce_perf\nAvg bus bandwidth : 100\nOut of bounds values : 0\n'),
        ('bundle/experimentA/log-exemplar-training_53_0.out', 'FATAL training failure\n'),
        ('bundle/experimentB/log-exemplar-training_54_0.out', 'iteration 1 / 1 elapsed time per iteration (ms): 100.0\n'),
        ('bundle/experimentB/nccl-all_reduce.out', '# Collective test starting: all_reduce_perf\nAvg bus bandwidth : 110\nOut of bounds values : 0\n'),
    ]
    report = analyze(entries, 'h100-spectrum-x')
    runs = {r['filename']: r['parsed'] for r in report['runs']}
    nested = runs[entries[0][0]]
    assert nested['job_id'] == '53'
    assert nested['metadata']['parent_job_failed'] is True
    other = runs[entries[3][0]]
    assert other['job_id'] == '54'
    assert not other['metadata'].get('parent_job_failed')


def test_duplicate_primary_workers_agree_but_ambiguous_context_is_unknown():
    from backend.troubleshooting import job_contexts, context_job_id
    names = [
        'root/jobA/log-exemplar-training_53_0.out',
        'root/jobA/log-exemplar-training_53_1.out',
        'root/jobA/sbatch_training_53.out',
        'root/jobB/log-exemplar-training_54_0.out',
        'root/jobB/log-exemplar-training_55_0.out',
    ]
    contexts = job_contexts(names)
    assert context_job_id('root/jobA/diagnostics/nccl-test.out', contexts) == '53'
    assert context_job_id('root/jobB/nccl-test.out', contexts) is None
    assert context_job_id('root/jobA_other/nccl-test.out', contexts) is None


def test_nonfatal_diagnostic_steps_do_not_fail_job(tmp_path):
    text = '[FAILED] exit code 1 (non-fatal, continuing)\nFailed non-fatal steps: 3\n'
    report = analyze([('diagnostic.log', disk_log(tmp_path, text))], 'h100-spectrum-x')
    assert not report['crashes']
    assert report['runs'][0]['parsed']['status'] != 'failed'
    assert not parse_log(text.splitlines())['metadata'].get('exit_failure')


def test_real_fatal_error_still_fails_job(tmp_path):
    text = '[rank0] FATAL: unable to initialize\n'
    report = analyze([('evidence.log', disk_log(tmp_path, text))], 'h100-spectrum-x')
    assert 'generic' in {group['id'] for group in report['crashes']}
    assert report['runs'][0]['parsed']['status'] == 'failed'
    assert parse_log(text.splitlines())['metadata']['exit_failure'] is True
