import copy
import pytest
from backend.performance import compare_training, training_context


def context(**overrides):
    value = dict(model='Llama3.1', model_size='405B', gpu='B300', gpus=512,
                 precision='fp8_cs', release='26.09', global_batch_size=3072,
                 sequence_length=8192, tensor_parallel=4, pipeline_parallel=8,
                 context_parallel=1, virtual_pipeline=4, expert_parallel=1,
                 expert_tensor_parallel='none', micro_batch_size=1)
    value.update(overrides)
    return value


def observed(**overrides):
    value = dict(**context(), status='passed', metrics=[dict(tflops_gpu=950), dict(time_ms=10000)])
    value.update(overrides)
    return value


def reference(**overrides):
    value = dict(**context(), ref_id='reference1', source_record_id='row1', job_id='job1',
                 source_url='https://beacon.example/record/1', tflops_gpu=1000, seconds_per_step=9.5)
    value.update(overrides)
    return value


def test_metric_direction_and_inclusive_threshold_boundaries():
    result = compare_training(observed(), [reference()])
    assert result['tier'] == 'exact' and result['classification'] == 'good' and result['verdict'] == 'pass'
    assert [metric['expected_pct'] for metric in result['metrics']] == [95, 95]
    assert all(metric['scored'] for metric in result['metrics'])
    result = compare_training(observed(metrics=[dict(tflops_gpu=900)]), [reference()])
    assert result['classification'] == 'bad' and result['verdict'] == 'fail'
    result = compare_training(observed(metrics=[dict(tflops_gpu=899)]), [reference()])
    assert result['classification'] == 'worst'


def test_lower_step_time_is_better_and_best_reference_is_minimum():
    refs = [reference(seconds_per_step=10), reference(ref_id='r2', job_id='job2', seconds_per_step=12)]
    result = compare_training(observed(metrics=[dict(time_ms=10000)]), refs, reference_statistic='best')
    assert result['metrics'][0]['expected'] == 10
    assert result['metrics'][0]['expected_pct'] == 100
    assert result['classification'] == 'good'
    result = compare_training(observed(metrics=[dict(time_ms=11000)]), refs)
    assert result['metrics'][0]['expected'] == 11
    assert result['metrics'][0]['expected_pct'] == 100


def test_missing_and_different_configuration_are_conditional_not_passes():
    ref = reference();del ref['release']
    result = compare_training(observed(), [ref])
    assert result['tier'] == 'contextual' and result['verdict'] == 'insufficient_evidence'
    assert result['classification'] == 'conditional' and 'release' in result['missing_context']
    assert not any(metric['scored'] for metric in result['metrics'])
    result = compare_training(observed(), [reference(sequence_length=4096)])
    assert result['verdict'] == 'insufficient_evidence' and 'sequence_length' in result['mismatches']
    assert compare_training(observed(), [reference(gpu='H100')])['tier'] == 'none'
    assert compare_training(observed(), [reference(precision='bf16')])['tier'] == 'none'


def test_failed_job_wins_even_when_faster_than_reference():
    for failure in (dict(status='failed'), dict(metadata=dict(parent_job_failed=True)), dict(metadata=dict(exit_failure=True))):
        result = compare_training(observed(metrics=[dict(tflops_gpu=2000)], **failure), [reference()])
        assert result['verdict'] == 'fail' and result['classification'] == 'failed'
        assert not result['metrics'][0]['scored']
    result = compare_training(observed(status='incomplete'), [reference()])
    assert result['classification'] == 'conditional' and result['verdict'] == 'insufficient_evidence'


def test_theoretical_and_unsourced_values_are_not_measurements():
    result = compare_training(observed(), [reference(kind='theoretical_sol'), reference(source_url=None), reference(is_measured=False), reference(raw_json=dict(is_sol=True))])
    assert result['tier'] == 'none' and result['metrics'] == []


def test_repeat_weighting_uses_one_median_per_job_not_each_artifact():
    refs = [reference(tflops_gpu=1000), reference(ref_id='r2', tflops_gpu=2000),
            reference(ref_id='r3', job_id='job2', tflops_gpu=500)]
    result = compare_training(observed(), refs)
    assert result['metrics'][0]['expected'] == 1000
    assert result['metrics'][0]['reference_jobs'] == 2
    assert result['provenance']['reference_records'] == 3
    best = compare_training(observed(), refs, reference_statistic='best')
    assert best['metrics'][0]['expected'] == 1500


def test_synthetic_path_context_keeps_unknown_release_and_sequence():
    run = dict(workload='Llama', metadata=dict(archive_member='pretrain_llama31_405b_fp8_cs_gpus512_tp4_pp8_cp1_vp4_ep1_etpNone_mbs1_gbs3072/log-exemplar-name_145_0.out', declared_gpu_model='B300'))
    extracted = training_context(run)
    assert extracted['context']['model'] == 'llama3.1'
    assert extracted['context']['model_size'] == '405b'
    assert extracted['context']['precision'] == 'fp8_cs'
    assert extracted['context']['gpus'] == 512
    assert extracted['context']['expert_tensor_parallel'] == 'none'
    assert extracted['context']['release'] is None and extracted['context']['sequence_length'] is None
    assert extracted['unverified'] == ['gpu']


def test_aggregation_and_policy_input_validation():
    result = compare_training(observed(metrics=[dict(tflops_gpu=1),dict(tflops_gpu=1000),dict(tflops_gpu=1001)]), [reference()])
    assert result['metrics'][0]['observed'] == 1000
    assert result['policy']['observed_statistic'] == 'median'
    with pytest.raises(ValueError):compare_training(observed(), [], good_pct=15, bad_pct=10)
    with pytest.raises(ValueError):compare_training(observed(), [], reference_statistic='largest')


def test_beacon_compact_workload_and_conditional_ratio_band():
    ref = reference(workload='deepseek-v3|671b', model=None, model_size=None)
    extracted = training_context(ref)
    assert extracted['context']['model'] == 'deepseek-v3'
    assert extracted['context']['model_size'] == '671b'
    run = observed(model='deepseek-v3', model_size=None, metrics=[dict(tflops_gpu=850)])
    result = compare_training(run, [ref])
    assert result['tier'] == 'contextual' and result['classification'] == 'conditional'
    assert result['candidate_band'] == 'worst'
    metric = result['metrics'][0]
    assert metric['performance_band'] == 'worst' and metric['delta_pct'] == -15
    assert not metric['scored']


def test_same_source_job_repeated_on_different_record_links_counts_once():
    refs = [reference(source_url='https://beacon.example/record/1', tflops_gpu=1000),
            reference(source_url='https://beacon.example/record/2', ref_id='r2', tflops_gpu=2000),
            reference(source_url='https://beacon.example/record/3', ref_id='r3', job_id='job2', tflops_gpu=500)]
    result = compare_training(observed(), refs)
    assert result['provenance']['reference_jobs'] == 2
    assert result['metrics'][0]['expected'] == 1000


def test_contextual_references_select_one_nearest_cohort_not_mixed_statistics():
    refs = [reference(release='26.09', global_batch_size=3072, tflops_gpu=1000),
            reference(ref_id='r2', job_id='job2', release='26.10', global_batch_size=6144, tflops_gpu=1900),
            reference(ref_id='r3', job_id='job3', release='26.10', global_batch_size=6144, tflops_gpu=2100)]
    result = compare_training(observed(release=None, global_batch_size=None, metrics=[dict(tflops_gpu=1000)]), refs)
    assert result['classification'] == 'conditional'
    assert result['metrics'][0]['expected'] == 2000
    assert result['provenance']['reference_jobs'] == 2
    assert result['provenance']['alternative_cohorts'] == 1
    assert result['provenance']['selected_reference_context']['release'] == '26.10'
    # A closer known release takes priority over a larger incompatible cohort.
    nearest = compare_training(observed(global_batch_size=None, metrics=[dict(tflops_gpu=1000)]), refs)
    assert nearest['metrics'][0]['expected'] == 1000
    assert len(nearest['references']) == 1
    assert nearest['provenance']['selected_reference_context']['release'] == '26.09'


def test_archive_release_declared_with_provenance_and_v_prefix_equivalence():
    run = observed(release=None, metadata=dict(upload_filename='ExampleCluster/llmb-archive_v26.08.01_20300101T000000Z.tar.zst'))
    extracted = training_context(run)
    assert extracted['context']['release'] == '26.08.01'
    assert extracted['provenance']['release'] == 'archive_filename_declared'
    assert compare_training(run, [reference(release='v26.08.01')])['tier'] == 'exact'
    explicit = observed(release='v26.09.01', metadata=dict(upload_filename='llmb-archive_v26.08.01.tar.zst'))
    assert training_context(explicit)['context']['release'] == '26.09.01'
    assert training_context(explicit)['provenance']['release'] == 'record'


def test_beacon_native_schema_normalizes_into_training_context():
    # Schema-compatible synthetic values; this is not a reference dataset.
    from backend.beacon import normalize_record
    raw = dict(id=123, model='llama3', model_size='70b', gpu='B300', scale=512,
               precision='fp8', precision_variant='mx', release='v26.08.01',
               global_batch_size=2048, sequence_length=4096, tensor_parallelism=1,
               pipeline_parallelism=4, context_parallelism=1, virtual_pipeline_parallelism=5,
               expert_model_parallel_size=1, expert_tensor_parallel_size=1,
               micro_batch_size=1, tflops_gpu_avg=1000, seconds_per_step=10)
    ref = normalize_record(raw, source_url='https://api.llmb.nvidia.com/v2/training/runs')
    extracted = training_context(ref)
    assert extracted['context']['model'] == 'llama3'
    assert extracted['context']['precision'] == 'fp8_mx'
    assert extracted['context']['release'] == '26.08.01'
    assert extracted['context']['expert_tensor_parallel'] == 1
    assert extracted['missing'] == []


def test_gpu_in_archive_filename_is_a_declaration_not_observed_hardware():
    run = dict(workload='Llama', metadata=dict(upload_filename='ExampleCluster_B300_llmb-archive_v26.08.01.tar.zst'))
    extracted = training_context(run)
    assert extracted['context']['gpu'] == 'B300'
    assert extracted['provenance']['gpu'] == 'filename_declaration_unverified'
    assert extracted['unverified'] == ['gpu']


def test_lossless_source_job_directory_supplies_explicit_model_variant():
    ref = reference(model='qwen3', model_size='235b', raw_json=dict(job_dir='/results/pretrain_qwen3_235b_a22b_bf16_gpus512_b300'))
    extracted = training_context(ref)
    assert extracted['context']['model'] == 'qwen3-a22b'
    assert extracted['context']['model_size'] == '235b'
    assert extracted['provenance']['model'] == 'experiment_path_variant'
    nemotron = reference(model='nemotron_3', model_size='120b', raw_json=dict(job_dir='/results/pretrain_nemotron_3_super_bf16_gpus512_b300'))
    assert training_context(nemotron)['context']['model'] == 'nemotron-3-super'
    # Missing source variant evidence must not be invented from family alone.
    assert training_context(reference(model='qwen3', model_size='235b'))['context']['model'] == 'qwen3'
    assert training_context(reference(model='llama3.1', model_size='70b'))['context']['model'] != 'llama3'


def test_compact_active_parameter_size_is_structurally_equivalent():
    first = training_context(reference(model='qwen3', model_size='235ba22b'))
    second = training_context(reference(model='qwen3', model_size='235b-a22b'))
    assert first['context'] == second['context']
    assert first['context']['model'] == 'qwen3-a22b'
    assert first['context']['model_size'] == '235b'
