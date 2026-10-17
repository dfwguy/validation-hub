import copy
import pytest
from backend.nccl_reference import compare_nccl, NCCLReferenceIndex, nccl_context, from_run_reference


def base(**changes):
    value = dict(gpu='B300',gpus=512,collective='all_reduce',fabric='Spectrum-X',network='Spectrum-X',
                 nccl_version='2.30.4',protocol='Simple',status='passed',nodes=64,placement_scope='inter_rack',rail_topology='8_rails',nic_model='CX8',link_speed_gbps=800,congestion_control='ECN_PFC',nccl_tests_version='2.17.6',nccl_environment={},metadata=dict(NCCL_TESTS_SPLIT_MASK='0x0',test_protocol=dict(
                 min_bytes=1024,max_bytes=1024,iterations=100,warmup=20,registration=True)),
                 metrics=[dict(message_bytes=1024,datatype='float',operation='sum',placement='out_of_place',busbw_gbps=95,time_us=10,errors=0)])
    value.update(changes);return value


def ref(**changes):
    value = base(ref_id='ref1',source_record_id='row1',source_url='https://beacon.example/nccl/1',job_id='job1')
    value['metrics'][0].update(busbw_gbps=100,time_us=9.5)
    value.update(changes);return value


def test_exact_direction_thresholds_and_failure_override():
    result = compare_nccl(base(),[ref()])
    assert result['tier']=='exact' and result['verdict']=='pass' and result['classification']=='good'
    assert [metric['expected_pct'] for metric in result['metrics']]==[95,95]
    failed=compare_nccl(base(metadata=dict(parent_job_failed=True)),[ref()])
    assert failed['verdict']=='fail' and failed['classification']=='failed'
    assert all(not metric['scored'] for metric in failed['metrics'])


def test_known_fabric_or_network_difference_never_matches():
    assert compare_nccl(base(),[ref(fabric='NVLink',network='NVLink')])['tier']=='none'
    assert compare_nccl(base(),[ref(fabric='InfiniBand',network='Quantum2')])['metrics']==[]
    assert compare_nccl(base(),[ref(gpu='GB300')])['tier']=='none'
    assert compare_nccl(base(),[ref(gpus=8)])['tier']=='none'
    assert compare_nccl(base(),[ref(collective='all_gather')])['tier']=='none'


def test_unknown_and_different_test_context_are_conditional():
    source=ref(nccl_version=None)
    result=compare_nccl(base(),[source])
    assert result['tier']=='contextual' and result['classification']=='conditional'
    assert 'nccl_version' in result['missing_context']
    assert all(not metric['scored'] for metric in result['metrics'])
    assert result['metrics'][0]['reference_context']['message_bytes'] == 1024
    assert compare_nccl(base(),[ref(protocol='LL128')])['tier']=='contextual'


def test_cohorts_do_not_mix_known_different_size_or_protocol_and_repeats():
    one=ref();two=ref(ref_id='ref2',job_id='job2');two['metrics'][0]['message_bytes']=2048;two['metrics'][0]['busbw_gbps']=1000
    result=compare_nccl(base(),NCCLReferenceIndex([one,two]))
    assert result['metrics'][0]['expected']==100
    duplicate=ref(ref_id='ref3',source_url='https://beacon.example/nccl/3');duplicate['metrics'][0]['busbw_gbps']=200
    third=ref(ref_id='ref4',job_id='job3');third['metrics'][0]['busbw_gbps']=50
    repeat=compare_nccl(base(),[one,duplicate,third])
    assert repeat['metrics'][0]['expected']==100
    assert repeat['metrics'][0]['reference_jobs']==2


def test_cached_match_does_not_confuse_observed_and_recipe_gpu():
    index=NCCLReferenceIndex([ref()])
    assert compare_nccl(base(),index)['tier']=='exact'
    declared=base(gpu=None,metadata=dict(declared_gpu_model='B300',test_protocol=base()['metadata']['test_protocol']))
    result=compare_nccl(declared,index)
    assert result['tier']=='contextual' and 'gpu' in result['missing_context']
    assert len(index.cache)==2


def test_legacy_verbs_transport_is_unknown_without_link_layer_evidence():
    run=base(fabric=None,network=None,transport='InfiniBand + NVLink')
    context=nccl_context(run)
    assert context['context']['fabric'] is None
    result=compare_nccl(run,[ref()])
    assert result['tier']=='none'
    legacy_source=ref(fabric=None,network=None,transport='InfiniBand + NVLink')
    assert compare_nccl(run,[legacy_source])['tier']=='contextual'
    explicit=base(fabric=None,network=None,transport='InfiniBand + NVLink',metadata=dict(link_layer='InfiniBand'))
    assert compare_nccl(explicit,[ref()])['tier']=='none'


def test_incomplete_sweep_or_bad_correctness_cannot_pass():
    run=base();run['metrics'].append(dict(message_bytes=2048,datatype='float',operation='sum',placement='out_of_place',busbw_gbps=95,errors=1))
    result=compare_nccl(run,[ref()])
    assert result['verdict']=='fail'
    assert all(not point['scored'] for point in result['metrics'])
    assert compare_nccl(base(status='incomplete'),[ref()])['classification']=='conditional'


def test_summary_reference_is_explicitly_contextual_and_theory_excluded():
    reference=ref(metrics=[],avg_bus_bw_gbps=100)
    result=compare_nccl(base(),[reference])
    assert result['tier']=='none' and result['metrics']==[]
    assert compare_nccl(base(),[ref(kind='theoretical_sol')])['tier']=='none'
    with pytest.raises(ValueError):compare_nccl(base(),[],good_pct=20,bad_pct=10)


def test_catalog_adapter_requires_clean_measured_source_and_preserves_unknowns():
    run=base(id='catalog1',source_kind='raw',workload='NCCL',hardware='B300',source=dict(path='/logs/a.out',sha256='sha1'))
    run['metadata']['correctness_errors']=0
    reference=from_run_reference(run)
    assert reference['ref_id']=='catalog1' and reference['source_url']=='/api/runs/catalog1'
    assert reference['source_sha256']=='sha1'
    assert from_run_reference(dict(run,status='incomplete')) is None
    assert from_run_reference(dict(run,source_kind='upload')) is None
    bad=copy.deepcopy(run);bad['metrics'][0]['errors']=1
    assert from_run_reference(bad) is None
    no_protocol=copy.deepcopy(run);no_protocol['metadata'].pop('test_protocol')
    assert 'test_protocol' not in from_run_reference(no_protocol)['metadata']


def test_same_source_job_hash_and_run_are_excluded_from_self_baselines():
    own=ref(cluster='customer',job_id='123',source_sha256='own-hash',ref_id='own')
    other=ref(cluster='customer',job_id='456',ref_id='other',source_sha256='other-hash')
    run=base(cluster='customer',job_id='123',source=dict(sha256='own-hash'),id='own')
    result=compare_nccl(run,[own,other])
    assert [r['ref_id'] for r in result['references']]==['other']
    assert compare_nccl(run,[own])['tier']=='none'
    samehash=ref(job_id='789',ref_id='samehash',source_sha256='own-hash')
    assert compare_nccl(run,[samehash])['tier']=='none'


def test_observed_zero_bandwidth_is_worst_and_invalid_latency_cannot_pass():
    run=base();run['metrics'][0]['busbw_gbps']=0
    result=compare_nccl(run,[ref()])
    assert result['classification']=='worst' and result['verdict']=='fail'
    assert result['metrics'][0]['expected_pct']==0
    run=base();run['metrics'][0]['time_us']=0
    result=compare_nccl(run,[ref()])
    assert result['verdict']=='insufficient_evidence' and 'valid.time_us' in result['missing_context']


@pytest.mark.parametrize('field,value',[('message_bytes',2048),('datatype','half'),('operation','max'),('placement','in_place')])
def test_distinct_known_point_context_never_provides_even_conditional_expectation(field,value):
    source=ref();source['metrics'][0][field]=value
    assert compare_nccl(base(),[source])['tier']=='none'
    assert compare_nccl(base(),[source])['metrics']==[]


def test_split_mask_difference_is_incomparable_and_cache_is_bounded():
    source=ref();source['metadata']['NCCL_TESTS_SPLIT_MASK']='0x7'
    assert compare_nccl(base(),[source])['tier']=='none'
    index=NCCLReferenceIndex([ref()],max_cache=2)
    for i in range(5):
        run=base(id=str(i),job_id=str(i),cluster='customer')
        compare_nccl(run,index)
    assert len(index.cache)==2


@pytest.mark.parametrize('field,value',[('nodes',1),('placement_scope','single_nvlink_domain'),('nic_model','CX7'),('link_speed_gbps',400),('rail_topology','4_rails')])
def test_known_physical_configuration_difference_never_matches(field,value):
    assert compare_nccl(base(),[ref(**{field:value})])['tier']=='none'


def test_missing_physical_context_is_not_exact_and_source_owner_cache_reuses_job():
    assert compare_nccl(base(nic_model=None),[ref()])['tier']=='contextual'
    own1=ref(ref_id='own1',job_id='123',cluster='customer',source_sha256='sha1')
    own2=ref(ref_id='own2',job_id='123',cluster='customer',source_sha256='sha2')
    other=ref(ref_id='other',job_id='456',cluster='customer')
    index=NCCLReferenceIndex([own1,own2,other])
    for identifier,sha in [('own1','sha1'),('own2','sha2')]:
        run=base(id=identifier,cluster='customer',job_id='123',source=dict(sha256=sha))
        assert compare_nccl(run,index)['tier']=='exact'
    assert len(index.cache)==1


def test_mixed_splits_and_metadata_correctness_override_nominal_speed():
    run=base();run['metadata']['mixed_split_masks']=True
    result=compare_nccl(run,[ref()])
    assert result['verdict']=='insufficient_evidence' and result['tier']!='exact'
    assert 'single_split_mask' in result['missing_context']
    run=base();run['metadata']['correctness_errors']=1
    assert compare_nccl(run,[ref()])['classification']=='failed'


def test_latency_alias_inside_metric_array_preserves_microsecond_units():
    run=base();run['metrics'][0]['latency_us']=run['metrics'][0].pop('time_us')
    source=ref();source['metrics'][0]['latency_us']=source['metrics'][0].pop('time_us')
    result=compare_nccl(run,[source])
    latency=next(point for point in result['metrics'] if point['metric']=='time_us')
    assert latency['observed']==10 and latency['expected']==9.5 and latency['unit']=='us'
    assert latency['expected_pct']==95


def gpucomms_schema_row():
    # Synthetic values in the verified GPUComms golden_data schema.
    import json
    return dict(id=123,tool='nccl-tests',tool_binary='all_reduce_perf',library_version='2.30.4',
        num_processes=8,num_nodes=1,slurm_job_id=456,tool_version='2.17.6',
        system_info=json.dumps(dict(gpu_arch_type='NVIDIA B200')),
        network_info=json.dumps(dict(network_name='IB',switch_type='Quantum2',nics='ConnectX-7')),
        test_config=json.dumps(dict(nthreads=1,ngpus=1,minbytes=1024,maxbytes=1024,iters=100,warmup_iters=20)),
        test_env_vars=json.dumps(dict(NCCL_PROTO='Simple',NCCL_TESTS_SPLIT_MASK='0x0')),
        test_result=json.dumps([dict(size=1024,type='float',redop='sum',**{
            'out_of_place.bus_bw':100,'out_of_place.alg_bw':70,'out_of_place.time':9.5,'out_of_place.nwrong':0})]))


def test_gpucomms_adapter_preserves_real_point_units_and_unknown_physical_scope():
    from backend.beacon_nccl import normalize_record
    source=normalize_record(gpucomms_schema_row(),source_url='https://gpucomms.nvidia.com/')
    assert source['gpu']=='B200' and source['gpus']==8 and source['nodes']==1
    assert source['metrics'][0]['busbw_gbps']==100
    assert source['metrics'][0]['latency_us']==9.5
    run=base(gpu='B200',gpus=8,nodes=1,fabric='InfiniBand',network='IB',nic_model='ConnectX-7')
    result=compare_nccl(run,[source])
    assert result['tier']=='contextual' and result['classification']=='conditional'
    bandwidth=next(point for point in result['metrics'] if point['metric']=='busbw_gbps')
    assert bandwidth['expected']==100 and bandwidth['expected_pct']==95 and bandwidth['unit']=='GB/s'
    assert 'placement_scope' in result['missing_context'] and 'link_speed_gbps' in result['missing_context']
    assert all(not point['scored'] for point in result['metrics'])


def test_gpucomms_adapter_never_compares_native_node_to_spx_or_another_size():
    from backend.beacon_nccl import normalize_record
    source=normalize_record(gpucomms_schema_row(),source_url='https://gpucomms.nvidia.com/')
    spectrum=base(gpu='B200',gpus=8,nodes=1,nic_model='ConnectX-7')
    assert compare_nccl(spectrum,[source])['metrics']==[]
    different_size=base(gpu='B200',gpus=8,nodes=1,fabric='InfiniBand',network='IB',nic_model='ConnectX-7')
    different_size['metrics'][0]['message_bytes']=2048
    assert compare_nccl(different_size,[source])['metrics']==[]


def test_rounded_zero_bandwidth_does_not_claim_a_hundred_percent_deficit():
    run=base();run['metrics'][0]['busbw_gbps']=0
    source=ref();source['metrics'][0]['busbw_gbps']=0.01
    result=compare_nccl(run,[source])
    assert not any(point['metric']=='busbw_gbps' for point in result['metrics'])
    assert 'bandwidth_reporting_resolution' in result['missing_context']
    assert result['provenance']['rounding_limited_measurements']==1
    assert result['verdict']=='insufficient_evidence' and result['candidate_band']=='good'
    assert run['metrics'][0]['busbw_gbps']==0 and source['metrics'][0]['busbw_gbps']==0.01


def test_recorded_extra_protocol_variants_are_separate_cohorts():
    ordinary=ref();ordinary['metadata']['test_protocol']['cudagraph']=0
    graphed=ref(ref_id='graphed',job_id='job2');graphed['metadata']['test_protocol']['cudagraph']=10
    graphed['metrics'][0]['busbw_gbps']=200
    result=compare_nccl(base(),[ordinary,graphed])
    bandwidth=next(point for point in result['metrics'] if point['metric']=='busbw_gbps')
    assert bandwidth['expected'] in (100,200) and bandwidth['expected']!=150
    assert result['tier']=='contextual' and 'test.additional_protocol' in result['missing_context']
    run=base();run['metadata']['test_protocol']['cuda_graph_launches']=10
    exact=compare_nccl(run,[ordinary,graphed])
    bandwidth=next(point for point in exact['metrics'] if point['metric']=='busbw_gbps')
    assert bandwidth['expected']==200 and bandwidth['reference_jobs']==1
    assert bandwidth['tier']=='exact'


def test_valid_repeats_from_archive_with_failure_label_remain_conditional():
    source=ref(source_archive_status='failed_label_in_source_notes',selected_group_status='passed',repeat_count=3)
    result=compare_nccl(base(),[source])
    assert result['candidate_band']=='good'
    assert result['classification']=='conditional' and result['verdict']=='insufficient_evidence'
    assert 'reference.reference_archive_caveat' in result['missing_context'] or 'reference_archive_caveat' in result['missing_context']
    assert result['references'][0]['source_archive_status']=='failed_label_in_source_notes'


@pytest.mark.parametrize('message_bytes',[None,0,-1])
def test_missing_or_nonpositive_message_bytes_never_matches_any_size(message_bytes):
    run=base();run['metrics'][0].update(message_bytes=message_bytes,time_us=0.06,busbw_gbps=0)
    source=ref();source['metrics'][0].update(message_bytes=2*1024**3,time_us=2937,busbw_gbps=0.355)
    result=compare_nccl(run,[source])
    assert result['metrics']==[] and result['candidate_band'] is None
    assert result['classification']=='unknown' and 'message_bytes' in result['missing_context']
    assert run['metrics'][0]['message_bytes']==message_bytes


def test_unknown_reference_message_size_never_establishes_pointwise_expectation():
    source=ref();source['metrics'][0].pop('message_bytes')
    assert compare_nccl(base(),[source])['metrics']==[]


def test_best_recorded_uses_best_within_each_job_not_median_and_retains_winner():
    one=ref(ref_id='slow',job_id='j1');one['metrics'][0]['busbw_gbps']=100
    fast=ref(ref_id='fast',job_id='j1');fast['metrics'][0]['busbw_gbps']=300
    middle=ref(ref_id='middle',job_id='j2');middle['metrics'][0]['busbw_gbps']=250
    index=NCCLReferenceIndex([one,fast,middle])
    result=compare_nccl(base(),index,reference_statistic='best')
    bw=next(m for m in result['metrics'] if m['metric']=='busbw_gbps')
    assert bw['expected']==300 and bw['reference_jobs']==2
    point=index.best_point(base(),1024,point=base()['metrics'][0])
    assert point['expected']==300 and point['reference_statistic']=='best'
    assert [r['ref_id'] for r in point['references']]==['fast']
    assert [r['ref_id'] for r in bw['reference_sources']]==['fast']
    # Best and median results must never share a cached aggregation.
    median=compare_nccl(base(),index)
    assert next(m for m in median['metrics'] if m['metric']=='busbw_gbps')['expected']==225


@pytest.mark.parametrize('version',[None,'2.31.2'])
def test_best_recorded_requires_matching_known_nccl_version(version):
    result=compare_nccl(base(nccl_version=version),[ref()],reference_statistic='best')
    assert result['metrics']==[] and result['candidate_band'] is None
    point=NCCLReferenceIndex([ref()]).best_point(base(nccl_version=version),1024)
    assert point['expected'] is None


def test_best_point_excludes_same_job_hash_and_unknown_fabric():
    own=ref(cluster='c',job_id='123',source_sha256='same',ref_id='own')
    index=NCCLReferenceIndex([own])
    assert index.best_point(base(cluster='c',job_id='123'),1024)['expected'] is None
    assert index.best_point(base(source=dict(sha256='same')),1024)['expected'] is None
    assert index.best_point(base(fabric=None,network=None),1024)['expected'] is None



def test_unknown_fabric_best_fallback_is_explicit_and_scoped_to_same_cluster():
    run=base(fabric=None,network=None,cluster='local',job_id='observed')
    local=ref(fabric=None,network=None,cluster='local',job_id='peer')
    remote=ref(fabric=None,network=None,cluster='remote',job_id='peer2',ref_id='remote')
    remote['metrics'][0]['busbw_gbps']=1000
    index=NCCLReferenceIndex([local,remote])
    assert index.best_point(run,1024)['expected'] is None
    result=index.best_point(run,1024,allow_same_cluster_unknown_fabric=True)
    assert result['expected']==100 and result['tier']=='contextual'
    assert result['references'][0]['cluster']=='local'
    assert result['references'][0]['reference_value']==100
    assert result['context_note']=='Best recorded on this cluster; fabric metadata incomplete'
    assert index.best_point(dict(run,cluster='other'),1024,allow_same_cluster_unknown_fabric=True)['expected'] is None
    assert index.best_point(dict(run,cluster=None),1024,allow_same_cluster_unknown_fabric=True)['expected'] is None
    assert index.best_point(run,0,allow_same_cluster_unknown_fabric=True)['expected'] is None
    assert index.best_point(dict(run,nccl_version='different'),1024,allow_same_cluster_unknown_fabric=True)['expected'] is None
    assert index.best_point(run,1024,allow_same_cluster_unknown_fabric=True)['reference_jobs']==1


def test_unknown_fabric_fallback_never_reaches_known_fabric_or_same_job():
    run=base(fabric=None,network=None,cluster='local',job_id='same')
    own=ref(fabric=None,network=None,cluster='local',job_id='same')
    known=ref(cluster='local',job_id='peer',ref_id='known')
    assert NCCLReferenceIndex([own,known]).best_point(run,1024,allow_same_cluster_unknown_fabric=True)['expected'] is None



def test_best_point_selects_highest_recorded_compatible_cohort_not_closest_tuning():
    closest=ref(ref_id='closest');closest['metrics'][0]['busbw_gbps']=100
    tuned=ref(ref_id='fast',job_id='fastjob',protocol='LL128')
    tuned['metadata']['test_protocol']['iterations']=200
    tuned['metrics'][0]['busbw_gbps']=300
    incompatible=ref(ref_id='wrongfabric',job_id='wrong',fabric='NVLink',network='NVLink')
    incompatible['metrics'][0]['busbw_gbps']=1000
    index=NCCLReferenceIndex([closest,tuned,incompatible])
    result=index.best_point(base(),1024,point=base()['metrics'][0])
    assert result['expected']==300 and result['tier']=='contextual'
    assert result['references'][0]['ref_id']=='fast'
    assert result['reference_jobs']==1
    assert 'protocol' in result['mismatches'] and 'test.iterations' in result['mismatches']
    comparison=compare_nccl(base(),index,reference_statistic='best')
    assert next(m for m in comparison['metrics'] if m['metric']=='busbw_gbps')['expected']==300
    assert compare_nccl(base(),index)['metrics'][0]['expected']==100


def test_best_latency_lookup_can_select_its_own_cohort_without_bandwidth_pooling():
    fastbw=ref(ref_id='bw');fastbw['metrics'][0].update(busbw_gbps=300,time_us=20)
    fastlat=ref(ref_id='lat',job_id='latjob',protocol='LL128');fastlat['metrics'][0].update(busbw_gbps=100,time_us=5)
    result=NCCLReferenceIndex([fastbw,fastlat]).best_point(base(),1024,metric='time_us',point=base()['metrics'][0])
    assert result['expected']==5 and result['references'][0]['ref_id']=='lat'
