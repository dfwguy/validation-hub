import copy,json,subprocess
import pytest
from fastapi.testclient import TestClient
from backend import assessment,benchmark_context,store
from backend.app import app
from backend.investigation import build
from backend.parser import parse_log
from backend.recipes import ScriptRequest,generate


def nccl_case():
    metadata=dict(test_protocol=dict(min_bytes=2147483648,max_bytes=2147483648,iterations=100,warmup=50,registration=0),
        nccl_environment={'NCCL_NVLS_ENABLE':'0'},placement_scope='cross-su/full-domains',rail_topology='rail-optimized',
        nic_model='CX8',link_speed_gbps=800,nccl_tests_version='source-commit',congestion_control='production')
    ref=dict(id='ref',job_id='123',cluster='reference',metadata=copy.deepcopy(metadata),source={'sha256':'refhash'})
    run=dict(status='passed',metadata=metadata,metrics=[dict(message_bytes=2147483648,busbw_gbps=96,errors=0)])
    c=dict(run=run,baseline_count=1,baseline_runs=[{'id':'ref'}],comparisons=[dict(message_bytes=2147483648,baseline_gbps=100,observed_gbps=96,placement='out_of_place',datatype='float',operation='sum',baseline_samples=1)])
    return c,{'ref':ref}


def test_exact_nccl_scores_and_protocol_changes_are_conditional():
    c,refs=nccl_case();result=assessment.historical_nccl(c,refs)
    assert result['classification']=='good' and result['verdict']=='pass'
    c['comparisons'][0]['observed_gbps']=80
    assert assessment.historical_nccl(c,refs)['classification']=='worst'
    c['run']['metadata']['test_protocol']['max_bytes']=16*1024**3
    result=assessment.historical_nccl(c,refs)
    assert result['classification']=='conditional' and result['candidate_band']=='worst'
    assert result['metrics'][0]['expected_pct']==80 and not result['metrics'][0]['scored']
    assert 'max_bytes' in result['mismatches']


def test_failed_rank_overrides_fast_nccl_points():
    c,refs=nccl_case();c['run']['metadata']['parent_job_failed']=True
    result=assessment.historical_nccl(c,refs)
    assert result['classification']=='failed' and result['verdict']=='fail'
    assert not result['metrics'][0]['scored']


def test_every_reference_artifact_protocol_is_checked():
    c,refs=nccl_case();refs['other']=copy.deepcopy(refs['ref'])
    refs['other']['metadata']['test_protocol']['warmup']=10
    c['baseline_runs'][0]['artifact_ids']=['ref','other']
    result=assessment.historical_nccl(c,refs)
    assert result['classification']=='conditional' and 'warmup' in result['mismatches']


def test_known_rounded_zero_reference_points_are_not_percentage_failures():
    c,refs=nccl_case()
    c['comparisons'].append(dict(message_bytes=8,baseline_gbps=0,observed_gbps=0))
    c['run']['metrics'].append(dict(message_bytes=8,busbw_gbps=0,errors=0))
    result=assessment.historical_nccl(c,refs)
    assert result['classification']=='good'
    assert result['provenance']['zero_reference_points_excluded']==1


def test_parser_keeps_exact_fixed_size_protocol_and_training_config():
    run=parse_log(['mpirun -np 128 alltoall_perf -b 2G -e 2G -n 100 -w 50 -R 2',
                   'sequence_length ................ 4096','tensor_model_parallel_size = 4','NCCL_MAX_NCHANNELS=16'])
    assert run['metadata']['test_protocol']==dict(min_bytes=2147483648,max_bytes=2147483648,iterations=100,warmup=50,registration=2)
    assert run['metadata']['sequence_length']==4096
    assert run['metadata']['tensor_model_parallel_size']==4
    assert run['metadata']['nccl_environment']['NCCL_MAX_NCHANNELS']=='16'


def test_benchmark_context_rejects_false_status_and_conflicting_measurement():
    with pytest.raises(ValueError):benchmark_context.parse('{"status":"passed"}')
    with pytest.raises(ValueError):benchmark_context.parse('{"test_protocol":{"registration":3}}')
    run=dict(workload='Llama',hardware='B300',metadata={'archive_member':'pretrain_llama3_70b_bf16_gpus512_tp4/model.out'})
    with pytest.raises(ValueError,match='contradicts'):benchmark_context.attach(run,{'precision':'fp8'})
    benchmark_context.attach(run,{'sequence_length':4096})
    assert run['metadata']['user_declared_benchmark_fields']==['sequence_length']


def test_script_fixed_2g_protocol_and_scoped_experiment(tmp_path):
    req=ScriptRequest(recipe_id='gb300-spectrum-x',nodes=32,gpus_per_node=4,image='image.sqsh',hca='=mlx5_0:1',interface='=site0',collective='alltoall',min_bytes=2147483648,max_bytes=2147483648,iterations=100,warmup=50,registration=2,tuning_preset='alltoall-qp4-split')
    generated=generate(req);path=tmp_path/'test.sh';path.write_text(generated['script'])
    assert subprocess.run(['bash','-n',str(path)]).returncode==0
    assert '-b 2147483648 -e 2147483648' in generated['script'] and '-R 2' in generated['script']
    assert 'NCCL_IB_QPS_PER_CONNECTION=4' in generated['script']
    assert 'NCCL_MAX_NCHANNELS' in generated['experiment']['cleared_inherited_tuning']
    assert 'NCCL_IB_TC' not in generated['experiment']['cleared_inherited_tuning']
    assert generated['script'].index('unset "$key"')<generated['script'].index('export NCCL_IB_QPS_PER_CONNECTION=4')
    with pytest.raises(ValueError,match='only'):generate(req.model_copy(update={'collective':'all_reduce'}))


def test_diagnostic_examples_are_guidance_and_pending_evidence_not_a_result():
    plan=build(dict(recipe={'fabric':'spectrum-x'},comparisons=[],crashes=[]))
    assert plan['evidence_state']=='awaiting_measured_evidence'
    assert 'Diagnostic examples do not establish measured customer results' in plan['customer_evidence_note']
    tuning=next(f for f in plan['findings'] if f['id']=='alltoall-ab')
    assert 'Conditional test candidates' in tuning['applicability']
    assert any('production policy' in caution for caution in tuning['cautions'])


def test_reference_api_isolated_and_missing_data_cannot_pass(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'catalog.sqlite'))
    monkeypatch.setenv('BEACON_DB',str(tmp_path/'beacon.sqlite'))
    client=TestClient(app)
    state=client.get('/api/performance/status').json()
    assert state['beacon']['records']==0
    assert state['policy']['good_pct']==5
    assert client.get('/api/performance/references').json()['items']==[]
    assert client.post('/api/performance/import',files={'file':('bad.json',b'not json')}).status_code==422
    assert client.get('/api/performance/references').json()['total']==0
    assert client.post('/api/performance/import',files={'file':('bad.json.gz',b'not gzip')}).status_code==422


def test_nccl_source_import_filters_and_reassessment_queue(tmp_path,monkeypatch):
    from backend import performance_api
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'catalog.sqlite'))
    queued=[]
    def queue(background,kind):
        queued.append(kind)
        return {'job_id':'queued-test','status':'queued'}
    monkeypatch.setattr(performance_api,'queue',queue)
    client=TestClient(app)
    assert client.get('/api/performance/nccl/status').json()['records']==0
    row=dict(id='external1',gpu='GB300',gpus=128,fabric='Spectrum-X',collective='alltoall',
        metrics=[dict(message_bytes=2147483648,busbw_gbps=178.84,errors=0)])
    response=client.post('/api/performance/nccl/import',files={'file':('actual.json',json.dumps([row]))})
    assert response.status_code==200,response.text
    assert queued==['nccl-reassess']
    assert client.get('/api/performance/nccl/status').json()['records']==1
    page=client.get('/api/performance/nccl/references?gpu=GB300&fabric=spectrum-x&gpus=128&collective=alltoall').json()
    assert page['total']==1
    assert page['items'][0]['metric_points']==1
    from backend import beacon_nccl
    assert beacon_nccl.references()[0]['metrics'][0]['busbw_gbps']==178.84
    assert client.get('/api/performance/nccl/references?gpu=B300').json()['total']==0
    assert client.post('/api/performance/nccl/import?source_url=https://example.com/',files={'file':('refs.json','[]')}).status_code==422
    assert client.post('/api/performance/nccl/import',files={'file':('invalid.json','not-json')}).status_code==422
    assert client.get('/api/performance/nccl/status').json()['records']==1


def test_external_nccl_index_loads_only_the_uploaded_execution_cohort(monkeypatch):
    from backend import beacon_nccl
    calls=[]
    monkeypatch.setattr(beacon_nccl,'references',lambda **kwargs:calls.append(kwargs) or [])
    monkeypatch.setattr(beacon_nccl,'status',lambda:dict(records=15379,snapshot={'snapshot_id':'cohort-scope-test'}))
    run=dict(gpu='GB300',gpus=128,collective='all_reduce')
    assessment.external_nccl_index(run)
    assessment.external_nccl_index(run)
    assert calls==[{'filters':{'gpu':'GB300','gpus':128,'collective':'all_reduce'}}]
    assessment.external_nccl_index(dict(gpu='B300',gpus=512,collective='alltoall'))
    assert calls[-1]['filters']==dict(gpu='B300',gpus=512,collective='alltoall')


def test_import_upload_reassessment_and_customer_share_one_source_backed_verdict(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'catalog.sqlite'))
    monkeypatch.setenv('BEACON_DB',str(tmp_path/'beacon.sqlite'))
    monkeypatch.setenv('VALIDATION_UPLOADS',str(tmp_path/'uploads'))
    client=TestClient(app)
    reference=dict(id='source1',job_id='reference-job',model='deepseek-v3',model_size='671b',gpu='B300',scale=8,precision='bf16',release='26.09',
        global_batch_size=8,sequence_length=4096,tensor_parallel=1,pipeline_parallel=1,context_parallel=1,virtual_pipeline=1,expert_parallel=1,expert_tensor_parallel=1,micro_batch_size=1,tflops_gpu=1000,seconds_per_step=1)
    imported=client.post('/api/performance/import',files={'file':('refs.json',json.dumps([reference]))})
    assert imported.status_code==200,imported.text
    customer=client.post('/api/customers',json={'name':'Isolated Test','clusters':['test-cluster']}).json()
    text='nodes=1 gpus=8\niteration 1/2 | elapsed time per iteration (ms): 1250.0\niteration 2/2 | elapsed time per iteration (ms): 1250.0\nGPU utilization: 800 MODEL_TFLOP/s/GPU\n'
    filename='pretrain_deepseek_v3_671b_bf16_gpus8_tp1_pp1_cp1_vp1_ep1_etp1_mbs1_gbs8_job7.log'
    response=client.post('/api/analyze',data={'recipe_id':'b300-spectrum-x','cluster':'test-cluster','customer_id':customer['id'],'hardware':'B300','context_json':json.dumps({'release':'26.09','sequence_length':4096})},files={'file':(filename,text)})
    assert response.status_code==200,response.text
    report=response.json();comparison=report['comparisons'][0];run_id=comparison['run_id']
    assert comparison['performance']['tier']=='exact'
    assert comparison['performance']['classification']=='worst'
    assert [m['expected_pct'] for m in comparison['performance']['metrics']]==[80,80]
    assert client.get('/api/runs/'+run_id).json()['performance']['classification']=='worst'
    listed=client.get('/api/runs?q=7').json()['items'][0]
    assert listed['performance']['classification']=='worst'
    assert listed['performance']['metrics'][0]['expected_pct']==80
    assert 'references' not in listed['performance']
    original=store.detail(run_id)
    reference.update(tflops_gpu=800,seconds_per_step=1.25)
    refreshed=client.post('/api/performance/import',files={'file':('refs.json',json.dumps([reference]))})
    assert refreshed.status_code==200,refreshed.text
    updated=client.get('/api/analyses/'+report['id']).json()
    assert updated['comparisons'][0]['performance']['classification']=='good'
    assert updated['baseline_attainment']['verdict']=='pass'
    assert store.detail(run_id)==original
    board=client.get('/api/customers').json()['items'][0]
    assert board['best_training_performance'][0]['performance']['classification']=='good'
    assert client.get('/api/runs/'+run_id).json()['performance']['classification']=='good'
    assert client.get('/api/runs?q=7').json()['items'][0]['performance']['classification']=='good'
    assert client.get('/api/performance/status').json()['jobs'][0]['status']=='complete'


def test_generic_gpu_recipes_never_guess_site_qos_policy():
    from backend import recipes
    for profile in recipes.profiles():
        assert not any(key in profile['expected_env'] for key in ('NCCL_IB_SL','NCCL_IB_TC','NCCL_IB_GID_INDEX','NCCL_IB_HCA','NCCL_SOCKET_IFNAME'))
