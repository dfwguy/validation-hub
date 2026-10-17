import pytest
from backend.parser import parse_log
from backend import store
from backend.compare import compare
LOG='''# Collective test starting: all_reduce_perf_mpi
NCCL INFO NCCL version 2.31.2
nodes=36 gpus=144
# Rank 0 Group 0 Pid 20 on host-1 device 0 [0008:06:00] NVIDIA GB300
8192 2048 float sum -1 15.0 120.0 160.0 0 14.0 125.0 166.0 0
# Out of bounds values : 0 OK
# Avg bus bandwidth : 163.0
'''

@pytest.fixture
def db(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'test.sqlite'))

def test_geometry_retains_allocation_over_partial_ranks():
    r=parse_log(LOG.splitlines())
    assert (r['nodes'],r['gpus'])==(36,144)
    assert r['status']=='passed'
    assert r['metadata']['observed_hosts']==1
    assert len(r['metrics'])==2

def test_error_after_completed_llama_is_failure():
    r=parse_log(['iteration 50/50 | elapsed time per iteration (ms): 123.0','FATAL training failed'],'llama.log')
    assert r['status']=='failed'

def test_mixed_collective_excluded():
    r=parse_log((LOG+'# Collective test starting: all_gather_perf_mpi\n').splitlines())
    assert r['collective']=='mixed'

def test_exact_baselines_and_policy(db):
    r=parse_log(LOG.splitlines(),hints={'transport':'InfiniBand + NVLink'})
    store.put_run(r,'reference.log','reference',100,'cluster')
    observed=parse_log(LOG.replace('160.0','130.0').splitlines(),hints={'transport':'InfiniBand + NVLink'})
    outcome=compare(observed,'cluster',10)
    assert outcome['baseline_count']==1
    assert outcome['verdict']=='fail'
    assert outcome['comparisons'][0]['deviation_pct']==pytest.approx(-18.75)
    observed['nodes']=288
    outcome=compare(observed,'cluster',10)
    assert outcome['verdict']=='insufficient_evidence'
    assert outcome['baseline_count']==0

def test_unknown_transport_and_failed_baselines(db):
    r=parse_log(LOG.splitlines())
    store.put_run(r,'reference.log','reference',100,'cluster')
    assert compare(r,'cluster')['verdict']=='insufficient_evidence'
    r['transport']='InfiniBand + NVLink'; r['status']='failed'
    store.put_run(r,'failed.log','failed',100,'cluster')
    observed=parse_log(LOG.splitlines(),hints={'transport':'InfiniBand + NVLink'})
    assert compare(observed,'cluster')['baseline_count']==0

def test_api_upload_validates_contradiction_and_size(db,tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import app
    monkeypatch.setenv('VALIDATION_UPLOADS',str(tmp_path/'uploads'))
    client=TestClient(app)
    response=client.post('/api/compare',files={'file':('result.txt',LOG)},data={'nodes':'1200'})
    assert response.status_code==422
    response=client.post('/api/compare',files={'file':('result.txt',LOG)},data={'nodes':'36','gpus':'144','transport':'InfiniBand + NVLink'})
    assert response.status_code==200
    assert response.json()['verdict']=='insufficient_evidence'
    assert client.get('/api/runs?nodes=36').json()['total']==1
    assert client.get('/api/compare').status_code==405


def test_upload_identity_cannot_replace_reference(db):
    r=parse_log(LOG.splitlines(),hints={'transport':'InfiniBand + NVLink'})
    ref=store.put_run(r,'reference.log','samehash',100,'cluster','raw')
    uploaded=store.put_run(r,'upload.log','samehash',100,'cluster','upload')
    assert ref['id']!=uploaded['id']
    assert compare(uploaded,'cluster')['baseline_count']==1


def test_membership_filters_join_measured_runs(db):
    job=parse_log([])
    job.update(job_id='12345',metadata={'racks_json':'["a01","a02"]','sus_json':'["SU1"]','nodes_json':'["host-1","host-2"]'})
    store.put_run(job,'jobs.parquet','jobs',100,'cluster','parquet_slurm_jobs')
    r=parse_log(LOG.splitlines());r['job_id']='12345'
    store.put_run(r,'reference.log','reference',100,'cluster','raw')
    assert store.search({'measured_only':True,'rack_ids':'a01,a02'})['total']==1
    assert store.search({'measured_only':True,'su_ids':'SU1','nodelist':'host-1'})['total']==1
    assert store.search({'measured_only':True,'rack_ids':'a03'})['total']==0


def test_scrub_and_observed_transport():
    from backend.security import scrub
    assert scrub({'password':'abc','command':'sshpass -p abc ssh x','Authorization':'Bearer secret'})=={'password':'[redacted]','command':'sshpass -p [redacted] ssh x','Authorization':'[redacted]'}
    r=parse_log((LOG+'NCCL INFO NET/IB network selected\nNCCL INFO NVLS multicast support is available on dev 0\n').splitlines())
    assert r['transport']=='InfiniBand + NVLink'


def test_slurm_roster_expansion_and_limits():
    from backend.hostlist import expand_hostlist
    hosts=expand_hostlist('ink50-dh01-h16-[11-18,28-37],ink50-dh01-h17-[11-18,28-37]')
    assert len(hosts)==36
    assert hosts[0]=='ink50-dh01-h16-11'
    assert hosts[-1]=='ink50-dh01-h17-37'
    with pytest.raises(ValueError):expand_hostlist('node[1-9999999]')


def test_disjoint_artifacts_of_one_job_contribute_all_points_once(db):
    first=parse_log(LOG.splitlines(),hints={'transport':'InfiniBand + NVLink'})
    first['job_id']='12345'
    second=parse_log(LOG.replace('float sum','int sum').splitlines(),hints={'transport':'InfiniBand + NVLink'})
    second['job_id']='12345'
    store.put_run(first,'float.log','float',100,'cluster')
    store.put_run(second,'int.log','int',100,'cluster')
    observed=dict(first);observed['metrics']=first['metrics']+second['metrics']
    result=compare(observed,'cluster')
    assert result['baseline_count']==1
    assert result['baseline_runs'][0]['artifact_count']==2
    assert result['verdict']=='pass'
    assert len(result['comparisons'])==4
    assert {p['baseline_samples'] for p in result['comparisons']}=={1}


def test_known_zero_reference_not_scored_missing_reference_stays_unknown(db):
    zero='8 2 float sum -1 15.0 0.0 0.0 0 14.0 0.0 0.0 0\n'
    text=LOG.replace('8192 2048 float',zero+'8192 2048 float')
    reference=parse_log(text.splitlines(),hints={'transport':'InfiniBand + NVLink'})
    store.put_run(reference,'reference.log','zero',100,'cluster')
    outcome=compare(reference,'cluster')
    assert outcome['verdict']=='pass'
    assert outcome['comparisons'][0]['verdict']=='not_scored'
    assert outcome['comparisons'][0]['baseline_samples']==1
    assert outcome['comparisons'][0]['deviation_pct'] is None
    zero_only=dict(reference);zero_only['metrics']=reference['metrics'][:2]
    assert compare(zero_only,'cluster')['verdict']=='insufficient_evidence'
    missing=dict(reference);missing['metrics']=[dict(reference['metrics'][0],message_bytes=4)]
    assert compare(missing,'cluster')['comparisons'][0]['verdict']=='no_baseline'
    assert compare(missing,'cluster')['verdict']=='insufficient_evidence'
    failed=dict(reference);failed['metrics']=[dict(reference['metrics'][2],busbw_gbps=0)]
    assert compare(failed,'cluster')['verdict']=='fail'
