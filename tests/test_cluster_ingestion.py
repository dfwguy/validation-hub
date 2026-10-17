"""New-cluster ingestion must preserve evidence and distinguish attainment from correctness."""
import io,json,zipfile,hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend import store
from backend.parser import parse_log
from backend.compare import compare
from test_workflows import LOG,setup_customer

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'catalog.sqlite'))
    monkeypatch.setenv('VALIDATION_UPLOADS',str(tmp_path/'lake'))
    # Initialize before exercising concurrent UI requests.
    with store.connection():pass
    return TestClient(app)

def baseline():
    run=parse_log((LOG+'NCCL INFO NVLS multicast support is available on dev 0\n').splitlines())
    return store.put_run(run,'known-good.log','trusted-reference',100,'Reference cluster')

def upload(c,text=LOG,**fields):
    response=c.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster',**fields},files={'file':('all_reduce_perf.log',text)})
    return response

def test_new_cluster_cross_cluster_attainment_and_receipts(client):
    baseline()
    text=LOG+'NCCL INFO NVLS multicast support is available on dev 0\n# ExampleCluster observation\n'
    response=upload(client,text)
    assert response.status_code==200,response.text
    report=response.json()
    # Matching bandwidth alone no longer certifies an unspecified test protocol.
    assert report['baseline_attainment']['verdict']=='insufficient_evidence'
    assert report['comparisons'][0]['performance']['candidate_band'] is None
    assert all(row['observed_gbps'] is None for row in report['comparisons'][0]['performance']['nccl_summary']['rows'])
    assert report['comparisons'][0]['baseline_runs'][0]['cluster']=='Reference cluster'
    assert report['comparisons'][0]['measurement']['peak_bus_bw_gbps']==166
    assert 'ExampleCluster' in client.get('/api/facets').json()['clusters']
    assert client.get('/api/runs?cluster=ExampleCluster&measured_only=true').json()['total']==1
    inventory=client.get('/api/lake/uploads').json()
    assert inventory['summary']['batches']==inventory['summary']['measured_runs']==1
    batch=inventory['items'][0]
    assert Path(batch['path']).read_bytes()==text.encode()
    assert batch['sha256']==hashlib.sha256(text.encode()).hexdigest()
    assert client.get('/api/storage').json()['ingested_uploads']['batches']==1
    assert client.get('/api/summary').json()['measurements']==2
    assert client.get('/api/analyses/'+report['id']).json()['baseline_attainment']['verdict']=='insufficient_evidence'
    assert upload(client,text).status_code==200
    assert client.get('/api/lake/uploads').json()['summary']['batches']==1
    assert client.get('/api/runs?cluster=ExampleCluster').json()['total']==1

def test_slow_run_is_ingested_and_gets_specific_recommendation(client):
    baseline()
    slow=LOG.replace('160.0','80.0').replace('166.0','83.0')+'NCCL INFO NVLS multicast support is available on dev 0\n'
    r=upload(client,slow);assert r.status_code==200,r.text
    a=r.json();assert a['baseline_attainment']['verdict']=='insufficient_evidence'
    # The retained 8 KiB diagnostics do not become a 16/32 GiB headline grade.
    assert a['comparisons'][0]['performance']['candidate_band'] is None
    assert a['comparisons'][0]['comparisons'][0]['deviation_pct']==-50
    assert a['baseline_attainment']['points_below']==2
    assert any(t['id'].startswith('regression-') and '-50.0%' in t['reason'] for t in a['follow_up_tests'])
    # Correctness-clean is distinct from meeting performance expectations.
    assert a['comparisons'][0]['run']['status']=='passed'
    assert client.get('/api/runs?cluster=ExampleCluster').json()['total']==1
    assert compare(a['comparisons'][0]['run'])['baseline_count']==1
    with store.connection() as db:assert db.execute('SELECT COUNT(*) FROM reference_reviews').fetchone()[0]==0

def test_unknown_baseline_is_not_passing_and_failed_results_are_retained(client):
    r=upload(client,LOG,transport='InfiniBand + NVLink');assert r.status_code==200
    assert r.json()['baseline_attainment']['verdict']=='insufficient_evidence'
    assert any(t['id'].startswith('reference-') for t in r.json()['follow_up_tests'])
    failed=LOG.replace('Out of bounds values : 0','Out of bounds values : 1')
    r=upload(client,failed);assert r.status_code==200
    assert r.json()['comparisons'][0]['run']['status']=='failed'
    assert client.get('/api/runs?cluster=ExampleCluster').json()['total']==2
    req={'run_id':r.json()['comparisons'][0]['run_id'],'reviewer':'Engineer','reason':'Reviewed for use as a reference'}
    assert client.post('/api/reference-reviews',json=req).status_code==422

def test_pinned_reference_and_tolerance_are_independent_of_destination(client):
    baseline()
    slow=LOG.replace('160.0','140.0')+'NCCL INFO NVLS multicast support is available on dev 0\n'
    r=upload(client,slow,reference_cluster='Reference cluster',tolerance_pct='20')
    assert r.json()['baseline_attainment']['verdict']=='insufficient_evidence'
    assert r.json()['comparisons'][0]['policy']['tolerance_pct']==20
    assert r.json()['comparisons'][0]['policy']['cluster_filter']=='Reference cluster'
    assert upload(client,slow,reference_cluster='ExampleCluster').json()['baseline_attainment']['verdict']=='insufficient_evidence'

def test_bundle_provenance_mixed_scales_and_customer_mapping(client):
    c,e=setup_customer(client)
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('../../all_reduce_job42_iter1.log',LOG)
        z.writestr('other/all_reduce_job42_iter2.log',LOG.replace('160.0','159.0'))
        z.writestr('rank.err','NCCL WARN unhandled cuda error')
    r=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster','customer_id':c['id'],'engagement_id':e['id']},files={'file':('exemplar.zip',b.getvalue())})
    assert r.status_code==200,r.text
    assert r.json()['ingestion']['member_count']==3
    assert r.json()['ingestion']['measured_runs']==2
    board=client.get('/api/customers').json()['items'][0]
    assert board['clusters']==['ExampleCluster']
    # The two collective artifacts share one job; the unassigned crash log is separate.
    assert board['validations']==2
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM upload_members').fetchone()[0]==3
        assert all(Path(row[0]).is_file() for row in db.execute("SELECT path FROM sources"))
    assert not Path(store.DB_PATH).parent.joinpath('all_reduce_job42_iter1.log').exists()

def test_reviewed_references_are_explicit_revocable_and_cannot_score_themselves(client):
    good=LOG+'NCCL INFO NVLS multicast support is available on dev 0\n'
    r=upload(client,good);assert r.status_code==200
    a=r.json();run=a['comparisons'][0]['run_id']
    assert a['baseline_attainment']['verdict']=='insufficient_evidence'
    req={'run_id':run,'reviewer':'Test engineer','reason':'Validated allocation and full correctness; accepted as a local reference'}
    assert client.post('/api/reference-reviews',json=req).status_code==200
    assert upload(client,good).json()['baseline_attainment']['verdict']=='insufficient_evidence'
    new=upload(client,good+'# independent rerun\n',cluster='ExampleCluster-2').json()
    assert new['baseline_attainment']['verdict']=='insufficient_evidence'
    assert new['comparisons'][0]['performance']['candidate_band'] is None
    assert new['comparisons'][0]['baseline_runs'][0]['review']['reviewer']=='Test engineer'
    assert client.post('/api/reference-reviews',json=dict(req,active=False)).status_code==200
    assert upload(client,good+'# independent rerun\n',cluster='ExampleCluster-2').json()['baseline_attainment']['verdict']=='insufficient_evidence'
    assert client.get('/api/lake/uploads').json()['summary']['reviewed_references']==0

def test_context_validation_is_atomic_for_bundle_and_can_enrich_missing_context(client):
    r=upload(client,LOG,nodes='99');assert r.status_code==422
    assert client.get('/api/runs?cluster=ExampleCluster').json()['total']==0
    assert client.get('/api/lake/uploads').json()['summary']['batches']==0
    r=upload(client,LOG);assert r.status_code==200
    r=upload(client,LOG,transport='InfiniBand + NVLink');assert r.status_code==200,r.text
    assert r.json()['comparisons'][0]['run']['transport']=='InfiniBand + NVLink'
    assert r.json()['comparisons'][0]['run']['metadata']['user_supplied_fields']==['transport']
    assert upload(client,LOG,transport='RoCE + NVLink').status_code==409
    assert upload(client,LOG,transport='InfiniBand + NVLink',tolerance_pct='nan').status_code==422

def test_training_numbers_are_ingested_without_using_nccl_as_training_target(client):
    text='nodes=2 gpus=16\niteration 1/1 | elapsed time per iteration (ms): 123.0\nGPU utilization: 456.0 MODEL_TFLOP/s/GPU\n'
    r=upload(client,text);assert r.status_code==200,r.text
    m=r.json()['comparisons'][0]['measurement']
    assert m['mean_step_time_ms']==123 and m['tflops_gpu']==456
    assert r.json()['baseline_attainment']['verdict']=='insufficient_evidence'
    assert client.get('/api/runs?cluster=ExampleCluster&workload=Training').json()['total']==1

def test_related_crash_artifact_disqualifies_clean_sibling_job(client):
    baseline()
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('all_reduce_job99.log',LOG+'NCCL INFO NVLS multicast support is available on dev 0\n')
        z.writestr('rank_job99.err','Connection refused')
    r=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster'},files={'file':('failed-job.zip',b.getvalue())})
    assert r.status_code==200,r.text
    a=r.json();assert a['baseline_attainment']['verdict']=='fail'
    assert a['comparisons'][0]['run']['metadata']['parent_job_failed']
    req={'run_id':a['comparisons'][0]['run_id'],'reviewer':'Engineer','reason':'Attempt to accept sibling output'}
    assert client.post('/api/reference-reviews',json=req).status_code==422

def test_partial_rank_counts_can_be_completed_on_reupload(client):
    partial=LOG.replace('nodes=2 gpus=16\n','')
    assert upload(client,partial).json()['comparisons'][0]['run']['gpus']==1
    r=upload(client,partial,nodes='2',gpus='16');assert r.status_code==200,r.text
    run=r.json()['comparisons'][0]['run']
    assert (run['nodes'],run['gpus'])==(2,16)
    assert run['metadata']['user_supplied_fields']==['gpus','nodes']
    assert upload(client,partial,nodes='4',gpus='32').status_code==409

def test_customer_upload_does_not_require_engagement_and_can_be_linked_later(client):
    c=client.post('/api/customers',json={'name':'ExampleCluster customer'}).json()
    r=upload(client,customer_id=c['id']);assert r.status_code==200,r.text
    report=r.json();run_id=report['comparisons'][0]['run_id']
    assert report['engagement_id'] is None
    board=client.get('/api/customers').json()['items'][0]
    assert board['validations']==1 and board['valid_logs']==1
    assert board['active_engagements']==0 and board['engagements']==[]
    e=client.post('/api/engagements',json={'customer_id':c['id'],'title':'ExampleCluster validation','recipe_id':'h100-infiniband'}).json()
    r=client.post('/api/customers/'+c['id']+'/evidence',json={'run_id':run_id,'engagement_id':e['id']})
    assert r.status_code==200,r.text
    assert upload(client,customer_id=c['id']).status_code==200
    evidence=client.get('/api/customers').json()['items'][0]['evidence']
    assert len(evidence)==1 and evidence[0]['engagement_id']==e['id']

def test_reupload_can_link_unassigned_customer_evidence_without_duplication(client):
    c,e=setup_customer(client)
    assert upload(client,customer_id=c['id']).status_code==200
    r=upload(client,customer_id=c['id'],engagement_id=e['id']);assert r.status_code==200,r.text
    evidence=client.get('/api/customers').json()['items'][0]['evidence']
    assert len(evidence)==1 and evidence[0]['engagement_id']==e['id']
    assert upload(client,customer_id='does-not-exist').status_code==404
    assert upload(client,engagement_id=e['id']).status_code==422
