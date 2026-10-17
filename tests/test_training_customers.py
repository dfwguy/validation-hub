import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend import store
from backend.engagements import training_result


def training(model='Llama'):
    return dict(workload=model,status='passed',nodes=64,gpus=512,hardware='B300',transport='Spectrum-X',iterations=2,tflops_gpu=500,metadata={'expected_iterations':2,'archive_member':f'jobs/{model}/training.log'},metrics=[{'iteration':1,'time_ms':100},{'iteration':2,'time_ms':90},{'tflops_gpu':500}],source_kind='upload',source_path='/evidence/training.log')


@pytest.mark.parametrize('mutation',[lambda r:r.update(status='incomplete'),lambda r:r['metadata'].update(parent_job_failed=True),lambda r:r['metadata'].update(exit_failure=True),lambda r:r['metrics'][0].update(time_ms=float('inf')),lambda r:r['metadata'].update(expected_iterations=3),lambda r:r.update(source_path=''),lambda r:r.update(tflops_gpu=float('nan'))])
def test_ineligible_training(mutation):
    run=training();mutation(run)
    assert training_result(run) is None


def test_training_customer_summary_and_explicit_standalone_designation(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'catalog.sqlite'))
    client=TestClient(app)
    customer=client.post('/api/customers',json={'name':'ExampleCluster','clusters':['ExampleCluster']}).json()
    first=store.put_run(training(),'/evidence/training.log','a'*64,100,cluster='ExampleCluster',kind='upload')
    store.put_run(training('DeepSeek'),'/evidence/deepseek.log','b'*64,100,cluster='ExampleCluster',kind='upload')
    bad=training();bad['metadata']['parent_job_failed']=True
    failed=store.put_run(bad,'/evidence/failed.log','c'*64,100,cluster='ExampleCluster',kind='upload')
    summary=client.get('/api/customers').json()['items'][0]
    assert summary['valid_logs']==2
    assert {r['workload'] for r in summary['best_training_performance']}=={'Llama','DeepSeek'}
    assert summary['best_training_performance'][0]['mean_step_time_ms']==95
    assert summary['final_results']==0
    response=client.post(f"/api/customers/{customer['id']}/evidence",json={'run_id':first['id'],'final':True,'exemplar':True})
    assert response.status_code==200,response.text
    summary=client.get('/api/customers').json()['items'][0]
    assert summary['final_results']==summary['exemplar_results']==1
    assert summary['evidence'][0]['tflops_gpu']==500
    assert summary['evidence'][0]['mean_step_time_ms']==95
    assert summary['evidence'][0]['source_sha256']=='a'*64
    assert client.post(f"/api/customers/{customer['id']}/evidence",json={'run_id':failed['id'],'final':True}).status_code==422
