import io,json,zipfile,subprocess
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend import store
from backend.parser import parse_log
from backend.compare import compare
from backend.troubleshooting import analyze,texts
from backend.recipes import generate,ScriptRequest

LOG='''# Collective test starting: all_reduce_perf_mpi
NCCL INFO NCCL version 2.31.2
nodes=2 gpus=16
# Rank 0 Group 0 Pid 20 on host-1 device 0 [0008:06:00] NVIDIA H100
NCCL_IB_DISABLE=0
NCCL_IB_HCA=mlx5_0:1
NCCL_SOCKET_IFNAME=eth0
NCCL INFO NET/IB : Using mlx5_0
8192 2048 float sum -1 15.0 120.0 160.0 0 14.0 125.0 166.0 0
# Out of bounds values : 0 OK
# Avg bus bandwidth : 163.0
'''
@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'test.sqlite'))
    monkeypatch.setenv('VALIDATION_UPLOADS',str(tmp_path/'uploads'))
    return TestClient(app)

def setup_customer(c,name='Acme',clusters=None):
    r=c.post('/api/customers',json={'name':name,'clusters':clusters or []});assert r.status_code==200,r.text
    customer=r.json()
    r=c.post('/api/engagements',json={'customer_id':customer['id'],'title':'H100 bringup','recipe_id':'h100-infiniband','status':'active'})
    assert r.status_code==200,r.text
    return customer,r.json()

def test_crash_only_log_unknowns_and_evidence():
    r=analyze([('rank-8.log','NCCL_IB_DISABLE=1\nNCCL WARN ibv_poll_cq failed error 12\npassword=secret')],'h100-infiniband')
    assert r['verdict']=='fail'
    assert next(c for c in r['checks'] if c['id']=='gpu')['status']=='unknown'
    g=next(g for g in r['crashes'] if g['id']=='rdma')
    assert g['evidence'][0]['line']==2
    assert any(t['priority']==1 for t in r['follow_up_tests'])

def test_unknown_is_not_pass_and_roce_uses_verbs():
    r=analyze([('x.log',LOG.replace('NCCL_IB_DISABLE=0',''))],'h100-spectrum-x')
    assert r['verdict']=='needs_evidence'
    assert next(c for c in r['checks'] if c['id']=='NCCL_IB_ROCE_VERSION_NUM')['status']=='unknown'
    assert next(c for c in r['checks'] if c['id']=='rdma')['status']=='pass'

def test_bundle_no_extraction_and_limits():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('../../rank.log',LOG)
    assert texts(b.getvalue(),'x.zip')[0][1]==LOG
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        for i in range(201):z.writestr(f'{i}.log','a')
    with pytest.raises(ValueError,match='200'):texts(b.getvalue(),'x.zip')
    with pytest.raises(ValueError,match='Binary'):texts(b'\0binary','x.log')
    with pytest.raises(ValueError,match='UTF-8'):texts(b'\xff','x.log')

def test_generator_quoting_and_bash_syntax(tmp_path):
    req=ScriptRequest(recipe_id='gb300-spectrum-x',nodes=2,gpus_per_node=4,image="/shared/image '$(touch /tmp/NEVER).sqsh",hca='=mlx5_0:1,mlx5_1:1',interface='=eth0')
    result=generate(req);script=tmp_path/'test.sbatch';script.write_text(result['script'])
    assert subprocess.run(['bash','-n',str(script)],capture_output=True).returncode==0
    assert 'NCCL_IB_ROCE_VERSION_NUM=2' in result['script']
    assert '0x0 0x3' in result['script']
    assert '-c 1' in result['script']
    assert 'printenv' not in result['script']
    with pytest.raises(ValueError):generate(req.model_copy(update={'collective':'$(id)'}))
    with pytest.raises(ValueError):ScriptRequest(recipe_id='h100-infiniband',image='bad\nimage',hca='mlx5_0',interface='eth0')

def test_split_mask_baselines_are_separate(client):
    ref=parse_log(LOG.splitlines(),hints={'transport':'InfiniBand + NVLink'})
    store.put_run(ref,'r.log','ref',100,'acme')
    split=parse_log((LOG+'NCCL_TESTS_SPLIT_MASK=0x7\n').splitlines(),hints={'transport':'InfiniBand + NVLink'})
    assert compare(split,'acme')['baseline_count']==0
    mixed=parse_log((LOG+'NCCL_TESTS_SPLIT_MASK=0x0\nNCCL_TESTS_SPLIT_MASK=0x7').splitlines())
    assert mixed['metadata']['mixed_split_masks']

def test_analysis_persistence_customer_totals_final_and_owner(client):
    c,e=setup_customer(client)
    data={'recipe_id':'h100-infiniband','customer_id':c['id'],'engagement_id':e['id']}
    response=client.post('/api/analyze',data=data,files={'file':('test.log',LOG)})
    assert response.status_code==200,response.text
    a=response.json();assert client.get('/api/analyses/'+a['id']).json()['id']==a['id']
    run_id=a['comparisons'][0]['run_id']
    body={'run_id':run_id,'engagement_id':e['id'],'final':True,'exemplar':True}
    assert client.post('/api/customers/'+c['id']+'/evidence',json=body).status_code==200
    r=client.get('/api/customers');assert r.status_code==200,r.text
    board=r.json()['items'][0]
    assert board['validations']==board['valid_logs']==board['final_results']==board['exemplar_results']==1
    assert board['best_performance'][0]['point']['message_bytes']==8192
    assert board['active_engagements']==1
    assert client.post('/api/analyze',data=data,files={'file':('same.log',LOG)}).status_code==200
    assert client.get('/api/customers').json()['items'][0]['validations']==1
    c2,e2=setup_customer(client,'Other')
    assert client.post('/api/customers/'+c2['id']+'/evidence',json=dict(body,engagement_id=e2['id'])).status_code==409
    assert client.post('/api/analyze',data=dict(data,customer_id=c2['id']),files={'file':('x.log',LOG)}).status_code==422

def test_crash_upload_and_designation_rejection(client):
    c,e=setup_customer(client)
    r=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','customer_id':c['id'],'engagement_id':e['id']},files={'file':('crash.err','NCCL WARN unhandled cuda error')})
    assert r.status_code==200,r.text
    assert r.json()['crashes']
    rid=r.json()['comparisons'][0]['run_id']
    assert client.post('/api/customers/'+c['id']+'/evidence',json={'run_id':rid,'engagement_id':e['id'],'final':True}).status_code==422

def test_customer_mapping_counts_distinct_jobs(client):
    c,e=setup_customer(client,clusters=['Acme-TX'])
    for i in range(3):
        run=parse_log(LOG.splitlines());run['job_id']='42'
        store.put_run(run,f'x{i}.log',f'hash{i}',100,'Acme-TX','raw')
    board=client.get('/api/customers').json()['items'][0]
    assert board['validations']==1 and board['records']==3
    assert client.post('/api/customers',json={'name':'Other','clusters':['Acme-TX']}).status_code==409
    assert client.put('/api/engagements/'+e['id'],json=dict(e,status='completed')).status_code==200
    assert client.get('/api/customers').json()['items'][0]['active_engagements']==0

def test_recipe_conflict_and_bad_inputs(client):
    c,e=setup_customer(client)
    assert client.post('/api/analyze',data={'recipe_id':'gb300-infiniband','customer_id':c['id'],'engagement_id':e['id']},files={'file':('x.log',LOG)}).status_code==422
    assert client.post('/api/analyze',data={'recipe_id':'no-such-profile'},files={'file':('x.log',LOG)}).status_code==422
    assert client.post('/api/analyze',data={'recipe_id':'h100-infiniband'},files={'file':('x.log',b'\0x')}).status_code==422
    assert client.post('/api/nccl-script',json={'recipe_id':'h100-infiniband','nodes':0}).status_code==422

def test_timeout_environment_is_not_crash_and_roce_is_not_ib():
    text=LOG+'NCCL_IB_TIMEOUT=20\nNCCL INFO NVLS multicast support is available on dev 0\n'
    result=analyze([('test.log',text)],'h100-spectrum-x')
    assert not any(g['id']=='timeout' for g in result['crashes'])
    assert result['runs'][0]['parsed']['transport'] is None
    assert analyze([('test.log',text+'Link layer: Ethernet')],'h100-spectrum-x')['runs'][0]['parsed']['transport']=='RoCE + NVLink'
