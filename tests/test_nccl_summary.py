import copy
import pytest
from backend.nccl_reference import NCCLReferenceIndex
from backend.nccl_summary import assess,SIZES


def case():
    point=dict(message_bytes=16*1024**3,busbw_gbps=400,datatype='float',operation='sum',placement='out_of_place',errors=0)
    run=dict(id='own',cluster='same',job_id='own-job',gpu='GB300',gpus=64,nodes=8,collective='all_gather',nccl_version='2.29.3',status='passed',metrics=[point],metadata={'correctness_errors':0})
    source=copy.deepcopy(run);source.update(ref_id='best',source_record_id='best',id='best',job_id='other-job',source_url='/api/runs/best')
    source['metrics'][0]['busbw_gbps']=500
    return run,source


def test_two_large_message_columns_use_best_bandwidth_and_keep_missing_size_blank():
    run,source=case()
    run['metrics'].append(dict(message_bytes=0,busbw_gbps=0,time_us=.06,errors=0))
    result=assess(run,[('Cluster history',NCCLReferenceIndex([source]))])
    summary=result['nccl_summary'];rows=summary['rows']
    assert SIZES==(16*1024**3,32*1024**3)
    assert len(rows)==2 and len(summary['improvements'])==4
    assert rows[0]['observed_gbps']==400 and rows[0]['reference_gbps']==500
    assert rows[0]['delta_pct']==pytest.approx(-20)
    assert rows[1]['observed_gbps'] is None and rows[1]['reference_status']=='not_measured'
    assert len(result['metrics'])==1 and result['metrics'][0]['metric']=='busbw_gbps'
    assert 'this cluster' in summary['context_note']


def test_best_value_has_exact_version_and_is_not_a_median_or_interpolation():
    run,source=case();second=copy.deepcopy(source);second.update(ref_id='second',job_id='second-job')
    second['metrics'][0]['busbw_gbps']=700
    other=copy.deepcopy(second);other.update(ref_id='wrong-version',job_id='third-job',nccl_version='2.30.7');other['metrics'][0]['busbw_gbps']=1000
    result=assess(run,[('history',NCCLReferenceIndex([source,second,other]))])
    assert result['nccl_summary']['rows'][0]['reference_gbps']==700
    run['nccl_version']=None
    assert assess(run,[('history',NCCLReferenceIndex([source]))])['nccl_summary']['rows'][0]['delta_pct'] is None


def test_job_failure_remains_visible_without_discarding_actual_bandwidth():
    run,source=case();run['metadata']['parent_job_failed']=True
    result=assess(run,[('history',NCCLReferenceIndex([source]))])
    assert result['classification']=='failed' and result['verdict']=='fail'
    assert result['nccl_summary']['rows'][0]['observed_gbps']==400
    assert result['nccl_summary']['rows'][0]['delta_pct'] is not None


def test_single_log_upload_persists_two_column_summary_for_explorer(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from backend import store
    from backend.app import app
    from test_workflows import LOG
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'catalog.sqlite'))
    monkeypatch.setenv('VALIDATION_UPLOADS',str(tmp_path/'uploads'))
    client=TestClient(app)
    text=LOG.replace('8192 2048','17179869184 4294967296')
    uploaded=client.post('/api/compare',files={'file':('all_reduce.log',text)})
    assert uploaded.status_code==200,uploaded.text
    result=uploaded.json();row=result['performance']['nccl_summary']['rows'][0]
    assert row['observed_gbps']==160
    listed=client.get('/api/runs').json()['items'][0]
    assert listed['performance']['nccl_summary']['rows'][0]['observed_gbps']==160
    assert client.get('/api/runs/'+result['run']['id']).json()['performance']['nccl_summary']['rows'][1]['observed_gbps'] is None


def test_native_rank_rounding_uses_display_bucket_but_exact_reference_bytes():
    run,source=case()
    run['metrics'][0]['message_bytes']-=128
    result=assess(run,[('history',NCCLReferenceIndex([source]))])
    row=result['nccl_summary']['rows'][0]
    assert row['observed_gbps']==400 and row['actual_message_bytes']==16*1024**3-128
    assert row['reference_gbps'] is None  # Never compare the nearby exact16GiB curve.
    source['metrics'][0]['message_bytes']-=128
    result=assess(run,[('history',NCCLReferenceIndex([source]))])
    assert result['nccl_summary']['rows'][0]['reference_gbps']==500
    assert result['metrics'][0]['message_bytes']==16*1024**3-128
