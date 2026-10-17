import gzip
import json
from pathlib import Path
import pytest
from backend import beacon


def test_compact_export_keeps_original_source_record_identity():
    original=beacon.normalize_record({'id':'record-123','model':'llama','gpu':'B300','scale':8},'https://api.llmb.nvidia.com/v2/training/runs')
    compact={key:original.get(key) for key in beacon.COMPACT_FIELDS}
    imported=beacon.normalize_record(compact)
    assert imported['source_record_id']=='record-123'
    assert imported['ref_id']==original['ref_id']


def test_actual_api_schema_normalization_preserves_config_and_provenance():
    raw={'id':42,'job_id':'job-51','model':'deepseek-v3','model_size':'671b','gpu':'b300','scale':256,
         'precision':'fp8','precision_variant':'MX','tflops_gpu_avg':520.5,'seconds_per_step':9.4,
         'job_timestamp':'2026-09-25T12:30:00Z','release':'26.09','tensor_parallelism':1,
         'pipeline_parallelism':8,'expert_model_parallel_size':8,'expert_tensor_parallel_size':1,
         'virtual_pipeline_parallelism':2,'global_batch_size':8192,'sequence_length':4096,
         'job_source':'llmb','unknown_future_column':{'exact':'retained'}}
    r=beacon.normalize_record(raw,'https://api.llmb.nvidia.com/v2/training/runs?page_size=200')
    assert (r['model'],r['model_size'],r['workload'])==('deepseek-v3','671b','deepseek-v3|671b')
    assert (r['gpu'],r['gpus'],r['precision'])==('B300',256,'fp8_mx')
    assert r['tflops_gpu']==520.5 and r['seconds_per_step']==9.4
    assert r['tensor_parallel']==1 and r['expert_tensor_parallel']==1 and r['virtual_pipeline']==2
    assert r['date_utc']=='2026-09-25T12:30:00+00:00'
    assert r['row_action']=='https://llmb.nvidia.com/beacon-v2/runs/42'
    assert r['raw_json']==raw
    assert r['source_mapping']['tflops_gpu']=='tflops_gpu_avg'
    assert r['ref_id']==beacon.normalize_record(raw,'https://api.llmb.nvidia.com/v2/training/runs?page_size=150')['ref_id']


def test_import_upserts_actual_rows_retains_compressed_raw_and_lists_filters(tmp_path):
    db=tmp_path/'refs.sqlite';row={'id':1,'model':'llama','gpu':'h100','scale':8,'precision':'bf16','tflops_gpu_avg':123}
    receipt=beacon.import_records([row],db_path=db)
    assert json.loads(gzip.decompress(Path(receipt['path']).read_bytes()))==row
    assert beacon.status(db)['records']==1
    assert beacon.list_references({'gpu':'H100'},db_path=db)['total']==1
    assert beacon.list_references({'gpu':'B300'},db_path=db)['total']==0
    beacon.import_records([{**row,'tflops_gpu_avg':124}],db_path=db)
    assert len(beacon.references(db))==1
    assert beacon.references(db)[0]['tflops_gpu']==124
    with pytest.raises(ValueError,match='No Beacon'):
        beacon.import_records([],db_path=db)
    assert beacon.status(db)['records']==1


@pytest.mark.parametrize('extension,body',[
    ('.json','{"items":[{"id":1,"gpu":"b300","scale":256}]}'),
    ('.jsonl','{"id":1,"gpu":"b300","scale":256}\n'),
    ('.csv','id,gpu,scale\n1,b300,256\n'),
])
def test_offline_export_formats_and_gzip(tmp_path,extension,body):
    path=tmp_path/('export'+extension+'.gz');path.write_bytes(gzip.compress(body.encode()))
    beacon.import_file(path,db_path=tmp_path/'db.sqlite')
    r=beacon.references(tmp_path/'db.sqlite')[0]
    assert r['gpu']=='B300' and r['scale']==256


def test_credentials_must_be_private_and_are_not_persisted(tmp_path,monkeypatch):
    path=tmp_path/'headers';path.write_text('{"Authorization":"Bearer test-secret"}');path.chmod(0o644)
    monkeypatch.setenv('BEACON_HEADERS_FILE',str(path));monkeypatch.delenv('BEACON_COOKIE_FILE',raising=False)
    with pytest.raises(ValueError,match='chmod 600'):beacon.private_headers()
    path.chmod(0o600)
    assert beacon.private_headers()['Authorization']=='Bearer test-secret'


def test_empty_store_is_explicit_and_has_no_fake_benchmarks(tmp_path):
    s=beacon.status(tmp_path/'empty.sqlite')
    assert s['records']==0 and s['status']=='login_required' and not s['authenticated']
    assert 'Network' in s['instructions']


def test_unknown_or_nonfinite_metrics_remain_missing():
    r=beacon.normalize_record({'gpu':'b300','scale':256,'tflops_gpu_avg':'NaN','seconds_per_step':'inf','date':'2026-09-25'})
    assert r['tflops_gpu'] is None and r['seconds_per_step'] is None and r['date_utc'] is None


def test_confirmed_beacon_cursor_pagination_and_provenance(tmp_path,monkeypatch):
    import io
    from email.message import Message
    from urllib.parse import parse_qs,urlsplit
    calls=[]
    class Response(io.BytesIO):
        def __init__(self,payload):
            super().__init__(json.dumps(payload).encode());self.headers=Message();self.headers['Content-Type']='application/json'
    class Opener:
        def open(self,request,timeout):
            calls.append(request.full_url)
            if len(calls)==1:return Response({'items':[{'id':1,'gpu':'h100'},{'id':2,'gpu':'b300'}],'total':3,'next_cursor':'cursor-2'})
            return Response({'items':[{'id':3,'gpu':'gb300'}],'next_cursor':None,'total':None})
    monkeypatch.setattr(beacon,'api_headers',lambda url:{'Accept':'application/json'})
    monkeypatch.setattr(beacon,'build_opener',lambda *args:Opener())
    receipt=beacon.fetch_all('https://api.llmb.nvidia.com/v2/training/runs?page_size=200',db_path=tmp_path/'db.sqlite')
    assert receipt['imported']==3
    query=parse_qs(urlsplit(calls[1]).query)
    assert query['c_d']==['after'] and query['c_id']==['cursor-2']
    status=beacon.status(tmp_path/'db.sqlite')
    assert status['authenticated'] and status['records']==3
    assert status['gpu_types']==['B300','GB300','H100']
    assert status['snapshot']['metadata']['all_gpu_flavors'] is True


def test_incomplete_api_page_does_not_replace_existing_references(tmp_path,monkeypatch):
    import io
    from email.message import Message
    class Response(io.BytesIO):
        def __init__(self):
            super().__init__(b'{"items":[{"id":2}],"total":10}');self.headers=Message();self.headers['Content-Type']='application/json'
    class Opener:
        def open(self,*args,**kwargs):return Response()
    db=tmp_path/'db.sqlite';beacon.import_records([{'id':1,'gpu':'b300'}],db_path=db)
    monkeypatch.setattr(beacon,'api_headers',lambda url:{})
    monkeypatch.setattr(beacon,'build_opener',lambda *args:Opener())
    with pytest.raises(ValueError,match='paginated'):
        beacon.fetch_all('https://api.llmb.nvidia.com/v2/training/runs',db_path=db)
    assert [r['source_record_id'] for r in beacon.references(db)]==['1']


def test_compact_snapshot_and_complete_scope_retirement(tmp_path):
    db=tmp_path/'db.sqlite';base='https://api.llmb.nvidia.com/v2/training/runs'
    a={'id':1,'gpu':'b300','model':'deepseek','scale':256,'tflops_gpu_avg':100}
    b={'id':2,'gpu':'h100','model':'llama','scale':8,'tflops_gpu_avg':200}
    beacon.import_records([a,b],base+'?page_size=150',db,metadata={'method':'authenticated_api','complete_all_pages':True,'reported_total':2})
    receipt=beacon.import_records([b],base+'?page_size=200',db,metadata={'method':'authenticated_api','complete_all_pages':True,'reported_total':1})
    assert [r['source_record_id'] for r in beacon.references(db)]==['2']
    compact=[json.loads(line) for line in gzip.decompress(Path(receipt['compact_path']).read_bytes()).decode().splitlines()]
    assert set(compact[0])==set(beacon.COMPACT_FIELDS)
    import hashlib
    assert hashlib.sha256(Path(receipt['compact_path']).read_bytes()).hexdigest()==receipt['compact_sha256']
    beacon.import_records([a],base,db)
    assert len(beacon.references(db))==2  # Partial/offline import appends.
    with pytest.raises(ValueError,match='Unique records'):
        beacon.import_records([a,a],base,db,metadata={'complete_all_pages':True,'reported_total':2})
    assert len(beacon.references(db))==2


def test_expired_credentials_preserve_existing_records_and_report_error(tmp_path,monkeypatch):
    db=tmp_path/'db.sqlite';beacon.import_records([{'id':1,'gpu':'b300'}],db_path=db)
    def expired(url):raise ValueError('Beacon session expired; refresh private credentials')
    monkeypatch.setattr(beacon,'api_headers',expired)
    with pytest.raises(ValueError,match='expired'):
        beacon.fetch_all('https://api.llmb.nvidia.com/v2/training/runs',db_path=db)
    assert beacon.status(db)['records']==1
    assert beacon.status(db)['status']=='refresh_failed'
    assert 'expired' in beacon.status(db)['error']
