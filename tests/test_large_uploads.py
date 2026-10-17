import io,json,tarfile,hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend import store,upload_io
from test_workflows import LOG

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'catalog.sqlite'))
    monkeypatch.setenv('VALIDATION_UPLOADS',str(tmp_path/'uploads'))
    return TestClient(app)

class ZeroReader:
    def read(self,size):return b'\0'*size

def create_large_tar(path,size):
    with tarfile.open(path,'w') as t:
        extra=tarfile.TarInfo('other/diagnostic.bin');extra.size=size;t.addfile(extra,ZeroReader())
        result=tarfile.TarInfo('all_reduce_perf.log');result.size=len(LOG.encode());t.addfile(result,io.BytesIO(LOG.encode()))

def test_file_larger_than_old_limit_ingests_from_disk(client,tmp_path):
    archive=tmp_path/'large.tar';create_large_tar(archive,34*1024**2)
    with archive.open('rb') as f:r=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster'},files={'file':('large.tar',f)})
    assert r.status_code==200,r.text
    assert r.json()['ingestion']['original_bytes']>32*1024**2
    assert r.json()['comparisons'][0]['measurement']['peak_bus_bw_gbps']==166
    receipt=next((tmp_path/'uploads/received').glob('*.receipt.json'))
    data=json.loads(receipt.read_text())
    assert data['size_bytes']==archive.stat().st_size
    assert receipt.with_suffix('').with_suffix('.upload').exists()

def test_oversize_input_rejected_without_large_allocation(client,monkeypatch):
    monkeypatch.setattr(upload_io,'UPLOAD_LIMIT',16)
    r=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster'},files={'file':('too-large.log',b'x'*17)})
    assert r.status_code==413
    assert '1 GB' in r.text

def test_original_survives_analysis_error(client,tmp_path):
    original=b'not a valid compressed archive'
    r=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster'},files={'file':('bad.tar.zst',original)})
    assert r.status_code==422
    sha=hashlib.sha256(original).hexdigest()
    assert (tmp_path/'uploads/received'/f'{sha}.upload').read_bytes()==original
    assert sha in r.json()['detail']

def test_large_archive_expansion_limit(client,tmp_path,monkeypatch):
    archive=tmp_path/'large.tar';create_large_tar(archive,2048)
    monkeypatch.setattr(upload_io,'EXPANDED_LIMIT',1024)
    with archive.open('rb') as f:r=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster'},files={'file':('large.tar',f)})
    assert r.status_code==422 and '32 GiB' in r.text
    assert list((tmp_path/'uploads/received').glob('*.upload'))

def test_streaming_index_keeps_original_crash_line_numbers(client,tmp_path):
    text=('ordinary unrelated log chatter\n'*2000)+'NCCL WARN unhandled cuda error\n'
    r=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster'},files={'file':('rank.log',text)})
    assert r.status_code==200,r.text
    cuda=next(g for g in r.json()['crashes'] if g['id']=='cuda')
    assert cuda['evidence'][0]['line']==2001


def test_llmb_rank_logs_share_job_and_keep_declared_allocation(client):
    files={'file':('log-exemplar-pretrain_deepseek_gpus512_bf16_53_2.out','NCCL INFO NCCL version 2.30.4\n')}
    r=client.post('/api/analyze',data={'recipe_id':'b300-infiniband','cluster':'ExampleCluster'},files=files)
    assert r.status_code==200,r.text
    run=r.json()['comparisons'][0]['run']
    assert run['job_id']=='53' and run['gpus']==512
    assert run['workload']=='DeepSeek'
    assert run['metadata']['allocation_context_source']=='LLMB experiment path'

def test_llmb_archive_limits_cover_retained_real_bundle():
    assert upload_io.EXPANDED_LIMIT>=13822870396
    assert upload_io.MEMBER_LIMIT>10641


def test_materialized_manifest_reuses_original_paths(client,tmp_path):
    archive=tmp_path/'small.tar';create_large_tar(archive,32)
    first=upload_io.disk_texts(archive,'small.tar')
    second=upload_io.disk_texts(archive,'small.tar')
    assert [(n,t.sha256,t.size) for n,t in first]==[(n,t.sha256,t.size) for n,t in second]
    assert all(t.path.exists() for _,t in second)
