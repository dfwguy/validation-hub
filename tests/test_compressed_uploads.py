import io,tarfile,hashlib
from pathlib import Path
import pytest,zstandard
from backend import troubleshooting,store
from fastapi.testclient import TestClient
from backend.app import app
from test_workflows import LOG

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',str(tmp_path/'catalog.sqlite'))
    monkeypatch.setenv('VALIDATION_UPLOADS',str(tmp_path/'uploads'))
    return TestClient(app)

def archive(compression='',name='results/all_reduce_perf.log',text=LOG):
    b=io.BytesIO();data=text.encode()
    with tarfile.open(fileobj=b,mode='w'+compression) as t:
        member=tarfile.TarInfo(name);member.size=len(data);t.addfile(member,io.BytesIO(data))
    return b.getvalue()

@pytest.mark.parametrize('filename,codec', [('logs.tar.gz','gz'),('logs.tgz','gz'),('logs.tar.zg','gz'),('logs.tar.zst','zst'),('logs.tzst','zst'),('logs.zst','zst')])
def test_compressed_archive_ingests_and_retains_original(client,filename,codec):
    data=archive(':gz') if codec=='gz' else zstandard.ZstdCompressor().compress(archive())
    response=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster'},files={'file':(filename,data)})
    assert response.status_code==200,response.text
    r=response.json();assert r['ingestion']['member_count']==1
    assert r['comparisons'][0]['measurement']['peak_bus_bw_gbps']==166
    original=client.get('/api/lake/uploads').json()['items'][0]
    assert Path(original['path']).read_bytes()==data
    assert original['sha256']==hashlib.sha256(data).hexdigest()

def test_zstd_size_limit_stops_before_reading_large_member(monkeypatch):
    data=zstandard.ZstdCompressor().compress(archive(text='x'*2048))
    monkeypatch.setattr(troubleshooting,'LIMIT',1024)
    with pytest.raises(ValueError,match='32 MiB'):troubleshooting.texts(data,'large.tar.zst')

def test_bad_zstd_returns_422_without_ingestion(client):
    for filename,data in [('broken.tar.zst',b'not a zstd stream'),('mistyped.tar.zg',b'not a gzip stream')]:
        r=client.post('/api/analyze',data={'recipe_id':'h100-infiniband','cluster':'ExampleCluster'},files={'file':(filename,data)})
        assert r.status_code==422,r.text
    assert client.get('/api/lake/uploads').json()['summary']['batches']==0

def test_zstd_members_never_extract_original_paths(tmp_path):
    data=zstandard.ZstdCompressor().compress(archive(name='../../escape.log'))
    assert troubleshooting.texts(data,'archive.tar.zst')[0][0]=='../../escape.log'
    assert not (tmp_path/'escape.log').exists()
