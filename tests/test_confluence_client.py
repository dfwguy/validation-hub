import json
import pytest
from backend import confluence_client


def test_authenticated_reader_refuses_external_paths_and_nonprivate_tokens(tmp_path):
    token=tmp_path/'token';token.write_text('private-token');token.chmod(0o644)
    reader=confluence_client.ConfluenceReader('user@example.com',token,cloud_id='00000000-0000-0000-0000-000000000000')
    with pytest.raises(ValueError,match='relative'):reader.get('https://example.com/api/v2/pages/1')
    with pytest.raises(ValueError,match='private'):reader.get('/api/v2/pages/1')


def test_search_follows_opaque_cursor_and_saves_no_credentials(tmp_path,monkeypatch):
    token=tmp_path/'token';token.write_text('private-token');token.chmod(0o600)
    reader=confluence_client.ConfluenceReader('user@example.com',token,cloud_id='00000000-0000-0000-0000-000000000000')
    seen=[]
    def fake_get(path,query=None):
        seen.append((path,query))
        if len(seen)==1:return {'results':[{'id':'1'}],'_links':{'next':'/rest/api/content/search?cursor=opaque&start=100'}}
        return {'results':[{'id':'2'}]}
    monkeypatch.setattr(reader,'get',fake_get)
    result=reader.search('title ~ "NCCL"')
    assert result['complete'] and result['count']==2
    assert seen[1]==('/rest/api/content/search?cursor=opaque&start=100',None)
    assert 'private-token' not in json.dumps(result)
    monkeypatch.setattr(reader,'get',lambda *args:{'results':[],'_links':{'next':'https://example.com/read'}})
    with pytest.raises(ValueError,match='Cross-origin'):reader.search('title ~ "NCCL"')
