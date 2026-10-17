import gzip,hashlib,json
from backend import wiki_sources


def test_indexed_wiki_evidence_is_versioned_searchable_and_separate_from_targets(tmp_path):
    source=tmp_path/'evidence';source.mkdir();db=tmp_path/'wiki.sqlite'
    meta=dict(id='123',title='GB300 Spectrum-X troubleshooting',version={'number':7,'createdAt':'2026-10-17T00:00:00Z'},_links={'webui':'/spaces/SW/pages/123'})
    body='<h1>Measured versus theoretical</h1><p>GB300 NCCL on Spectrum-X</p><ac:plain-text-body><![CDATA[NCCL_PROTO=Simple]]></ac:plain-text-body><script>hidden_secret</script>'
    payload=json.dumps(dict(meta,body={'storage':{'value':body}})).encode()
    (source/'123.metadata.json').write_text(json.dumps(meta));(source/'123.html').write_text(body);(source/'123.page.json.gz').write_bytes(gzip.compress(payload))
    assert wiki_sources.ingest_directory(source,db)['indexed']==1
    status=wiki_sources.status(db)
    assert status['read_only'] is True and status['pages']==1
    assert 'not scored' in status['caveat']
    result=wiki_sources.sources(q='Spectrum-X NCCL',gpu='GB300',db_path=db)
    row=result['items'][0]
    assert row['version']==7 and row['category']=='troubleshooting'
    assert row['url']=='https://nvidia.atlassian.net/wiki/spaces/SW/pages/123'
    assert row['payload_sha256']==hashlib.sha256(payload).hexdigest()
    assert 'hidden_secret' not in row['excerpt']
    assert 'NCCL_PROTO=Simple' in row['excerpt']
    assert 'raw_path' not in row
    assert wiki_sources.sources(gpu='B300',db_path=db)['total']==0
    assert wiki_sources.sources(q="' OR 1=1 --",db_path=db)['total']==0
    wiki_sources.ingest_directory(source,db)
    assert wiki_sources.status(db)['pages']==1


def test_only_verified_versioned_guidance_enters_recipe_investigation(tmp_path,monkeypatch):
    from backend import investigation
    root=tmp_path/'source';root.mkdir();db=tmp_path/'wiki.sqlite';monkeypatch.setenv('CONFLUENCE_EVIDENCE_DB',str(db))
    meta=dict(id='42',title='Spectrum-X troubleshooting',version={'number':3})
    payload=json.dumps(dict(meta,body={'storage':{'value':'<p>Correlate queues and retransmissions.</p>'}})).encode();sha=hashlib.sha256(payload).hexdigest()
    (root/'42.metadata.json').write_text(json.dumps(meta));(root/'42.html').write_text('<p>Correlate queues and retransmissions.</p>');(root/'42.page.json.gz').write_bytes(gzip.compress(payload))
    evidence=[dict(page_id='42',page_version=3,source_sha256=sha,source_url='https://nvidia.atlassian.net/wiki/spaces/SW/pages/42',section_path=['Queues'])]
    f=dict(id='spx-application-flow-3950118452',title='Correlate queues',kind='troubleshooting',summary='Compare counters at the same time.',evidence=evidence,applicability={'fabric':['Spectrum-X']},diagnostic_branches=[dict(condition='Localized retransmits',investigate='physical link')])
    site=dict(f,id='site-specific-recipe',title='Site-specific change',site_specific=True,applicability={'fabric':['Spectrum-X'],'cluster':['different-site']})
    (root/'review-findings-report-ui.json').write_text(json.dumps({'findings':[f,site]}))
    wiki_sources.ingest_directory(root,db)
    plan=investigation.build({'recipe':{'fabric':'spectrum-x'}})
    assert any(r['id']=='wiki-'+f['id'] for r in plan['findings'])
    assert not any(r['id']=='wiki-'+site['id'] for r in plan['findings'])
    assert not any(r['id'].startswith('wiki-') for r in investigation.build({'recipe':{'fabric':'infiniband'}})['findings'])
    meta['version']['number']=4;(root/'42.metadata.json').write_text(json.dumps(meta))
    wiki_sources.ingest_directory(root,db)
    assert wiki_sources.knowledge(db_path=db)['total']==2
    payload=json.dumps(dict(meta,body={'storage':{'value':'<p>Updated source.</p>'}})).encode()
    (root/'42.page.json.gz').write_bytes(gzip.compress(payload))
    wiki_sources.ingest_directory(root,db)
    assert wiki_sources.knowledge(db_path=db)['total']==0
    assert len(list((root/'versions'/'42').glob('*.page.json.gz')))==2
