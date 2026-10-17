"""Local inventory of read-only Confluence evidence, with source versions and hashes."""
import datetime,gzip,hashlib,json,os,re,sqlite3
from html.parser import HTMLParser
from pathlib import Path

from . import config
SITE=config.ATLASSIAN_SITE

def _path(db_path=None):return Path(db_path) if db_path else config.data_path('CONFLUENCE_EVIDENCE_DB','confluence.sqlite')

def _db(db_path=None):
    p=_path(db_path);p.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(p,timeout=30);db.row_factory=sqlite3.Row
    db.execute('''CREATE TABLE IF NOT EXISTS wiki_sources(page_id TEXT PRIMARY KEY,title TEXT,url TEXT,version INTEGER,updated_at TEXT,category TEXT,gpus TEXT,text TEXT,body_sha256 TEXT,payload_sha256 TEXT,raw_path TEXT,indexed_at TEXT)''')
    db.execute('CREATE TABLE IF NOT EXISTS wiki_meta(key TEXT PRIMARY KEY,value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS wiki_findings(id TEXT PRIMARY KEY,title TEXT,kind TEXT,payload TEXT)')
    return db

class _Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.skip+=1
        if tag in ('p','br','tr','h1','h2','h3','li','pre'):self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in ('script','style') and self.skip:self.skip-=1
    def handle_data(self,text):
        if not self.skip and text.strip():self.parts.append(text.strip()+' ')
    def unknown_decl(self,data):
        if data.startswith('CDATA['):self.handle_data(data[6:])

def _category(title):
    t=title.lower()
    if any(v in t for v in ('troubleshoot','issue','regression','debug')):return 'troubleshooting'
    if any(v in t for v in ('recipe','configuration','bring-up','deployment','firmware')):return 'configuration'
    if any(v in t for v in ('result','performance report','benchmark','perf data','dashboard')):return 'benchmark evidence'
    return 'reference guide'

def ingest_directory(folder,db_path=None):
    folder=Path(folder);count=0;skipped=[];now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _db(db_path) as db:
        for p in sorted(folder.glob('*.metadata.json')):
            pid=p.name.removesuffix('.metadata.json');body=folder/(pid+'.html');raw=folder/(pid+'.page.json.gz')
            if not body.exists() or not raw.exists():continue
            try:
                payload=gzip.decompress(raw.read_bytes());document=json.loads(payload)
                if str(document.get('id'))!=pid:raise ValueError('Source page ID mismatch')
                html=document['body']['storage']['value']
                if not isinstance(html,str):raise ValueError('Missing source HTML')
                meta={k:v for k,v in document.items() if k!='body'}
            except (OSError,EOFError,ValueError,KeyError,TypeError):
                skipped.append(pid);continue
            parser=_Text();parser.feed(html)
            payload_sha=hashlib.sha256(payload).hexdigest()
            archive=folder/'versions'/pid/(payload_sha+'.page.json.gz');archive.parent.mkdir(parents=True,exist_ok=True)
            if not archive.exists():archive.write_bytes(gzip.compress(payload,mtime=0))
            text=re.sub(r'[ \t]+',' ',''.join(parser.parts));text=re.sub(r'\n\s*\n+','\n',text).strip()
            gpus=sorted(set(re.findall(r'\b(?:GB300|GB200|B300|B200|H200|H100|A100|VR200)\b',text+' '+meta.get('title',''),re.I)))
            gpus=sorted(set(g.upper() for g in gpus));webui=meta.get('_links',{}).get('webui','')
            url=SITE+webui if webui.startswith('/') else SITE+'/pages/viewpage.action?pageId='+pid
            version=meta.get('version',{});title=meta.get('title',pid)
            db.execute('INSERT OR REPLACE INTO wiki_sources VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(pid,title,url,version.get('number'),version.get('createdAt'),_category(title),json.dumps(gpus),text,hashlib.sha256(html.encode()).hexdigest(),payload_sha,str(archive),now))
            count+=1
        manifest=folder/'wiki-discovery-manifest.json'
        if manifest.exists():
            discovery=json.loads(manifest.read_text())
            receipt=dict(discovered_pages=discovery.get('unique_pages'),queries=[{k:v for k,v in q.items() if k!='items'} for q in discovery.get('queries',[])],
                         method='Cursor-paginated relevant-topic searches; downloaded pages selected for benchmark, recipe and troubleshooting evidence.')
            db.execute('INSERT OR REPLACE INTO wiki_meta VALUES(?,?)',('discovery',json.dumps(receipt)))
        reviewed=folder/'review-findings-report-ui.json'
        if reviewed.exists():
            db.execute('DELETE FROM wiki_findings')
            for finding in json.loads(reviewed.read_text()).get('findings',[]):
                evidence=finding.get('evidence',[])
                if not evidence or not all(str(e.get('source_url','')).startswith(SITE+'/') for e in evidence):continue
                valid=True
                for e in evidence:
                    current=db.execute('SELECT version,payload_sha256 FROM wiki_sources WHERE page_id=?',(e.get('page_id'),)).fetchone()
                    if not current or current[0]!=e.get('page_version') or current[1]!=e.get('source_sha256'):valid=False;break
                if valid:db.execute('INSERT OR REPLACE INTO wiki_findings VALUES(?,?,?,?)',(finding['id'],finding['title'],finding['kind'],json.dumps(finding)))
    return dict(indexed=count,skipped_incomplete=len(skipped),site=SITE,read_only=True)

def sources(q='',category='',gpu='',limit=50,offset=0,db_path=None):
    clauses=[];args=[]
    for token in re.findall(r'[\w-]+',q)[:12]:clauses.append('(title LIKE ? OR text LIKE ?)');args.extend(['%'+token+'%']*2)
    if category:clauses.append('category=?');args.append(category)
    if gpu:clauses.append('gpus LIKE ?');args.append('%"'+gpu.upper()+'"%')
    where=' WHERE '+' AND '.join(clauses) if clauses else '';limit=max(1,min(int(limit),200));offset=max(0,int(offset))
    with _db(db_path) as db:
        total=db.execute('SELECT count(*) FROM wiki_sources'+where,args).fetchone()[0]
        rows=[dict(r) for r in db.execute('SELECT page_id,title,url,version,updated_at,category,gpus,substr(text,1,360) AS excerpt,body_sha256,payload_sha256,indexed_at FROM wiki_sources'+where+' ORDER BY title LIMIT ? OFFSET ?',args+[limit,offset])]
        categories=[r[0] for r in db.execute('SELECT DISTINCT category FROM wiki_sources ORDER BY category')]
    for row in rows:row['gpus']=json.loads(row['gpus'])
    return dict(items=rows,total=total,limit=limit,offset=offset,facets=dict(category=categories))

def status(db_path=None):
    with _db(db_path) as db:
        count,latest=db.execute('SELECT count(*),max(indexed_at) FROM wiki_sources').fetchone()
        categories={r[0]:r[1] for r in db.execute('SELECT category,count(*) FROM wiki_sources GROUP BY category')}
        discovery=db.execute("SELECT value FROM wiki_meta WHERE key='discovery'").fetchone()
        findings=db.execute('SELECT count(*) FROM wiki_findings').fetchone()[0]
    return dict(site=SITE,pages=count,indexed_at=latest,categories=categories,read_only=True,access_method='Confluence REST API',
                scope='Relevant benchmark, recipe, validation and troubleshooting sources; not a full copy of the NVIDIA wiki.',
                reviewed_findings=findings,discovery=json.loads(discovery[0]) if discovery else None,
                caveat='Source pages include measured results, expectations, theoretical limits and guidance. Indexed pages alone are not scored benchmark references.')


def knowledge(q='',limit=50,db_path=None):
    with _db(db_path) as db:
        rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM wiki_findings ORDER BY title')]
    terms=re.findall(r'[\w-]+',q.lower())[:12]
    rows=[r for r in rows if all(t in json.dumps(r).lower() for t in terms)]
    return dict(items=rows[:max(1,min(int(limit),200))],total=len(rows),automatic_changes=False)
