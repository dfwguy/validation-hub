import os, json, sqlite3, hashlib
from pathlib import Path
from datetime import datetime, timezone
from .security import scrub
from .hostlist import expand_hostlist
from contextlib import contextmanager
from .config import data_path
DB_PATH = str(data_path('VALIDATION_DB','catalog/validation.sqlite'))

@contextmanager
def connection():
    Path(DB_PATH).parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(DB_PATH,timeout=60); db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY,title TEXT,cluster TEXT,workload TEXT,collective TEXT,nodes INTEGER,gpus INTEGER,racks REAL,sus REAL,hardware TEXT,transport TEXT,nccl_version TEXT,status TEXT,avg_bus_bw_gbps REAL,peak_bus_bw_gbps REAL,tflops_gpu REAL,iterations INTEGER,source_path TEXT,source_kind TEXT,created_at TEXT,payload TEXT,sha256 TEXT); CREATE INDEX IF NOT EXISTS scope_idx ON runs(cluster,workload,nodes,gpus,collective,status); CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY,sha256 TEXT,size_bytes INTEGER,kind TEXT,cluster TEXT,details TEXT);''')
    columns={r[1] for r in db.execute('PRAGMA table_info(runs)')}
    if 'job_id' not in columns: db.execute('ALTER TABLE runs ADD COLUMN job_id TEXT')
    db.execute('CREATE INDEX IF NOT EXISTS job_idx ON runs(cluster,source_kind,job_id)')
    try:
        with db:
            yield db
    finally:
        db.close()

def put_run(result,path,sha256,size,cluster='Unknown',kind='raw',suffix='',display_name=None):
    payload=scrub(dict(result))
    payload.update(id=hashlib.sha256((sha256+cluster+kind+suffix).encode()).hexdigest()[:24],title=Path(path).name+suffix,cluster=cluster,source_path=str(path),source_kind=kind,created_at=datetime.now(timezone.utc).isoformat(),source=dict(path=str(path),sha256=sha256,size_bytes=size))
    if display_name:payload['title']=display_name
    keys=['id','title','cluster','workload','collective','nodes','gpus','racks','sus','hardware','transport','nccl_version','status','avg_bus_bw_gbps','peak_bus_bw_gbps','tflops_gpu','iterations','source_path','source_kind','created_at','job_id']
    with connection() as db:
        db.execute('INSERT OR REPLACE INTO runs ('+','.join(keys)+',payload,sha256) VALUES ('+','.join('?' for _ in range(len(keys)+2))+')',[payload.get(k) for k in keys]+[json.dumps(payload,allow_nan=False),sha256])
    return payload

def source(path,sha256,size,kind,cluster,details=None):
    with connection() as db: db.execute('INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?)',(str(path),sha256,size,kind,cluster,json.dumps(details or {})))

def search(filters=None,limit=50,offset=0):
    filters=filters or {}; clauses=[]; values=[]
    for field in ('cluster','workload','collective','nodes','gpus','racks','sus','hardware','status','nccl_version','transport'):
        if filters.get(field) is not None and filters[field]!='': clauses.append(field+'=?'); values.append(filters[field])
    for field,op in [('min_nodes','>='),('max_nodes','<=')]:
        if filters.get(field) is not None: clauses.append('nodes'+op+'?');values.append(filters[field])
    for key,column in [('rack_ids','racks_json'),('su_ids','sus_json'),('nodelist','nodes_json')]:
        if filters.get(key):
            for member in (expand_hostlist(filters[key]) if key=='nodelist' else filters[key].split(',')):
                clauses.append("EXISTS (SELECT 1 FROM json_each(COALESCE(json_extract(runs.payload,'$.metadata."+column+"'),(SELECT json_extract(j.payload,'$.metadata."+column+"') FROM runs j WHERE j.cluster=runs.cluster AND j.source_kind='parquet_slurm_jobs' AND j.job_id=runs.job_id LIMIT 1))) WHERE value=?)");values.append(member.strip())
    if filters.get('measured_only'): clauses.append("(avg_bus_bw_gbps IS NOT NULL OR tflops_gpu IS NOT NULL OR iterations IS NOT NULL)")
    if filters.get('q') and filters['q'].isdigit():
        clauses.append('job_id=?');values.append(filters['q'])
    elif filters.get('q'):
        clauses.append('(title LIKE ? OR source_path LIKE ? OR payload LIKE ?)'); values.extend(['%'+filters['q']+'%']*3)
    where=' WHERE '+' AND '.join(clauses) if clauses else ''
    with connection() as db:
        total=db.execute('SELECT COUNT(*) FROM runs'+where,values).fetchone()[0]
        items=[dict(r) for r in db.execute('SELECT * FROM runs'+where+' ORDER BY CASE WHEN source_kind="raw" AND (avg_bus_bw_gbps IS NOT NULL OR iterations IS NOT NULL) THEN 0 WHEN avg_bus_bw_gbps IS NOT NULL OR tflops_gpu IS NOT NULL THEN 1 ELSE 2 END,created_at DESC,id LIMIT ? OFFSET ?',values+[limit,offset])]
    for item in items: item.pop('payload',None);item.pop('sha256',None)
    return dict(items=items,total=total)

def detail(id):
    with connection() as db: row=db.execute('SELECT payload FROM runs WHERE id=?',(id,)).fetchone()
    return json.loads(row[0]) if row else None
