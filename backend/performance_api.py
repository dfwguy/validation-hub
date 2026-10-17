"""Reference inventory and durable, auditable performance reassessments."""
import gzip
import hashlib
import json
import os
from . import config
from pathlib import Path
import tempfile
import threading
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from . import assessment, beacon, engagements, store, beacon_nccl, wiki_sources

router=APIRouter(prefix='/api/performance')
_work_lock=threading.Lock()


def now():return datetime.now(timezone.utc).isoformat()


def init(db):
    db.executescript('''CREATE TABLE IF NOT EXISTS performance_jobs(id TEXT PRIMARY KEY,kind TEXT,status TEXT,created_at TEXT,updated_at TEXT,details TEXT);
    CREATE TABLE IF NOT EXISTS performance_history(id TEXT PRIMARY KEY,analysis_id TEXT,created_at TEXT,snapshot_id TEXT,payload BLOB,sha256 TEXT);
    CREATE TABLE IF NOT EXISTS run_performance(run_id TEXT PRIMARY KEY,analysis_id TEXT,snapshot_id TEXT,created_at TEXT,payload TEXT);''')


def cache_assessments(db,report):
    snapshot=(report.get('performance',{}).get('source_snapshot') or {}).get('snapshot_id')
    for comparison in report.get('comparisons',[]):
        if comparison.get('performance'):
            db.execute('INSERT OR REPLACE INTO run_performance VALUES(?,?,?,?,?)',(comparison['run_id'],report.get('id'),snapshot,now(),json.dumps(comparison['performance'],allow_nan=False)))


def run_assessments(identifiers):
    identifiers=list(set(identifiers))
    if not identifiers:return {}
    with store.connection() as db:
        init(db)
        result={}
        for offset in range(0,len(identifiers),500):
            batch=identifiers[offset:offset+500]
            result.update({r['run_id']:json.loads(r['payload']) for r in db.execute('SELECT run_id,payload FROM run_performance WHERE run_id IN ('+','.join('?' for _ in batch)+')',batch)})
    return result


def update_job(identifier,status,details):
    with store.connection() as db:
        init(db);db.execute('UPDATE performance_jobs SET status=?,updated_at=?,details=? WHERE id=?',(status,now(),json.dumps(details,allow_nan=False),identifier))


def reassess_analysis(identifier):
    with store.connection() as db:
        engagements.init(db);init(db)
        row=db.execute('SELECT report,filename FROM analyses WHERE id=?',(identifier,)).fetchone()
    if not row:raise ValueError('Analysis not found')
    original=row['report'];report=json.loads(original);report['filename']=row['filename']
    assessment.enrich(report)
    revision=dict(analysis_id=identifier,performance=report['performance'],
        assessments=[dict(run_id=c['run_id'],performance=c['performance']) for c in report['comparisons'] if c.get('performance')],
        investigation=report['investigation'],previous_report_sha256=hashlib.sha256(original.encode()).hexdigest())
    payload=json.dumps(revision,allow_nan=False,separators=(',',':')).encode()
    sha=hashlib.sha256(payload).hexdigest();stamp=now()
    snapshot=(report['performance'].get('source_snapshot') or {}).get('snapshot_id')
    with store.connection() as db:
        init(db)
        current=db.execute('SELECT report FROM analyses WHERE id=?',(identifier,)).fetchone()
        if not current or current[0]!=original:raise ValueError('Analysis changed during reassessment; retry against the latest evidence')
        db.execute('UPDATE analyses SET report=? WHERE id=?',(json.dumps(report,allow_nan=False),identifier))
        db.execute('INSERT OR IGNORE INTO performance_history VALUES(?,?,?,?,?,?)',(sha,identifier,stamp,snapshot,gzip.compress(payload,mtime=0),sha))
        cache_assessments(db,report)
    return dict(analysis_id=identifier,revision=sha,performance=report['performance'],baseline_attainment=report['baseline_attainment'])


def reassess_all(job_id=None):
    with store.connection() as db:
        engagements.init(db);ids=[r[0] for r in db.execute('SELECT id FROM analyses ORDER BY created_at')]
    completed=[];errors=[]
    for identifier in ids:
        try:completed.append(reassess_analysis(identifier))
        except Exception as error:errors.append(dict(analysis_id=identifier,error=str(error) if isinstance(error,ValueError) else type(error).__name__))
        if job_id:update_job(job_id,'running',dict(completed=len(completed),total=len(ids),errors=errors))
    result=dict(completed=len(completed),total=len(ids),errors=errors)
    if job_id:update_job(job_id,'complete' if not errors else 'partial',result)
    return result


def run_job(identifier,kind,analysis_id=None):
    if not _work_lock.acquire(blocking=False):
        update_job(identifier,'failed',dict(error='Another reference job is running; retry after it completes'));return
    try:
        update_job(identifier,'running',{})
        if kind=='nccl-reassess':
            root=Path(__file__).resolve().parents[1]
            output=config.uploads_root()/'nccl-assessments'/(identifier+'.json')
            output.parent.mkdir(parents=True,exist_ok=True)
            process=subprocess.Popen([sys.executable,'-u',str(root/'tools/assess_nccl_catalog.py'),'--write','--output',str(output)],cwd=root,env={**os.environ,'VALIDATION_DB':str(store.DB_PATH)},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            for line in process.stdout:
                try:update_job(identifier,'running',json.loads(line))
                except ValueError:continue
            error=process.stderr.read();code=process.wait()
            if code:raise ValueError('NCCL catalog assessment failed; existing source evidence is retained. '+(error.splitlines()[-1][:300] if error else ''))
            catalog=json.loads(output.read_text());reports=reassess_all()
            update_job(identifier,'complete' if not reports['errors'] else 'partial',dict(catalog=catalog,reports=reports));return
        if kind=='sync':
            result=beacon.fetch_all(config.BEACON_API_URL+'?page_size=200&sort_by=job_timestamp_desc&include_total=true',
                progress=lambda p:update_job(identifier,'running',p))
            update_job(identifier,'running',dict(imported=result.get('imported'),phase='reassessing'))
        if analysis_id:
            result=reassess_analysis(analysis_id);update_job(identifier,'complete',result)
        else:reassess_all(identifier)
    except Exception as error:
        update_job(identifier,'failed',dict(error=str(error) if isinstance(error,ValueError) else type(error).__name__,instructions=beacon.INSTRUCTIONS))
    finally:_work_lock.release()


def queue(background,kind,analysis_id=None):
    identifier=uuid.uuid4().hex;stamp=now()
    with store.connection() as db:
        init(db);db.execute('INSERT INTO performance_jobs VALUES(?,?,?,?,?,?)',(identifier,kind,'queued',stamp,stamp,'{}'))
    background.add_task(run_job,identifier,kind,analysis_id)
    return dict(job_id=identifier,status='queued')


@router.get('/status')
def status():
    result=assessment.reference_status()
    with store.connection() as db:
        init(db);rows=[dict(r) for r in db.execute('SELECT * FROM performance_jobs ORDER BY created_at DESC LIMIT 10')]
    for row in rows:row['details']=json.loads(row['details'])
    result['jobs']=rows
    return result


@router.get('/references')
def references(gpu:str='',workload:str='',precision:str='',scale:int|None=None,release:str='',q:str='',limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
    return beacon.list_references(dict(gpu=gpu,workload=workload,precision=precision,scale=scale,release=release,q=q),limit,offset)


@router.get('/nccl/references')
def nccl_references(gpu:str='',fabric:str='',gpus:int|None=None,collective:str='',limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
    return beacon_nccl.list_references(dict(gpu=gpu,fabric=fabric,gpus=gpus,collective=collective),limit,offset)


@router.get('/nccl/status')
def nccl_status():return beacon_nccl.status()


@router.get('/confluence/status')
def confluence_status():return wiki_sources.status()


@router.get('/confluence/sources')
def confluence_sources(q:str='',category:str='',gpu:str='',limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
    return wiki_sources.sources(q,category,gpu,limit,offset)


@router.get('/confluence/knowledge')
def confluence_knowledge(q:str='',limit:int=Query(50,ge=1,le=200)):
    return wiki_sources.knowledge(q,limit)


@router.post('/nccl/import')
async def import_nccl_references(background:BackgroundTasks,file:UploadFile=File(...),source_url:str=config.GPUCOMMS_URL):
    from urllib.parse import urlsplit
    source=urlsplit(source_url)
    if source.scheme!='https' or source.hostname not in config.REFERENCE_HOSTS:raise HTTPException(422,'Use a source URL from the NCCL benchmark service or the referenced NVIDIA Confluence pages')
    content=await file.read(beacon.MAX_IMPORT_BYTES+1)
    if len(content)>beacon.MAX_IMPORT_BYTES:raise HTTPException(413,'NCCL reference export limit is 50 MiB')
    suffix=''.join(Path(file.filename or 'export.json').suffixes)[-30:]
    if suffix not in ('.json','.jsonl','.csv','.json.gz','.jsonl.gz','.csv.gz','.gz'):raise HTTPException(422,'Import JSON, JSONL, CSV or gzip-compressed NCCL reference exports')
    try:
        with tempfile.TemporaryDirectory(prefix='validation-nccl-reference-') as folder:
            path=Path(folder)/('reference'+suffix);path.write_bytes(content)
            result=beacon_nccl.import_file(path,source_url)
    except (ValueError,UnicodeError,TypeError,KeyError,OSError,EOFError) as error:raise HTTPException(422,str(error))
    result['reassessment']=queue(background,'nccl-reassess')
    return result


@router.post('/nccl/reassess')
def reassess_nccl(background:BackgroundTasks):return queue(background,'nccl-reassess')


@router.post('/import')
async def import_references(background:BackgroundTasks,file:UploadFile=File(...)):
    content=await file.read(beacon.MAX_IMPORT_BYTES+1)
    if len(content)>beacon.MAX_IMPORT_BYTES:raise HTTPException(413,'Reference export limit is 50 MiB')
    suffix=''.join(Path(file.filename or 'export.json').suffixes)[-30:]
    if suffix not in ('.json','.jsonl','.csv','.json.gz','.jsonl.gz','.csv.gz','.gz','.har'):raise HTTPException(422,'Import JSON, JSONL, CSV or gzip-compressed reference exports')
    try:
        with tempfile.TemporaryDirectory(prefix='validation-reference-') as folder:
            path=Path(folder)/('reference'+suffix);path.write_bytes(content)
            result=beacon.import_file(path)
    except (ValueError,UnicodeError,TypeError,KeyError,OSError,EOFError) as error:raise HTTPException(422,str(error))
    result['reassessment']=queue(background,'reassess')
    return result


@router.post('/sync')
def sync(background:BackgroundTasks):return queue(background,'sync')


@router.post('/reassess')
def reassess(background:BackgroundTasks,analysis_id:str=''):
    if analysis_id:
        with store.connection() as db:
            engagements.init(db)
            if not db.execute('SELECT 1 FROM analyses WHERE id=?',(analysis_id,)).fetchone():raise HTTPException(404,'Analysis not found')
    return queue(background,'reassess',analysis_id or None)


@router.get('/snapshot/{snapshot_id}')
def snapshot(snapshot_id:str):
    # Paths come only from the connector's own catalog, never the request path.
    if len(snapshot_id)!=32 or any(c not in '0123456789abcdef' for c in snapshot_id):raise HTTPException(404,'Snapshot not found')
    with beacon._connect() as db:row=db.execute('SELECT path,metadata_json FROM beacon_snapshots WHERE snapshot_id=?',(snapshot_id,)).fetchone()
    if not row:raise HTTPException(404,'Snapshot not found')
    metadata=json.loads(row['metadata_json']);path=Path(metadata.get('compact_path') or row['path'])
    if not path.is_file():raise HTTPException(404,'Snapshot artifact is unavailable')
    return FileResponse(path,media_type='application/gzip',filename='beacon-'+snapshot_id+'.jsonl.gz')
