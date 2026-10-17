"""Evidence-backed Beacon reference imports and an authenticated JSON connector.

No benchmark rows are seeded. Credentials stay in private local files, never in
reference payloads. Unknown columns survive unchanged in raw_json and snapshots.
"""
import base64
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
from http.cookies import SimpleCookie
import uuid
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler

from . import config
BASE_URL = config.BEACON_BASE_URL
MAX_IMPORT_BYTES = 50 * 1024**2
COMPACT_FIELDS=('date_utc','job_id','workload','gpu','scale','precision','release','source','tflops_gpu','seconds_per_step','cluster','username','row_action')
INSTRUCTIONS = ('Sign in to llmb.nvidia.com, open Beacon Runs, then DevTools > Network > Fetch/XHR and reload. '
                'Export the JSON response containing run rows, or export CSV/JSON from Beacon. '
                'For automatic refresh, configure the observed API URL and a private local '
                'BEACON_HEADERS_FILE (JSON request headers) or BEACON_COOKIE_FILE (Cookie header value); '
                'both must have mode 600. Do not paste credentials into chat.')
ALIASES = {
 'source_record_id': ['id','run_id','runId','_id'], 'date_utc': ['date_utc','dateUTC','date','timestamp','job_timestamp','created_at','createdAt','start_time','startTime'],
 'job_id': ['job_id','jobId','jobid','slurm_job_id','slurmJobId'],
 'workload': ['workload','workload_name','workloadName','benchmark'], 'model': ['model','model_name','modelName'],
 'model_size': ['model_size','modelSize','parameters'], 'gpu': ['gpu','gpu_type','gpuType','gpu_model','gpuModel','hardware'],
 'scale': ['scale','gpu_count','gpuCount','num_gpus','numGpus','gpus'], 'precision': ['precision','dtype','data_type'],
 'release': ['release','release_version','releaseVersion','llmb_version','llmbVersion','version'],
 'cluster': ['cluster','cluster_name','clusterName'], 'username': ['username','user','submitted_by','submittedBy','owner'],
 'tflops_gpu': ['tflops_gpu_avg','tflops_gpu','tflops_per_gpu','tflopsPerGpu','tflops','TFLOPS','TFLOP/s/GPU','model_tflops_gpu'],
 'seconds_per_step': ['seconds_per_step','secondsPerStep','sPerStep','s_per_step','s/step','step_time_s','stepTimeSeconds','iteration_time_s'],
 'row_action': ['row_action','rowAction','action','actions','url','href'],
 'global_batch_size': ['global_batch_size','globalBatchSize','global_batch','gbs'],
 'sequence_length': ['sequence_length','sequenceLength','seq_length','seqLength','seq_len'],
 'tensor_parallel': ['tensor_parallel','tensor_parallel_size','tensor_parallelism','tensorModelParallelSize','tp'],
 'pipeline_parallel': ['pipeline_parallel','pipeline_parallel_size','pipeline_parallelism','pipelineModelParallelSize','pp'],
 'context_parallel': ['context_parallel','context_parallel_size','context_parallelism','contextParallelSize','cp'],
 'expert_parallel': ['expert_parallel','expert_parallel_size','expert_model_parallel_size','expertModelParallelSize','ep'],
 'virtual_pipeline': ['virtual_pipeline','virtual_pipeline_size','virtual_pipeline_parallelism','virtualPipelineModelParallelSize','vp'],
 'expert_tensor_parallel': ['expert_tensor_parallel','expert_tensor_parallel_size','expertTensorParallelSize','etp'],
 'micro_batch_size': ['micro_batch_size','microBatchSize','micro_batch','mbs'],
}


def _utc(): return datetime.now(timezone.utc).isoformat()

def _db_path(db_path=None): return Path(db_path) if db_path else config.data_path('BEACON_DB','beacon.sqlite')

def _connect(db_path=None):
    path = _db_path(db_path); path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60); db.row_factory = sqlite3.Row
    db.executescript('''CREATE TABLE IF NOT EXISTS beacon_runs (ref_id TEXT PRIMARY KEY, source_url TEXT NOT NULL, source_record_id TEXT, gpu TEXT, workload TEXT, precision TEXT, scale INTEGER, release TEXT, normalized_json TEXT NOT NULL, snapshot_id TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS beacon_scope ON beacon_runs(gpu,workload,precision,scale);
    CREATE TABLE IF NOT EXISTS beacon_snapshots (snapshot_id TEXT PRIMARY KEY, created_at TEXT, source_url TEXT, records INTEGER, path TEXT, sha256 TEXT, metadata_json TEXT);
    CREATE TABLE IF NOT EXISTS beacon_state (key TEXT PRIMARY KEY, value TEXT);''')
    return db


def init(db_path=None):
    with _connect(db_path) as db: pass


def _flatten(raw, prefix=''):
    result = {}
    for key, value in raw.items():
        path = prefix+str(key)
        if isinstance(value,dict): result.update(_flatten(value,path+'.'))
        else: result[path] = value
    return result


def _number(value, integer=False):
    try:
        if isinstance(value,bool): return None
        n = float(str(value).replace(',',''))
        if not math.isfinite(n) or n < 0 or integer and not n.is_integer(): return None
        return int(n) if integer else n
    except (ValueError,TypeError): return None


def source_scope(source_url):
    parts=urlsplit(source_url)
    pagination={'page_size','sort_by','include_total','c_d','c_id','page','pageSize','offset','limit'}
    return urlunsplit(parts._replace(query=urlencode(sorted((k,v) for k,v in parse_qsl(parts.query) if k not in pagination)),fragment=''))


def normalize_record(raw, source_url=BASE_URL, refreshed_at=None):
    if not isinstance(raw,dict): raise ValueError('Each Beacon row must be a JSON object')
    flat = _flatten(raw); mapping = {}; result = {}
    def pick(names):
        for alias in names:
            for path,value in flat.items():
                if (path == alias or path.rsplit('.',1)[-1] == alias) and value is not None and value != '': return path,value
        return None,None
    for field,names in ALIASES.items():
        path,value = pick(names); result[field] = value
        if path is not None: mapping[field] = path
    # Some exports label GPU flavor as `gpus`; numeric `gpus` means scale.
    if result['gpu'] is None and isinstance(raw.get('gpus'),str) and _number(raw['gpus']) is None:
        result['gpu'] = raw['gpus']; mapping['gpu'] = 'gpus'
    for field in ('scale','global_batch_size','sequence_length','tensor_parallel','pipeline_parallel','context_parallel','expert_parallel','virtual_pipeline','expert_tensor_parallel','micro_batch_size'):
        result[field] = _number(result[field], integer=True)
    for field in ('tflops_gpu','seconds_per_step'): result[field] = _number(result[field])
    if result['seconds_per_step'] is None:
        path,value=pick(['step_time_ms','stepTimeMs','iteration_time_ms','time_ms'])
        n=_number(value)
        if n is not None: result['seconds_per_step']=n/1000;mapping['seconds_per_step']=path+' / 1000'
    value=result['date_utc']; result['date_utc']=None
    if value is not None:
        try:
            if isinstance(value,(int,float)): date=datetime.fromtimestamp(value/1000 if value>1e12 else value,timezone.utc)
            else: date=datetime.fromisoformat(str(value).replace('Z','+00:00'))
            if date.tzinfo is not None: result['date_utc']=date.astimezone(timezone.utc).isoformat()
        except (ValueError,OverflowError,OSError): pass
    for field in ('job_id','source_record_id'):
        if result[field] is not None:result[field]=str(result[field])
    if result['source_record_id'] is None and isinstance(result.get('row_action'),str):
        action=urlsplit(result['row_action'])
        match=re.fullmatch(r'/beacon-v2/runs/([A-Za-z0-9_-]{1,128})',action.path)
        if action.scheme=='https' and action.hostname==urlsplit(BASE_URL).hostname and match:
            result['source_record_id']=match[1];mapping['source_record_id']='row_action record identifier'
    if isinstance(result['gpu'],str):result['gpu']=result['gpu'].upper()
    if isinstance(result['precision'],str):result['precision']=result['precision'].lower()
    if result['workload'] is None and result['model'] is not None:
        result['workload']=str(result['model'])+('|'+str(result['model_size']) if result['model_size'] is not None else '')
        mapping['workload']='model + model_size'
    result['precision_base']=result['precision']
    variant=raw.get('precision_variant')
    if variant and result['precision']:
        result['precision']=result['precision']+'_'+str(variant).lower()
        mapping['precision']='precision + precision_variant'
    result['job_source']=raw.get('job_source')
    result['framework']=raw.get('framework_name')
    result['framework_version']=raw.get('framework_version')
    result['status']=raw.get('status')
    if result['row_action'] is None and result['source_record_id']:
        result['row_action']=BASE_URL+'/'+result['source_record_id']
        mapping['row_action']='Beacon UI route /beacon-v2/runs/{id}'
    result['gpus']=result['scale']
    if result['model'] is None and isinstance(result['workload'],str):
        parts=result['workload'].split('|',1);result['model']=parts[0]
        if len(parts)==2 and result['model_size'] is None:result['model_size']=parts[1]
    raw_json=json.dumps(raw,sort_keys=True,separators=(',',':'),allow_nan=False)
    identity=result['source_record_id'] or hashlib.sha256(raw_json.encode()).hexdigest()
    source_identity=urlunsplit(urlsplit(source_url)._replace(query='',fragment=''))
    if source_identity in (BASE_URL,'https://api.llmb.nvidia.com/v2/training/runs'):
        source_identity='https://api.llmb.nvidia.com/v2/training/runs'
    result.update(ref_id=hashlib.sha256((source_identity+'\0'+identity).encode()).hexdigest()[:32],source='Beacon',
                  source_url=source_url,source_scope=source_scope(source_url),refreshed_at=refreshed_at or _utc(),raw_json=raw,
                  raw_sha256=hashlib.sha256(raw_json.encode()).hexdigest(),source_mapping=mapping)
    return result


def import_records(rows, source_url=BASE_URL, db_path=None, metadata=None, max_bytes=MAX_IMPORT_BYTES):
    now=_utc(); normalized=[]; raw_lines=[]; total_bytes=0
    for row in rows:
        raw=json.dumps(row,separators=(',',':'),allow_nan=False)+'\n';total_bytes+=len(raw.encode())
        if total_bytes>max_bytes:raise ValueError('Beacon import exceeds 50 MiB expanded text')
        raw_lines.append(raw);normalized.append(normalize_record(row,source_url,now))
    if not normalized: raise ValueError('No Beacon reference rows found; existing references preserved')
    unique_count=len({r['ref_id'] for r in normalized})
    complete=bool((metadata or {}).get('complete_all_pages'))
    if complete and (metadata or {}).get('reported_total') is not None and unique_count!=(metadata or {})['reported_total']:
        raise ValueError('Unique records do not equal the reported total; existing references preserved')
    payload=''.join(raw_lines).encode();sha=hashlib.sha256(payload).hexdigest();snapshot_id=hashlib.sha256((source_url+'\0'+sha).encode()).hexdigest()[:32]
    root=_db_path(db_path).parent/'beacon-snapshots';root.mkdir(parents=True,exist_ok=True)
    path=root/(sha+'.jsonl.gz')
    if not path.exists():
        temp=path.with_suffix('.partial');temp.write_bytes(gzip.compress(payload,mtime=0));temp.replace(path)
    compact_payload=''.join(json.dumps({**{key:r.get(key) for key in COMPACT_FIELDS},'source':r.get('job_source') or r.get('source')},separators=(',',':'),allow_nan=False)+'\n' for r in normalized).encode()
    compact_bytes=gzip.compress(compact_payload,mtime=0);compact_sha=hashlib.sha256(compact_bytes).hexdigest()
    compact_path=root/(compact_sha+'.compact.jsonl.gz')
    if not compact_path.exists():
        temp=compact_path.with_suffix('.partial');temp.write_bytes(compact_bytes);temp.replace(compact_path)
    meta=dict(metadata or {},source_scope=source_scope(source_url),unique_records=unique_count,duplicate_records=len(normalized)-unique_count,
              compact_path=str(compact_path),compact_sha256=compact_sha,compact_fields=list(COMPACT_FIELDS),compact_bytes=len(compact_bytes),
              gpu_types=sorted({r['gpu'] for r in normalized if r.get('gpu')}),
              compressed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),expanded_bytes=len(payload),compressed_bytes=path.stat().st_size,raw_format='JSONL',complete_scope='Imported rows only; source filters and pagination determine coverage.')
    with _connect(db_path) as db:
        db.execute('INSERT OR REPLACE INTO beacon_snapshots VALUES(?,?,?,?,?,?,?)',(snapshot_id,now,source_url,len(normalized),str(path),sha,json.dumps(meta)))
        for r in normalized:
            db.execute('INSERT OR REPLACE INTO beacon_runs VALUES(?,?,?,?,?,?,?,?,?,?)',(r['ref_id'],source_url,r['source_record_id'],r['gpu'],r['workload'],r['precision'],r['scale'],r['release'],json.dumps(r,allow_nan=False),snapshot_id))
        if complete:
            db.execute("DELETE FROM beacon_runs WHERE json_extract(normalized_json,'$.source_scope')=? AND snapshot_id!=?",(source_scope(source_url),snapshot_id))
        db.execute('INSERT OR REPLACE INTO beacon_state VALUES(?,?)',('last_sync',now))
        db.execute('INSERT OR REPLACE INTO beacon_state VALUES(?,?)',('status','imported'))
        db.execute("DELETE FROM beacon_state WHERE key='error'")
    return dict(imported=len(normalized),unique_records=unique_count,compact_path=str(compact_path),compact_sha256=compact_sha,snapshot_id=snapshot_id,sha256=sha,path=str(path),last_sync=now)


def extract_rows(payload):
    if isinstance(payload,list): return payload
    if not isinstance(payload,dict):raise ValueError('Expected a JSON list or an object containing run rows')
    for key in ('runs','rows','results','items','records','data'):
        value=payload.get(key)
        if isinstance(value,list):return value
        if isinstance(value,dict):
            try:return extract_rows(value)
            except ValueError:pass
    raise ValueError('JSON response has no recognizable run-row list')


def import_file(path, source_url=BASE_URL, db_path=None):
    path=Path(path)
    if path.stat().st_size>MAX_IMPORT_BYTES:raise ValueError('Beacon import exceeds 50 MiB')
    with path.open('rb') as stream: magic=stream.read(2)
    opener=gzip.open if magic==b'\x1f\x8b' else open
    with opener(path,'rb') as stream:data=stream.read(MAX_IMPORT_BYTES+1)
    if len(data)>MAX_IMPORT_BYTES:raise ValueError('Beacon import exceeds 50 MiB expanded text')
    text=data.decode('utf-8-sig');name=path.name.lower().removesuffix('.gz')
    if name.endswith('.csv'): rows=list(csv.DictReader(io.StringIO(text)))
    elif name.endswith(('.jsonl','.ndjson')):rows=[json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload=json.loads(text)
        if isinstance(payload,dict) and 'log' in payload and isinstance(payload['log'],dict):
            rows=[]
            for entry in payload['log'].get('entries',[]):
                request_path=urlsplit(entry.get('request',{}).get('url','')).path.rstrip('/')
                if request_path!='/v2/training/runs':continue
                content=entry.get('response',{}).get('content',{});body=content.get('text','')
                if content.get('encoding')=='base64':body=base64.b64decode(body).decode()
                try:rows.extend(extract_rows(json.loads(body)))
                except (ValueError,TypeError):pass
        else:rows=extract_rows(payload)
    return import_records(rows,source_url,db_path,metadata={'import_filename':path.name,'import_sha256':hashlib.sha256(data).hexdigest(),'method':'offline_export'})


def list_references(filters=None, limit=100, offset=0, db_path=None):
    filters=filters or {};where=[];values=[]
    for key in ('gpu','workload','precision','scale','release'):
        if filters.get(key) not in (None,''):
            where.append(key+'=?');values.append(filters[key])
    if filters.get('q'):
        where.append('normalized_json LIKE ?');values.append('%'+str(filters['q'])+'%')
    clause=' WHERE '+' AND '.join(where) if where else ''
    limit=max(1,min(int(limit),10000));offset=max(0,int(offset))
    with _connect(db_path) as db:
        total=db.execute('SELECT COUNT(*) FROM beacon_runs'+clause,values).fetchone()[0]
        items=[json.loads(row[0]) for row in db.execute('SELECT normalized_json FROM beacon_runs'+clause+' ORDER BY ref_id LIMIT ? OFFSET ?',values+[limit,offset])]
        facets={key:[row[0] for row in db.execute('SELECT DISTINCT '+key+' FROM beacon_runs WHERE '+key+' IS NOT NULL ORDER BY '+key)] for key in ('gpu','workload','precision','scale','release')}
        row=db.execute('SELECT * FROM beacon_snapshots ORDER BY created_at DESC LIMIT 1').fetchone()
    snapshot=dict(row) if row else None
    if snapshot:snapshot['metadata']=json.loads(snapshot.pop('metadata_json'))
    return dict(items=items,total=total,limit=limit,offset=offset,facets=facets,snapshot=snapshot)


def references(db_path=None,filters=None):
    page=list_references(filters,10000,0,db_path);items=page['items']
    while len(items)<page['total']:items.extend(list_references(filters,10000,len(items),db_path)['items'])
    return items


def status(db_path=None):
    page=list_references(limit=1,db_path=db_path)
    with _connect(db_path) as db: state=dict(db.execute('SELECT key,value FROM beacon_state'))
    return dict(status=state.get('status','login_required'),authenticated=bool(page['snapshot'] and page['snapshot']['metadata'].get('method')=='authenticated_api'),records=page['total'],
                gpu_types=page['facets']['gpu'],last_sync=state.get('last_sync'),error=state.get('error'),
                instructions=INSTRUCTIONS,source_url=BASE_URL,snapshot=page['snapshot'])


def private_headers():
    headers={'Accept':'application/json'}
    for env in ('BEACON_HEADERS_FILE','BEACON_COOKIE_FILE'):
        name=os.environ.get(env)
        if not name:continue
        path=Path(name);mode=stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:raise ValueError(env+' must be private (chmod 600)')
        if env=='BEACON_HEADERS_FILE':
            data=json.loads(path.read_text())
            if not isinstance(data,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in data.items()):raise ValueError('Headers file must contain a JSON object of strings')
            headers.update(data)
        else:headers['Cookie']=path.read_text().strip()
    return headers


class _SameOriginRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if urlsplit(newurl).scheme!='https' or urlsplit(req.full_url).netloc!=urlsplit(newurl).netloc or '/signin' in urlsplit(newurl).path:raise ValueError('Beacon authentication required; sign in and refresh the private credential file')
        return super().redirect_request(req,fp,code,msg,headers,newurl)




def _persist_session_cookies(response):
    name=os.environ.get('BEACON_COOKIE_FILE')
    if not name:return
    updates={}
    for header in response.headers.get_all('Set-Cookie',[]):
        cookie=SimpleCookie();cookie.load(header)
        for key,morsel in cookie.items():
            if re.search(r'(?:next-auth|authjs)\.session-token(?:\.\d+)?$',key):updates[key]=morsel.value
    if not updates:return
    path=Path(name)
    current=SimpleCookie();current.load(path.read_text().strip())
    values={key:m.value for key,m in current.items()}
    # Rotated chunked sessions must replace every old session-token chunk.
    values={key:value for key,value in values.items() if not re.search(r'(?:next-auth|authjs)\.session-token(?:\.\d+)?$',key)}
    values.update(updates)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w') as stream:stream.write('; '.join(key+'='+value for key,value in values.items())+'\n')
        temp.replace(path)
    finally:
        if temp.exists():temp.unlink()


def api_headers(api_url):
    headers=private_headers()
    if urlsplit(api_url).hostname==urlsplit(config.BEACON_API_URL).hostname and not any(key.lower()=='authorization' for key in headers):
        # This is the same supported session exchange performed by Beacon's UI.
        opener=build_opener(_SameOriginRedirect())
        with opener.open(Request(urlunsplit((urlsplit(BASE_URL).scheme,urlsplit(BASE_URL).netloc,'/api/auth/session','','')),headers=headers),timeout=30) as response:
            session=json.load(response)
            _persist_session_cookies(response)
        token=session.get('idToken')
        if not token:raise ValueError('Beacon session expired; sign in and refresh the private cookie file')
        headers.pop('Cookie',None);headers['Authorization']='Bearer '+token
    return headers


def _fetch_all(api_url, db_path=None, page_param=None, page_size_param=None, page_size=200, max_pages=10000, progress=None):
    if urlsplit(api_url).scheme!='https' or urlsplit(api_url).hostname not in config.BEACON_HOSTS:raise ValueError('Use the confirmed HTTPS llmb.nvidia.com JSON API URL from browser Network')
    if any(re.search(r'token|secret|cookie|password',k,re.I) for k,_ in parse_qsl(urlsplit(api_url).query)):raise ValueError('Put credentials in a private headers file, not the API URL')
    headers=api_headers(api_url);opener=build_opener(_SameOriginRedirect());url=api_url;seen=set();rows=[];pages=0;expected_total=None
    while url:
        if url in seen:raise ValueError('Pagination repeated a URL; import stopped without updating references')
        seen.add(url);pages+=1
        if pages>max_pages:raise ValueError('Pagination safety limit reached; existing references preserved')
        with opener.open(Request(url,headers=headers),timeout=45) as response:
            if 'json' not in response.headers.get('Content-Type',''):raise ValueError('Beacon returned a login page or non-JSON response')
            data=response.read(MAX_IMPORT_BYTES+1)
            if len(data)>MAX_IMPORT_BYTES:raise ValueError('Beacon page exceeds 50 MiB')
            payload=json.loads(data)
        batch=extract_rows(payload);rows.extend(batch)
        if progress:progress({'pages':pages,'records':len(rows)})
        if isinstance(payload,dict):
            total=payload.get('total',payload.get('totalCount'))
            if isinstance(total,int):expected_total=total
            cursor=payload.get('next_cursor')
            next_link=payload.get('next') or payload.get('next_url') or (payload.get('links') or {}).get('next')
        else:next_link=None;cursor=None
        if next_link:
            if not isinstance(next_link,str):raise ValueError('Unsupported next-page format; configure the observed API pagination')
            url=urljoin(url,next_link)
            if urlsplit(url).scheme!='https' or urlsplit(url).netloc!=urlsplit(api_url).netloc:raise ValueError('Cross-origin pagination refused')
        elif cursor:
            parts=urlsplit(api_url);query=dict(parse_qsl(parts.query));query.update(c_d='after',c_id=str(cursor),include_total='false')
            url=urlunsplit(parts._replace(query=urlencode(query)))
        elif page_param and batch and (expected_total is None and len(batch)>=page_size or expected_total is not None and len(rows)<expected_total):
            parts=urlsplit(api_url);query=dict(parse_qsl(parts.query));query[page_param]=str(pages+1)
            if page_size_param:query[page_size_param]=str(page_size)
            url=urlunsplit(parts._replace(query=urlencode(query)))
        else:url=None
    if expected_total is not None and len(rows)<expected_total:raise ValueError('Response is paginated; provide the observed page parameter or next-page API URL')
    return import_records(rows,api_url,db_path,metadata={'method':'authenticated_api','pages':pages,'reported_total':expected_total,'complete_all_pages':True,'filters':dict(parse_qsl(urlsplit(source_scope(api_url)).query)),'all_gpu_flavors':not any(k in dict(parse_qsl(urlsplit(api_url).query)) for k in ('gpus','gpu'))},max_bytes=512*1024**2)


def fetch_all(api_url, db_path=None, page_param=None, page_size_param=None, page_size=200, max_pages=10000, progress=None):
    try:
        return _fetch_all(api_url,db_path,page_param,page_size_param,page_size,max_pages,progress)
    except Exception as exc:
        # Preserve last successful records and snapshot on every refresh error.
        error=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
        with _connect(db_path) as db:
            for key,value in [('status','refresh_failed'),('error',error),('last_attempt',_utc())]:
                db.execute('INSERT OR REPLACE INTO beacon_state VALUES(?,?)',(key,value))
        raise
