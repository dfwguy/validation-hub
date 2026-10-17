from . import config
import json,hashlib,shutil,os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,UploadFile,File,Form,Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from . import store
from .parser import parse_log
from .compare import compare,eligible,baseline_identity
from .security import scrub
ROOT=Path(__file__).resolve().parents[1]
@asynccontextmanager
async def lifespan(application):
    from .performance_api import init
    with store.connection() as db:
        init(db)
        db.execute("UPDATE performance_jobs SET status='failed',details=? WHERE status IN ('queued','running')",(json.dumps({'error':'Service restarted before this job completed; retry the reference refresh or reassessment. Existing references remain available.'}),))
    yield

app=FastAPI(title='Cluster Validation Hub',version='2.0',lifespan=lifespan)
POLICY='Baselines require complete historical or explicitly reviewed uploaded NCCL results, zero correctness errors and exact nodes, GPUs, hardware, transport, collective, NCCL version, message size, datatype, operation and placement. Source-reported parquet and scheduler verdicts remain distinct.'

def storage():
    result={}
    for p in (config.data_root()/'catalog/storage-inventory.json',):
        if p.exists():
            try:result=json.loads(p.read_text());break
            except ValueError:pass
    mirror=config.data_root()/'catalog/mirror-status.json'
    if mirror.exists():
        try: result['mirror_status']=json.loads(mirror.read_text())
        except ValueError:pass
    result['local_archive_count']=sum(1 for _ in (config.data_root()/'raw').rglob('*.tar.gz')) if (config.data_root()/'raw').exists() else 0
    disk_root=config.data_root();disk_root.mkdir(parents=True,exist_ok=True)
    usage=shutil.disk_usage(disk_root)
    result['disk']={'path':str(disk_root),'total_bytes':usage.total,'used_bytes':usage.used,'free_bytes':usage.free}
    with store.connection() as db:result['ingested_uploads']=lake.inventory(db)
    return result

@app.get('/api/health')
def health():return {'ok':True,'database':store.DB_PATH}

@app.get('/api/storage')
def get_storage():return storage()

@app.get('/api/summary')
def summary():
    with store.connection() as db:
        total=db.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
        clusters=[dict(name=r[0],runs=r[1]) for r in db.execute('SELECT cluster,COUNT(*) FROM runs GROUP BY cluster')]
        workloads=[dict(name=r[0],runs=r[1]) for r in db.execute('SELECT workload,COUNT(*) FROM runs GROUP BY workload ORDER BY COUNT(*) DESC')]
        sources=db.execute('SELECT COUNT(*) FROM sources').fetchone()[0]
        eligible_runs=[json.loads(r[0]) for r in db.execute("SELECT payload FROM runs WHERE source_kind='raw' AND status='passed' AND workload='NCCL'")]
        lake.init(db)
        for row in db.execute('SELECT r.payload FROM runs r JOIN reference_reviews v ON v.run_id=r.id WHERE v.active=1'):
            candidate=json.loads(row[0]);candidate['_reference_review']={'active':1};eligible_runs.append(candidate)
        qualified=len({baseline_identity(r) for r in eligible_runs if eligible(r)})
        max_nodes,max_gpus=db.execute('SELECT MAX(nodes),MAX(gpus) FROM runs').fetchone()
        measured=db.execute("SELECT COUNT(*) FROM runs WHERE source_kind IN ('raw','upload') AND (avg_bus_bw_gbps IS NOT NULL OR iterations IS NOT NULL)").fetchone()[0]
    return dict(runs=total,measurements=measured,clusters=clusters,workloads=workloads,sources=sources,qualified_baselines=qualified,max_nodes=max_nodes,max_gpus=max_gpus,baseline_policy=POLICY,storage=storage())

@app.get('/api/facets')
def facets():
    names={'clusters':'cluster','workloads':'workload','collectives':'collective','hardware':'hardware','statuses':'status','node_counts':'nodes','nccl_versions':'nccl_version','transports':'transport'}
    with store.connection() as db:return {name:[r[0] for r in db.execute('SELECT DISTINCT '+col+' FROM runs WHERE '+col+' IS NOT NULL ORDER BY '+col)] for name,col in names.items()}

@app.get('/api/runs')
def runs(q:str='',measured_only:bool=False,rack_ids:str='',su_ids:str='',nodelist:str='',cluster:str='',workload:str='',collective:str='',nodes:int|None=None,min_nodes:int|None=None,max_nodes:int|None=None,gpus:int|None=None,racks:float|None=None,sus:float|None=None,hardware:str='',status:str='',nccl_version:str='',transport:str='',limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
    try:
        result=store.search(locals(),limit,offset)
        from .performance_api import run_assessments
        assessments=run_assessments([r['id'] for r in result['items']])
        for item in result['items']:
            evaluation=assessments.get(item['id'])
            if evaluation:
                # Lists need the rating and the weakest normalized points;
                # full evidence remains available in the run detail.
                summary={k:evaluation.get(k) for k in ('classification','candidate_band','tier','verdict','missing_context','mismatches','policy','provenance','nccl_summary')}
                metrics=sorted(evaluation.get('metrics',[]),key=lambda m:m.get('expected_pct',float('inf')))
                summary.update(metrics=metrics[:2],point_count=len(metrics))
                item['performance']=summary
        return result
    except ValueError as error: raise HTTPException(422,str(error))

@app.get('/api/runs/{id}')
def run(id:str):
    result=store.detail(id)
    if not result:raise HTTPException(404,'Run not found')
    from .performance_api import run_assessments
    evaluation=run_assessments([id]).get(id)
    if evaluation:result['performance']=evaluation
    return scrub(result)

@app.post('/api/compare')
async def upload(file:UploadFile=File(...),cluster:str=Form(''),nodes:int|None=Form(None),gpus:int|None=Form(None),hardware:str=Form(''),transport:str=Form(''),nccl_version:str=Form(''),collective:str=Form(''),tolerance_pct:float=Form(10)):
    if not 0<=tolerance_pct<=100:raise HTTPException(422,'Tolerance must be between 0 and 100 percent')
    content=await file.read(32*1024*1024+1)
    if len(content)>32*1024*1024:raise HTTPException(413,'Upload limit is 32 MiB per log')
    if b'\x00' in content:raise HTTPException(422,'Upload a UTF-8 NCCL text log')
    try:text=content.decode('utf-8')
    except UnicodeDecodeError:raise HTTPException(422,'Upload a UTF-8 NCCL text log')
    run=parse_log(text.splitlines(),file.filename or 'uploaded.log')
    if run['workload']!='NCCL' or not any('busbw_gbps' in m for m in run['metrics']):raise HTTPException(422,'No nccl-tests performance rows were found in this file')
    warnings=[]
    provided=dict(nodes=nodes,gpus=gpus,hardware=hardware or None,transport=transport or None,nccl_version=nccl_version or None,collective=collective or None)
    for key,value in provided.items():
        if value is not None:
            if run.get(key) is not None and run[key]!=value:raise HTTPException(422,f'{key} contradicts metadata measured in the uploaded log ({run[key]})')
            run[key]=value
    run['metadata']['user_supplied_fields']=[k for k,v in provided.items() if v is not None]
    sha=hashlib.sha256(content).hexdigest();path=config.uploads_root()/(sha+'.log');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    run=store.put_run(run,path,sha,len(content),cluster or 'User uploads','upload')
    result=compare(run,cluster or None,tolerance_pct)
    result['performance']=assessment.nccl(result)
    from .performance_api import init,cache_assessments
    with store.connection() as db:
        init(db);cache_assessments(db,{'comparisons':[{'run_id':run['id'],'performance':result['performance']}]})
    result['policy']['user_supplied_fields']=run['metadata']['user_supplied_fields']
    return result

# Troubleshooting and customer workflows share the existing evidence catalog.
from .engagements import router as engagements_router
from . import engagements, recipes, troubleshooting, lake, upload_io, assessment, benchmark_context
from .performance_api import router as performance_router
from fastapi.responses import PlainTextResponse
import uuid, zipfile, tarfile, zstandard
app.include_router(engagements_router)
app.include_router(lake.router)
app.include_router(performance_router)

@app.get('/api/recipes')
def get_recipes():return dict(items=recipes.profiles(),version=recipes.VERSION)

@app.post('/api/nccl-script')
def nccl_script(req:recipes.ScriptRequest):
    try:return recipes.generate(req)
    except ValueError as e:raise HTTPException(422,str(e))

@app.get('/api/system-check-script')
def system_check_script():
    return PlainTextResponse(SYSTEM_CHECK_SCRIPT,headers={'Content-Disposition':'attachment; filename="validation-system-checks.sh"'})

SYSTEM_CHECK_SCRIPT = r'''#!/usr/bin/env bash
# Run on each allocated node; capture stdout/stderr to a .log file.
# Within a Slurm allocation: srun --ntasks-per-node=1 bash validation-system-checks.sh
set +x
set -u
hostname
for key in NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_IB_ROCE_VERSION_NUM NCCL_IB_SL NCCL_NET NCCL_IB_ECE_ENABLE NCCL_TESTS_SPLIT_MASK NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_NVLS_ENABLE NCCL_MNNVL_ENABLE; do
  if [[ -v "$key" ]]; then printf '%s=%s\n' "$key" "${!key}"; fi
done
for cmd in 'lscpu' 'numactl -H' 'nvidia-smi -q' 'nvidia-smi topo -m' 'ibstat' 'ibdev2netdev' 'rdma link show' 'lspci -vv'; do
  printf '\n=== %s ===\n' "$cmd"
  bash -c "$cmd" || echo "UNAVAILABLE: $cmd"
done
printf '\n=== CPU governors ===\n'
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
printf '\n=== NUMA balancing ===\n'
cat /proc/sys/kernel/numa_balancing 2>/dev/null || true
'''

@app.post('/api/analyze')
async def analyze_upload(file:UploadFile=File(...),recipe_id:str=Form(...),customer_id:str=Form(''),engagement_id:str=Form(''),cluster:str=Form(''),reference_cluster:str=Form(''),tolerance_pct:float=Form(10),nodes:int|None=Form(None),gpus:int|None=Form(None),hardware:str=Form(''),transport:str=Form(''),nccl_version:str=Form(''),context_json:str=Form('')):
    content=await run_in_threadpool(upload_io.receive,file.file,file.filename or 'upload.log')
    try:
        return await run_in_threadpool(ingest_upload,content,file.filename or 'upload.log',recipe_id,customer_id,engagement_id,cluster,reference_cluster,tolerance_pct,nodes,gpus,hardware,transport,nccl_version,context_json)
    except HTTPException as error:
        error.detail=f'{error.detail}. Original upload retained on server (SHA-256: {content.stem}).'
        raise

def ingest_upload(content,filename,recipe_id,customer_id='',engagement_id='',cluster='',reference_cluster='',tolerance_pct=10,nodes=None,gpus=None,hardware='',transport='',nccl_version='',context_json=''):
    import math
    try:declared_context=benchmark_context.parse(context_json)
    except ValueError as error:raise HTTPException(422,str(error))
    cluster=cluster.strip() or 'User uploads'
    reference_cluster=reference_cluster.strip()
    if len(cluster)>120 or any(ord(c)<32 for c in cluster):raise HTTPException(422,'Cluster name must be at most 120 characters without control characters')
    if not math.isfinite(tolerance_pct) or not 0<=tolerance_pct<=100:raise HTTPException(422,'Tolerance must be between 0 and 100 percent')
    if any(v is not None and v<=0 for v in (nodes,gpus)):raise HTTPException(422,'Node and GPU counts must be positive')
    with store.connection() as db:
        engagements.init(db);lake.init(db)
        from .performance_api import init as init_performance
        init_performance(db)
        if customer_id and not db.execute('SELECT 1 FROM customers WHERE id=?',(customer_id,)).fetchone():raise HTTPException(404,'Customer not found')
        if engagement_id:
            engagements.validate_owner(db,customer_id,engagement_id)
            er=db.execute('SELECT recipe_id FROM engagements WHERE id=?',(engagement_id,)).fetchone()
            if er[0]!=recipe_id:raise HTTPException(422,'Selected recipe does not match the engagement recipe')
        for c in db.execute('SELECT id,clusters FROM customers'):
            if cluster in json.loads(c['clusters']) and customer_id and c['id']!=customer_id:raise HTTPException(409,'Cluster is assigned to another customer')
    content_size=content.stat().st_size if isinstance(content,Path) else len(content)
    if content_size>upload_io.UPLOAD_LIMIT:raise HTTPException(413,'Upload limit is 1 GB per file')
    if not content_size:raise HTTPException(422,'Upload is empty')
    try:
        entries=troubleshooting.texts(content,filename or 'upload.log')
        report=troubleshooting.analyze(entries,recipe_id)
    except (ValueError,zipfile.BadZipFile,tarfile.TarError,zstandard.ZstdError,EOFError,OSError,RuntimeError) as e:raise HTTPException(422,str(e))
    provided=dict(nodes=nodes,gpus=gpus,hardware=hardware.strip() or None,transport=transport.strip() or None,nccl_version=nccl_version.strip() or None)
    # Validate the complete bundle before writing catalog records or source files.
    members=report.pop('runs')
    for entry in members:
        parsed=entry['parsed'];supplied=[]
        parsed['metadata']['upload_filename']=filename
        try:benchmark_context.attach(parsed,declared_context)
        except ValueError as error:raise HTTPException(422,f"{entry['filename']}: {error}")
        for key,value in provided.items():
            if value is None:continue
            observed=parsed.get(key)
            inferred=key in parsed['metadata'].get('inferred_fields',[])
            if observed is not None and observed!=value and (not inferred or value<observed):raise HTTPException(422,f"{entry['filename']}: {key} contradicts observed log context ({observed})")
            parsed[key]=value;supplied.append(key)
        parsed['metadata']['user_supplied_fields']=supplied
        if parsed['metadata'].get('mixed_split_masks'):parsed['status']='incomplete'
    sha=upload_io.digest_file(content) if isinstance(content,Path) else hashlib.sha256(content).hexdigest()
    batch_id=hashlib.sha256((sha+'\0'+cluster).encode()).hexdigest()[:32]
    context=json.dumps([recipe_id,recipes.VERSION,customer_id,engagement_id,cluster,reference_cluster,tolerance_pct,provided]+([declared_context] if declared_context else []),sort_keys=True)
    id=hashlib.sha256((sha+context).encode()).hexdigest()[:32]
    # Ownership validation precedes upserts, including for repeated uploads.
    with store.connection() as db:
        for entry in members:
            run_id=hashlib.sha256((entry['sha256']+cluster+'upload').encode()).hexdigest()[:24]
            owner=db.execute('SELECT customer_id,engagement_id FROM customer_evidence WHERE run_id=?',(run_id,)).fetchone()
            if owner and customer_id and (owner[0]!=customer_id or (owner[1] and engagement_id and owner[1]!=engagement_id)):raise HTTPException(409,'Identical run is already assigned to another engagement')
            previous=db.execute('SELECT payload FROM runs WHERE id=?',(run_id,)).fetchone()
            if previous:
                old=json.loads(previous[0])
                try:benchmark_context.attach(old,declared_context)
                except ValueError as error:raise HTTPException(409,f'Existing benchmark context differs: {error}')
                # Same content cannot silently change its declared allocation or transport on retry.
                for key,value in provided.items():
                    if value is None:continue
                    declared=key not in old.get('metadata',{}).get('inferred_fields',[]) or key in old.get('metadata',{}).get('user_supplied_fields',[])
                    if declared and old.get(key) is not None and entry['parsed'].get(key)!=old[key]:raise HTTPException(409,f'Existing run context differs for {key}; retain the original evidence and correct the source log')
    base=config.uploads_root()
    root=base/hashlib.sha256(cluster.encode()).hexdigest()[:20]
    root.mkdir(parents=True,exist_ok=True)
    suffix=next((s for s in ('.tar.gz','.tar.zst','.tar.zg','.tzst','.tgz','.zip','.tar') if (filename or '').lower().endswith(s)),'.log')
    original=root/(sha+suffix)
    if not original.exists():
        if isinstance(content,Path):shutil.copyfile(content,original)
        else:original.write_bytes(content)
    comparisons=[]
    for ordinal,(entry,(_,member_text)) in enumerate(zip(members,entries)):
        path=root/(entry['sha256']+'.log')
        if not path.exists():
            if isinstance(member_text,upload_io.DiskLog):shutil.copyfile(member_text.path,path)
            else:path.write_bytes(member_text.encode('utf-8'))
        parsed=entry['parsed']
        parsed['metadata']['upload_batch_id']=batch_id
        parsed['metadata']['archive_member']=entry['filename']
        # Preserve prior audit metadata on an idempotent retry.
        run_id=hashlib.sha256((entry['sha256']+cluster+'upload').encode()).hexdigest()[:24]
        run=store.detail(run_id)
        if run:
            changed=False
            if declared_context:
                benchmark_context.attach(run,declared_context);changed=True
            for key,value in provided.items():
                if value is not None:
                    inferred=key in run['metadata'].get('inferred_fields',[]) and key not in run['metadata'].get('user_supplied_fields',[])
                    if run.get(key) is None or inferred:run[key]=value;changed=True
                    if key not in run['metadata'].get('user_supplied_fields',[]):changed=True
            if changed:
                run['metadata']['user_supplied_fields']=sorted(set(run['metadata'].get('user_supplied_fields',[])+[k for k,v in provided.items() if v is not None]))
                with store.connection() as db:db.execute('UPDATE runs SET nodes=?,gpus=?,hardware=?,transport=?,nccl_version=?,payload=? WHERE id=?',(run.get('nodes'),run.get('gpus'),run.get('hardware'),run.get('transport'),run.get('nccl_version'),json.dumps(run,allow_nan=False),run_id))
        else:run=store.put_run(parsed,path,entry['sha256'],entry['size_bytes'],cluster,'upload',display_name=Path(entry['filename']).name)
        outcome=compare(run,reference_cluster or None,tolerance_pct,excluded_hashes={e['sha256'] for e in members})
        comparisons.append(dict(filename=entry['filename'],run_id=run['id'],measurement=lake.metrics_summary(run),**outcome))
        store.source(path,entry['sha256'],entry['size_bytes'],'upload_member',cluster,dict(original_sha256=sha,member=entry['filename'],batch_id=batch_id))
    stamp=engagements.now()
    report['follow_up_tests']=sorted(lake.recommendations(comparisons)+report['follow_up_tests'],key=lambda t:t['priority'])
    report.update(id=id,filename=filename,customer_id=customer_id or None,engagement_id=engagement_id or None,cluster=cluster,reference_cluster=reference_cluster or None,comparisons=comparisons,baseline_attainment=lake.attainment(comparisons),created_at=stamp,
                  ingestion=dict(batch_id=batch_id,original_sha256=sha,original_bytes=content_size,member_count=len(members),measured_runs=sum(bool(c['run']['metrics']) for c in comparisons),reference_policy='Uploads are observations. Reusable references require an explicit engineer review; the current upload is excluded from its own comparison.'))
    if report['baseline_attainment']['verdict']=='fail':report['verdict']='fail'
    assessment.enrich(report)
    store.source(original,sha,content_size,'upload_batch',cluster,dict(batch_id=batch_id,filename=filename,member_count=len(members)))
    with store.connection() as db:
        db.execute('INSERT OR IGNORE INTO upload_batches VALUES(?,?,?,?,?,?,?,?,?)',(batch_id,cluster,sha,filename or 'upload.log',str(original),content_size,len(members),report['ingestion']['measured_runs'],stamp))
        for ordinal,entry in enumerate(members):
            db.execute('INSERT OR IGNORE INTO upload_members VALUES(?,?,?,?,?)',(batch_id,ordinal,entry['filename'],entry['sha256'],comparisons[ordinal]['run_id']))
        db.execute('INSERT OR REPLACE INTO analyses VALUES(?,?,?,?,?,?,?,?)',(id,customer_id or None,engagement_id or None,sha,filename or 'upload.log',recipe_id,json.dumps(scrub(report),allow_nan=False),stamp))
        from .performance_api import cache_assessments
        cache_assessments(db,report)
        if customer_id:
            for c in comparisons:
                owner=db.execute('SELECT customer_id,engagement_id FROM customer_evidence WHERE run_id=?',(c['run_id'],)).fetchone()
                if owner and not owner['engagement_id'] and engagement_id:
                    db.execute('UPDATE customer_evidence SET engagement_id=? WHERE customer_id=? AND run_id=?',(engagement_id,customer_id,c['run_id']))
                elif not owner:db.execute('INSERT INTO customer_evidence(customer_id,engagement_id,run_id) VALUES(?,?,?)',(customer_id,engagement_id,c['run_id']))
            if cluster!='User uploads':
                current=json.loads(db.execute('SELECT clusters FROM customers WHERE id=?',(customer_id,)).fetchone()[0])
                if cluster not in current:db.execute('UPDATE customers SET clusters=? WHERE id=?',(json.dumps(current+[cluster]),customer_id))
    return scrub(report)

frontend=ROOT/'frontend'
frontend.mkdir(exist_ok=True)
app.mount('/static',StaticFiles(directory=frontend),name='static')
@app.get('/')
def index():return FileResponse(frontend/'index.html')
