import json,hashlib,shutil,os
from pathlib import Path
from fastapi import FastAPI,HTTPException,UploadFile,File,Form,Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from . import store
from .parser import parse_log
from .compare import compare,eligible,baseline_identity
from .security import scrub
ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='Cluster Validation Hub',version='1.0')
POLICY='Baselines require complete raw NCCL results, zero correctness errors and exact nodes, GPUs, hardware, transport, collective, NCCL version, message size, datatype, operation and placement. Source-reported parquet and scheduler verdicts remain distinct.'

def storage():
    result={}
    for p in (ROOT/'data/storage-inventory.json',Path('/data/validation-hub/catalog/storage-inventory.json')):
        if p.exists():
            try:result=json.loads(p.read_text());break
            except ValueError:pass
    mirror=Path('/data/validation-hub/catalog/mirror-status.json')
    if mirror.exists():
        try: result['mirror_status']=json.loads(mirror.read_text())
        except ValueError:pass
    result['local_archive_count']=sum(1 for _ in Path('/data/validation-hub/raw').rglob('*.tar.gz')) if Path('/data/validation-hub/raw').exists() else 0
    usage=shutil.disk_usage('/data' if Path('/data').exists() else ROOT)
    result['disk']={'path':'/data','total_bytes':usage.total,'used_bytes':usage.used,'free_bytes':usage.free}
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
        qualified=len({baseline_identity(r) for r in eligible_runs if eligible(r)})
        max_nodes,max_gpus=db.execute('SELECT MAX(nodes),MAX(gpus) FROM runs').fetchone()
        measured=db.execute("SELECT COUNT(*) FROM runs WHERE source_kind='raw' AND (avg_bus_bw_gbps IS NOT NULL OR iterations IS NOT NULL)").fetchone()[0]
    return dict(runs=total,measurements=measured,clusters=clusters,workloads=workloads,sources=sources,qualified_baselines=qualified,max_nodes=max_nodes,max_gpus=max_gpus,baseline_policy=POLICY,storage=storage())

@app.get('/api/facets')
def facets():
    names={'clusters':'cluster','workloads':'workload','collectives':'collective','hardware':'hardware','statuses':'status','node_counts':'nodes','nccl_versions':'nccl_version','transports':'transport'}
    with store.connection() as db:return {name:[r[0] for r in db.execute('SELECT DISTINCT '+col+' FROM runs WHERE '+col+' IS NOT NULL ORDER BY '+col)] for name,col in names.items()}

@app.get('/api/runs')
def runs(q:str='',measured_only:bool=False,rack_ids:str='',su_ids:str='',nodelist:str='',cluster:str='',workload:str='',collective:str='',nodes:int|None=None,min_nodes:int|None=None,max_nodes:int|None=None,gpus:int|None=None,racks:float|None=None,sus:float|None=None,hardware:str='',status:str='',nccl_version:str='',transport:str='',limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
    try: return store.search(locals(),limit,offset)
    except ValueError as error: raise HTTPException(422,str(error))

@app.get('/api/runs/{id}')
def run(id:str):
    result=store.detail(id)
    if not result:raise HTTPException(404,'Run not found')
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
    sha=hashlib.sha256(content).hexdigest();path=Path(os.environ.get('VALIDATION_UPLOADS','/data/validation-hub/uploads'))/(sha+'.log');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    run=store.put_run(run,path,sha,len(content),cluster or 'User uploads','upload')
    result=compare(run,cluster or None,tolerance_pct)
    result['policy']['user_supplied_fields']=run['metadata']['user_supplied_fields']
    return result

# Troubleshooting and customer workflows share the existing evidence catalog.
from .engagements import router as engagements_router
from . import engagements, recipes, troubleshooting
from fastapi.responses import PlainTextResponse
import uuid, zipfile, tarfile
app.include_router(engagements_router)

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
async def analyze_upload(file:UploadFile=File(...),recipe_id:str=Form(...),customer_id:str=Form(''),engagement_id:str=Form(''),cluster:str=Form('')):
    if customer_id or engagement_id:
        with store.connection() as db:
            engagements.init(db);engagements.validate_owner(db,customer_id,engagement_id)
            er=db.execute('SELECT recipe_id FROM engagements WHERE id=?',(engagement_id,)).fetchone()
            if er[0]!=recipe_id:raise HTTPException(422,'Selected recipe does not match the engagement recipe')
            for c in db.execute('SELECT id,clusters FROM customers WHERE id!=?',(customer_id,)):
                if cluster and cluster in json.loads(c['clusters']):raise HTTPException(409,'Cluster is assigned to another customer')
    content=await file.read(troubleshooting.LIMIT+1)
    if len(content)>troubleshooting.LIMIT:raise HTTPException(413,'Upload limit is 32 MiB')
    if not content:raise HTTPException(422,'Upload is empty')
    try:
        entries=troubleshooting.texts(content,file.filename or 'upload.log')
        report=troubleshooting.analyze(entries,recipe_id)
    except (ValueError,zipfile.BadZipFile,tarfile.TarError,EOFError,OSError,RuntimeError) as e:raise HTTPException(422,str(e))
    sha=hashlib.sha256(content).hexdigest()
    id=hashlib.sha256((sha+recipe_id+recipes.VERSION+customer_id+engagement_id+cluster).encode()).hexdigest()[:32]
    comparisons=[]
    for entry,(_,member_text) in zip(report.pop('runs'),entries):
        parsed=entry['parsed']
        if parsed['metadata'].get('mixed_split_masks'):parsed['status']='incomplete'
        # Materialize members by content hash only; archive names remain display metadata.
        path=Path(os.environ.get('VALIDATION_UPLOADS','/data/validation-hub/uploads'))/(entry['sha256']+'.log')
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(member_text.encode('utf-8'))
        run=store.put_run(parsed,path,entry['sha256'],entry['size_bytes'],cluster or 'User uploads','upload',display_name=Path(entry['filename']).name)
        comparisons.append(dict(filename=entry['filename'],run_id=run['id'],**compare(run,cluster or None)))
    report.update(id=id,customer_id=customer_id or None,engagement_id=engagement_id or None,comparisons=comparisons,created_at=engagements.now())
    with store.connection() as db:
        engagements.init(db)
        db.execute('INSERT OR REPLACE INTO analyses VALUES(?,?,?,?,?,?,?,?)',(id,customer_id or None,engagement_id or None,sha,file.filename or 'upload.log',recipe_id,json.dumps(report,allow_nan=False),report['created_at']))
        if customer_id:
            for c in comparisons:
                owner=db.execute('SELECT customer_id,engagement_id FROM customer_evidence WHERE run_id=?',(c['run_id'],)).fetchone()
                if owner and (owner[0]!=customer_id or owner[1]!=engagement_id):raise HTTPException(409,'Identical run is already assigned to another engagement; use its existing evidence record')
                db.execute('INSERT OR IGNORE INTO customer_evidence(customer_id,engagement_id,run_id) VALUES(?,?,?)',(customer_id,engagement_id,c['run_id']))
    return scrub(report)

frontend=ROOT/'frontend'
frontend.mkdir(exist_ok=True)
app.mount('/static',StaticFiles(directory=frontend),name='static')
@app.get('/')
def index():return FileResponse(frontend/'index.html')
