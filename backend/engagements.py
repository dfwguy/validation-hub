"""Customer ownership and engagement state, separate from immutable source evidence."""
import json, uuid, math, statistics
from pathlib import PurePosixPath
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from . import store

router=APIRouter(prefix='/api')
def now():return datetime.now(timezone.utc).isoformat()
def init(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS customers(id TEXT PRIMARY KEY,name TEXT NOT NULL COLLATE NOCASE UNIQUE,clusters TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS engagements(id TEXT PRIMARY KEY,customer_id TEXT NOT NULL,title TEXT NOT NULL,status TEXT NOT NULL,recipe_id TEXT NOT NULL,notes TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY,customer_id TEXT,engagement_id TEXT,sha256 TEXT NOT NULL,filename TEXT NOT NULL,recipe_id TEXT NOT NULL,report TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS customer_evidence(customer_id TEXT NOT NULL,engagement_id TEXT NOT NULL,run_id TEXT NOT NULL,final INTEGER NOT NULL DEFAULT 0,exemplar INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(customer_id,engagement_id,run_id));
    CREATE INDEX IF NOT EXISTS analyses_customer ON analyses(customer_id,engagement_id);
    ''')

class Customer(BaseModel):
    name:str=Field(min_length=1,max_length=120)
    clusters:list[str]=Field(default_factory=list,max_length=100)
class Engagement(BaseModel):
    customer_id:str
    title:str=Field(min_length=1,max_length=200)
    status:str=Field('planned',pattern='^(planned|active|completed|blocked)$')
    recipe_id:str
    notes:str=Field('',max_length=4000)
class Evidence(BaseModel):
    run_id:str
    engagement_id:str=""
    final:bool=False
    exemplar:bool=False

def validate_owner(db,customer_id,engagement_id):
    if not customer_id or not engagement_id:raise HTTPException(422,'Select both customer and engagement')
    if not db.execute('SELECT 1 FROM engagements WHERE id=? AND customer_id=?',(engagement_id,customer_id)).fetchone():raise HTTPException(422,'Engagement does not belong to this customer')

def training_result(run):
    """Only completed, source-backed training observations can become final evidence."""
    meta=run.get('metadata',{})
    if run.get('workload') in ('Unknown','NCCL',None) or run.get('status')!='passed' or meta.get('parent_job_failed') or meta.get('exit_failure'):return None
    if run.get('source_kind') not in ('raw','upload') or not (run.get('source_path') or run.get('source',{}).get('path')):return None
    metrics=run.get('metrics',[])
    def positive(v):return isinstance(v,(int,float)) and math.isfinite(v) and v>0
    values=[m[k] for m in metrics for k in ('time_ms','tflops_gpu') if k in m]
    if not values or not all(positive(v) for v in values):return None
    steps=[m for m in metrics if 'iteration' in m and 'time_ms' in m]
    expected=meta.get('expected_iterations')
    if not isinstance(expected,int) or expected<=0 or not steps or max(m['iteration'] for m in steps)!=expected:return None
    if run.get('tflops_gpu') is not None and not positive(run['tflops_gpu']):return None
    identity={k:meta.get(k) for k in ('model','model_name','model_size','precision','dtype','global_batch','global_batch_size','sequence_length') if meta.get(k) is not None}
    identity['source_configuration']=str(PurePosixPath(meta.get('archive_member') or run.get('source_path','')).parent)
    return dict(workload=run['workload'],configuration=identity,tflops_gpu=run.get('tflops_gpu'),mean_step_time_ms=statistics.mean(m['time_ms'] for m in steps),iterations=run.get('iterations'),source_path=run.get('source_path'),source_sha256=run.get('sha256') or run.get('source',{}).get('sha256'))

@router.post('/customers')
def create_customer(req:Customer):
    import sqlite3
    name=req.name.strip()
    if not name:raise HTTPException(422,'Customer name is required')
    with store.connection() as db:
        init(db)
        clusters=list(dict.fromkeys(c.strip() for c in req.clusters if c.strip()))
        existing=[json.loads(r[0]) for r in db.execute('SELECT clusters FROM customers')]
        if set(clusters).intersection(c for group in existing for c in group):raise HTTPException(409,'A cluster is already assigned to another customer')
        id=uuid.uuid4().hex
        try:db.execute('INSERT INTO customers VALUES(?,?,?,?)',(id,name,json.dumps(clusters),now()))
        except sqlite3.IntegrityError:raise HTTPException(409,'Customer already exists')
    return dict(id=id,name=name,clusters=clusters)

@router.put('/customers/{id}')
def update_customer(id:str,req:Customer):
    with store.connection() as db:
        init(db)
        if not db.execute('SELECT 1 FROM customers WHERE id=?',(id,)).fetchone():raise HTTPException(404,'Customer not found')
        clusters=list(dict.fromkeys(c.strip() for c in req.clusters if c.strip()))
        existing=[json.loads(r[0]) for r in db.execute('SELECT clusters FROM customers WHERE id!=?',(id,))]
        if set(clusters).intersection(c for group in existing for c in group):raise HTTPException(409,'A cluster is already assigned to another customer')
        import sqlite3
        if not req.name.strip():raise HTTPException(422,'Customer name is required')
        try:db.execute('UPDATE customers SET name=?,clusters=? WHERE id=?',(req.name.strip(),json.dumps(clusters),id))
        except sqlite3.IntegrityError:raise HTTPException(409,'Customer already exists')
    return dict(id=id)

@router.post('/engagements')
def create_engagement(req:Engagement):
    from .recipes import profile
    if not profile(req.recipe_id):raise HTTPException(422,'Unknown recipe')
    if not req.title.strip():raise HTTPException(422,'Engagement title is required')
    with store.connection() as db:
        init(db)
        if not db.execute('SELECT 1 FROM customers WHERE id=?',(req.customer_id,)).fetchone():raise HTTPException(404,'Customer not found')
        id=uuid.uuid4().hex
        db.execute('INSERT INTO engagements VALUES(?,?,?,?,?,?,?)',(id,req.customer_id,req.title.strip(),req.status,req.recipe_id,req.notes,now()))
    return dict(id=id,**req.model_dump())

@router.put('/engagements/{id}')
def update_engagement(id:str,req:Engagement):
    from .recipes import profile
    if not profile(req.recipe_id) or not req.title.strip():raise HTTPException(422,'Valid recipe and title required')
    with store.connection() as db:
        init(db);validate_owner(db,req.customer_id,id)
        db.execute('UPDATE engagements SET title=?,status=?,recipe_id=?,notes=? WHERE id=?',(req.title.strip(),req.status,req.recipe_id,req.notes,id))
    return dict(id=id,**req.model_dump())

@router.post('/customers/{id}/evidence')
def attach_evidence(id:str,req:Evidence):
    with store.connection() as db:
        init(db)
        if req.engagement_id:validate_owner(db,id,req.engagement_id)
        elif not db.execute('SELECT 1 FROM customers WHERE id=?',(id,)).fetchone():raise HTTPException(404,'Customer not found')
        run=db.execute('SELECT status,workload,payload,source_kind FROM runs WHERE id=?',(req.run_id,)).fetchone()
        if not run:raise HTTPException(404,'Run not found')
        parsed=json.loads(run['payload'])
        if req.final or req.exemplar:
            nccl_valid=(run['workload']=='NCCL' and run['status']=='passed' and any('busbw_gbps' in m for m in parsed.get('metrics',[])) and all(isinstance(m['busbw_gbps'],(int,float)) and math.isfinite(m['busbw_gbps']) and m['busbw_gbps']>=0 for m in parsed.get('metrics',[]) if 'busbw_gbps' in m) and parsed.get('metadata',{}).get('correctness_errors')==0 and not parsed.get('metadata',{}).get('parent_job_failed'))
            if not nccl_valid and not training_result({**parsed,'source_kind':run['source_kind']}):
                raise HTTPException(422,'Final / Exemplar results require complete source-backed measured training or correctness-clean NCCL evidence')
        owner=db.execute('SELECT customer_id,engagement_id FROM customer_evidence WHERE run_id=?',(req.run_id,)).fetchone()
        if owner and (owner[0]!=id or (owner[1] and owner[1]!=req.engagement_id)):raise HTTPException(409,'Run is already assigned to another engagement')
        # Cluster ownership must agree for imported evidence.
        cluster=parsed.get('cluster')
        for other in db.execute('SELECT id,clusters FROM customers WHERE id!=?',(id,)):
            if cluster in json.loads(other['clusters']):raise HTTPException(409,'Run belongs to a cluster assigned to another customer')
        if owner and not owner['engagement_id']:db.execute('DELETE FROM customer_evidence WHERE customer_id=? AND run_id=? AND engagement_id=?',(id,req.run_id,''))
        db.execute('INSERT OR REPLACE INTO customer_evidence VALUES(?,?,?,?,?)',(id,req.engagement_id,req.run_id,req.final,req.exemplar))
    return dict(ok=True)

@router.get('/customers')
def customers():
    with store.connection() as db:
        init(db)
        items=[]
        for c in db.execute('SELECT * FROM customers ORDER BY name'):
            item=dict(c);item['clusters']=json.loads(item['clusters'])
            item['engagements']=[dict(e) for e in db.execute('SELECT * FROM engagements WHERE customer_id=? ORDER BY created_at DESC',(c['id'],))]
            es=item['engagements'];item['active_engagements']=sum(e['status']=='active' for e in es);item['planned_engagements']=sum(e['status']=='planned' for e in es)
            cluster_clause='cluster IN ('+','.join('?' for _ in item['clusters'])+')' if item['clusters'] else '0'
            where='('+cluster_clause+' OR id IN (SELECT run_id FROM customer_evidence WHERE customer_id=?))'
            args=item['clusters']+[c['id']]
            # Scheduler artifacts sharing a job are one validation. Files without a job use content identity.
            counts=db.execute('SELECT COUNT(*) records,COUNT(DISTINCT cluster || char(31) || COALESCE(job_id,sha256,id)) validations FROM runs WHERE '+where,args).fetchone()
            item.update(dict(counts))
            valid="status='passed' AND workload='NCCL' AND source_kind IN ('raw','upload') AND avg_bus_bw_gbps IS NOT NULL AND json_extract(payload,'$.metadata.correctness_errors')=0 AND COALESCE(json_extract(payload,'$.metadata.parent_job_failed'),0)=0"
            item['valid_logs']=db.execute('SELECT COUNT(DISTINCT sha256) FROM runs WHERE '+where+' AND '+valid,args).fetchone()[0]
            flags=db.execute('SELECT SUM(final),SUM(exemplar) FROM customer_evidence WHERE customer_id=?',(c['id'],)).fetchone()
            item['final_results']=flags[0] or 0;item['exemplar_results']=flags[1] or 0
            item['best_performance']=[]
            best_sql="SELECT * FROM (SELECT id,hardware,transport,nodes,gpus,collective,nccl_version,peak_bus_bw_gbps,payload,json_extract(payload,'$.metadata.NCCL_TESTS_SPLIT_MASK') split_mask,ROW_NUMBER() OVER (PARTITION BY hardware,transport,nodes,gpus,collective,nccl_version,COALESCE(json_extract(payload,'$.metadata.NCCL_TESTS_SPLIT_MASK'),'0x0') ORDER BY peak_bus_bw_gbps DESC) ranking FROM runs WHERE "+where+' AND '+valid+') WHERE ranking=1 ORDER BY peak_bus_bw_gbps DESC LIMIT 12'
            for row in db.execute(best_sql,args):
                best=dict(row);payload=json.loads(best.pop('payload'));best.pop('ranking')
                best['point']=next((m for m in payload.get('metrics',[]) if m.get('busbw_gbps')==best['peak_bus_bw_gbps']),None)
                item['best_performance'].append(best)
            training=[]
            for row in db.execute("SELECT payload,source_kind,sha256 FROM runs WHERE "+where+" AND workload NOT IN ('NCCL','Unknown') AND status='passed' AND source_kind IN ('raw','upload') AND (tflops_gpu IS NOT NULL OR iterations IS NOT NULL)",args):
                parsed=json.loads(row['payload']);result=training_result({**parsed,'source_kind':row['source_kind'],'sha256':row['sha256']})
                if result:training.append({**{k:parsed.get(k) for k in ('id','hardware','transport','nodes','gpus','collective','nccl_version')},**result})
            item['valid_logs']+=len({r['source_sha256'] or r['id'] for r in training})
            grouped={}
            for result in training:
                key=(result['workload'],result['hardware'],result['transport'],result['nodes'],result['gpus'],json.dumps(result['configuration'],sort_keys=True))
                score=(result['tflops_gpu'] or 0,-result['mean_step_time_ms'])
                if key not in grouped or score>(grouped[key]['tflops_gpu'] or 0,-grouped[key]['mean_step_time_ms']):grouped[key]=result
            item['best_training_performance']=list(grouped.values())[:100]
            item['training_configurations']=len(grouped)
            item['evidence_total']=db.execute('SELECT COUNT(*) FROM customer_evidence WHERE customer_id=?',(c['id'],)).fetchone()[0]
            item['analyses']=[dict(r) for r in db.execute('SELECT id,engagement_id,filename,recipe_id,created_at FROM analyses WHERE customer_id=? ORDER BY created_at DESC LIMIT 30',(c['id'],))]
            item['evidence']=[dict(r) for r in db.execute('SELECT e.*,r.title,r.status,r.hardware,r.nodes,r.gpus,r.collective,r.workload,r.peak_bus_bw_gbps,r.tflops_gpu,r.payload FROM customer_evidence e JOIN runs r ON r.id=e.run_id WHERE e.customer_id=? ORDER BY r.created_at DESC LIMIT 100',(c['id'],))]
            for evidence in item['evidence']:
                payload=json.loads(evidence.pop('payload'));summary=training_result({**payload,'source_kind':payload.get('source_kind','upload')})
                if summary:evidence.update(summary)
            from .performance_api import run_assessments
            displayed=item['best_performance']+item['best_training_performance']+item['evidence']
            assessments=run_assessments([r.get('id') or r.get('run_id') for r in displayed if r.get('id') or r.get('run_id')])
            for result in displayed:
                evaluation=assessments.get(result.get('id') or result.get('run_id'))
                if evaluation:result['performance']=evaluation
            items.append(item)
    return dict(items=items,count_policy='Validations deduplicate cluster + job ID, or SHA-256 when job ID is absent. Records include scheduler and result artifacts. Final and Exemplar are explicit engineer designations, not certification.')

@router.get('/analyses/{id}')
def get_analysis(id:str):
    with store.connection() as db:
        init(db);row=db.execute('SELECT report FROM analyses WHERE id=?',(id,)).fetchone()
    if not row:raise HTTPException(404,'Analysis not found')
    return json.loads(row[0])
