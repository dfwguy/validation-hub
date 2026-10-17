"""Uploaded evidence receipts and explicit reviews for reusable measured references."""
import json
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from . import store

router=APIRouter(prefix='/api')

def init(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS upload_batches (
      id TEXT PRIMARY KEY, cluster TEXT NOT NULL, sha256 TEXT NOT NULL,
      filename TEXT NOT NULL, path TEXT NOT NULL, size_bytes INTEGER NOT NULL,
      member_count INTEGER NOT NULL, measured_runs INTEGER NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS upload_members (
      batch_id TEXT NOT NULL, ordinal INTEGER NOT NULL, filename TEXT NOT NULL,
      sha256 TEXT NOT NULL, run_id TEXT NOT NULL, PRIMARY KEY(batch_id,ordinal));
    CREATE TABLE IF NOT EXISTS reference_reviews (
      run_id TEXT PRIMARY KEY, reviewer TEXT NOT NULL, reason TEXT NOT NULL,
      active INTEGER NOT NULL, reviewed_at TEXT NOT NULL);
    ''')
    db.execute('CREATE INDEX IF NOT EXISTS reference_scope_idx ON runs(workload,status,nodes,gpus,hardware,transport,nccl_version,collective,source_kind)')

def inventory(db):
    init(db)
    result=dict(db.execute('SELECT COUNT(*) batches,COALESCE(SUM(size_bytes),0) original_bytes,COALESCE(SUM(member_count),0) members FROM upload_batches').fetchone())
    result['measured_runs']=db.execute('SELECT COUNT(DISTINCT m.run_id) FROM upload_members m JOIN runs r ON r.id=m.run_id WHERE r.avg_bus_bw_gbps IS NOT NULL OR r.tflops_gpu IS NOT NULL OR r.iterations IS NOT NULL').fetchone()[0]
    result['reviewed_references']=db.execute('SELECT COUNT(*) FROM reference_reviews WHERE active=1').fetchone()[0]
    return result

@router.get('/lake/uploads')
def uploads(limit:int=Query(30,ge=1,le=100)):
    with store.connection() as db:
        summary=inventory(db)
        items=[dict(r) for r in db.execute('SELECT * FROM upload_batches ORDER BY created_at DESC LIMIT ?',(limit,))]
    return dict(summary=summary,items=items)

class ReferenceReview(BaseModel):
    run_id:str
    reviewer:str=Field(min_length=1,max_length=120)
    reason:str=Field(min_length=10,max_length=2000)
    active:bool=True

@router.post('/reference-reviews')
def review(req:ReferenceReview):
    from .compare import eligible
    with store.connection() as db:
        init(db)
        row=db.execute('SELECT payload FROM runs WHERE id=?',(req.run_id,)).fetchone()
        if not row:raise HTTPException(404,'Run not found')
        run=json.loads(row[0])
        if run.get('source_kind')!='upload':raise HTTPException(422,'Review applies to uploaded evidence')
        if not req.reviewer.strip() or len(req.reason.strip())<10:raise HTTPException(422,'Record the engineer name and review rationale')
        if req.active:
            candidate=dict(run,source_kind='raw')
            if not eligible(candidate) or not run.get('metrics') or any(m.get('errors')!=0 for m in run['metrics'] if 'busbw_gbps' in m):
                raise HTTPException(422,'Reference requires complete correctness-clean NCCL measurements, exact configuration, and a single split mask')
            if not any(m.get('busbw_gbps',0)>0 for m in run['metrics']):raise HTTPException(422,'Reference has no positive measured bandwidth')
            if not db.execute('SELECT 1 FROM upload_members WHERE run_id=?',(req.run_id,)).fetchone():raise HTTPException(422,'Reference requires an ingestion receipt')
        stamp=datetime.now(timezone.utc).isoformat()
        db.execute('INSERT OR REPLACE INTO reference_reviews VALUES(?,?,?,?,?)',(req.run_id,req.reviewer.strip(),req.reason.strip(),req.active,stamp))
    return dict(run_id=req.run_id,active=req.active,reviewer=req.reviewer.strip(),reviewed_at=stamp)

def metrics_summary(run):
    import statistics
    timings=[m['time_ms'] for m in run.get('metrics',[]) if 'time_ms' in m]
    return {**{k:run.get(k) for k in ('workload','collective','nodes','gpus','hardware','transport','nccl_version','status','avg_bus_bw_gbps','peak_bus_bw_gbps','tflops_gpu','iterations')},
            'mean_step_time_ms':statistics.mean(timings) if timings else None,
            'measured_points':len(run.get('metrics',[]))}

def attainment(comparisons):
    verdicts=[c['verdict'] for c in comparisons if c['run'].get('metrics')]
    points=[p for c in comparisons for p in c.get('comparisons',[])]
    return dict(verdict='fail' if 'fail' in verdicts else 'pass' if verdicts and all(v=='pass' for v in verdicts) else 'insufficient_evidence',
                runs_met=verdicts.count('pass'),runs_below_or_failed=verdicts.count('fail'),runs_unscored=verdicts.count('insufficient_evidence'),
                measured_runs=len(verdicts),points_met=sum(p['verdict']=='pass' for p in points),points_below=sum(p['verdict']=='fail' for p in points),
                missing_reference_points=sum(p['verdict']=='no_baseline' for p in points))

def recommendations(comparisons):
    suggestions=[]
    for c in comparisons:
        bad=[p for p in c.get('comparisons',[]) if p.get('deviation_pct') is not None and p['verdict']=='fail' and p.get('correctness_errors')==0]
        if bad:
            worst=min(bad,key=lambda p:p['deviation_pct'])
            suggestions.append(dict(id='regression-'+c['run_id'],title='Investigate measured bandwidth regression',priority=1,
                reason=f"{c['filename']}: {len(bad)} points miss the selected tolerance. Worst deviation {worst['deviation_pct']:.1f}% at {worst['message_bytes']} bytes ({worst['placement']}).",
                command='Reproduce the same collective, message range, placement and split mask on one node, a node pair, then target scale. Keep GPU/NIC placement fixed and collect 3 repeats. Review HCA selection and link counters where performance first drops.',
                success='Zero correctness errors and a stable median within the selected tolerance against the same reference configuration.',
                run_id=c['run_id'],reference_clusters=sorted({b['cluster'] for b in c.get('baseline_runs',[])})))
        if c['verdict']=='insufficient_evidence' and c['run'].get('metrics'):
            suggestions.append(dict(id='reference-'+c['run_id'],title='Complete the baseline context',priority=2,
                reason=f"{c['filename']}: baseline attainment is unknown; this is not a passing result.",
                command='Supply missing allocation, GPU model, transport and NCCL version. Select a compatible reference cluster or ingest and review a trusted reference at the same configuration. Training metrics need a matching workload target; NCCL targets cannot score training throughput.',
                success='Each reported point has compatible source-backed reference evidence.',run_id=c['run_id']))
    return suggestions


@router.get('/lake/received')
def received_uploads():
    import os,re
    from . import config
    from pathlib import Path
    root=config.uploads_root()/'received'
    items=[]
    for path in root.glob('*.receipt.json'):
        sha=path.name.removesuffix('.receipt.json')
        if not re.fullmatch('[0-9a-f]{64}',sha):continue
        try:
            receipt=json.loads(path.read_text())
            if not path.with_name(sha+'.upload').is_file():continue
            state=path.with_name(sha+'.recovery-status.json')
            try:recovery=json.loads(state.read_text()) if state.exists() else {}
            except (ValueError,OSError):recovery={}
            items.append({**{k:receipt.get(k) for k in ('sha256','filename','size_bytes','received_at')},'recovery':recovery})
        except (ValueError,OSError):continue
    return {'items':sorted(items,key=lambda i:i.get('received_at') or '',reverse=True)[:30]}
