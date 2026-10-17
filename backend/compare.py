import statistics
from . import store
MATCH_FIELDS=('nodes','gpus','hardware','transport','nccl_version','collective')

def eligible(run):
    return run.get('source_kind')=='raw' and run.get('status')=='passed' and run.get('workload')=='NCCL' and run.get('collective')!='mixed' and all(run.get(k) is not None and run.get(k)!='' for k in MATCH_FIELDS) and run.get('metadata',{}).get('correctness_errors')==0 and not run.get('metadata',{}).get('parent_job_failed') and not run.get('metadata',{}).get('mixed_split_masks')

def baseline_identity(run):
    return (run.get('cluster'),run.get('job_id') or run.get('source',{}).get('sha256'),run.get('collective'),split_mask(run))

def compare(run,cluster=None,tolerance_pct=10):
    warnings=[]; missing=[k for k in MATCH_FIELDS if run.get(k) is None or run.get(k)=='']
    policy={'tolerance_pct':tolerance_pct,'basis':'Each reference job contributes one median across all of its eligible artifacts for each point. The baseline is the median across reference jobs at each exact message size, datatype, reduction operation and placement from complete correctness-clean runs. Bandwidth below baseline by more than tolerance fails. This is a configurable comparison policy, not an NVIDIA certification.','match_fields':list(MATCH_FIELDS),'cluster_filter':cluster,'baseline_eligibility':'passed NCCL runs with completed summary and zero correctness errors'}
    if run.get('metadata',{}).get('mixed_split_masks'):missing.append('single_split_mask')
    if run.get('collective')=='mixed': missing.append('single_collective');warnings.append('Mixed collective logs cannot define one comparable run; upload one collective output per file.')
    if missing: warnings.append('Missing comparison metadata: '+', '.join(missing)+'. Provide these fields to establish a compatible baseline.')
    if run['status']!='passed': warnings.append('Uploaded log is incomplete, lacks correctness evidence, or reports an error; performance acceptance cannot establish correctness.')
    filters={k:run.get(k) for k in MATCH_FIELDS};filters.update(workload='NCCL',status='passed')
    if cluster:filters['cluster']=cluster
    matches=store.search(filters,limit=10000)['items'] if not missing else []
    candidates=[]
    ids=[r['id'] for r in matches]
    with store.connection() as db:
        rows=db.execute('SELECT payload FROM runs WHERE id IN ('+','.join('?' for _ in ids)+')',ids) if ids else []
        import json
        for row in rows:
            r=json.loads(row[0])
            if not eligible(r):continue
            if split_mask(r)!=split_mask(run) or r.get('metadata',{}).get('mixed_split_masks'):continue
            candidates.append(r)
    points={}; jobs={}; grouped_runs={}
    for r in candidates:
        identity=baseline_identity(r)
        grouped_runs.setdefault(identity,[]).append(r)
        per_job=jobs.setdefault(identity,{})
        for m in r.get('metrics',[]):
            if m.get('busbw_gbps') is None or m.get('errors')!=0:continue
            key=(m.get('message_bytes'),m.get('datatype'),m.get('operation'),m.get('placement'))
            per_job.setdefault(key,[]).append(m['busbw_gbps'])
    for per_job in jobs.values():
        for k,v in per_job.items():points.setdefault(k,[]).append(statistics.median(v))
    comparisons=[]
    for m in run.get('metrics',[]):
        if 'busbw_gbps' not in m:continue
        key=(m.get('message_bytes'),m.get('datatype'),m.get('operation'),m.get('placement')); refs=points.get(key,[])
        median=statistics.median(refs) if refs else None
        delta=100*(m['busbw_gbps']/median-1) if median and median>0 else None
        verdict='fail' if m.get('errors',0) else 'no_baseline' if not refs else 'not_scored' if median==0 else 'fail' if delta is not None and delta < -tolerance_pct else 'pass'
        comparisons.append(dict(message_bytes=m['message_bytes'],datatype=m.get('datatype'),operation=m.get('operation'),placement=m.get('placement'),observed_gbps=m['busbw_gbps'],baseline_gbps=median,deviation_pct=delta,baseline_samples=len(refs),verdict=verdict))
    if len(matches)==10000: warnings.append('Baseline query reached the 10,000-run cap; narrow the cluster filter to compare a smaller population.')
    if not candidates:warnings.append('No complete reference run matches the specified geometry, hardware, collective, transport and NCCL version.')
    if not comparisons:warnings.append('No NCCL performance table found in uploaded file.')
    if any(c['verdict']=='no_baseline' for c in comparisons):warnings.append('Some measurements have no matching message size, datatype, operation or placement baseline.')
    scored=[c for c in comparisons if c['verdict'] in ('pass','fail')]
    if any(c['verdict']=='not_scored' for c in comparisons):warnings.append('Reference bandwidth rounds to zero for some sizes; these known reference points cannot produce a relative percentage and are excluded from performance scoring.')
    verdict='fail' if run['status']=='failed' or any(c['verdict']=='fail' for c in comparisons) else 'insufficient_evidence' if run['status']!='passed' or not scored or any(c['verdict']=='no_baseline' for c in comparisons) else 'pass'
    return dict(run=run,verdict=verdict,baseline_count=len(grouped_runs),baseline_runs=[dict({k:rs[0].get(k) for k in ('id','title','source_path','cluster')},artifact_count=len(rs),artifact_ids=[r['id'] for r in rs]) for rs in grouped_runs.values()],comparisons=comparisons,warnings=warnings,policy=policy)


def split_mask(run):
    value=run.get('metadata',{}).get('NCCL_TESTS_SPLIT_MASK','0')
    try:return int(value,0)
    except (ValueError,TypeError):return value
