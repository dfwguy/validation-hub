"""Attach explainable training/NCCL acceptance and investigation to evidence."""
import collections
import copy
from datetime import datetime, timezone
import math
import statistics
import threading
from . import beacon, investigation, lake, performance, store

POLICY = dict(good_pct=5, bad_pct=10, reference_statistic='median',
              description='Good: no more than 5% below expected; bad: more than 5% through 10%; worst: more than 10%. Failed correctness overrides speed. Conditional comparisons are unscored. This is a hub comparison policy, not certification.')
PROTOCOL_FIELDS = ('min_bytes', 'max_bytes', 'iterations', 'warmup', 'registration')


def band(percent):
    deficit = 100-percent
    return 'good' if deficit <= POLICY['good_pct']+1e-9 else 'bad' if deficit <= POLICY['bad_pct']+1e-9 else 'worst'


def historical_nccl(outcome, reference_cache=None):
    run = outcome['run']; meta = run.get('metadata', {})
    reference_cache = reference_cache if reference_cache is not None else {}
    references = []
    observed_protocol = meta.get('test_protocol', {})
    missing = [key for key in PROTOCOL_FIELDS if key not in observed_protocol]
    conflicts = []
    sources=[]
    for source in outcome.get('baseline_runs',[]):
        sources.extend(dict(source,id=identifier) for identifier in source.get('artifact_ids',[source['id']]))
    needed=list({s['id'] for s in sources if s['id'] not in reference_cache})
    if needed:
        import json
        with store.connection() as db:
            for offset in range(0,len(needed),500):
                batch=needed[offset:offset+500]
                for row in db.execute('SELECT id,payload FROM runs WHERE id IN ('+','.join('?' for _ in batch)+')',batch):reference_cache[row['id']]=json.loads(row['payload'])
        for identifier in needed:reference_cache.setdefault(identifier,None)
    for source in sources:
        identifier = source['id']
        ref = reference_cache[identifier] or {}
        reference_protocol = ref.get('metadata', {}).get('test_protocol', {})
        for key in PROTOCOL_FIELDS:
            if key not in reference_protocol:missing.append('reference.'+key)
            elif key in observed_protocol and reference_protocol[key] != observed_protocol[key]:conflicts.append(key)
        observed_env = meta.get('nccl_environment')
        ref_env = ref.get('metadata', {}).get('nccl_environment')
        if observed_env is None or ref_env is None:missing.append('recorded_environment')
        elif observed_env != ref_env:conflicts.append('nccl_environment')
        for key in ('placement_scope','rail_topology','nic_model','link_speed_gbps','nccl_tests_version','congestion_control'):
            left,right=meta.get(key),ref.get('metadata',{}).get(key)
            if left is None or right is None:missing.append(key)
            elif left!=right:conflicts.append(key)
        references.append(dict(ref_id=identifier,job_id=ref.get('job_id'),cluster=ref.get('cluster'),
            source_url='/api/runs/'+identifier,source_path=ref.get('source_path'),
            source_sha256=ref.get('source',{}).get('sha256'),artifact_count=source.get('artifact_count'),test_protocol=reference_protocol))
    points=[]
    for point in outcome.get('comparisons', []):
        expected=point.get('baseline_gbps');observed=point.get('observed_gbps')
        if not expected or expected<=0 or observed is None or not math.isfinite(observed):continue
        percent=100*observed/expected
        points.append(dict(metric='busbw_gbps',message_bytes=point['message_bytes'],placement=point.get('placement'),datatype=point.get('datatype'),operation=point.get('operation'),
            observed=observed,expected=expected,expected_pct=percent,delta_pct=percent-100,deficit_pct=100-percent,
            performance_band=band(percent),classification='conditional',scored=False,unit='GB/s bus bandwidth',direction='higher_better',reference_jobs=point.get('baseline_samples',0)))
    zero_points=sum(p.get('baseline_gbps')==0 for p in outcome.get('comparisons',[]))
    required_points=len(outcome.get('comparisons',[]))-zero_points
    complete=bool(points) and len(points)==required_points and bool(references)
    exact=complete and not missing and not conflicts
    failed=run.get('status')=='failed' or meta.get('exit_failure') or meta.get('parent_job_failed') or any(m.get('errors') for m in run.get('metrics',[]))
    nominal=max((p['performance_band'] for p in points),key=lambda x:('good','bad','worst').index(x),default=None)
    classification='failed' if failed else nominal if exact and run.get('status')=='passed' else 'conditional' if points else 'unknown'
    verdict='fail' if failed or exact and nominal in ('bad','worst') else 'pass' if exact and nominal=='good' and run.get('status')=='passed' else 'insufficient_evidence'
    for point in points:point.update(scored=bool(exact and not failed and run.get('status')=='passed'),classification=classification if exact or failed else 'conditional')
    return dict(verdict=verdict,classification=classification,candidate_band=nominal,tier='exact' if exact else 'contextual' if references else 'none',
        metrics=points,references=references,missing_context=sorted(set(missing)),mismatches=sorted(set(conflicts)),
        provenance=dict(reference_jobs=outcome.get('baseline_count',0),point_count=len(points),zero_reference_points_excluded=zero_points,observed_protocol=observed_protocol),policy=POLICY,
        recommendations=[dict(title='Reproduce the exact NCCL reference protocol',reason='A sweep point and a fixed-size result, or different topology/tuning policies, cannot establish the same acceptance target.',
            command='Use the recipe generator with the reference message bytes, scale, warmups, iterations, registration and site placement. Collect at least three repeats and all environment overrides.',
            expected='Zero errors and a reproducible median within the selected threshold against a source-linked identical test.',rollback='Restore defaults and the site-approved fabric policy after every isolated A/B experiment.')] if not exact or nominal in ('bad','worst') else [])


_nccl_lock=threading.Lock()
_nccl_snapshot=None
_nccl_indexes=collections.OrderedDict()


def external_nccl_index(run):
    global _nccl_snapshot
    from . import beacon_nccl,nccl_reference
    state=beacon_nccl.status()
    snapshot=(state.get('snapshot') or {}).get('snapshot_id')
    key=(str(beacon_nccl._path()),snapshot,state.get('records',0))
    context=nccl_reference.nccl_context(run)['context']
    core=(context['gpu'],context['gpus'],beacon_nccl._collective(context['collective']))
    with _nccl_lock:
        if key!=_nccl_snapshot:
            _nccl_indexes.clear();_nccl_snapshot=key
        if core not in _nccl_indexes:
            filters=dict(zip(('gpu','gpus','collective'),core))
            # Core fields are mandatory for matching; never index a million
            # unrelated curve points when a single uploaded cohort is needed.
            refs=beacon_nccl.references(filters=filters) if all(v is not None for v in core) else []
            _nccl_indexes[core]=nccl_reference.NCCLReferenceIndex(refs)
            while len(_nccl_indexes)>16:_nccl_indexes.popitem(last=False)
        _nccl_indexes.move_to_end(core)
        return _nccl_indexes[core],state


_historical_indexes=collections.OrderedDict()


def historical_nccl_index(run):
    import json
    from .nccl_reference import NCCLReferenceIndex,nccl_context,from_run_reference
    context=nccl_context(run)['context'];core=tuple(context[k] for k in ('gpu','gpus','collective','nccl_version'))
    if any(v is None for v in core):return NCCLReferenceIndex([])
    from .beacon_nccl import _collective
    values=(core[0],core[1],_collective(core[2]),core[3])
    with store.connection() as db:
        lake.init(db)
        sql="FROM runs r LEFT JOIN reference_reviews v ON v.run_id=r.id WHERE r.workload='NCCL' AND r.status='passed' AND r.hardware=? AND r.gpus=? AND r.collective=? AND r.nccl_version=? AND (r.source_kind='raw' OR v.active=1)"
        stamp=tuple(db.execute('SELECT count(*),max(r.created_at),max(v.reviewed_at) '+sql,values).fetchone())
        key=(str(store.DB_PATH),core,stamp)
        with _nccl_lock:
            if key not in _historical_indexes:
                refs=[]
                for row in db.execute('SELECT r.payload,v.active,v.reviewer,v.reason,v.reviewed_at '+sql,values):
                    candidate=json.loads(row['payload']);candidate.get('metadata',{}).pop('hosts',None)
                    if row['active']==1:candidate['_reference_review']={k:row[k] for k in ('active','reviewer','reason','reviewed_at')}
                    ref=from_run_reference(candidate)
                    if ref:refs.append(ref)
                _historical_indexes[key]=NCCLReferenceIndex(refs,max_cache=1024)
                while len(_historical_indexes)>8:_historical_indexes.popitem(last=False)
            _historical_indexes.move_to_end(key)
            return _historical_indexes[key]


def nccl(outcome, reference_cache=None):
    from .nccl_summary import assess
    run=outcome['run'];external,state=external_nccl_index(run)
    result=assess(run,[('External measured references',external),('Cluster historical results',historical_nccl_index(run))])
    result['provenance']['source_snapshot']=state.get('snapshot')
    return result


def reference_status():
    state=beacon.status()
    with store.connection() as db:
        rows=list(db.execute("SELECT hardware,COUNT(*) records FROM runs WHERE workload='NCCL' AND source_kind='raw' AND status='passed' AND avg_bus_bw_gbps IS NOT NULL GROUP BY hardware"))
    from . import beacon_nccl
    return dict(beacon=state,nccl_external=beacon_nccl.status(),nccl=dict(records=sum(r['records'] for r in rows),gpu_types=[r['hardware'] for r in rows if r['hardware']],
        coverage=[dict(r) for r in rows],source='Local source-backed historical raw logs and explicitly reviewed uploads; exact context required'),policy=POLICY)


def enrich(report, references=None, status=None):
    """Operate on an analysis payload; the caller decides when to persist it."""
    state=beacon.status() if status is None else status
    counts=collections.Counter(); cache={}; scoped={}
    for comparison in report.get('comparisons',[]):
        run=comparison['run']
        if not run.get('metrics'):continue
        if run.get('workload')=='NCCL':evaluation=nccl(comparison,cache)
        else:
            candidate=copy.deepcopy(run)
            candidate.setdefault('metadata',{}).setdefault('upload_filename',report.get('filename') or report.get('ingestion',{}).get('filename'))
            if references is None:
                context=performance.training_context(candidate)['context']
                key=(context.get('gpu'),context.get('gpus'),context.get('precision'))
                if key not in scoped:
                    filters={k:v for k,v in zip(('gpu','scale','precision'),key) if v is not None}
                    scoped[key]=beacon.references(filters=filters) if filters.get('gpu') else []
                candidates=scoped[key]
            else:candidates=references
            evaluation=performance.compare_training(candidate,candidates)
        comparison['performance']=evaluation
        comparison['verdict']=evaluation['verdict']
        counts[evaluation['classification']]+=1
    report['performance']=dict(ratings={key:counts[key] for key in ('good','bad','worst','unknown','conditional','failed')},
        policy=POLICY,reference_count=state.get('records',len(references) if references is not None else 0),source_status=state.get('status','unknown'),
        source_snapshot=state.get('snapshot'),generated_at=datetime.now(timezone.utc).isoformat())
    report['baseline_attainment']=lake.attainment(report.get('comparisons',[]))
    if report['baseline_attainment']['verdict']=='fail':report['verdict']='fail'
    report['investigation']=investigation.build(report)
    return report
