"""Compact NCCL best-recorded comparisons at useful large message sizes."""
import math
from .nccl_reference import nccl_context

VERSION='2026-10-17.simple-nccl-2'
SIZES=tuple(2**30*n for n in (16,32))


def _number(value):
    try:
        n=float(value)
        return n if math.isfinite(n) and n>=0 else None
    except (TypeError,ValueError):return None


def _band(delta):return 'good' if delta>=-5-1e-9 else 'bad' if delta>=-10-1e-9 else 'worst'


def _points(run):
    return [p for p in run.get('metrics',[]) if _number(p.get('busbw_gbps')) is not None and _number(p.get('message_bytes')) not in (None,0)]


def _observed(points,size,gpus=None):
    candidates=[p for p in points if p.get('message_bytes')==size]
    if not candidates:
        # nccl-tests rounds collective counts down to a rank/datatype multiple.
        # Bucket that tiny loss for display; reference matching still uses the
        # exact ACTUAL bytes, never the nominal column size or interpolation.
        limit=(_number(gpus) or 0)*8
        candidates=[p for p in points if 0<size-p['message_bytes']<=limit]
    if not candidates:return None
    oop=[p for p in candidates if nccl_context({},p)['context']['placement']=='out_of_place']
    return max(oop or candidates,key=lambda p:float(p['busbw_gbps']))


def assess(run,indexes):
    """Compare actual observations against independent exact-size/version records.

    indexes is an ordered sequence of (source label, NCCLReferenceIndex). Missing
    fabric may use explicitly scoped same-cluster historical evidence only.
    No small-message, algorithm-bandwidth or latency ratios enter this summary.
    """
    points=_points(run);rows=[];references={};metrics=[];missing=set();mismatches=set();used_sources=set()
    template={}
    for field in ('datatype','operation','placement','protocol','split_mask'):
        values={str(p[field]) for p in points if p.get(field) is not None}
        if len(values)==1:template[field]=next(p[field] for p in points if p.get(field) is not None)
    template['placement']='out_of_place'
    meta=run.get('metadata') or {}
    failed=run.get('status')=='failed' or bool(meta.get('parent_job_failed') or meta.get('exit_failure')) or meta.get('correctness_errors') not in (None,0) or any(p.get('errors') for p in run.get('metrics',[]))
    for size in SIZES:
        observed_point=_observed(points,size,run.get('gpus'))
        observed=_number(observed_point.get('busbw_gbps')) if observed_point else None
        probe=observed_point or template
        matches=[]
        for label,index in indexes:
            if index is None:continue
            actual_size=observed_point['message_bytes'] if observed_point else size
            found=index.best_point(run,actual_size,point=probe,allow_same_cluster_unknown_fabric=True)
            expected=_number(found.get('expected'))
            if expected is not None and expected>0:matches.append((expected,label,found))
        winner=max(matches,key=lambda value:value[0]) if matches else None
        expected,label,match=winner if winner else (None,None,{})
        delta=100*(observed/expected-1) if observed is not None and expected else None
        rating=_band(delta) if delta is not None else 'unknown'
        ref=next(iter(match.get('references',[])),{})
        status='matched' if delta is not None else 'not_measured' if observed is None else 'no_matching_reference'
        source_url=ref.get('row_action') or ref.get('source_csv_url') or ref.get('source_url')
        row=dict(message_bytes=size,actual_message_bytes=observed_point['message_bytes'] if observed_point else None,observed_gbps=observed,reference_gbps=expected,delta_pct=delta,
                 expected_pct=delta+100 if delta is not None else None,classification=rating,
                 reference_status=status,reference_jobs=match.get('reference_jobs',0),
                 placement=nccl_context(run,probe)['context']['placement'],source_url=source_url,
                 reference_job_id=ref.get('job_id'),reference_tier=match.get('tier','none'),
                 reference_source=label,context_note=match.get('context_note'))
        rows.append(row)
        if winner:
            used_sources.add(label);missing.update(match.get('missing_context',[]));mismatches.update(match.get('mismatches',[]))
            for item in match.get('references',[]):references[item.get('ref_id') or item.get('source_url')]=item
        if delta is not None:
            metrics.append(dict(metric='busbw_gbps',message_bytes=row['actual_message_bytes'],display_message_bytes=size,placement=row['placement'],observed=observed,expected=expected,
                expected_pct=delta+100,delta_pct=delta,deficit_pct=-delta,performance_band=rating,classification='conditional',scored=False,
                unit='GB/s',direction='higher_better',reference_statistic='best',reference_jobs=row['reference_jobs'],
                reference_sources=match.get('references',[])))
    bands=[r['classification'] for r in rows if r['delta_pct'] is not None]
    nominal=max(bands,key=lambda b:('good','bad','worst').index(b),default=None)
    exact=bool(metrics) and all(r['reference_tier']=='exact' for r in rows if r['delta_pct'] is not None)
    classification='failed' if failed else nominal if exact and run.get('status')=='passed' else 'conditional' if metrics else 'unknown'
    verdict='fail' if failed or classification in ('bad','worst') else 'pass' if classification=='good' else 'insufficient_evidence'
    for m in metrics:m.update(classification=classification,scored=bool(exact and not failed and run.get('status')=='passed'))
    context_note='Best recorded for the same GPU scale, collective and NCCL version; out-of-place results are preferred.'
    if any(r.get('context_note') for r in rows):context_note='Best recorded on this cluster; fabric metadata is incomplete.'
    elif missing or mismatches:context_note+=' Some network or test settings are unverified.'
    if failed:context_note='Another artifact in this job failed; bandwidth values are shown for diagnosis.' if meta.get('parent_job_failed') else 'This run failed; bandwidth values are shown for diagnosis.'
    collective=run.get('collective') or 'collective'
    improvements=[
        dict(title='Check links and rails',detail='Check per-rail traffic, link errors and retransmissions. Identify idle or unhealthy paths before tuning.'),
        dict(title='Compare the best-run configuration',detail='Compare the linked reference job’s NCCL environment, container, algorithm, protocol and registration. Change one setting at a time.'),
        dict(title='Verify allocation and topology',detail='Check GPU/NIC affinity and complete NVLink domains. Compare one-domain, two-domain and full-scale tests to locate where performance drops.'),
        dict(title='Repeat the sizes that are below reference',detail=f'Run {collective} at each affected size with the same settings and at least three repeats. Retain zero-error logs and verify the gain before accepting a change.')]
    summary=dict(version=VERSION,rows=rows,best_policy='Best independent recorded bus bandwidth at the same exact message size and NCCL version; no interpolation.',
                 unit='GB/s',message_unit='GiB',nccl_version=run.get('nccl_version'),collective=collective,
                 classification=classification,candidate_band=nominal,context_note=context_note,improvements=improvements,
                 reference_sources=sorted(used_sources),failed=bool(failed))
    return dict(verdict=verdict,classification=classification,candidate_band=nominal,tier='exact' if exact else 'contextual' if references else 'none',
                metrics=metrics,references=list(references.values())[:20],missing_context=sorted(missing),mismatches=sorted(mismatches),
                policy=dict(reference_statistic='best',good_pct=5,bad_pct=10,summary_message_bytes=list(SIZES)),
                provenance=dict(scoring_version=VERSION,reference_source=' + '.join(sorted(used_sources)) or 'No matching recorded reference',point_count=len(metrics)),
                recommendations=[],nccl_summary=summary)
