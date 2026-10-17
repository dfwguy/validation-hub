"""Cache independent NCCL comparisons for measured catalog runs, without re-parsing logs."""
import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import store,beacon_nccl
from backend.nccl_reference import NCCLReferenceIndex,compare_nccl,from_run_reference,nccl_context
from backend.nccl_summary import assess
from backend.performance_api import init,now


def catalog_runs():
    with store.connection() as db:
        for row in db.execute("SELECT payload FROM runs WHERE workload='NCCL' AND source_kind IN ('raw','upload') AND (avg_bus_bw_gbps IS NOT NULL OR peak_bus_bw_gbps IS NOT NULL) ORDER BY cluster,job_id,collective,id"):
            run=json.loads(row[0]);run['metadata'].pop('hosts',None)
            yield run


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--limit',type=int);parser.add_argument('--run-ids-file',type=Path,help='JSON array of observation IDs to reassess; reference coverage stays complete');parser.add_argument('--write',action='store_true');parser.add_argument('--output',type=Path,default=Path('artifacts/nccl-catalog-assessment.json'));args=parser.parse_args()
    started=time.monotonic();runs=list(catalog_runs());refs=[ref for run in runs if (ref:=from_run_reference(run))]
    with store.connection() as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'reference_reviews' in tables:
            reviewed={r['run_id']:dict(r) for r in db.execute('SELECT * FROM reference_reviews WHERE active=1')}
            for run in runs:
                if run['id'] in reviewed:
                    run['_reference_review']=reviewed[run['id']]
                    if (ref:=from_run_reference(run)):refs.append(ref)
    print(json.dumps(dict(stage='indexing',observations=len(runs),eligible_reference_artifacts=len(refs),elapsed=round(time.monotonic()-started,1))),flush=True)
    historical=NCCLReferenceIndex(refs,max_cache=1024)
    state=beacon_nccl.status();snapshot=(state.get('snapshot') or {}).get('snapshot_id')
    observed_contexts={tuple(nccl_context(run)['context'][field] for field in ('gpu','gpus','collective','fabric')) for run in runs}
    with beacon_nccl._db() as db:
        source_scopes=[dict(zip(('gpu','gpus','fabric','collective'),row)) for row in db.execute('SELECT DISTINCT gpu,gpus,fabric,collective FROM nccl_references')]
    eligible_cores=set()
    for source in source_scopes:
        context=nccl_context(source)['context']
        key=tuple(context[field] for field in ('gpu','gpus','collective','fabric'))
        if key in observed_contexts:eligible_cores.add((source['gpu'],source['gpus'],source['collective']))
    external_rows=[]
    for core in eligible_cores:external_rows.extend(beacon_nccl.references(filters=dict(zip(('gpu','gpus','collective'),core))))
    external=NCCLReferenceIndex(external_rows,max_cache=1024)
    fingerprint=hashlib.sha256('\n'.join(sorted(ref['ref_id'] for ref in refs)).encode()).hexdigest()
    counts=collections.Counter();bands=collections.Counter();source_counts=collections.Counter();pending=[];examples=[];scanned=0
    selected=runs
    if args.run_ids_file:
        ids=set(json.loads(args.run_ids_file.read_text()));selected=[run for run in runs if run['id'] in ids]
    if args.limit:selected=selected[:args.limit]
    for run in selected:
        result=assess(run,[('External measured references',external),('Cluster historical results',historical)])
        origin=result['provenance']['reference_source']
        result['provenance'].update(reference_source=origin,source_snapshot=state.get('snapshot') if origin.startswith('External') else None,catalog_reference_fingerprint=fingerprint)
        # Source-job counts stay complete; keep bounded source samples in cache.
        if len(result['references'])>20:
            result['provenance']['references_displayed']=20
            result['references']=result['references'][:20]
        counts[result['classification']]+=1;bands[result.get('candidate_band') or 'unknown']+=1;source_counts[origin]+=1
        if len(examples)<5:examples.append(dict(run_id=run['id'],job_id=run.get('job_id'),classification=result['classification'],candidate_band=result.get('candidate_band'),point_count=len(result['metrics'])))
        if args.write:pending.append((run['id'],None,snapshot,now(),json.dumps(result,allow_nan=False,separators=(',',':'))))
        scanned+=1
        if len(pending)>=100:
            with store.connection() as db:
                init(db);db.executemany('INSERT OR REPLACE INTO run_performance VALUES(?,?,?,?,?)',pending)
            pending.clear()
        if scanned%100==0:print(json.dumps(dict(stage='assessing',runs=scanned,total=len(selected),elapsed=round(time.monotonic()-started,1),ratings=dict(counts))),flush=True)
    if pending:
        with store.connection() as db:
            init(db);db.executemany('INSERT OR REPLACE INTO run_performance VALUES(?,?,?,?,?)',pending)
    result=dict(runs=scanned,eligible_reference_artifacts=len(refs),external_source_records=state.get('records',0),external_reference_records=len(external_rows),ratings=dict(counts),bands=dict(bands),sources=dict(source_counts),examples=examples,elapsed_seconds=round(time.monotonic()-started,2),written=args.write,reference_fingerprint=fingerprint)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
