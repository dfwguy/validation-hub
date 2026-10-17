"""Incremental source catalog and measured NCCL / Llama ingestion."""
import argparse, hashlib, json, re, gzip
from pathlib import Path
from datetime import datetime,timezone
from . import store
from .parser import parse_log,COLLECTIVES
from .security import scrub
from .hostlist import expand_hostlist

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def parquet(path,cluster,sha):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path); names=pf.schema_arrow.names
    kind=path.parent.name if path.name.startswith('part-') else path.stem
    if kind not in ('run_results','slurm_jobs'):
        store.source(path,sha,path.stat().st_size,'parquet',cluster,{'rows':pf.metadata.num_rows,'columns':names,'ingest_version':2,'mtime_ns':path.stat().st_mtime_ns})
        return 0
    count=0
    cols=['id','title','cluster','workload','collective','nodes','gpus','racks','sus','hardware','transport','nccl_version','status','avg_bus_bw_gbps','peak_bus_bw_gbps','tflops_gpu','iterations','source_path','source_kind','created_at','payload','sha256','job_id']
    sql='INSERT OR REPLACE INTO runs ('+','.join(cols)+') VALUES ('+','.join('?' for _ in cols)+')'
    with store.connection() as db:
        jobs={}
        if kind=='run_results':
            for j in db.execute("SELECT job_id,nodes,gpus,racks,sus,status,title FROM runs WHERE cluster=? AND source_kind='parquet_slurm_jobs'",(cluster,)):
                jobs[j['job_id']]=dict(j)
        for batch in pf.iter_batches(batch_size=2000):
            output=[]
            for row in batch.to_pylist():
                metadata=scrub({k:v for k,v in row.items() if v is not None and v!=''})
                if kind=='slurm_jobs' and row.get('nodelist') and (not row.get('nodes_json') or row.get('nodes_json')=='[]'):
                    try:
                        metadata['nodes_json']=json.dumps(expand_hostlist(row['nodelist']))
                        metadata['roster_source']='expanded_recorded_slurm_nodelist'
                    except ValueError: metadata['roster_source']='unresolved_slurm_nodelist'
                workload=(row.get('workload') or row.get('workload_class') or 'Unknown').lower()
                label=(str(row.get('job_name',''))+' '+str(row.get('test',''))).lower()
                if workload=='nccl' or 'nccl' in label or any(c in label for c in COLLECTIVES) or 'allreduce' in label:workload='NCCL'
                elif workload in ('llm','llama','nemo') or any(s in label for s in ('llama','llm','nemo')):workload='Llama'
                else:workload=workload.upper() if workload in ('dcgm','hpl') else workload.title()
                metrics_json={}
                try:metrics_json=json.loads(row.get('metrics_json') or '{}')
                except (ValueError,TypeError):pass
                metrics_json=scrub(metrics_json)
                bus=metrics_json.get('busbw',{})
                if not isinstance(bus,dict):bus={}
                collective=next(iter(bus)) if len(bus)==1 else next((c for c in COLLECTIVES if c in label),None)
                bw=bus.get(collective) if collective else None
                if isinstance(bw,dict):bw=None
                verdict=str(row.get('verdict','')).upper(); state=str(row.get('state','')).upper()
                status={'PASS':'reported_pass','PASSED':'reported_pass','FAIL':'failed','FAILED':'failed','PARTIAL':'incomplete'}.get(verdict,'job_completed' if state=='COMPLETED' else 'failed' if state in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') else 'unknown')
                matched=jobs.get(str(row.get('job_id') or ''),{})
                nodes=row.get('node_count',matched.get('nodes'))
                if nodes is None and str(row.get('nnodes','')).isdigit():nodes=int(row['nnodes'])
                gpus=matched.get('gpus')
                m=re.search(r'(?:gres/)?gpu(?:[:=][\w]+)?=(\d+)',str(row.get('req_tres','')))
                if m:gpus=int(m[1])
                payload=dict(id=hashlib.sha256((sha+str(count)+cluster).encode()).hexdigest()[:24],title=row.get('job_name') or row.get('artifact_name') or f'Job {row.get("job_id",count)}',cluster=cluster,workload=workload,collective=collective,nodes=nodes,gpus=gpus,racks=row.get('rack_count',matched.get('racks')),sus=row.get('su_count',matched.get('sus')),hardware=None,transport=None,nccl_version=None,status=status,avg_bus_bw_gbps=bw,peak_bus_bw_gbps=None,tflops_gpu=None,iterations=None,source_path=row.get('artifact_path') or row.get('result_path') or str(path),source_kind='parquet_'+kind,created_at=row.get('start') or datetime.fromtimestamp(row.get('mtime') or row.get('captured_at') or 0,timezone.utc).isoformat(),metadata=metadata,metrics=[],source={'path':str(path),'sha256':sha,'size_bytes':path.stat().st_size,'row':count},job_id=row.get('job_id'))
                metadata['metrics_original']=metrics_json
                if matched: metadata['scheduler_context']=matched
                metadata['geometry_source']='matched_slurm_job' if matched else 'source_row' if nodes is not None else 'unknown'
                metadata['evidence_note']='Scheduler completion or source-reported verdict is retained separately from correctness-qualified raw performance measurements.'
                output.append([json.dumps(payload,allow_nan=False) if k=='payload' else sha if k=='sha256' else payload.get(k) for k in cols]);count+=1
            db.executemany(sql,output)
    store.source(path,sha,path.stat().st_size,'parquet',cluster,{'rows':pf.metadata.num_rows,'columns':names,'ingest_version':2,'mtime_ns':path.stat().st_mtime_ns})
    return count

def infer_hints(path):
    hints={}; parent=path.parent
    # A wrapper's allocation and transport contract supplies context absent from rank logs.
    for directory in (parent,parent.parent,parent.parent.parent):
        candidates=list(directory.glob('*.out'))
        if directory.name.startswith('job-'):candidates+=list((directory/'01_JOB_OUTPUT').glob('*.out'))
        for candidate in candidates[:10]:
            if candidate==path:continue
            with open(candidate,errors='replace') as f: head=[next(f,'') for _ in range(60)]
            facts=parse_log(head,str(candidate))
            if any('TRANSPORT_CONTRACT' in l or 'ALLOCATION_GATE' in l for l in head):
                hints.setdefault('metadata',{}).update({k:v for k,v in facts['metadata'].items() if k in ('ACTIVE_DIAG','RAS_DIAG','global_batch','steps','NCCL_RUN_DIAGNOSTICS','NCCL_RUN_RAS_DIAGNOSTICS')})
                for key in ('nodes','gpus','racks','transport','nccl_version','hardware'):
                    if facts.get(key) is not None:hints[key]=facts[key]
    return hints

def ingest_path(root,cluster):
    root=Path(root); count=0; failures=[]
    paths=[root] if root.is_file() else sorted(root.rglob('*'),key=lambda p:(0 if 'slurm_jobs' in str(p) else 1,str(p)))
    for path in paths:
        if not path.is_file():continue
        if path.suffix not in ('.parquet','.raw','.out','.txt','.log','.err'):continue
        if path.name.endswith('.live.log') or 'nccl-info' in path.name or path.stat().st_size>1200*1024*1024:continue
        try:
            context_path=Path(str(path)+'.context.json')
            context_mtime=context_path.stat().st_mtime_ns if context_path.exists() else None
            version=2 if path.suffix=='.parquet' else 3
            with store.connection() as db:
                existing=db.execute('SELECT sha256,details,size_bytes FROM sources WHERE path=?',(str(path),)).fetchone()
            details=json.loads(existing['details']) if existing else {}
            if existing and details.get('ingest_version')==version and existing['size_bytes']==path.stat().st_size and details.get('mtime_ns')==path.stat().st_mtime_ns and details.get('context_mtime_ns')==context_mtime:continue
            sha=digest(path)
            if existing and existing['sha256']==sha and details.get('ingest_version')==version and details.get('context_mtime_ns')==context_mtime:
                details['mtime_ns']=path.stat().st_mtime_ns
                store.source(path,sha,path.stat().st_size,'parquet' if path.suffix=='.parquet' else 'raw',cluster,details)
                continue
            if path.suffix=='.parquet':count+=parquet(path,cluster,sha);continue
            with open(path,errors='replace') as f:
                lines=(line for line in f if any(token in line for token in ('iteration','MODEL_TFLOP','FATAL','Traceback','CUDA out of memory','NCCL231','GB300','ALLOCATION_GATE','TRANSPORT_CONTRACT','STRICT_ACCEPT'))) if 'llama' in path.name.lower() and path.stat().st_size>50*1024*1024 else f
                run=parse_log(lines,str(path),infer_hints(path))
            provenance_path=Path(str(path)+'.provenance.json')
            if provenance_path.exists():
                run['metadata']['archive_provenance']=scrub(json.loads(provenance_path.read_text()))
            if context_path.exists():
                context=json.loads(context_path.read_text())
                for key in ('transport','nccl_version'):
                    if not run.get(key) and context.get(key):run[key]=context[key]
                run['metadata']['transport_context']=scrub(context)
                run['metadata']['transport_context_path']=str(context_path)
            store.source(path,sha,path.stat().st_size,'raw',cluster,{'workload':run['workload'],'parsed_measurements':len(run['metrics']),'ingest_version':3,'mtime_ns':path.stat().st_mtime_ns,'context_mtime_ns':context_mtime})
            if run['workload']=='Unknown' and not run['metrics']:continue
            m=re.search(r'(?:job-|train-|diag-)(\d{4,})',str(path))
            if not m: m=re.search(r'_(\d{5,})_(?:gb300|gb200)|[/-](\d{5,})[/-]',str(path))
            if m:
                run['job_id']=next(g for g in m.groups() if g)
                with store.connection() as db:
                    j=db.execute("SELECT nodes,gpus,racks,sus,status FROM runs WHERE cluster=? AND job_id=? AND source_kind='parquet_slurm_jobs' LIMIT 1",(cluster,run['job_id'])).fetchone()
                if j:
                    for key in ('nodes','gpus','racks','sus'):
                        if j[key] is not None and (run[key] is None or key in ('nodes','gpus') and run['metadata'].get('observed_ranks',0)<(j['gpus'] or 0)):run[key]=j[key]
                    run['metadata']['scheduler_state']=j['status']
                    if j['status']=='failed':run['metadata']['parent_job_failed']=True
            if 'FAILED' in str(path): run['metadata']['parent_job_failed']=True
            store.put_run(run,path,sha,path.stat().st_size,cluster);count+=1
        except Exception as e:failures.append({'path':str(path),'error':str(e)})
    return {'runs_ingested':count,'errors':failures}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--cluster',required=True);args=p.parse_args(); print(json.dumps(ingest_path(args.path,args.cluster)))
