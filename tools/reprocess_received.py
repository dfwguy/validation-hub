"""Reprocess a retained archive using recorded recovery context; no network upload."""
import argparse,json,time
from pathlib import Path
from backend import store,troubleshooting,upload_io
from backend.app import ingest_upload


def main():
    parser=argparse.ArgumentParser();parser.add_argument('path',type=Path);args=parser.parse_args()
    receipt=json.loads(args.path.with_suffix('.receipt.json').read_text());context=receipt['recovery_context']
    with store.connection() as db:
        customer=db.execute('SELECT id FROM customers WHERE name=?',(context['customer_name'],)).fetchone()
    if not customer:raise ValueError('Recovery customer is not registered')
    status=args.path.with_suffix('.recovery-status.json');start=time.monotonic()
    def progress(stage,**fields):
        data=dict(stage=stage,elapsed_seconds=round(time.monotonic()-start,1),**fields)
        status.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True)
    original_index=upload_io.DiskLog._index;scanned=0;last_scan=time.monotonic()
    def index(text):
        nonlocal scanned,last_scan
        fresh=text._lines is None;result=original_index(text)
        if fresh:
            scanned+=1
            if scanned%250==0 or time.monotonic()-last_scan>15:progress('analyzing',scanned_log_files=scanned);last_scan=time.monotonic()
        return result
    upload_io.DiskLog._index=index
    original_analyze=troubleshooting.analyze
    def analyze(entries,recipe):
        progress('analyzing',log_members=len(entries),recipe_id=recipe)
        result=original_analyze(entries,recipe)
        progress('indexing',log_members=len(entries));return result
    troubleshooting.analyze=analyze
    original_put=store.put_run;count=0;last=time.monotonic()
    def put(*a,**kw):
        nonlocal count,last
        result=original_put(*a,**kw);count+=1
        if count%250==0 or time.monotonic()-last>15:progress('indexing',indexed_runs=count);last=time.monotonic()
        return result
    store.put_run=put
    progress('starting',filename=receipt['filename'])
    try:
        report=ingest_upload(args.path,receipt['filename'],context['recipe_id'],customer_id=customer['id'],engagement_id=context.get('engagement_id') or '',cluster=context['cluster'])
        progress('complete',analysis_id=report['id'],ingestion=report['ingestion'],baseline_attainment=report['baseline_attainment'],crash_groups=len(report['crashes']))
    except Exception as error:
        progress('failed',error=str(error));raise

if __name__=='__main__':main()
