"""Normalize source-reported multi-collective summary metrics without assigning raw correctness."""
import json,hashlib
from . import store

def derive_collectives():
    count=0
    with store.connection() as db:
        records=db.execute("SELECT payload FROM runs WHERE source_kind='parquet_run_results' AND workload='NCCL' AND avg_bus_bw_gbps IS NULL").fetchall()
        cols=['id','title','cluster','workload','collective','nodes','gpus','racks','sus','hardware','transport','nccl_version','status','avg_bus_bw_gbps','peak_bus_bw_gbps','tflops_gpu','iterations','source_path','source_kind','created_at','payload','sha256','job_id']
        output=[]
        for record in records:
            original=json.loads(record[0]); bws=original.get('metadata',{}).get('metrics_original',{}).get('busbw',{})
            if not isinstance(bws,dict) or len(bws)<2:continue
            for collective,bw in bws.items():
                if not isinstance(bw,(int,float)):continue
                r=dict(original);r.update(id=hashlib.sha256((original['id']+collective).encode()).hexdigest()[:24],title=original['title']+' · '+collective,collective=collective,avg_bus_bw_gbps=bw,source_kind='parquet_collective_summary')
                output.append([json.dumps(r) if k=='payload' else r['source']['sha256'] if k=='sha256' else r.get(k) for k in cols])
                count+=1
        db.executemany('INSERT OR REPLACE INTO runs ('+','.join(cols)+') VALUES ('+','.join('?' for _ in cols)+')',output)
    return count
if __name__=='__main__':print(json.dumps({'collective_summaries_normalized':derive_collectives()}))
