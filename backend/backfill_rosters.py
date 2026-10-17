import json
from . import store
from .hostlist import expand_hostlist

def backfill():
    count=0;errors=0
    with store.connection() as db:
        records=db.execute("SELECT id,payload FROM runs WHERE source_kind='parquet_slurm_jobs' AND nodes>1 AND json_extract(payload,'$.metadata.nodes_json')='[]'").fetchall()
        output=[]
        for record in records:
            r=json.loads(record['payload']); metadata=r['metadata']
            try:
                nodes=expand_hostlist(metadata.get('nodelist',''))
                if len(nodes)==r['nodes']:
                    metadata['nodes_json']=json.dumps(nodes);metadata['roster_source']='expanded_recorded_slurm_nodelist';output.append((json.dumps(r),r['id']));count+=1
                else:errors+=1
            except ValueError:errors+=1
        db.executemany('UPDATE runs SET payload=? WHERE id=?',output)
    return dict(jobs_backfilled=count,unresolved=errors)
if __name__=='__main__':print(json.dumps(backfill()))
