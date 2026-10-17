"""Select and ingest local receipted archives for an explicitly named cluster."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from backend import config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=config.data_root())
    parser.add_argument('--archives-dir',type=Path,required=True)
    parser.add_argument('--cluster',required=True)
    parser.add_argument('--database',type=Path)
    parser.add_argument('--budget-gib',type=float,default=5)
    parser.add_argument('--reserve-gib',type=float,default=1)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',args.cluster):parser.error('Use a cluster label containing letters, digits, dot, underscore or hyphen')
    root=args.root.resolve();catalog=root/'catalog';catalog.mkdir(parents=True,exist_ok=True)
    with (catalog/'refresh.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time()
        extraction=subprocess.run([sys.executable,str(PROJECT/'tools/extract_evidence.py'),'--root',str(root),
            '--archives-dir',str(args.archives_dir),'--cluster',args.cluster,'--budget-gib',str(args.budget_gib),
            '--reserve-gib',str(args.reserve_gib)],capture_output=True,text=True)
        results={}
        if extraction.returncode==0:
            from backend import store
            from backend.ingest import ingest_path
            store.DB_PATH=str(args.database or (config.data_path('VALIDATION_DB','catalog/validation.sqlite') if root==config.data_root() else catalog/'validation.sqlite'))
            path=root/'selected'/args.cluster
            if path.exists():results[args.cluster]=ingest_path(path,args.cluster)
        status=dict(started_at=started,finished_at=time.time(),extraction_exit_code=extraction.returncode,
            extraction_output=extraction.stdout[-2000:],extraction_error=extraction.stderr[-1000:],ingestion=results)
        target=catalog/'refresh-status.json';temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(status,indent=2));temporary.replace(target)
        print(json.dumps(status),flush=True)
        if extraction.returncode or any(x.get('errors') for x in results.values()):raise SystemExit(1)


if __name__=='__main__':main()
