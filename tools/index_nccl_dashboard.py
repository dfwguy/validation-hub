"""Import observed NCCL dashboard curves; no target or SOL measurements."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import nccl_dashboard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path)
    parser.add_argument('--snapshot-root',type=Path,default=nccl_dashboard.DEFAULT_SNAPSHOTS)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--platform');parser.add_argument('--collective');parser.add_argument('--gpus',type=int)
    parser.add_argument('--kernel');parser.add_argument('--limit-curves',type=int)
    args=parser.parse_args()
    result=nccl_dashboard.index(db_path=args.db,snapshot_root=args.snapshot_root,max_workers=args.workers,
        platform=args.platform,collective=args.collective,gpus=args.gpus,kernel=args.kernel,limit=args.limit_curves,
        progress=lambda row:print(json.dumps(row),flush=True))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
