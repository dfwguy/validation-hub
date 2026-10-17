#!/usr/bin/env python3
"""Import actual Beacon exports or index its observed authenticated JSON API."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import beacon


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group();source.add_argument('--file');source.add_argument('--api-url')
    p.add_argument('--source-url',default=beacon.BASE_URL);p.add_argument('--db');p.add_argument('--page-param');p.add_argument('--page-size-param');p.add_argument('--page-size',type=int,default=200)
    args=p.parse_args()
    try:
        if args.file:result=beacon.import_file(args.file,args.source_url,args.db)
        elif args.api_url:result=beacon.fetch_all(args.api_url,args.db,args.page_param,args.page_size_param,args.page_size,progress=lambda p: print(json.dumps(p),flush=True) if p['pages']%10==0 else None)
        else:result=beacon.status(args.db)
        print(json.dumps(result,indent=2));return 0
    except Exception as exc:
        # Do not print request headers, cookies, or HTTP error response bodies.
        print(json.dumps({'status':'error','error':str(exc) if isinstance(exc,ValueError) else type(exc).__name__,'instructions':beacon.INSTRUCTIONS}),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
