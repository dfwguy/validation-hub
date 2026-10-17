#!/usr/bin/env python3
"""Refresh relevant Confluence evidence with GET requests, then index local sources."""
import argparse,concurrent.futures,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import wiki_sources,config
from backend.confluence_client import ConfluenceReader,SITE

QUERIES=[
 'type=page AND (title ~ "NCCL" OR title ~ "Exemplar")',
 'type=page AND (title ~ "Spectrum-X" OR title ~ "SPCX") AND (text ~ "recipe" OR text ~ "validation" OR text ~ "performance")',
 'type=page AND (title ~ "troubleshooting" OR title ~ "tech bulletin") AND (text ~ "NCCL" OR text ~ "GB300")',
 'type=page AND (title ~ "GB300" OR title ~ "B300" OR title ~ "GB200" OR title ~ "B200" OR title ~ "H100" OR title ~ "H200") AND (title ~ "performance" OR title ~ "benchmark" OR title ~ "validation" OR title ~ "recipe")'
]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,default=config.evidence_root());parser.add_argument('--refresh',action='store_true');parser.add_argument('--email');parser.add_argument('--token-file');parser.add_argument('--db');args=parser.parse_args()
    if args.refresh:
        client=ConfluenceReader(args.email,args.token_file)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:queries=list(ex.map(client.search,QUERIES))
        pages={}
        for query in queries:
            for row in query['items']:pages[row['id']]=row
        args.folder.mkdir(parents=True,exist_ok=True)
        (args.folder/'wiki-discovery-manifest.json').write_text(json.dumps(dict(site=SITE,queries=[{k:v for k,v in q.items() if k!='items'} for q in queries],pages=list(pages.values()),unique_pages=len(pages)),indent=2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for i,result in enumerate(ex.map(lambda pid:client.save_page(pid,args.folder),pages),1):
                if i%100==0:print(json.dumps(dict(downloaded=i,total=len(pages))),flush=True)
    print(json.dumps(wiki_sources.ingest_directory(args.folder,args.db),indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
