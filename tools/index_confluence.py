#!/usr/bin/env python3
"""Index locally retained, authorized read-only wiki evidence; never writes wiki."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.confluence_reference import index_page
from backend import config

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('page_ids',nargs='+');p.add_argument('--evidence-dir',type=Path,default=config.evidence_root());p.add_argument('--db');p.add_argument('--import-measured',action='store_true');p.add_argument('--output');args=p.parse_args()
 results=[index_page(page_id,args.evidence_dir,args.db,args.import_measured) for page_id in args.page_ids]
 if args.output:Path(args.output).write_text(json.dumps(results,indent=2))
 print(json.dumps([{'page_id':r['page_id'],'page_version':r['page_version'],'tables':len(r['tables']),'measured_references':len(r['references']),'receipt':r.get('receipt')} for r in results]))
if __name__=='__main__':main()
