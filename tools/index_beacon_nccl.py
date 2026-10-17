#!/usr/bin/env python3
"""Index actual NCCL exports or an observed authenticated NCCL JSON API."""
import argparse,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import beacon_nccl



def public_gpucomms_context(chromium=None):
 """Create a fresh public-site session, without Beacon credentials or cookies."""
 try:from playwright.sync_api import sync_playwright
 except ImportError:raise ValueError('GPUComms refresh requires the browser Python environment with Playwright installed')
 with sync_playwright() as playwright:
  launch={'headless':True,'args':['--no-sandbox']}
  if chromium:launch['executable_path']=chromium
  browser=playwright.chromium.launch(**launch)
  try:
   context=browser.new_context();page=context.new_page();state={'options':{}}
   def receive(response):
    if '/_dash-update-component' not in response.url:return
    try:
     result=response.json().get('response',{})
     for key,value in result.items():
      if key.startswith('{') and 'filter-input' in key:
       state['options'][json.loads(key)['field']]=value.get('data',[])
     if 'filtered-tests-count-detail' in result:
      count=int(str(result['filtered-tests-count-detail']['children']).replace(',',''))
      if count:
       state.update(template=response.request.post_data_json,headers=response.request.all_headers(),advertised_total=count)
    except (ValueError,KeyError,TypeError):pass
   page.on('response',receive)
   page.goto('https://gpucomms.nvidia.com/',wait_until='domcontentloaded',timeout=60000)
   for _ in range(120):
    page.wait_for_timeout(500)
    if state.get('template') and all(key in state['options'] for key in ('num_nodes','tool_binary','cluster_name','library_and_version','name')):break
   if not state.get('template'):raise ValueError('GPUComms did not initialize its public golden dataset')
   headers=state['headers'];headers['Cookie']='; '.join(c['name']+'='+c['value'] for c in context.cookies() if c['name']=='gcperf_session')
   # Remove the original browser Cookie spelling before setting our explicit
   # public-session-only header, preventing unrelated analytics cookie leakage.
   headers.pop('cookie',None);state['headers']=headers
   return state
  finally:browser.close()


def main():
 p=argparse.ArgumentParser(description=__doc__);source=p.add_mutually_exclusive_group();source.add_argument('--file');source.add_argument('--api-url');source.add_argument('--gpucomms',action='store_true');p.add_argument('--chromium',default=os.environ.get('VALIDATION_CHROMIUM')); p.add_argument('--source-url',default=beacon_nccl.SOURCE_URL);p.add_argument('--db');args=p.parse_args()
 try:
  if args.gpucomms:
   context=public_gpucomms_context(args.chromium)
   result=beacon_nccl.fetch_gpucomms(context,args.db,progress=lambda p:print(json.dumps(p),flush=True))
  elif args.file:result=beacon_nccl.import_file(args.file,args.source_url,args.db)
  elif args.api_url:result=beacon_nccl.fetch_all(args.api_url,args.db,progress=lambda p:print(json.dumps(p),flush=True) if p['pages']%10==0 else None)
  else:result=beacon_nccl.status(args.db)
  print(json.dumps(result,indent=2));return 0
 except Exception as exc:
  print(json.dumps({'status':'error','error':str(exc) if isinstance(exc,ValueError) else type(exc).__name__}),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
