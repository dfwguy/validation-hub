"""NCCL reference evidence adapter; no fabricated or training-derived baselines.

The official Exemplar troubleshooting guide links GPUComms golden NCCL results.
Beacon training TFLOPS are never substituted for communication measurements.
"""
from datetime import datetime,timezone
import gzip,hashlib,json,os,re,sqlite3,tempfile,uuid
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit,urlencode,parse_qsl,urljoin
from urllib.request import Request,build_opener
from . import beacon,config

SOURCE_URL=config.GPUCOMMS_URL
DISCOVERY={'checked_at':'2026-10-17','schema_url':'https://api.llmb.nvidia.com/openapi.json',
 'beacon_page_url':'https://llmb.nvidia.com/beacon-v2/microbenchmarks','source_url':SOURCE_URL,
 'discovery_url':'https://exemplar.nvidia.com/resources/performance-troubleshooting.json',
 'source_dataset':'golden_data','finding':'Official Exemplar guide directs NCCL reference comparisons to GPUComms; its public golden_data dataset is retrieved through normal bounded filter queries. Beacon training data is a separate source.'}
FIELDS={
 'source_record_id':['source_record_id','id','run_id','reference_id'],'job_id':['job_id','jobId','slurm_job_id'],
 'gpu':['gpu','gpu_model','gpu_type','hardware'],'gpus':['gpus','scale','num_gpus','gpu_count'],
 'nodes':['nodes','num_nodes'],'fabric':['fabric','network','network_type','transport'],
 'collective':['collective','collective_type','test','operation_name'],
 'nccl_version':['nccl_version','ncclVersion'],'protocol':['protocol','nccl_protocol','NCCL_PROTO'],
 'algorithm':['algorithm','nccl_algorithm','NCCL_ALGO'],'split_mask':['split_mask','NCCL_TESTS_SPLIT_MASK'],
 'datatype':['datatype','dtype','data_type'],'operation':['operation','reduction_op','reduce_op'],
 'placement':['placement','memory_placement'],'message_bytes':['message_bytes','size_bytes','bytes'],
 'busbw_gbps':['busbw_gbps','busbw_GBps','bus_bandwidth_gb_s','busbw','bus_bandwidth'],
 'algbw_gbps':['algbw_gbps','algbw_GBps','algorithm_bandwidth_gb_s','algbw','algorithm_bandwidth'],
 'latency_us':['latency_us','time_us'],'errors':['errors','correctness_errors','wrong'],
 'status':['status','test_status'],'date_utc':['date_utc','job_timestamp','timestamp','created_at'],
 'cluster':['cluster','cluster_name'],'release':['release','llmb_version'],
}
COMPACT_FIELDS=('ref_id','job_id','gpu','gpus','fabric','collective','nccl_version','protocol','algorithm','split_mask','date_utc','source_url','metrics')


def _path(db_path=None):return Path(db_path) if db_path else config.data_path('BEACON_NCCL_DB','beacon-nccl.sqlite')

def _db(db_path=None):
 p=_path(db_path);p.parent.mkdir(parents=True,exist_ok=True)
 db=sqlite3.connect(p,timeout=60);db.row_factory=sqlite3.Row
 db.executescript('''CREATE TABLE IF NOT EXISTS nccl_references(ref_id TEXT PRIMARY KEY,source_scope TEXT,source_record_id TEXT,gpu TEXT,gpus INTEGER,fabric TEXT,collective TEXT,normalized_json TEXT,snapshot_id TEXT);
 CREATE INDEX IF NOT EXISTS nccl_reference_scope ON nccl_references(gpu,gpus,fabric,collective);
 CREATE INDEX IF NOT EXISTS nccl_reference_snapshot ON nccl_references(snapshot_id);
 CREATE TABLE IF NOT EXISTS nccl_snapshots(snapshot_id TEXT PRIMARY KEY,created_at TEXT,source_url TEXT,records INTEGER,path TEXT,sha256 TEXT,metadata_json TEXT);''')
 return db


def init(db_path=None):
 with _db(db_path):pass


def _pick(flat,names):
 for name in names:
  for key,value in flat.items():
   if (key==name or key.rsplit('.',1)[-1]==name) and value is not None and value!='':return key,value
 return None,None


def _fabric(value):
 if not isinstance(value,str):return None
 text=value.lower();has_nv='nvlink' in text
 if re.search(r'spectrum[ -]?x',text):base='spectrum-x'
 elif 'roce' in text:base='roce'
 elif 'infiniband' in text or text.strip()=='ib':base='infiniband'
 elif text.strip() in ('nvlink','nvls'):base='nvlink'
 elif text.strip() in ('socket','tcp','ethernet','efa'):base=text.strip()
 else:return None
 return base+'+nvlink' if has_nv and base!='nvlink' else base


def _collective(value):
 if not isinstance(value,str):return None
 key=value.lower().replace('_perf_mpi','').replace('_perf','').replace('-','_')
 return {'allreduce':'all_reduce','allgather':'all_gather','reducescatter':'reduce_scatter','all_to_all':'alltoall','send_recv':'sendrecv'}.get(key,key)


def _bandwidth(value,path,unit):
 n=beacon._number(value)
 if n is None:return None
 if unit is None:
  # Only a unit-bearing field may establish GB/s without a separate unit.
  return n if path and (path.endswith('_gbps') or 'GBps' in path or path.endswith('_gb_s')) else None
 factor={'GB/s':1,'GBps':1,'Gb/s':1/8,'Gbps':1/8,'MB/s':.001,'MiB/s':2**20/1e9,'GiB/s':2**30/1e9}.get(str(unit))
 return n*factor if factor is not None else None



def _json_value(value,default):
 if isinstance(value,str):
  try:return json.loads(value)
  except (ValueError,TypeError):return default
 return value if value is not None else default


def _registration(value):
 if value is None:return None
 return {'disabled':0,'local_register_all':1,'symmetric_register_all':2,'0':0,'1':1,'2':2}.get(str(value).lower())


def _gpucomms_view(raw):
 """Adapt verified golden_data columns without imputing missing test context."""
 system=_json_value(raw.get('system_info'),{});network=_json_value(raw.get('network_info'),{})
 config=_json_value(raw.get('test_config'),{});env=_json_value(raw.get('test_env_vars'),{})
 results=_json_value(raw.get('test_result'),[])
 view=dict(raw);gpu_raw=system.get('gpu_arch_type');gpu=None
 if isinstance(gpu_raw,str):
  match=re.search(r'\b(GB300|GB200|B300|B200|H200|H100|A100|V100|H800|H20|A800)\b',gpu_raw,re.I)
  gpu=match[1].upper() if match else gpu_raw
 processes=beacon._number(raw.get('num_processes'),True)
 threads=beacon._number(config.get('nthreads'),True);per_thread=beacon._number(config.get('ngpus'),True)
 gpus=processes*threads*per_thread if all(v is not None for v in (processes,threads,per_thread)) else None
 metrics=[]
 if isinstance(results,list):
  for point in results:
   if not isinstance(point,dict):continue
   for placement in ('in_place','out_of_place'):
    bw=point.get(placement+'.bus_bw');alg=point.get(placement+'.alg_bw');time=point.get(placement+'.time')
    if not any(v is not None for v in (bw,alg,time)):continue
    metrics.append(dict(message_bytes=point.get('size'),datatype=point.get('type'),operation=point.get('redop'),
       placement=placement,busbw_gbps=bw,algbw_gbps=alg,latency_us=time,errors=point.get(placement+'.nwrong'),
       bandwidth_unit='GB/s',actual_iterations=point.get('actual_iterations')))
 network_name=network.get('network_name');switch_type=network.get('switch_type')
 # A switch family alone does not establish a Spectrum-X deployment recipe.
 fabric_text=network_name
 if any(re.search(r'spectrum[ -]?x',str(v),re.I) for v in (network_name,switch_type)):
  fabric_text='Spectrum-X'
 view.update(gpu=gpu,gpus=gpus,nodes=raw.get('num_nodes'),fabric=fabric_text,collective=raw.get('tool_binary'),
   nccl_version=raw.get('library_version'),job_id=raw.get('slurm_job_id'),date_utc=raw.get('created_timestamp'),
   protocol=env.get('NCCL_PROTO'),algorithm=env.get('NCCL_ALGO'),split_mask=env.get('NCCL_TESTS_SPLIT_MASK'),
   metrics=metrics,bandwidth_unit='GB/s',nic_model=network.get('nics'),link_speed_gbps=network.get('link_speed_gbps'),
   placement_scope=None,rail_topology=network.get('rail_topology'),congestion_control=network.get('congestion_control'),
   nccl_tests_version=raw.get('tool_version'),nccl_environment=env,gpu_model_raw=gpu_raw,network_context=network,
   fabric_evidence_scope='Source network configuration; actual per-collective data-plane transport is not established.',
   test_protocol={**{key:config.get(key) for key in ('minbytes','maxbytes','iters','warmup_iters','agg_iters','nthreads','ngpus','check','cudagraph','blocking','parallel_init','local_register','stepfactor')},
     'min_bytes':config.get('minbytes'),'max_bytes':config.get('maxbytes'),'iterations':config.get('iters'),'warmup':config.get('warmup_iters'),'registration':_registration(config.get('local_register'))},
   unit_provenance={'tool':'nccl-tests','bandwidth':'GB/s','latency':'microseconds','reference':'https://github.com/NVIDIA/nccl-tests/blob/master/src/common.cu'},
   _source_mapping={'gpu':'system_info.gpu_arch_type','gpus':'num_processes * test_config.nthreads * test_config.ngpus',
      'fabric':'network_info.network_name (site configuration)','nic_model':'network_info.nics','nccl_version':'library_version',
      'metrics':'test_result NCCL native in_place/out_of_place columns','protocol':'test_env_vars.NCCL_PROTO','split_mask':'test_env_vars.NCCL_TESTS_SPLIT_MASK'})
 return view


def normalize_record(raw,source_url=SOURCE_URL,refreshed_at=None):
 if not isinstance(raw,dict):raise ValueError('Each NCCL reference must be a JSON object')
 original_raw=raw
 if raw.get('tool')=='nccl-tests' and 'test_result' in raw:
  raw=_gpucomms_view(raw)
 flat=beacon._flatten({k:v for k,v in raw.items() if k not in ('metrics','points','results')});r={};mapping={}
 for field,aliases in FIELDS.items():
  path,value=_pick(flat,aliases);r[field]=value
  if path:mapping[field]=path
 if isinstance(r['gpu'],str):r['gpu']=r['gpu'].upper()
 for field in ('gpus','nodes','message_bytes','errors'):r[field]=beacon._number(r[field],integer=True)
 r['scale']=r['gpus'];r['network']=r['fabric'];r['fabric']=_fabric(r['fabric']);r['collective']=_collective(r['collective'])
 if r['nccl_version'] is not None:r['nccl_version']=str(r['nccl_version'])
 base_point={key:r.get(key) for key in ('message_bytes','datatype','operation','placement','protocol','algorithm','split_mask','latency_us','errors','collective')}
 points=next((raw[key] for key in ('metrics','points','results') if isinstance(raw.get(key),list)),None)
 if points is None:points=[raw]
 normalized=[]
 for point in points:
  if not isinstance(point,dict):continue
  pf=beacon._flatten(point);metric=dict(base_point);pm={}
  for field in base_point:
   path,value=_pick(pf,FIELDS[field])
   if path:metric[field]=value;pm[field]=path
  metric['message_bytes']=beacon._number(metric['message_bytes'],integer=True)
  metric['errors']=beacon._number(metric['errors'],integer=True)
  metric['latency_us']=beacon._number(metric['latency_us'])
  for field in ('busbw_gbps','algbw_gbps'):
   path,value=_pick(pf,FIELDS[field]);unit_path,unit=_pick(pf,['bandwidth_unit','bandwidth_units','bw_unit','units.bandwidth'])
   if unit is None:_,unit=_pick(flat,['bandwidth_unit','bandwidth_units','bw_unit','units.bandwidth'])
   metric[field]=_bandwidth(value,path,unit)
   if path:pm[field]=path
  metric['bandwidth_unit']='GB/s';metric['source_mapping']=pm
  if point.get('actual_iterations') is not None:metric['actual_iterations']=beacon._number(point['actual_iterations'],True)
  metric['collective']=_collective(metric.get('collective'))
  if metric.get('placement') in ('in-place','inplace'):metric['placement']='in_place'
  if metric.get('placement') in ('out-of-place','outofplace'):metric['placement']='out_of_place'
  if any(metric.get(field) is not None for field in ('busbw_gbps','algbw_gbps','latency_us')):normalized.append(metric)
 r['metrics']=normalized
 # Single point keys are also convenient for point-oriented consumers.
 if len(normalized)==1:r.update({k:v for k,v in normalized[0].items() if k!='source_mapping'})
 for field in ('job_id','source_record_id'):
  if r[field] is not None:r[field]=str(r[field])
 raw_text=json.dumps(original_raw,sort_keys=True,separators=(',',':'),allow_nan=False);raw_sha=hashlib.sha256(raw_text.encode()).hexdigest()
 r.update(ref_id=hashlib.sha256((beacon.source_scope(source_url)+'\0'+(r['source_record_id'] or raw_sha)).encode()).hexdigest()[:32],
          source=('GPUComms' if urlsplit(source_url).hostname=='gpucomms.nvidia.com' else 'NVIDIA Confluence' if urlsplit(source_url).hostname=='nvidia.atlassian.net' else 'NVIDIA NCCL Performance Dashboard' if 'nccl-perf-dashboard-' in (urlsplit(source_url).hostname or '') else 'Beacon NCCL'),source_url=source_url,source_scope=beacon.source_scope(source_url),raw_json=original_raw,raw_sha256=raw_sha,
          source_mapping=mapping,refreshed_at=refreshed_at or datetime.now(timezone.utc).isoformat())
 for key in ('row_action','source_csv_url','source_metadata_url','page_id','page_version','page_title','section','table_index','row_index','column_header','confidence','reference_kind','reference_role','reference_origin','is_measured','theoretical','kind','source_provenance','provenance','source_test_profile','optimization_profile','source_recipe_scope','execution_status','test_protocol','nic_model','nic_type','link_speed_gbps','placement_scope','rail_topology','congestion_control','nccl_tests_version','nccl_environment','gpu_model_raw','network_context','fabric_evidence_scope','unit_provenance','source_archive_status','selected_group_status','repeat_count','reference_statistic','raw_asset_sha256','raw_asset_path','message_size_source','metadata'):
  if key in original_raw:r[key]=original_raw[key]
 if raw is not original_raw:
  for key in ('nic_model','link_speed_gbps','placement_scope','rail_topology','congestion_control','nccl_tests_version','nccl_environment','test_protocol','gpu_model_raw','network_context','fabric_evidence_scope','unit_provenance'):
   r[key]=raw.get(key)
  r['source_mapping'].update(raw.get('_source_mapping',{}))
 return r


def import_records(rows,source_url=SOURCE_URL,db_path=None,metadata=None,max_bytes=beacon.MAX_IMPORT_BYTES):
 """Stream reference records and both compressed snapshots with bounded memory."""
 root=_path(db_path).parent/'beacon-nccl-snapshots';root.mkdir(parents=True,exist_ok=True)
 raw_temp=root/(uuid.uuid4().hex+'.raw.partial');compact_temp=root/(uuid.uuid4().hex+'.compact.partial')
 now=datetime.now(timezone.utc).isoformat();digest=hashlib.sha256();total=0;count=0;measured=0;points=0
 identities=set();gpus=set();networks=set();meta=dict(metadata or {});snapshot_id=uuid.uuid4().hex
 try:
  with _db(db_path) as db:
   if meta.get('complete_all_pages'):db.execute('DELETE FROM nccl_references WHERE source_scope=?',(beacon.source_scope(source_url),))
   with raw_temp.open('wb') as raw_file, compact_temp.open('wb') as compact_file:
    with gzip.GzipFile(fileobj=raw_file,mode='wb',filename='',mtime=0) as raw_out, gzip.GzipFile(fileobj=compact_file,mode='wb',filename='',mtime=0) as compact_out:
     for row in rows:
      line=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode();total+=len(line)
      if total>max_bytes:raise ValueError('NCCL reference import exceeds its expanded size limit')
      ref=normalize_record(row,source_url,now);count+=1;measured+=bool(ref['metrics']);points+=len(ref['metrics'])
      identities.add(ref['ref_id'])
      if ref.get('gpu'):gpus.add(ref['gpu'])
      if ref.get('fabric'):networks.add(ref['fabric'])
      digest.update(line);raw_out.write(line)
      compact_out.write((json.dumps({k:ref.get(k) for k in COMPACT_FIELDS},separators=(',',':'))+'\n').encode())
      db.execute('INSERT OR REPLACE INTO nccl_references VALUES(?,?,?,?,?,?,?,?,?)',(ref['ref_id'],ref['source_scope'],ref['source_record_id'],ref['gpu'],ref['gpus'],ref['fabric'],ref['collective'],json.dumps(ref,allow_nan=False),snapshot_id))
   if not measured:raise ValueError('No measured NCCL bandwidth or latency with known units; existing references preserved')
   unique=len(identities)
   if meta.get('complete_all_pages') and meta.get('reported_total') is not None and unique!=meta['reported_total']:raise ValueError('Unique records do not match reported total; existing references preserved')
   sha=digest.hexdigest();path=root/(sha+'.jsonl.gz');compact_sha=hashlib.sha256(compact_temp.read_bytes()).hexdigest();compact_path=root/(compact_sha+'.compact.jsonl.gz')
   raw_temp.replace(path);compact_temp.replace(compact_path)
   with path.open('rb') as stream:compressed_sha=hashlib.file_digest(stream,'sha256').hexdigest()
   meta.update(unique_records=unique,duplicate_records=count-unique,measured_records=measured,metric_points=points,
     expanded_bytes=total,compressed_bytes=path.stat().st_size,compressed_sha256=compressed_sha,
     compact_path=str(compact_path),compact_sha256=compact_sha,compact_bytes=compact_path.stat().st_size,
     gpu_types=sorted(gpus),networks=sorted(networks))
   db.execute('INSERT OR REPLACE INTO nccl_snapshots VALUES(?,?,?,?,?,?,?)',(snapshot_id,now,source_url,count,str(path),sha,json.dumps(meta)))
  return dict(imported=count,unique_records=unique,measured_records=measured,metric_points=points,snapshot_id=snapshot_id,path=str(path),sha256=sha,compact_path=str(compact_path),compact_sha256=compact_sha)
 finally:
  for path in (raw_temp,compact_temp):
   if path.exists():path.unlink()


def import_file(path,source_url=SOURCE_URL,db_path=None):
 # Reuse the bounded, credential-free format reader without writing training refs.
 path=Path(path)
 if path.stat().st_size>beacon.MAX_IMPORT_BYTES:raise ValueError('NCCL import exceeds 50 MiB')
 with path.open('rb') as f:magic=f.read(2)
 with (gzip.open(path,'rb') if magic==b'\x1f\x8b' else path.open('rb')) as f:data=f.read(beacon.MAX_IMPORT_BYTES+1)
 if len(data)>beacon.MAX_IMPORT_BYTES:raise ValueError('NCCL import exceeds 50 MiB expanded text')
 text=data.decode('utf-8-sig');name=path.name.lower().removesuffix('.gz')
 if name.endswith('.csv'):
  import csv,io
  rows=list(csv.DictReader(io.StringIO(text)))
 elif name.endswith(('.jsonl','.ndjson')):rows=[json.loads(line) for line in text.splitlines() if line.strip()]
 else:rows=beacon.extract_rows(json.loads(text))
 return import_records(rows,source_url,db_path,{'method':'offline_export','import_filename':path.name,'import_sha256':hashlib.sha256(data).hexdigest()})


def list_references(filters=None,limit=100,offset=0,db_path=None,include_raw=False,include_metrics=False):
 filters=filters or {};clauses=[];values=[]
 for key in ('gpu','gpus','fabric','collective'):
  if filters.get(key) not in (None,''):clauses.append(key+'=?');values.append(filters[key])
 where=' WHERE '+' AND '.join(clauses) if clauses else '';limit=max(1,min(10000,int(limit)));offset=max(0,int(offset))
 with _db(db_path) as db:
  count=db.execute('SELECT COUNT(*) FROM nccl_references'+where,values).fetchone()[0]
  removed=[]
  if not include_raw:removed.append('$.raw_json')
  if not include_metrics:removed.append('$.metrics')
  projection='json_remove(normalized_json,'+','.join("'"+key+"'" for key in removed)+')' if removed else 'normalized_json'
  rows=[]
  preview="'[]'" if include_metrics else "(SELECT json_group_array(json(value)) FROM (SELECT value FROM json_each(normalized_json,'$.metrics') ORDER BY CASE WHEN json_extract(value,'$.message_bytes')=2147483648 THEN 0 ELSE 1 END,json_extract(value,'$.message_bytes') DESC,key LIMIT 3))"
  for item in db.execute('SELECT '+projection+",json_array_length(normalized_json,'$.metrics'),"+preview+' FROM nccl_references'+where+' ORDER BY ref_id LIMIT ? OFFSET ?',values+[limit,offset]):
   row=json.loads(item[0]);row['metric_points']=item[1];row['metric_preview']=json.loads(item[2])
   if include_metrics and not include_raw:
    for point in row.get('metrics',[]):point.pop('source_mapping',None)
   rows.append(row)
  facets={key:[row[0] for row in db.execute('SELECT DISTINCT '+key+' FROM nccl_references WHERE '+key+' IS NOT NULL ORDER BY '+key)] for key in ('gpu','gpus','fabric','collective')}
  snap=db.execute('SELECT * FROM nccl_snapshots ORDER BY created_at DESC LIMIT 1').fetchone()
 snapshot=dict(snap) if snap else None
 if snapshot:snapshot['metadata']=json.loads(snapshot.pop('metadata_json'))
 return dict(items=rows,total=count,limit=limit,offset=offset,facets=facets,snapshot=snapshot)


def references(db_path=None,filters=None,include_raw=False):
 # Matcher references retain all points, but do not duplicate full raw curves.
 page=list_references(filters,10000,0,db_path,include_raw=include_raw,include_metrics=True);rows=page['items']
 while len(rows)<page['total']:rows.extend(list_references(filters,10000,len(rows),db_path,include_raw=include_raw,include_metrics=True)['items'])
 for row in rows:
  for point in row.get('metrics',[]):point.pop('source_mapping',None)
 return rows


def status(db_path=None):
 page=list_references(limit=1,db_path=db_path)
 sources=[];measured=0;points=0
 with _db(db_path) as db:
  for group in db.execute('SELECT r.snapshot_id,count(*) AS active_records,s.records AS imported_records,s.source_url,s.created_at,s.metadata_json FROM nccl_references r LEFT JOIN nccl_snapshots s ON s.snapshot_id=r.snapshot_id GROUP BY r.snapshot_id'):
   meta=json.loads(group['metadata_json'] or '{}');active=group['active_records']
   if meta.get('measured_records')==group['imported_records']:count=active
   else:count=db.execute("SELECT count(*) FROM nccl_references WHERE snapshot_id=? AND json_array_length(normalized_json,'$.metrics')>0",(group['snapshot_id'],)).fetchone()[0]
   if active==group['imported_records'] and meta.get('metric_points') is not None:metric_points=meta['metric_points']
   else:metric_points=db.execute("SELECT coalesce(sum(json_array_length(normalized_json,'$.metrics')),0) FROM nccl_references WHERE snapshot_id=?",(group['snapshot_id'],)).fetchone()[0]
   measured+=count;points+=metric_points
   sources.append(dict(source_url=group['source_url'],snapshot_id=group['snapshot_id'],records=active,measured_records=count,metric_points=metric_points,created_at=group['created_at']))
 return dict(status='imported' if page['total'] else 'source_available',records=page['total'],gpu_types=page['facets']['gpu'],networks=page['facets']['fabric'],
             measured_records=measured,metric_points=points,sources=sources,
             snapshot=page['snapshot'],discovery=DISCOVERY,source_url=SOURCE_URL,requires_authentication=False,
             instructions='References include imported measured sources with per-source provenance. Refresh the applicable connector or import an actual NCCL export. Missing transport, link speed and test context remain unknown; training TFLOPS are never substituted.')


def fetch_all(api_url,db_path=None,max_pages=10000,progress=None):
 origin=urlsplit(api_url)
 if origin.scheme!='https' or origin.hostname not in config.BEACON_HOSTS:raise ValueError('NCCL connector requires the observed HTTPS llmb.nvidia.com API URL; credentials cannot be sent to other domains')
 if '/training/' in origin.path:raise ValueError('The training endpoint cannot establish NCCL bandwidth references')
 if any(re.search(r'token|cookie|secret|password',key,re.I) for key,_ in parse_qsl(origin.query)):raise ValueError('Credentials belong in private files, not the API URL')
 headers=beacon.api_headers(api_url);opener=build_opener(beacon._SameOriginRedirect());url=api_url;seen=set();rows=[];total=None;pages=0
 while url:
  if url in seen or pages>=max_pages:raise ValueError('Incomplete or repeated pagination; existing NCCL references preserved')
  seen.add(url);pages+=1
  with opener.open(Request(url,headers=headers),timeout=45) as response:
   if 'json' not in response.headers.get('Content-Type',''):raise ValueError('NCCL endpoint returned a login page or non-JSON response')
   data=response.read(beacon.MAX_IMPORT_BYTES+1)
   if len(data)>beacon.MAX_IMPORT_BYTES:raise ValueError('NCCL response page exceeds 50 MiB')
   payload=json.loads(data)
  batch=beacon.extract_rows(payload);rows.extend(batch)
  if progress:progress(dict(pages=pages,records=len(rows)))
  next_link=None;cursor=None
  if isinstance(payload,dict):
   if isinstance(payload.get('total'),int):total=payload['total']
   next_link=payload.get('next') or payload.get('next_url');cursor=payload.get('next_cursor')
  if next_link:
   url=urljoin(url,next_link)
   if urlsplit(url).scheme!='https' or urlsplit(url).netloc!=origin.netloc:raise ValueError('Cross-origin NCCL pagination refused')
  elif cursor:
   query=dict(parse_qsl(origin.query));query.update(c_d='after',c_id=str(cursor),include_total='false');url=urlunsplit(origin._replace(query=urlencode(query)))
  else:url=None
 if total is not None and len(rows)!=total:raise ValueError('NCCL pagination count does not match total; existing references preserved')
 return import_records(rows,api_url,db_path,{'method':'authenticated_api','pages':pages,'reported_total':total,'complete_all_pages':True},512*1024**2)


def fetch_gpucomms(session_context,db_path=None,max_workers=4,progress=None,max_queries=4000):
 """Read the public golden dataset through its normal <=200-row UI queries.

 The context contains only this public site's own browser session and CSRF
 state. Beacon/NVIDIA login credentials are explicitly refused on this host.
 """
 import copy
 from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
 from http.cookies import SimpleCookie
 headers={k:v for k,v in session_context['headers'].items() if not k.startswith(':') and k.lower() not in ('content-length','host','accept-encoding')}
 if any(k.lower()=='authorization' for k in headers):raise ValueError('GPUComms public queries must not receive Beacon authorization headers')
 cookie=SimpleCookie();cookie.load(next((v for k,v in headers.items() if k.lower()=='cookie'),''))
 if any(key!='gcperf_session' for key in cookie):raise ValueError('Only the public GPUComms session cookie may be used on gpucomms.nvidia.com')
 template=session_context['template'];options=session_context['options'];expected=int(session_context['advertised_total'])
 dims=[('num_nodes','integer'),('tool_binary','string'),('cluster_name','string'),('library_and_version','string'),('name','string')]
 destination=config.GPUCOMMS_URL+'_dash-update-component'
 def query(criteria,depth):
  body=copy.deepcopy(template);body['inputs'][1]['value']=criteria;body['changedPropIds']=['filter-criteria-store.data']
  opener=build_opener(beacon._SameOriginRedirect())
  with opener.open(Request(destination,data=json.dumps(body).encode(),headers=headers),timeout=90) as response:
   data=response.read(100*1024**2+1)
   if len(data)>100*1024**2:raise ValueError('GPUComms partition exceeded response size bound')
   result=json.loads(data)['response']
  count=int(str(result['filtered-tests-count-detail']['children']).replace(',',''))
  rows=result.get('filtered-data-table',{}).get('rowData',[])
  return criteria,depth,count,rows
 root=_path(db_path).parent/'beacon-nccl-snapshots';root.mkdir(parents=True,exist_ok=True)
 raw_path=root/(uuid.uuid4().hex+'.gpucomms.partial.jsonl');seen=set();queries=0;leaves=0;source_ids=set()
 try:
  with raw_path.open('w') as out,ThreadPoolExecutor(max_workers=max_workers) as executor:
   pending={executor.submit(query,[],0)}
   while pending:
    done,pending=wait(pending,return_when=FIRST_COMPLETED)
    for future in done:
     criteria,depth,count,rows=future.result();queries+=1
     if queries>max_queries:raise ValueError('GPUComms partition safety limit reached; existing references preserved')
     if count<=200:
      if len(rows)!=count:raise ValueError('GPUComms partition was truncated; existing references preserved')
      leaves+=1
      for row in rows:
       identity=row.get('id')
       if identity is None:raise ValueError('GPUComms result has no source identifier')
       if identity not in source_ids:
        source_ids.add(identity);out.write(json.dumps(row,separators=(',',':'))+'\n')
     else:
      if depth>=len(dims):raise ValueError('GPUComms group remains above its public 200-record limit; finer source filter required')
      field,field_type=dims[depth]
      for option in options.get(field,[]):
       value=option.get('value') if isinstance(option,dict) else option
       child=criteria+[dict(field=field,db_field_type=field_type,values=[str(value)])]
       key=json.dumps(child,sort_keys=True)
       if key not in seen:seen.add(key);pending.add(executor.submit(query,child,depth+1))
     if progress and queries%10==0:progress(dict(queries=queries,records=len(source_ids),expected=expected,pending=len(pending)))
  if len(source_ids)!=expected:raise ValueError('GPUComms unique partition rows do not equal the advertised total; existing references preserved')
  with raw_path.open() as stream:
   receipt=import_records((json.loads(line) for line in stream),config.GPUCOMMS_URL,db_path,
      dict(method='public_ui_partition_queries',source_dataset='golden_data',complete_all_pages=True,reported_total=expected,
           test_tool='nccl-tests',queries=queries,leaf_partitions=leaves,query_record_limit=200,discovery_source='https://exemplar.nvidia.com/resources/performance-troubleshooting.json'),
      max_bytes=2*1024**3)
  return receipt
 finally:
  if raw_path.exists():raw_path.unlink()


def normalize_protocol_aliases(db_path=None):
 """Add canonical names to retained native NCCL test configuration, in place.

 Native keys and immutable raw source snapshots remain intact. This also
 upgrades an index produced by an older long-running fetch worker.
 """
 mappings={'min_bytes':'minbytes','max_bytes':'maxbytes','iterations':'iters','warmup':'warmup_iters'}
 args=[]
 for canonical,native in mappings.items():
  args.extend(["'$.test_protocol."+canonical+"'", "COALESCE(json_extract(normalized_json,'$.test_protocol."+canonical+"'),json_extract(normalized_json,'$.test_protocol."+native+"'))"])
 register="CASE lower(CAST(json_extract(normalized_json,'$.test_protocol.local_register') AS TEXT)) WHEN 'disabled' THEN 0 WHEN 'local_register_all' THEN 1 WHEN 'symmetric_register_all' THEN 2 WHEN '0' THEN 0 WHEN '1' THEN 1 WHEN '2' THEN 2 ELSE NULL END"
 args.extend(["'$.test_protocol.registration'",register])
 provenance=dict(version=1,applied_at=datetime.now(timezone.utc).isoformat(),native_keys_retained=True,
   aliases=mappings,registration={'disabled':0,'LOCAL_REGISTER_ALL':1,'SYMMETRIC_REGISTER_ALL':2},raw_snapshot_unchanged=True)
 with _db(db_path) as db:
  cursor=db.execute('UPDATE nccl_references SET normalized_json=json_set(normalized_json,'+','.join(args)+") WHERE source_scope='https://gpucomms.nvidia.com/'")
  provenance['records_updated']=cursor.rowcount
  db.execute("UPDATE nccl_snapshots SET metadata_json=json_set(metadata_json,'$.protocol_normalization',json(?)) WHERE snapshot_id=(SELECT snapshot_id FROM nccl_snapshots ORDER BY created_at DESC LIMIT 1)",(json.dumps(provenance),))
 return provenance
