"""Observed NCCL release curves from NVIDIA's linked performance dashboard.

Only catalogued *_observed.csv measurements are imported. Manual target curves
and theoretical SOL remain separate source annotations, never measurements.
"""
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shlex
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from . import config
BASE_URL = config.NCCL_DASHBOARD_URL
GUIDANCE_URL = os.environ.get('NCCL_DASHBOARD_GUIDANCE_URL','')
DEFAULT_SNAPSHOTS = config.data_path('NCCL_DASHBOARD_SNAPSHOTS','nccl-dashboard-snapshots')
CURVE = re.compile(r'^(?P<platform>[^/]+)/(?P<release>[^/]+)/(?P<collective>[^/]+)/(?P<kernel>.+?)/nvls_(?P<nvls>on|off)/(?P<domains>\d+)nvld/(?P=collective)_(?P=domains)nvld_(?P=domains)nvl(?P<width>\d+)_observed\.csv$')


def utcnow(): return datetime.now(timezone.utc).isoformat()


def safe_path(path):
    value = PurePosixPath(path)
    if urlsplit(path).scheme or urlsplit(path).netloc or value.is_absolute() or '..' in value.parts:
        raise ValueError('Dashboard asset must be a relative same-origin path')
    return str(value)


class SnapshotFetcher:
    def __init__(self, directory, base_url=BASE_URL, timeout=30):
        self.directory = Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.base_url = base_url.rstrip('/')+'/';self.timeout = timeout
        self.assets = {};self.lock = threading.Lock()

    def __call__(self,path,optional=False):
        path = safe_path(path);url = urljoin(self.base_url,path)
        for attempt in range(3):
            try:
                with urlopen(Request(url,headers={'User-Agent':'Validation-Hub-NCCL-Reference/1.0','Accept':'application/json,text/csv,text/plain'}),timeout=self.timeout) as response:
                    if urlsplit(response.url).netloc != urlsplit(self.base_url).netloc:
                        raise ValueError('Dashboard redirected away from its source origin')
                    body = response.read(4*1024*1024+1)
                    if len(body)>4*1024*1024:raise ValueError('Dashboard asset exceeds 4 MiB')
                    headers = {key:response.headers[key] for key in ('ETag','Last-Modified','Content-Type') if response.headers.get(key)}
                sha = hashlib.sha256(body).hexdigest();target=self.directory/(sha+'.gz')
                if not target.exists():target.write_bytes(gzip.compress(body,mtime=0))
                with self.lock:self.assets[path]=dict(url=url,status=200,sha256=sha,bytes=len(body),gzip_path=str(target),headers=headers)
                return body
            except HTTPError as error:
                if error.code==404 and optional:
                    with self.lock:self.assets[path]=dict(url=url,status=404)
                    return None
                if error.code not in (429,500,502,503,504) or attempt==2:raise
                time.sleep(0.5*(attempt+1))
        raise RuntimeError('Dashboard request did not complete')


def command_context(command):
    """Read explicit command flags; do not supply executable defaults."""
    try:tokens=shlex.split(command or '')
    except ValueError:tokens=[]
    # srun commonly wraps the executable and its flags in bash -c quotes.
    # Tokenize those command arguments without evaluating any shell content.
    nested=[]
    for token in tokens:
        if 'perf' in token and re.search(r'\s-[A-Za-z]',token):
            try:nested.extend(shlex.split(token))
            except ValueError:pass
    tokens.extend(nested)
    aliases={'-b':'min_bytes','-e':'max_bytes','-n':'iterations','-w':'warmup','-R':'registration',
             '-G':'cuda_graph_launches','-x':'device_implementation','-D':'custom_gin_kernel',
             '-f':'step_factor','-t':'threads','-g':'gpus_per_thread','-d':'datatype','-o':'operation'}
    result={};environment={};nodes=None
    def size(value):
        m=re.fullmatch(r'(\d+(?:\.\d+)?)([kKmMgG]?)',value)
        if not m:return None
        return int(float(m[1])*{'':1,'k':1024,'m':1024**2,'g':1024**3}[m[2].lower()])
    for index,token in enumerate(tokens):
        env=re.fullmatch(r'(NCCL_[A-Z0-9_]+)=(.*)',token)
        if env:environment[env[1]]=env[2]
        if token=='-N' and index+1<len(tokens) and tokens[index+1].isdigit():nodes=int(tokens[index+1])
        elif re.fullmatch(r'-N\d+',token):nodes=int(token[2:])
        elif token.startswith('--nodes=') and token.split('=',1)[1].isdigit():nodes=int(token.split('=',1)[1])
        for flag,field in aliases.items():
            if token==flag and index+1<len(tokens):value=tokens[index+1]
            elif token.startswith(flag) and len(token)>len(flag):value=token[len(flag):]
            else:continue
            if field in ('min_bytes','max_bytes'):parsed=size(value)
            elif field in ('datatype','operation'):parsed=value
            else:parsed=int(value) if value.isdigit() else None
            if parsed is not None:result[field]=parsed
    return dict(test_protocol={k:v for k,v in result.items() if k not in ('datatype','operation')},
                environment=environment,nodes=nodes,datatype=result.get('datatype'),operation=result.get('operation'))


def _positive(value):
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) and value>=0 else None


def parse_curve(path,csv_bytes,metadata,platform=None,metadata_url=None,base_url=BASE_URL):
    """Create one source record, preserving a curve's complete declared geometry."""
    match=CURVE.fullmatch(safe_path(path))
    if not match:raise ValueError('Unsupported observed NCCL curve path: '+path)
    ctx=match.groupdict();metadata=metadata or {};platform=platform or {}
    if not isinstance(metadata,dict):raise ValueError('Curve metadata must be an object')
    command=metadata.get('observed_cmd') or ''
    parsed=command_context(command)
    notes=str(metadata.get('collection_notes') or '')
    source=str(metadata.get('observed_source') or '')
    evidence=notes+' '+source
    gpu=platform.get('tag') or ctx['platform'].split('_',1)[0].upper()
    domains,width=int(ctx['domains']),int(ctx['width'])
    site_fabric='RoCE' if '_roce' in ctx['platform'] else 'InfiniBand' if ctx['platform'].endswith('_ib') else None
    spectrum=bool(re.search(r'spectrum[ -]?x',evidence,re.I))
    # One explicitly declared NVLink domain is distinct from a cross-domain
    # network test. The site's NIC/fabric remains separately recorded.
    fabric='NVLink' if domains==1 else 'Spectrum-X' if spectrum else site_fabric
    network=fabric
    nodes=parsed['nodes']
    if nodes is None:
        node_match=re.search(r'\b(\d+)\s+nodes?\b',evidence,re.I)
        if node_match:nodes=int(node_match[1])
    nic_match=re.search(r'\bcx(\d+)\b',ctx['platform'],re.I)
    nic_model='ConnectX-'+nic_match[1] if nic_match else None
    repeat=re.search(r'\b(\d+)\s+(?:successful\s+)?runs?\b',evidence,re.I)
    repeat_count=int(repeat[1]) if repeat else None
    archive_failed=bool(re.search(r'(?:archive|tarball)[^.]{0,100}(?:label(?:ed)?[^.]{0,20})?\bFAILED\b',evidence,re.I))
    correct=bool(re.search(r'(?:errors?\s*=\s*0|zero\s+(?:out-of-place\s+)?#wrong)',notes,re.I))
    completed=bool(re.search(r'(?:successful runs?|complete result footers|completed \d+/\d+)',notes,re.I))
    selected_passed=correct and completed
    placement='out_of_place' if re.search(r'out-of-place\s+busbw\s+only',notes,re.I) else None
    protocol=dict(parsed['test_protocol'])
    protocol.update(nvlink_domains=domains,gpus_per_nvlink_domain=width,dashboard_kernel=ctx['kernel'],dashboard_nvls=ctx['nvls'])
    metrics=[]
    text=csv_bytes.decode('utf-8-sig') if isinstance(csv_bytes,bytes) else csv_bytes
    reader=csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {'bytes_transferred','busbw_GB_s'}.issubset(reader.fieldnames):
        raise ValueError('Observed CSV lacks bytes_transferred/busbw_GB_s columns')
    for row in reader:
        size=_positive(row.get('bytes_transferred'));bandwidth=_positive(row.get('busbw_GB_s'))
        if size is None or bandwidth is None or not size.is_integer():raise ValueError('Observed CSV contains invalid measurement')
        metrics.append(dict(message_bytes=int(size),busbw_gbps=bandwidth,datatype=parsed['datatype'],operation=parsed['operation'],
                            placement=placement,errors=0 if selected_passed else None,bandwidth_unit='GB/s'))
    if not metrics:raise ValueError('Observed CSV is empty')
    csv_url=urljoin(base_url,'data/'+path)
    job=re.search(r'\bjob\s+(\d+)\b',notes,re.I)
    record=dict(id=path,source_record_id=path,gpu=gpu,gpus=domains*width,nodes=nodes,fabric=fabric,network=network,
        collective=ctx['collective'],nccl_version=ctx['release'],job_id=job[1] if job else None,
        cluster=metadata.get('observed_cluster'),status='passed' if selected_passed else None,
        metrics=metrics,bandwidth_unit='GB/s',nic_model=nic_model,link_speed_gbps=None,
        placement_scope='single_nvlink_domain' if domains==1 else 'multiple_nvlink_domains',
        rail_topology=None,congestion_control=None,nccl_tests_version=None,
        test_protocol=protocol,nccl_environment=parsed['environment'] if command else None,
        protocol=parsed['environment'].get('NCCL_PROTO'),split_mask=parsed['environment'].get('NCCL_TESTS_SPLIT_MASK'),
        reference_kind='measured_observed',is_measured=True,theoretical=False,
        reference_origin='nvidia_nccl_performance_dashboard',source_csv_url=csv_url,
        source_metadata_url=metadata_url,row_action=csv_url,source_url=csv_url,
        selected_group_status='passed' if selected_passed else 'completion_not_provided',
        source_archive_status='failed_label_in_source_notes' if archive_failed else 'not_provided',
        archive_status_caveat=notes if archive_failed else None,repeat_count=repeat_count,
        reference_statistic='mean_per_message_size' if re.search(r'averag(?:e|ed)',evidence,re.I) else 'source_observation',
        nvlink_domains=domains,gpus_per_nvlink_domain=width,kernel=ctx['kernel'],nvls=ctx['nvls'],
        site_fabric=site_fabric,source_fabric_evidence='Explicit single NVLink domain' if domains==1 else 'Spectrum-X explicitly named in observed collection notes' if spectrum else 'Published platform network label',
        source_command=command,source_notes=notes,source_metadata=metadata,
        source_platform=platform,unit_provenance=dict(bandwidth='CSV busbw_GB_s, decimal GB/s',message_bytes='CSV bytes_transferred'),
        theoretical_sol=dict(value=metadata.get('sol_GBs'),basis=metadata.get('sol_basis'),used_as_measurement=False),
        source_provenance=dict(csv_url=csv_url,metadata_url=metadata_url,source_notes=notes,source_command=command,
                               source_archive_status='failed_label_in_source_notes' if archive_failed else 'not_provided',
                               selected_group_status='passed' if selected_passed else 'completion_not_provided'),
        source_test_profile=dict(nvlink_domains=domains,gpus_per_nvlink_domain=width,kernel=ctx['kernel'],nvls=ctx['nvls'],site_fabric=site_fabric),
        raw_asset_sha256=hashlib.sha256(csv_bytes if isinstance(csv_bytes,bytes) else csv_bytes.encode()).hexdigest(),
        metadata=dict(correctness_errors=0 if selected_passed else None,nvlink_domains=domains,gpus_per_nvlink_domain=width))
    return record


def collect(*,base_url=BASE_URL,snapshot_root=DEFAULT_SNAPSHOTS,progress=None,max_workers=4,
            platform=None,collective=None,gpus=None,kernel=None,limit=None,getter=None):
    if not 1<=max_workers<=4:raise ValueError('Use one to four concurrent source requests')
    root=Path(snapshot_root);directory=root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);directory.mkdir(parents=True,exist_ok=True)
    fetcher=getter or SnapshotFetcher(directory/'assets',base_url)
    raw_catalog=fetcher('catalog.json');raw_platforms=fetcher('platforms/manifest.json')
    catalog=json.loads(raw_catalog);platforms=json.loads(raw_platforms)
    if not isinstance(catalog,dict) or not isinstance(catalog.get('observed'),list):raise ValueError('Dashboard catalog lacks observed list')
    platform_map={item['id']:item for item in platforms.get('platforms',[])}
    paths=list(dict.fromkeys(catalog['observed']))
    selected=[]
    for path in paths:
        match=CURVE.fullmatch(safe_path(path))
        if not match:raise ValueError('Unrecognized observed catalog geometry: '+path)
        values=match.groupdict()
        if platform and values['platform']!=platform:continue
        if collective and values['collective']!=collective:continue
        if kernel and values['kernel']!=kernel:continue
        if gpus and int(values['domains'])*int(values['width'])!=int(gpus):continue
        selected.append(path)
    if limit is not None:selected=selected[:max(0,int(limit))]
    if not selected:raise ValueError('No observed curves match the requested scope')
    manifest=dict(base_url=base_url,created_at=utcnow(),catalog_records=len(paths),selected_records=len(selected),
                  filters=dict(platform=platform,collective=collective,gpus=gpus,kernel=kernel,limit=limit),
                  policy='Only observed CSV measurements; target curves and SOL excluded',curves=[],assets={})
    manifest_path=directory/'manifest.json';started=time.monotonic()
    def one(path):
        csv_data=fetcher('data/'+path)
        stem=path.removesuffix('_observed.csv');metadata=None;metadata_path=None
        for candidate in (stem+'_target.meta.json',stem+'.meta.json'):
            content=fetcher('data/'+candidate,optional=True)
            if content is not None:
                metadata=json.loads(content);metadata_path='data/'+candidate;break
        metadata_url=urljoin(base_url,metadata_path) if metadata_path else None
        record=parse_curve(path,csv_data,metadata,platform_map.get(path.split('/',1)[0]),metadata_url,base_url)
        asset=getattr(fetcher,'assets',{}).get('data/'+path,{})
        if asset.get('gzip_path'):record['raw_asset_path']=asset['gzip_path']
        return record,dict(path=path,metadata_path=metadata_path,points=len(record['metrics']),
                           csv_sha256=hashlib.sha256(csv_data).hexdigest(),selected_group_status=record['selected_group_status'])
    rows=[]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures={pool.submit(one,path):path for path in selected}
            for future in concurrent.futures.as_completed(futures):
                record,entry=future.result();rows.append(record);manifest['curves'].append(entry)
                if progress and (len(rows)%10==0 or len(rows)==len(selected)):
                    progress(dict(stage='collecting_observed_curves',completed=len(rows),total=len(selected),elapsed_seconds=round(time.monotonic()-started,1)))
                if len(rows)%10==0:
                    manifest['assets']=dict(getattr(fetcher,'assets',{}));manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        rows.sort(key=lambda row:row['id']);manifest['curves'].sort(key=lambda row:row['path'])
        manifest.update(complete=True,metric_points=sum(len(row['metrics']) for row in rows),elapsed_seconds=round(time.monotonic()-started,2),assets=dict(getattr(fetcher,'assets',{})))
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        return rows,manifest_path,manifest
    except BaseException as error:
        manifest.update(complete=False,error=type(error).__name__,assets=dict(getattr(fetcher,'assets',{})))
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');raise


def index(*,db_path=None,progress=None,**options):
    from . import beacon_nccl
    rows,path,manifest=collect(progress=progress,**options)
    complete = len(rows)==manifest['catalog_records'] and not any(manifest['filters'].values())
    result=beacon_nccl.import_records(rows,source_url=options.get('base_url',BASE_URL),db_path=db_path,
        metadata=dict(method='official_nccl_dashboard_observed_catalog',complete_all_pages=complete,
                      reported_total=len(rows),manifest_path=str(path),catalog_records=manifest['catalog_records'],
                      platforms=sorted({row['source_platform'].get('id') for row in rows}),
                      source_guidance_url=GUIDANCE_URL,source_policy=manifest['policy']),max_bytes=64*1024**2)
    result.update(manifest_path=str(path),catalog_records=manifest['catalog_records'],complete_catalog=complete)
    return result
