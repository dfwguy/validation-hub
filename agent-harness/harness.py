#!/usr/bin/env python3
"""Provider-neutral, local ValidationHub deployment with private durable receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import build_opener, ProxyHandler, Request, HTTPRedirectHandler
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
FIELDS = {'schema_version','project_name','runtime_dir','persistent_volume','host_port','startup_timeout_seconds','environment_file','secrets_dir','secrets_read_only'}
ENV_FIELDS = {'BEACON_BASE_URL','BEACON_API_URL','GPUCOMMS_URL','NCCL_DASHBOARD_URL','NCCL_DASHBOARD_GUIDANCE_URL','ATLASSIAN_SITE','ATLASSIAN_CLOUD_ID','ATLASSIAN_EMAIL','BEACON_HEADERS_FILE','BEACON_COOKIE_FILE','ATLASSIAN_TOKEN_FILE'}
SECRET_FIELDS = {'BEACON_HEADERS_FILE','BEACON_COOKIE_FILE','ATLASSIAN_TOKEN_FILE'}


class Blocked(ValueError):
    pass


def contained(path, parent):
    return path == parent or parent in path.parents


def absolute(value, field):
    if not isinstance(value,str) or not value or not Path(value).is_absolute():raise Blocked(field+' must be an absolute path')
    return Path(value).resolve()


def load_config(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise Blocked('Duplicate configuration key: '+key)
            result[key]=value
        return result
    data=json.loads(Path(path).read_text(),object_pairs_hook=pairs)
    if not isinstance(data,dict) or set(data)-FIELDS:raise Blocked('Configuration has unsupported fields')
    required={'schema_version','project_name','runtime_dir','persistent_volume'}
    if required-set(data):raise Blocked('Missing configuration fields: '+','.join(sorted(required-set(data))))
    if type(data['schema_version']) is not int or data['schema_version']!=1:raise Blocked('Unsupported schema_version')
    for key in ('project_name','persistent_volume'):
        if not isinstance(data[key],str) or not re.fullmatch(r'[a-z][a-z0-9_-]{2,62}',data[key]):raise Blocked(key+' must be a lowercase deployment-local name (3–63 characters)')
    root=absolute(data['runtime_dir'],'runtime_dir')
    if contained(root,REPO) or root==Path('/'):raise Blocked('runtime_dir must be outside the checkout and cannot be /')
    data['runtime_dir']=str(root)
    for field,default,low,high in [('host_port',8081,1024,65535),('startup_timeout_seconds',120,1,1800)]:
        value=data.setdefault(field,default)
        if type(value) is not int or not low<=value<=high:raise Blocked(field+' is out of range')
    for field in ('environment_file','secrets_dir'):
        if data.get(field) is not None:
            target=absolute(data[field],field)
            if contained(target,REPO):raise Blocked(field+' must be outside the checkout')
            data[field]=str(target)
    value=data.setdefault('secrets_read_only',True)
    if type(value) is not bool:raise Blocked('secrets_read_only must be a boolean')
    return data


def private(path,directory=False):
    if not (path.is_dir() if directory else path.is_file()):raise Blocked('Required private '+('directory' if directory else 'file')+' is missing: '+str(path))
    if stat.S_IMODE(path.stat().st_mode)&0o077:raise Blocked('Private path allows group/world access: '+str(path))


def validate_environment(config):
    result={};name=config.get('environment_file')
    if name:
        path=Path(name);private(path)
        for line in path.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith('#'):continue
            key,sep,value=line.partition('=')
            if not sep or key not in ENV_FIELDS or key in result:raise Blocked('Environment file requires unique allowlisted KEY=value settings')
            if any(char in value for char in ('\x00','\n','$','`')):raise Blocked('Environment values must be literal settings, without expansion or shell syntax')
            if value and (key.endswith('_URL') or key=='ATLASSIAN_SITE'):
                parts=urlsplit(value)
                if parts.scheme!='https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:raise Blocked('Integration settings must be HTTPS URLs without credentials, query or fragment')
            result[key]=value
    secrets=config.get('secrets_dir')
    if secrets:
        private(Path(secrets),directory=True)
        if Path(secrets).stat().st_uid!=10001:raise Blocked('Mounted private secrets directory must be owned by container UID10001, including read-only mounts')
    for key in SECRET_FIELDS:
        value=result.get(key)
        if not value:continue
        if not secrets:raise Blocked('Credential file settings require secrets_dir')
        prefix='/run/validation-secrets/'
        if not value.startswith(prefix) or '/' in value[len(prefix):] or value[len(prefix):] in ('','.','..'):raise Blocked('Credential settings must name a file directly under /run/validation-secrets/')
        host=Path(secrets)/value[len(prefix):];private(host)
        if host.stat().st_uid!=10001:raise Blocked('Mounted private credential files must be readable by container UID10001; provision ownership outside the harness')
    if result.get('BEACON_COOKIE_FILE') and config['secrets_read_only']:raise Blocked('Beacon cookie rotation requires secrets_read_only=false and a UID10001-owned writable secrets directory')
    if secrets and not config['secrets_read_only'] and Path(secrets).stat().st_uid!=10001:raise Blocked('Writable secrets directory must be owned by container UID10001')
    return result


def compose_text(config):
    template=(REPO/'deploy/compose.yaml').read_text()
    substitutions=[('      context: ..','      context: '+json.dumps(str(REPO))),
                   ('    image: nvis-ova-validation-hub:local','    image: '+config['project_name']+':local'),
                   ('      - "127.0.0.1:8081:8081"','      - "127.0.0.1:'+str(config['host_port'])+':8081"')]
    for original,replacement in substitutions:
        if template.count(original)!=1:raise Blocked('Deployment template changed; review the harness renderer before continuing')
        template=template.replace(original,replacement)
    original='    env_file:\n      - ../.env\n'
    if template.count(original)!=1:raise Blocked('Deployment template environment contract changed')
    template=template.replace(original,'    env_file:\n      - '+json.dumps(config['environment_file'])+'\n' if config.get('environment_file') else '')
    if config.get('secrets_dir'):
        mode='ro' if config['secrets_read_only'] else 'rw'
        template=template.replace('      - validation-data:/var/lib/validation-hub','      - validation-data:/var/lib/validation-hub\n      - '+json.dumps(config['secrets_dir']+':/run/validation-secrets:'+mode))
    template=template.rstrip()+'\n    name: '+config['persistent_volume']+'\n'
    return template


def compose_command(config,*args):
    return ['docker','compose','--project-name',config['project_name'],'--file',str(Path(config['runtime_dir'])/'compose.yaml'),*args]


def source_identity():
    result=dict(source_commit=None,source_dirty=None,source_archive_manifest_sha256=None)
    try:
        top=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=REPO,capture_output=True,text=True,timeout=5)
        if top.returncode==0 and Path(top.stdout.strip()).resolve()==REPO:
            commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,timeout=5)
            dirty=subprocess.run(['git','status','--porcelain'],cwd=REPO,capture_output=True,text=True,timeout=5)
            if commit.returncode==0:result.update(source_commit=commit.stdout.strip(),source_dirty=bool(dirty.stdout.strip()))
    except (OSError,subprocess.TimeoutExpired):pass
    manifest=REPO/'FILE_MANIFEST.json'
    if manifest.is_file():
        content=manifest.read_bytes()
        result['source_archive_manifest_sha256']=hashlib.sha256(content).hexdigest()
        if result['source_commit'] is None:
            try:
                archive_commit=json.loads(content).get('commit')
                if isinstance(archive_commit,str) and re.fullmatch(r'[0-9a-f]{40}',archive_commit):result['source_commit']=archive_commit
            except (ValueError,AttributeError):pass
    return result


def plan(config):
    rendered=compose_text(config)
    digest=hashlib.sha256()
    for path in sorted([REPO/'Dockerfile',REPO/'requirements.lock',*REPO.glob('backend/*.py'),*REPO.glob('frontend/*'),*REPO.glob('tools/*.py'),*REPO.glob('agent-harness/*.py')]):
        if path.is_file():digest.update(str(path.relative_to(REPO)).encode()+b'\0'+path.read_bytes())
    return dict(source_identity(),status='PLANNED',source_sha256=digest.hexdigest(),provider='any local agent or operator',project=config['project_name'],
        url='http://127.0.0.1:'+str(config['host_port']),persistent_volume=config['persistent_volume'],
        template_sha256=hashlib.sha256((REPO/'deploy/compose.yaml').read_bytes()).hexdigest(),
        rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),runtime_dir=config['runtime_dir'],
        commands=[compose_command(config,'config','--quiet'),compose_command(config,'up','--build','--detach')],
        data_policy='Stop preserves the persistent volume; the harness never removes volumes or imports customer data.',
        rollback='No automatic rollback is implemented. A failed deployment may have changed containers; inspect its receipt, preserve data, and redeploy an approved known revision.')


def run(command,timeout=120):
    try:
        result=subprocess.run(command,cwd=REPO,capture_output=True,text=True,timeout=timeout)
    except FileNotFoundError:raise Blocked('Docker is unavailable; install it through the approved platform process (the harness never installs packages)') from None
    except subprocess.TimeoutExpired:raise Blocked('Command timed out; inspect local container state before retrying') from None
    if result.returncode:raise Blocked('Command failed (exit '+str(result.returncode)+'): '+' '.join(command[:3])+'; inspect Docker locally; raw process output is not copied into receipts')
    return result.stdout.strip()


def doctor(config):
    validate_environment(config);compose_text(config)
    root=Path(config['runtime_dir'])
    if root.exists():private(root,directory=True)
    if os.environ.get('DOCKER_HOST') and not os.environ['DOCKER_HOST'].startswith('unix://'):raise Blocked('Only a local Unix-socket Docker endpoint is supported')
    if not shutil.which('docker'):raise Blocked('Docker is unavailable; plan works without Docker, deployment requires an approved local Docker Engine + Compose plugin')
    endpoint=run(['docker','context','inspect','--format','{{.Endpoints.docker.Host}}'])
    if not endpoint.startswith('unix://'):raise Blocked('Docker context must use a local Unix socket')
    run(['docker','compose','version']);run(['docker','info','--format','{{.ServerVersion}}'])
    return dict(status='READY',checks=['strict configuration','private external runtime and credential paths','local Docker endpoint','Compose plugin','Docker daemon access'],gpu_tests_executed=False)


def secure_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w') as output:output.write(value)
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def receipt(config,action,result):
    root=Path(config['runtime_dir']);root.mkdir(parents=True,exist_ok=True,mode=0o700);private(root,directory=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    record=dict(source_identity(),source_sha256=plan(config)['source_sha256'],action=action,observed_at=datetime.now(timezone.utc).isoformat(),project=config['project_name'],result=result)
    target=root/'receipts'/(stamp+'-'+action+'.json');secure_write(target,json.dumps(record,indent=2)+'\n')
    secure_write(root/'latest.json',json.dumps(record,indent=2)+'\n')
    return str(target)


def verify(config):
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    opener=build_opener(ProxyHandler({}),NoRedirect());base='http://127.0.0.1:'+str(config['host_port']);evidence={}
    for route in ('/api/health','/api/summary'):
        try:
            with opener.open(Request(base+route,headers={'Accept':'application/json'}),timeout=5) as response:
                if response.status!=200 or response.url!=base+route:raise Blocked('Verification endpoint redirected or did not return200')
                raw=response.read(1024*1024+1)
                if len(raw)>1024*1024:raise Blocked('Verification response exceeds1MiB')
                data=json.loads(raw)
        except (HTTPError,URLError,TimeoutError,OSError,json.JSONDecodeError) as error:raise Blocked('Local HTTP verification failed: '+type(error).__name__) from None
        if not isinstance(data,dict):raise Blocked('Verification response must be a JSON object')
        if route=='/api/health' and data.get('ok') is not True:raise Blocked('Health endpoint is not ready')
        if route=='/api/summary' and (type(data.get('runs')) is not int or data['runs']<0):raise Blocked('Summary lacks a valid record count')
        # Receipts retain readiness/counts, never customer names or raw payloads.
        evidence[route]=dict(status=200,ok=data.get('ok')) if route.endswith('health') else dict(status=200,runs=data['runs'],source_files=data.get('sources'),cluster_count=len(data.get('clusters',[])))
    return dict(status='VERIFIED',url=base,evidence=evidence,gpu_tests_executed=False)


def container_evidence(config):
    raw=run(compose_command(config,'ps','--format','json'))
    try:
        rows=json.loads(raw)
        if isinstance(rows,dict):rows=[rows]
    except json.JSONDecodeError:
        try:rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
        except json.JSONDecodeError:raise Blocked('Compose service-state evidence is not valid JSON') from None
    if not isinstance(rows,list):raise Blocked('Compose service-state evidence is missing')
    matches=[row for row in rows if row.get('Service')=='validation-hub' and row.get('State')=='running']
    if len(matches)!=1:raise Blocked('Exactly one running validation-hub Compose service is required')
    row=matches[0]
    if not any(p.get('URL')=='127.0.0.1' and p.get('PublishedPort')==config['host_port'] and p.get('TargetPort')==8081 for p in row.get('Publishers') or []):raise Blocked('Compose service does not publish the reviewed localhost port')
    return dict(service='validation-hub',state='running',container_id=row.get('ID'),image=row.get('Image'),localhost_port=config['host_port'])


def execute(action,config):
    if action=='plan':return plan(config)
    if action=='doctor':return doctor(config)
    if action=='report':
        target=Path(config['runtime_dir'])/'latest.json'
        if not target.exists():raise Blocked('No receipt exists; no deployment success can be inferred')
        private(target);record=json.loads(target.read_text())
        if record.get('project')!=config['project_name']:raise Blocked('Receipt belongs to a different Compose project')
        return record
    if action=='verify':
        target=Path(config['runtime_dir'])/'compose.yaml'
        if not target.exists() or target.read_text()!=compose_text(config):raise Blocked('No matching rendered deployment exists; deploy this configuration before verification')
        previous=Path(config['runtime_dir'])/'latest.json'
        if previous.exists() and json.loads(previous.read_text()).get('project')!=config['project_name']:raise Blocked('Runtime directory belongs to a different Compose project')
        doctor(config)
        try:
            service=container_evidence(config)
            result=verify(config);result['compose_service']=service
        except Blocked as error:
            target=receipt(config,action,dict(status='FAILED',error=str(error)))
            raise Blocked(str(error)+'; receipt: '+target) from None
        result['receipt']=receipt(config,action,result);return result
    doctor(config)
    root=Path(config['runtime_dir']);root.mkdir(parents=True,exist_ok=True,mode=0o700);private(root,directory=True)
    previous=root/'latest.json'
    if previous.exists() and json.loads(previous.read_text()).get('project')!=config['project_name']:raise Blocked('Runtime directory belongs to a different Compose project')
    rendered=compose_text(config);target=root/'compose.yaml'
    if action=='stop':
        if not target.exists() or target.read_text()!=rendered:raise Blocked('No matching rendered deployment exists; refusing to stop another deployment')
        run(compose_command(config,'down'),timeout=120)
        result=dict(status='STOPPED',persistent_volume_preserved=config['persistent_volume'])
    else:
        secure_write(target,rendered);run(compose_command(config,'config','--quiet'))
        try:
            run(compose_command(config,'up','--build','--detach'),timeout=1800)
            deadline=time.monotonic()+config['startup_timeout_seconds']
            while True:
                try:result=verify(config);result['compose_service']=container_evidence(config);break
                except Blocked:
                    if time.monotonic()>=deadline:raise
                    time.sleep(1)
            result['status']='DEPLOYED';result['rendered_sha256']=hashlib.sha256(rendered.encode()).hexdigest()
        except Blocked as error:
            failure=dict(status='FAILED',error=str(error),next_action='Inspect Docker Compose state; data is preserved. No automatic rollback was attempted.')
            failure['receipt']=receipt(config,action,failure);raise Blocked(json.dumps(failure)) from None
    result['receipt']=receipt(config,action,result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['doctor','plan','deploy','verify','stop','report']);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv)
    try:result=execute(args.action,load_config(args.config))
    except (Blocked,OSError,json.JSONDecodeError) as error:
        print(json.dumps(dict(status='BLOCKED',error=str(error))),file=sys.stderr)
        return 2
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':sys.exit(main())
