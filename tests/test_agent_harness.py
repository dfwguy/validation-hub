import copy
import importlib.util
import json
from pathlib import Path
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest

SPEC=importlib.util.spec_from_file_location('validation_harness',Path(__file__).resolve().parents[1]/'agent-harness/harness.py')
h=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(h)


@pytest.fixture
def config(tmp_path):
    return dict(schema_version=1,project_name='test-hub',runtime_dir=str(tmp_path/'runtime'),persistent_volume='test-hub-data',host_port=8081,startup_timeout_seconds=1,secrets_read_only=True)


def test_plan_is_docker_free_and_has_no_state_mutation(config,monkeypatch):
    monkeypatch.setattr(h,'run',lambda *args,**kwargs:pytest.fail('Plan executed Docker'))
    result=h.execute('plan',config)
    assert result['status']=='PLANNED' and len(result['source_sha256'])==64
    assert not Path(config['runtime_dir']).exists()
    rendered=h.compose_text(config)
    assert '127.0.0.1:8081:8081' in rendered and 'name: test-hub-data' in rendered
    assert 'env_file:' not in rendered and '../.env' not in rendered
    assert 'read_only: true' in rendered and 'cap_drop:' in rendered
    assert '--volumes' not in str(result['commands']) and 'sudo' not in str(result['commands'])


@pytest.mark.parametrize('change',[dict(password='no'),dict(host_port=80),dict(host_port=True),dict(runtime_dir='/'),dict(project_name='x; touch /tmp/x'),dict(schema_version=True)])
def test_strict_config_rejects_unsafe_unknown_or_wrong_types(config,tmp_path,change):
    config.update(change);file=tmp_path/'config.json';file.write_text(json.dumps(config))
    with pytest.raises(h.Blocked):h.load_config(file)


def test_duplicate_keys_and_repository_runtime_refused(config,tmp_path):
    file=tmp_path/'config.json';file.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(h.Blocked,match='Duplicate'):h.load_config(file)
    config['runtime_dir']=str(h.REPO/'var/runtime');file.write_text(json.dumps(config))
    with pytest.raises(h.Blocked,match='outside'):h.load_config(file)


def test_private_environment_rejects_secrets_inline_and_shell_expansion(config,tmp_path):
    file=tmp_path/'runtime.env';file.chmod(0o600) if file.exists() else None
    config['environment_file']=str(file)
    for value in ['BEACON_TOKEN=secret\n','BEACON_BASE_URL=https://user:secret@example.test\n','ATLASSIAN_EMAIL=$(bad)\n']:
        file.write_text(value);file.chmod(0o600)
        with pytest.raises(h.Blocked):h.validate_environment(config)
    file.write_text('ATLASSIAN_EMAIL=operator@example.test\n');file.chmod(0o644)
    with pytest.raises(h.Blocked,match='group/world'):h.validate_environment(config)


def test_doctor_missing_docker_is_actionable(config,monkeypatch):
    monkeypatch.setattr(h.shutil,'which',lambda name:None)
    with pytest.raises(h.Blocked,match='Docker is unavailable'):h.doctor(config)
    assert not Path(config['runtime_dir']).exists()


def test_http_verify_retains_aggregate_evidence_not_customer_payload(config,monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body={'ok':True} if self.path=='/api/health' else {'runs':7,'sources':2,'clusters':[{'name':'private-customer','runs':7}]}
            payload=json.dumps(body).encode();self.send_response(200);self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        config['host_port']=server.server_port
        root=Path(config['runtime_dir']);root.mkdir(mode=0o700);h.secure_write(root/'compose.yaml',h.compose_text(config))
        monkeypatch.setattr(h,'doctor',lambda _:dict(status='READY'))
        monkeypatch.setattr(h,'container_evidence',lambda _:dict(service='validation-hub',state='running'))
        result=h.execute('verify',config)
        assert result['status']=='VERIFIED' and result['evidence']['/api/summary']['runs']==7
        assert 'private-customer' not in json.dumps(result)
        target=Path(result['receipt']);assert stat.S_IMODE(target.stat().st_mode)==0o600
        assert stat.S_IMODE(target.parent.parent.stat().st_mode)==0o700
        assert h.execute('report',config)['result']['status']=='VERIFIED'
    finally:server.shutdown();server.server_close();thread.join()


def test_redirect_is_never_followed(config):
    class Handler(BaseHTTPRequestHandler):
        calls=0
        def do_GET(self):
            Handler.calls+=1;self.send_response(302);self.send_header('Location','http://example.invalid/');self.end_headers()
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        config['host_port']=server.server_port
        with pytest.raises(h.Blocked):h.verify(config)
        assert Handler.calls==1
    finally:server.shutdown();server.server_close();thread.join()


def test_deploy_records_failure_without_rollback_or_volume_deletion(config,monkeypatch):
    commands=[];monkeypatch.setattr(h,'doctor',lambda _:dict(status='READY'))
    def run(command,**kwargs):
        commands.append(command)
        if 'up' in command:raise h.Blocked('synthetic build failure')
        return ''
    monkeypatch.setattr(h,'run',run)
    with pytest.raises(h.Blocked,match='No automatic rollback'):h.execute('deploy',config)
    receipt=json.loads((Path(config['runtime_dir'])/'latest.json').read_text())
    assert receipt['result']['status']=='FAILED'
    assert not any('down' in cmd or '--volumes' in cmd for cmd in commands)
    assert 'synthetic build failure' in receipt['result']['error']


def test_deploy_requires_running_compose_service_and_stop_preserves_volume(config,monkeypatch):
    commands=[];monkeypatch.setattr(h,'doctor',lambda _:dict(status='READY'));monkeypatch.setattr(h,'verify',lambda _:dict(status='VERIFIED',evidence={}))
    def run(command,**kwargs):
        commands.append(command)
        return json.dumps([dict(Service='validation-hub',State='running',ID='synthetic-id',Image='test-hub:local',Publishers=[dict(URL='127.0.0.1',PublishedPort=8081,TargetPort=8081)])]) if 'ps' in command else ''
    monkeypatch.setattr(h,'run',run)
    assert h.execute('deploy',config)['status']=='DEPLOYED'
    result=h.execute('stop',config)
    assert result['status']=='STOPPED' and result['persistent_volume_preserved']=='test-hub-data'
    assert commands[-1][-1]=='down' and '--volumes' not in commands[-1]
    wrong=copy.deepcopy(config);wrong['host_port']=8082
    with pytest.raises(h.Blocked,match='matching rendered'):h.execute('stop',wrong)



def test_standalone_verify_refuses_unowned_existing_http_service(config,monkeypatch):
    monkeypatch.setattr(h,'verify',lambda _:pytest.fail('HTTP verification reached an unowned service'))
    with pytest.raises(h.Blocked,match='matching rendered'):h.execute('verify',config)
    root=Path(config['runtime_dir']);root.mkdir(mode=0o700);h.secure_write(root/'compose.yaml',h.compose_text(config))
    monkeypatch.setattr(h,'doctor',lambda _:dict(status='READY'))
    monkeypatch.setattr(h,'container_evidence',lambda _:(_ for _ in ()).throw(h.Blocked('No running owned service')))
    with pytest.raises(h.Blocked,match='owned service'):h.execute('verify',config)
    record=json.loads((root/'latest.json').read_text())
    assert record['result']['status']=='FAILED'


def test_source_archive_identity_without_git(tmp_path,monkeypatch):
    manifest=tmp_path/'FILE_MANIFEST.json';manifest.write_text(json.dumps({'schema_version':1,'commit':'b'*40,'files':[]}))
    monkeypatch.setattr(h,'REPO',tmp_path)
    def unavailable(*args,**kwargs):raise FileNotFoundError('Git is not installed')
    monkeypatch.setattr(h.subprocess,'run',unavailable)
    identity=h.source_identity()
    assert identity['source_commit']=='b'*40 and identity['source_dirty'] is None
    assert identity['source_archive_manifest_sha256']==h.hashlib.sha256(manifest.read_bytes()).hexdigest()
