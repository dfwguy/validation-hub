"""Xid decoding and host attribution use only what the log states (synthetic fixtures)."""
import re
import pytest
from fastapi.testclient import TestClient
from backend import crash_enrichment as ce
from backend import store
from backend.app import app
from backend.troubleshooting import SIGNATURES, analyze
from backend.upload_io import DiskLog

NCCL_WARN = ('gpu-node-05:100:200 [0] transport/net_ib.cc:1 NCCL WARN NET/IB : Got completion from peer '
             '10.0.0.12<45760> with status=12 opcode=0 len=0 vendor err 129 (Recv) error')
XID_79 = 'Nov  4 12:00:01 gpu-node-07 kernel: NVRM: Xid (PCI:0000:3b:00): 79, pid=1, GPU has fallen off the bus.'
XID_45 = 'Nov  4 12:00:02 gpu-node-07 kernel: NVRM: Xid (PCI:0000:3b:00): 45, pid=1, Preemptive cleanup, due to previous errors'
XID_13 = 'Nov  4 12:00:03 gpu-node-08 kernel: NVRM: Xid (0000:86:00): 13, pid=2, Graphics Engine Exception'
SCHEDULER = 'slurmstepd: error: *** STEP 1001.0 ON gpu-node-07 CANCELLED DUE TO NODE FAILURE ***'
UNATTRIBUTED = 'NCCL WARN collective failed with no host information'
LOG = '\n'.join([NCCL_WARN, XID_79, XID_45, XID_13, SCHEDULER, UNATTRIBUTED]) + '\n'


def disk_log(tmp_path, text, name='evidence.log'):
    path = tmp_path / name
    path.write_text(text)
    return DiskLog(path, len(text.encode()), 'sha-' + name, '')


# --- Xid decoding ---------------------------------------------------------

@pytest.mark.parametrize('line,expected', [
    ('NVRM: Xid (PCI:0000:5a:00): 79, GPU has fallen off the bus.', (79, '0000:5a:00')),
    ('NVRM: Xid (0000:03:00): 14, Channel 00000001', (14, '0000:03:00')),
    ('Xid 13', (13, None)), ('Xid: 31', (31, None)), ('xid=48', (48, None)),
    ('txid=12', None), ('xid=12345', None), ('Xid without a code', None), ('nothing here', None),
])
def test_decode_xid_accepts_kernel_and_bare_forms_and_rejects_identifiers(line, expected):
    assert ce.decode_xid(line) == expected


def test_signature_matches_kernel_xid_format_the_old_pattern_missed():
    kernel = 'NVRM: Xid (PCI:0000:3b:00): 79, pid=1'
    assert re.search(r'Xid\s*[:=]?\s*\d+', kernel) is None  # pre-change pattern
    pattern = next(p for key, _, p, _ in SIGNATURES if key == 'cuda')
    assert re.search(pattern, kernel, re.I)
    for bare in ('Xid 13', 'Xid: 31', 'Xid=48'):
        assert re.search(pattern, bare, re.I)
    assert not re.search(pattern, 'txid=12 xid=12345', re.I)


def test_known_xids_cite_the_catalog_and_unknown_codes_are_not_invented():
    for code in ce.XID_CATALOG:
        row = ce.describe_xid(code)
        assert row['name'] and row['action'] and row['domain'] != 'unclassified'
    unknown = ce.describe_xid(777)
    assert unknown['name'] is None and unknown['domain'] == 'unclassified'
    assert 'catalog' in unknown['action'] and 'Reset' not in unknown['action']
    # Xid 45 is cleanup after another event: it must never steer to node hardware.
    assert ce.describe_xid(45)['domain'] == 'follow_on' and 'follow_on' not in ce.NODE_ACTION_DOMAINS


# --- host extraction --------------------------------------------------------

@pytest.mark.parametrize('line,expected', [
    (NCCL_WARN, [('gpu-node-05', 'reporter'), ('10.0.0.12', 'peer_address')]),
    ('3: gpu-node-05:12345:12399 [3] NCCL WARN x failed', [('gpu-node-05', 'reporter')]),
    ('[1,3]<stdout>:gpu-node-05:1:2 [0] NCCL WARN x failed', [('gpu-node-05', 'reporter')]),
    (XID_79, [('gpu-node-07', 'reporter')]),
    ('2026-11-04T12:00:01+00:00 gpu-node-09 kernel: NVRM: Xid 79', [('gpu-node-09', 'reporter')]),
    (SCHEDULER, [('gpu-node-07', 'named')]),
    ('slurmstepd: error: *** JOB 53 ON node001 CANCELLED AT 2026-09-25T00:33:32 ***', [('node001', 'named')]),
    ('socketStartConnect: Connect to 10.1.2.3<33445> failed : Connection refused', [('10.1.2.3', 'peer_address')]),
    (UNATTRIBUTED, []), ('illegal memory access', []), ('NCCL INFO NET/IB : Using mlx5_0', []),
])
def test_hosts_in_line_reports_only_what_the_line_states(line, expected):
    assert ce.hosts_in_line(line) == expected


# --- analysis integration -----------------------------------------------------

def check_report(report):
    groups = {g['id']: g for g in report['crashes']}
    xids = groups['cuda']['xids']
    # Hardware-implicating Xids lead; an Xid 45 follow-on is last.
    assert [x['code'] for x in xids] == [79, 13, 45]
    assert xids[0]['domain'] == 'hardware' and xids[0]['devices'] == [dict(device='0000:3b:00', count=1)]
    assert xids[0]['hosts'] == [dict(host='gpu-node-07', count=1)]
    assert xids[1]['devices'] == [dict(device='0000:86:00', count=1)]
    assert xids[0]['first_seen']['line'] == 2
    assert groups['cuda']['hosts'] == [dict(host='gpu-node-07', role='reporter', count=2), dict(host='gpu-node-08', role='reporter', count=1)]
    items = {(i['host'], i['role']): i for i in report['host_attribution']['items']}
    # The NCCL line matches both rdma and generic but is attributed once, to rdma.
    assert items[('gpu-node-05', 'reporter')]['count'] == 1 and items[('gpu-node-05', 'reporter')]['signatures'] == {'rdma': 1}
    assert items[('gpu-node-07', 'reporter')]['signatures'] == {'cuda': 2}
    assert items[('gpu-node-07', 'named')]['signatures'] == {'scheduler': 1}
    assert items[('10.0.0.12', 'peer_address')]['count'] == 1
    assert "isn't necessarily at fault" in report['host_attribution']['note']
    return items


def test_in_memory_analysis_decodes_xids_and_attributes_hosts_once():
    report = analyze([('crash.log', LOG)], 'h100-infiniband')
    check_report(report)
    command = next(t for t in report['follow_up_tests'] if t['id'] == 'crash-cuda')['command']
    assert 'Xid 79 (GPU has fallen off the bus, 1x)' in command
    assert 'Xid 13' not in command and 'Xid 45' not in command  # only node-level guidance is appended


def test_disk_backed_analysis_matches_in_memory(tmp_path):
    expected = analyze([('crash.log', LOG)], 'h100-infiniband')
    disk = analyze([('crash.log', disk_log(tmp_path, LOG, 'crash.log'))], 'h100-infiniband')
    check_report(disk)
    assert disk['host_attribution'] == expected['host_attribution']
    assert [g.get('xids') for g in disk['crashes']] == [g.get('xids') for g in expected['crashes']]


def test_absent_host_and_device_stay_unknown():
    report = analyze([('crash.log', 'illegal memory access\nXid 13\n')], 'h100-infiniband')
    cuda = next(g for g in report['crashes'] if g['id'] == 'cuda')
    assert cuda['hosts'] == [] and report['host_attribution']['items'] == []
    assert cuda['xids'][0]['code'] == 13 and cuda['xids'][0]['devices'] == [] and cuda['xids'][0]['hosts'] == []


def test_unclassified_xid_gets_no_node_guidance():
    report = analyze([('crash.log', 'NVRM: Xid (PCI:0000:3b:00): 777, pid=1, something new\n')], 'h100-infiniband')
    row = report['crashes'][0]['xids'][0]
    assert row['domain'] == 'unclassified' and row['name'] is None
    assert 'Xid 777' not in next(t for t in report['follow_up_tests'] if t['id'] == 'crash-cuda')['command']


def test_log_without_crash_lines_has_empty_attribution():
    report = analyze([('ok.log', 'NCCL INFO NET/IB : Using mlx5_0\n')], 'h100-infiniband')
    assert report['crashes'] == [] and report['host_attribution']['items'] == []


# --- collector bounds ---------------------------------------------------------

def test_line_numbers_are_deduplicated_per_file_only():
    collector = ce.Collector()
    collector.start_file()
    collector.observe('rdma', NCCL_WARN, 'a.log', 1)
    collector.observe('generic', NCCL_WARN, 'a.log', 1)   # same line, second signature
    collector.start_file()
    collector.observe('rdma', NCCL_WARN, 'b.log', 1)      # same number, different file
    row = next(i for i in collector.attribution()['items'] if i['role'] == 'reporter')
    assert row['count'] == 2 and row['signatures'] == {'rdma': 2} and row['files'] == 2


def test_tracked_hosts_are_bounded_and_truncation_is_reported(monkeypatch):
    monkeypatch.setattr(ce, 'MAX_TRACKED_KEYS', 3)
    collector = ce.Collector()
    collector.start_file()
    for n in range(10):
        collector.observe('rdma', f'host-{n}:1:2 [0] NCCL WARN x error', 'a.log', n + 1)
    result = collector.attribution()
    assert result['total_hosts'] == 3 and result['truncated'] is True


def test_dedupe_memory_is_bounded(monkeypatch):
    monkeypatch.setattr(ce, 'MAX_DEDUPE_LINES', 2)
    collector = ce.Collector()
    collector.start_file()
    for n in range(5):
        collector.observe('rdma', 'host-1:1:2 [0] NCCL WARN x error', 'a.log', n + 1)
    result = collector.attribution()
    assert result['items'][0]['count'] == 2 and result['truncated'] is True


# --- cross-analysis aggregation -------------------------------------------------

def row(identifier, host, count, role='reporter', created='2026-11-01T00:00:00+00:00'):
    return dict(id=identifier, filename=identifier + '.log', created_at=created,
                host_attribution=dict(items=[dict(host=host, role=role, count=count, signatures={'cuda': count})]))


def test_aggregate_counts_each_analysis_once_and_keeps_roles_apart():
    rows = [row('a', 'n1', 50), row('b', 'n1', 1, created='2026-11-03T00:00:00+00:00'), row('c', 'n2', 500), row('d', 'n1', 1, role='named'),
            dict(id='old', filename='old.log', created_at='2026-10-01T00:00:00+00:00')]  # saved before this feature
    result = ce.aggregate(rows, min_analyses=2)
    assert [(i['host'], i['role'], i['analyses'], i['lines']) for i in result['items']] == [('n1', 'reporter', 2, 51)]
    assert result['items'][0]['last_seen'] == '2026-11-03T00:00:00+00:00'
    assert ce.aggregate(rows, min_analyses=1)['total'] == 3 and result['analyses_considered'] == 5


# --- API ------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'DB_PATH', str(tmp_path / 'test.sqlite'))
    monkeypatch.setenv('VALIDATION_UPLOADS', str(tmp_path / 'uploads'))
    return TestClient(app)


def upload(client, cluster, text, name):
    response = client.post('/api/analyze', data=dict(recipe_id='h100-infiniband', cluster=cluster), files={'file': (name, text.encode())})
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_persists_attribution_and_endpoint_finds_repeat_hosts(client):
    first = upload(client, 'Synthetic-A', LOG, 'job-1.log')
    assert first['host_attribution']['items']
    upload(client, 'Synthetic-A', LOG + 'extra evidence line\n', 'job-2.log')   # distinct bytes, same node
    upload(client, 'Synthetic-B', XID_79 + '\n', 'other-cluster.log')
    result = client.get('/api/host-attribution', params=dict(cluster='Synthetic-A')).json()
    by_key = {(i['host'], i['role']): i for i in result['items']}
    assert by_key[('gpu-node-07', 'reporter')]['analyses'] == 2
    assert by_key[('gpu-node-07', 'named')]['analyses'] == 2 and result['analyses_considered'] == 2
    assert client.get('/api/host-attribution', params=dict(cluster='Synthetic-A', min_analyses=3)).json()['items'] == []
    # Another cluster's analysis is not counted, and no filter spans every cluster.
    assert client.get('/api/host-attribution', params=dict(cluster='Synthetic-B', min_analyses=1)).json()['analyses_considered'] == 1
    assert client.get('/api/host-attribution', params=dict(min_analyses=3)).json()['items'][0]['analyses'] == 3


def test_endpoint_validates_parameters_and_handles_an_empty_catalog(client):
    assert client.get('/api/host-attribution').json()['items'] == []
    assert client.get('/api/host-attribution', params=dict(min_analyses=0)).status_code == 422
    assert client.get('/api/host-attribution', params=dict(limit=1000)).status_code == 422
