"""Xid decoding and host attribution for crash-signature hits.

Everything comes from the matching log lines. A host, device or Xid the log
doesn't state is left out. Attribution is a lead, not a diagnosis: the host that
wrote a line may only be the victim of a peer's failure, so each host carries
the role it was seen in.
"""
import re
from collections import Counter

# Meanings and actions follow NVIDIA's Xid catalog (docs.nvidia.com/deploy/xid-errors,
# checked 2026-11-04). The catalog doesn't label codes application/hardware, so
# `domain` is our reading of its wording. Codes not listed get no meaning or advice.
XID_CATALOG = {
    13: dict(name='Graphics Engine Exception', domain='application',
             action='General user-application fault. Restart the application and debug it (cuda-gdb, Compute Sanitizer); rarely hardware or system software.'),
    31: dict(name='GPU memory page fault', domain='application',
             action='Typically an application bug, but can be a driver or hardware bug. Repeated faults on the same GPU across workloads point to a hardware check.'),
    43: dict(name='GPU stopped processing', domain='application',
             action='Usually a user-application error rather than a driver bug. Restart the application; contact support if it recurs without an application cause.'),
    45: dict(name='Preemptive cleanup, due to previous errors', domain='follow_on',
             action='Follows a user or system action (Ctrl-C, GPU reset, SIGKILL) or another Xid. Investigate the earlier Xid on this GPU, not this one.'),
    48: dict(name='Double Bit ECC Error', domain='hardware',
             action='Uncorrectable GPU memory error. Drain the node, reset the GPU, and run field diagnostics; with Xid 63/64 follow the drain-and-reset path.'),
    63: dict(name='GPU memory remapping event', domain='ecc_memory',
             action='A row remap was recorded. Informational on its own; a GPU reset is needed for it to take effect. Track recurrence on this GPU.'),
    64: dict(name='GPU memory remapping failure', domain='ecc_memory',
             action='A row remap could not be recorded. Reset the GPU and contact support if it recurs.'),
    74: dict(name='NVLINK Error', domain='hardware',
             action='May indicate a failure of the link itself or the remote device (documented for A100/H100). A GPU reset or node reboot is needed; contact the hardware vendor if it recurs.'),
    79: dict(name='GPU has fallen off the bus', domain='hardware',
             action='Often a PCIe-link hardware failure; a driver or GPU failure is also possible. Drain the node and review PCIe, power and BMC/system logs before returning it to service.'),
    92: dict(name='High single-bit ECC error rate', domain='ecc_memory',
             action='Correctable-error rate is high. Monitor the GPU and contact support per site policy if it persists.'),
    94: dict(name='Contained memory error', domain='ecc_memory',
             action='Contained to one application. Restart the application and reset the GPU at a convenient time; Xid 45 often accompanies it.'),
    95: dict(name='Uncontained memory error', domain='ecc_memory',
             action='Can affect multiple applications. The GPU must be reset before applications are restarted.'),
    119: dict(name='GSP RPC Timeout', domain='driver_firmware',
              action='The GSP firmware did not answer. Reset the GPU; if it recurs after a power cycle, follow the GPU debug guidelines and check the driver release.'),
    120: dict(name='GSP Error', domain='driver_firmware',
              action='A GSP firmware error. Reset the GPU; if it recurs after a power cycle, follow the GPU debug guidelines and check the driver release.'),
}
# Domains where the catalog points at the GPU or node rather than the workload.
NODE_ACTION_DOMAINS = ('hardware', 'ecc_memory', 'driver_firmware')

# Kernel form "NVRM: Xid (PCI:0000:5a:00): 79, ..." (the PCI: prefix is optional), or
# bare "Xid 13" / "Xid: 31" / "Xid=48". Three digits at most, so "xid=12345" and
# "txid=1" don't match.
XID_PATTERN = r'\bXid\b(?:\s*\([^)]*\))?\s*[:=]?\s*\d{1,3}\b'
_XID = re.compile(r'\bXid\b(?:\s*\((?P<device>[^)]*)\))?\s*[:=]?\s*(?P<code>\d{1,3})\b', re.I)

_HOST = r'[A-Za-z0-9][A-Za-z0-9_.-]*'
# `host:pid:tid [dev] ... NCCL`, optionally behind an srun label ("3: ") or an
# mpirun rank prefix ("[1,3]<stdout>:").
_NCCL_PREFIX = re.compile(r'^\s*(?:\[\d+,\d+\]<std(?:out|err)>:\s*)?(?:\d+:\s+)?(?P<host>' + _HOST + r'):\d+:\d+\s+\[\d+\]')
# Syslog/journal: "Nov  4 12:00:01 node-12 kernel: ..." or ISO timestamp.
_SYSLOG = re.compile(r'^\s*(?:[A-Z][a-z]{2}\s+\d{1,2}\s+\d\d:\d\d:\d\d|\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d\S*)\s+(?P<host>' + _HOST + r')\s+\S+:')
# Scheduler: "*** STEP 1001.0 ON node-3 CANCELLED ..." names the node.
_SCHEDULER = re.compile(r'\b(?:STEP|JOB)\s+\S+\s+ON\s+(?P<host>' + _HOST + r')\b', re.I)
# NCCL connection errors name the remote end by address; we don't resolve it.
_PEER = re.compile(r'\b(?:peer|Connect to)\s+(?P<address>\d{1,3}(?:\.\d{1,3}){3})<\d+>', re.I)

ROLE_NOTE = {
    'reporter': 'Host that wrote the matching line. It may be the victim of a peer failure, not the cause.',
    'named': 'Host named in the scheduler message that reported the failure.',
    'peer_address': 'Remote address in a connection error. Not resolved to a hostname.',
}
ATTRIBUTION_NOTE = ("Hosts come only from the matching log lines. The host that logged an error isn't necessarily "
                    'at fault; check port counters and GPU health, and reproduce on the same node before draining.')

MAX_TRACKED_KEYS = 5000        # distinct (host, role, signature) keys per analysis
MAX_DEDUPE_LINES = 100_000     # attributed lines remembered per file
TOP_HOSTS = 25


def decode_xid(line):
    """Return (code, device) for the first Xid in the line, or None."""
    match = _XID.search(line)
    if not match:
        return None
    device = (match.group('device') or '').strip() or None
    if device and device.upper().startswith('PCI:'):
        device = device[4:]
    return int(match.group('code')), device


def describe_xid(code):
    entry = XID_CATALOG.get(code)
    if entry:
        return dict(code=code, name=entry['name'], domain=entry['domain'], action=entry['action'])
    return dict(code=code, name=None, domain='unclassified',
                action=f'Xid {code} is not in this tool\'s table; look it up in NVIDIA\'s Xid catalog for the installed driver release.')


def hosts_in_line(line):
    """Return [(name, role)] stated by one log line; empty when it states none."""
    found = []
    match = _SCHEDULER.search(line)
    if match:
        found.append((match.group('host'), 'named'))
    match = _NCCL_PREFIX.match(line) or _SYSLOG.match(line)
    if match:
        found.append((match.group('host'), 'reporter'))
    match = _PEER.search(line)
    if match:
        found.append((match.group('address'), 'peer_address'))
    return found


class Collector:
    """Collects Xid and host evidence for one analysis, with bounded memory."""

    def __init__(self):
        self._hosts = Counter()      # (name, role, signature) -> lines
        self._first = {}             # (name, role) -> (file, line)
        self._files = {}             # (name, role) -> set(file) (capped)
        self._xids = {}              # code -> dict
        self._seen = set()
        self.truncated = False

    def start_file(self):
        """Call once per log file: line numbers are only unique within a file."""
        self._seen = set()

    def observe(self, signature, line, file, line_no):
        if signature == 'cuda':
            self._observe_xid(line, file, line_no)
        # A line can match several signatures. Count it once, under the first.
        if line_no in self._seen:
            return
        hosts = hosts_in_line(line)
        if not hosts:
            return
        if len(self._seen) >= MAX_DEDUPE_LINES:
            self.truncated = True
            return
        self._seen.add(line_no)
        for name, role in hosts:
            key = (name, role, signature)
            if key not in self._hosts and len(self._hosts) >= MAX_TRACKED_KEYS:
                self.truncated = True
                continue
            self._hosts[key] += 1
            self._first.setdefault((name, role), dict(file=file, line=line_no))
            files = self._files.setdefault((name, role), set())
            if len(files) < 20:
                files.add(file)

    def _observe_xid(self, line, file, line_no):
        decoded = decode_xid(line)
        if not decoded:
            return
        code, device = decoded
        entry = self._xids.setdefault(code, dict(count=0, devices=Counter(), hosts=Counter(), first=dict(file=file, line=line_no)))
        entry['count'] += 1
        if device and (device in entry['devices'] or len(entry['devices']) < 16):
            entry['devices'][device] += 1
        for name, role in hosts_in_line(line):
            if role == 'reporter' and (name in entry['hosts'] or len(entry['hosts']) < 64):
                entry['hosts'][name] += 1

    def xids(self):
        """Decoded Xids, hardware-implicating domains first, then by frequency."""
        rows = []
        for code, entry in self._xids.items():
            row = describe_xid(code)
            row.update(count=entry['count'], first_seen=entry['first'],
                       devices=[dict(device=d, count=c) for d, c in entry['devices'].most_common(8)],
                       hosts=[dict(host=h, count=c) for h, c in entry['hosts'].most_common(8)])
            rows.append(row)
        rows.sort(key=lambda r: (r['domain'] not in NODE_ACTION_DOMAINS, -r['count'], r['code']))
        return rows

    def signature_hosts(self, signature, limit=8):
        totals = Counter()
        for (name, role, sig), count in self._hosts.items():
            if sig == signature:
                totals[(name, role)] += count
        return [dict(host=n, role=r, count=c) for (n, r), c in totals.most_common(limit)]

    def attribution(self):
        """Per-host roll-up across signatures, most-implicated first."""
        merged = {}
        for (name, role, sig), count in self._hosts.items():
            row = merged.setdefault((name, role), dict(host=name, role=role, count=0, signatures={}))
            row['count'] += count
            row['signatures'][sig] = row['signatures'].get(sig, 0) + count
        rows = sorted(merged.values(), key=lambda r: (-r['count'], r['host'], r['role']))
        for row in rows:
            row['first_seen'] = self._first[(row['host'], row['role'])]
            row['files'] = len(self._files[(row['host'], row['role'])])
        return dict(items=rows[:TOP_HOSTS], total_hosts=len(rows), truncated=self.truncated or len(rows) > TOP_HOSTS,
                    roles=ROLE_NOTE, note=ATTRIBUTION_NOTE)


def aggregate(rows, min_analyses=2, limit=50):
    """Hosts that recur across saved analyses.

    `rows` hold id, filename, created_at and the `host_attribution` mapping built
    above. A host counts once per analysis, so one noisy job can't make a node
    look like a repeat offender. Roles are kept apart.
    """
    merged = {}
    for row in rows:
        attribution = row.get('host_attribution') or {}
        for item in attribution.get('items', []):
            key = (item['host'], item['role'])
            entry = merged.setdefault(key, dict(host=item['host'], role=item['role'], analyses=0, lines=0, signatures={}, last_seen=None, sample_analyses=[]))
            entry['analyses'] += 1
            entry['lines'] += item['count']
            for sig, count in item.get('signatures', {}).items():
                entry['signatures'][sig] = entry['signatures'].get(sig, 0) + count
            if row.get('created_at') and (entry['last_seen'] is None or row['created_at'] > entry['last_seen']):
                entry['last_seen'] = row['created_at']
            if len(entry['sample_analyses']) < 5:
                entry['sample_analyses'].append(dict(id=row['id'], filename=row.get('filename')))
    items = [e for e in merged.values() if e['analyses'] >= min_analyses]
    items.sort(key=lambda e: (-e['analyses'], -e['lines'], e['host'], e['role']))
    return dict(items=items[:limit], total=len(items), min_analyses=min_analyses, analyses_considered=len(rows),
                roles=ROLE_NOTE, note=ATTRIBUTION_NOTE)
