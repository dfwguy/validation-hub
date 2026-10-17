"""Select bounded validation evidence from operator-supplied, receipted tar.zst archives.

Never restore full archives. Every output keeps archive/member provenance and SHA256;
a SQLite ledger makes repeated scans idempotent while the mirror is still running.
"""
from __future__ import annotations
import argparse, fcntl, fnmatch, hashlib, io, json, os, re, shutil, sqlite3, tarfile, time, sys
from pathlib import Path, PurePosixPath
import zstandard
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import config

PATTERNS=('result-*.txt','all*default*.txt','reduce*default*.txt','send*default*.txt','broadcast*default*.txt','summary*.tsv','slurm*.out')
EXTRACTION_POLICY='validation-summary-v5-selected-transport'
SENSITIVE=re.compile(r'(?i)(password|passwd|secret|token|authorization|private[_ -]?key|api[_ -]?key)\s*[=:]|-----BEGIN .*PRIVATE KEY-----|bearer\s+[A-Za-z0-9_.-]+')
MAX_MEMBER=32*2**20
MAX_ARCHIVE_SCAN=2*2**30

class BoundedReader(io.RawIOBase):
    def __init__(self, raw, limit):self.raw,self.limit,self.count=raw,limit,0
    def readable(self):return True
    def readinto(self,b):
        if self.count>=self.limit:raise ValueError('Archive decompression scan limit reached')
        n=self.raw.readinto(memoryview(b)[:min(len(b),self.limit-self.count)])
        self.count+=n;return n

def safe_member(member):
    p=PurePosixPath(member.name)
    allowed=any(fnmatch.fnmatch(p.name.lower(),q) for q in PATTERNS) or (p.name.lower()=='stdout.txt' and any(re.fullmatch(r'rank0(?:_.*)?',part) for part in p.parts))
    return member.isfile() and not p.is_absolute() and '..' not in p.parts and member.size<=MAX_MEMBER and allowed and not any(x in p.name.lower() for x in ('rank','debug','env','config'))

CONTEXT_PREFIX=2*2**20
CONTEXT_FACT=re.compile(r'NET/IB|Using network|MNNVL|NVLS|NVLink|NCCL version|NCCL_(?:IB_HCA|IB_DISABLE|SOCKET_IFNAME|NVLS_ENABLE|MNNVL_ENABLE|ALGO|PROTO)',re.I)

def context_targets(member):
    p=PurePosixPath(member.name)
    if not member.isfile() or p.is_absolute() or '..' in p.parts:return []
    if re.fullmatch(r'rank0(?:_.*)?',p.parent.name) and 'nccl_debug' in p.name:
        return [p.parent/'stdout.txt']
    if '-rank0-nccl.' in p.name:
        stem=p.name.split('-rank0-nccl.')[0]
        return [p.parent/(stem+'.txt'),p.parent/('result-'+stem+'.txt')]
    return []

def transport_context(data, member, remote, archive_sha256):
    evidence=[];ib_ports=False;selected_backend=False;nvlink=False;version=None
    for lineno,line in enumerate(data.decode('utf-8',errors='replace').splitlines(),1):
        if SENSITIVE.search(line) or not CONTEXT_FACT.search(line):continue
        # Positive initialization evidence only: availability/config flags alone do not prove active NVLink.
        if re.search(r'NET/IB.*Using.*?/IB(?:/|\s|\[)',line):ib_ports=True
        if re.search(r'Using network.*(?:\bIB\b|RDMA)|via NET/IB',line,re.I):selected_backend=True
        if re.search(r'MNNVL\s+1\b|via NVLS|NVLS.*(?:connected|Connection)|NVLink.*(?:connected|Connection)',line,re.I):nvlink=True
        match=re.search(r'NCCL version ([0-9.]+)',line)
        if match:version=match.group(1)
        if len(evidence)<128:evidence.append({'lineno':lineno,'line':line[:1024]})
    ib=ib_ports and selected_backend
    transport='InfiniBand + NVLink' if ib and nvlink else 'InfiniBand' if ib else 'NVLink' if nvlink else None
    return {'schema_version':1,'transport':transport,'nccl_version':version,'observed_ib_selection':ib,'observed_ib_ports':ib_ports,'observed_rdma_backend_selection':selected_backend,'observed_nvlink_initialization':nvlink,'remote_archive':remote,'archive_sha256':archive_sha256,'debug_member':member,'prefix_bytes':len(data),'prefix_sha256':hashlib.sha256(data).hexdigest(),'read_limit_bytes':CONTEXT_PREFIX,'evidence':evidence,'scope':'Bounded rank-zero initialization evidence; preserves unknown transport when facts are insufficient.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=config.data_root())
    ap.add_argument('--archives-dir',type=Path,required=True,help='Local directory containing tar.zst archives with verified .mirror.json receipts')
    ap.add_argument('--cluster',required=True,help='Deployment-local cluster label; used under selected/')
    ap.add_argument('--budget-gib',type=float,default=5)
    ap.add_argument('--reserve-gib',type=float,default=1,help='Free disk capacity to preserve')
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',args.cluster):ap.error('Use a cluster label containing letters, digits, dot, underscore or hyphen')
    if args.budget_gib<=0 or args.reserve_gib<0:ap.error('Budget must be positive and reserve must be nonnegative')
    if not args.archives_dir.is_dir():ap.error('Archives directory does not exist')
    root=Path(args.root);selected=root/'selected';selected.mkdir(parents=True,exist_ok=True)
    (root/'catalog').mkdir(parents=True,exist_ok=True)
    lock=(root/'catalog/extracted-evidence.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=sqlite3.connect(root/'catalog/extracted-evidence.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS archives(path TEXT PRIMARY KEY,sha256 TEXT,status TEXT,selected_files INTEGER,finished_at REAL,error TEXT); CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY,remote_archive TEXT,local_archive TEXT,member TEXT,original_sha256 TEXT,sha256 TEXT,local_path TEXT,size_bytes INTEGER,redacted_lines INTEGER,extracted_at REAL);')
    used=db.execute('SELECT coalesce(sum(size_bytes),0) FROM evidence').fetchone()[0];budget=int(args.budget_gib*2**30)
    counts={'archives':0,'files':0,'bytes':0,'errors':[]}
    for archive in sorted(args.archives_dir.glob('*.tar.zst')):
        receipt=archive.with_name(archive.name+'.mirror.json')
        if not receipt.exists():continue
        info=json.loads(receipt.read_text())
        fingerprint=info['sha256']+':'+EXTRACTION_POLICY+':'+args.cluster
        if not info.get('verified_size') or archive.stat().st_size!=info.get('size_bytes'):continue
        if db.execute("SELECT 1 FROM archives WHERE path=? AND sha256=? AND status IN ('complete','bounded')",(str(archive),fingerprint)).fetchone():continue
        status,error,nfiles='complete','',0
        pending_context=[]
        try:
            with archive.open('rb') as compressed, zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(compressed) as stream:
                with tarfile.open(fileobj=io.BufferedReader(BoundedReader(stream,MAX_ARCHIVE_SCAN)),mode='r|') as tar:
                    for member in tar:
                        targets=context_targets(member)
                        if targets:
                            prefix=tar.extractfile(member).read(CONTEXT_PREFIX)
                            ctx=transport_context(prefix,member.name,info['remote_path'],info['sha256'])
                            if ctx['evidence']:pending_context.append((targets,ctx))
                            continue
                        if not safe_member(member):continue
                        if used+member.size>budget:raise ValueError('Selected evidence budget reached')
                        if shutil.disk_usage(root).free<int(args.reserve_gib*2**30):raise ValueError('Reserved filesystem capacity reached')
                        data=tar.extractfile(member).read(MAX_MEMBER+1)
                        if len(data)>MAX_MEMBER:continue
                        original=hashlib.sha256(data).hexdigest()
                        text=data.decode('utf-8',errors='replace');redacted=0;lines=[]
                        for line in text.splitlines(keepends=True):
                            if SENSITIVE.search(line):line='[REDACTED sensitive operational line]\n';redacted+=1
                            lines.append(line)
                        data=''.join(lines).encode()
                        if len(data)>MAX_MEMBER:continue
                        if used+len(data)>budget:raise ValueError('Selected evidence budget reached')
                        digest=hashlib.sha256(data).hexdigest()
                        eid=hashlib.sha256((args.cluster+'\0'+info['remote_path']+'\0'+member.name+'\0'+original).encode()).hexdigest()
                        if db.execute('SELECT 1 FROM evidence WHERE id=?',(eid,)).fetchone():continue
                        out=selected/args.cluster/hashlib.sha256(info['remote_path'].encode()).hexdigest()[:12]/PurePosixPath(member.name)
                        out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix(out.suffix+'.partial');temp.write_bytes(data);temp.replace(out)
                        row=(eid,info['remote_path'],str(archive),member.name,original,digest,str(out),len(data),redacted,time.time())
                        db.execute('INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?)',row);db.commit()
                        out.with_name(out.name+'.provenance.json').write_text(json.dumps(dict(zip(('id','remote_archive','local_archive','member','original_sha256','sha256','local_path','size_bytes','redacted_lines','extracted_at'),row)),indent=2))
                        used+=len(data);nfiles+=1;counts['files']+=1;counts['bytes']+=len(data)
        except Exception as exc:
            status='bounded' if isinstance(exc,ValueError) else 'error';error=str(exc);counts['errors'].append({'archive':str(archive),'error':error})
        for targets,ctx in pending_context:
            for target in targets:
                out=selected/args.cluster/hashlib.sha256(info['remote_path'].encode()).hexdigest()[:12]/target
                if not out.exists():continue
                sidecar=out.with_name(out.name+'.context.json')
                temp=sidecar.with_suffix('.json.partial');temp.write_text(json.dumps(ctx,indent=2));temp.replace(sidecar)
        db.execute('INSERT OR REPLACE INTO archives VALUES(?,?,?,?,?,?)',(str(archive),fingerprint,status,nfiles,time.time(),error));db.commit();counts['archives']+=1
        if used>=budget:break
    db.row_factory=sqlite3.Row
    index={'updated_at':time.time(),'total_files':db.execute('SELECT count(*) FROM evidence').fetchone()[0],'total_bytes':used,'budget_bytes':budget,'this_scan':counts,'evidence':[dict(row) for row in db.execute('SELECT * FROM evidence')]}
    temp=root/'catalog/extracted-evidence.json.tmp';temp.write_text(json.dumps(index,indent=2));temp.replace(root/'catalog/extracted-evidence.json')
    print(json.dumps({k:v for k,v in index.items() if k!='evidence'},indent=2))

if __name__=='__main__':main()
