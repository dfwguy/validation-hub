"""Deterministic evidence checks and crash signatures with explicit uncertainty."""
import io, re, tarfile, zipfile, hashlib
from pathlib import PurePosixPath
from .parser import parse_log
from .recipes import profile, VERSION
from .security import scrub

LIMIT=32*1024*1024

def texts(content,name):
    """Read bounded text members in memory; never extract archive paths."""
    entries=[]; total=0
    def add(path,data):
        nonlocal total
        total+=len(data)
        if total>LIMIT:raise ValueError('Expanded text exceeds 32 MiB')
        if b'\0' in data:raise ValueError('Binary content found; upload UTF-8 logs')
        try:text=data.decode('utf-8')
        except UnicodeDecodeError:raise ValueError('Upload UTF-8 text logs')
        entries.append((str(path),text))
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            members=[m for m in z.infolist() if not m.is_dir()]
            if len(members)>200:raise ValueError('Archive limit is 200 files')
            if sum(m.file_size for m in members)>LIMIT:raise ValueError('Expanded archive exceeds 32 MiB')
            for m in members:
                if m.file_size+total>LIMIT:raise ValueError('Expanded archive exceeds 32 MiB')
                if PurePosixPath(m.filename).suffix.lower() in ('.log','.txt','.out','.err'):
                    with z.open(m) as f:add(m.filename,f.read(LIMIT-total+1))
    elif name.lower().endswith(('.tar','.tar.gz','.tgz')):
        with tarfile.open(fileobj=io.BytesIO(content),mode='r|*') as t:
            expanded=0
            for index,m in enumerate(t):
                if index>=200:raise ValueError('Archive limit is 200 members')
                expanded+=m.size
                if expanded>LIMIT:raise ValueError('Expanded archive exceeds 32 MiB')
                if m.isfile():
                    if m.size+total>LIMIT:raise ValueError('Expanded archive exceeds 32 MiB')
                    if PurePosixPath(m.name).suffix.lower() in ('.log','.txt','.out','.err'):
                        with t.extractfile(m) as f:add(m.name,f.read(LIMIT-total+1))
    else:add(name,content)
    if not entries:raise ValueError('No .log, .txt, .out or .err text files found')
    return entries

SIGNATURES=[
 ('rdma','RDMA completion / fabric failure',r'ibv_poll_cq.*(?:error|failed)|Got completion.*(?:error|status)|IB.*(?:retry exceeded|error 12)|transport retry counter exceeded','Inspect the affected port counters and link health; reproduce on the same node pair before changing timeouts.'),
 ('bootstrap','Bootstrap connectivity failure',r'No route to host|Connection refused|Network is unreachable|Bootstrap.*(?:failed|no socket)|socketStartConnect.*failed','Verify bootstrap interface, routing, firewall and name resolution on every allocated node.'),
 ('memory','GPU or host memory exhaustion',r'CUDA out of memory|out of memory|OOM-kill|cudaErrorMemoryAllocation','Reduce maximum message size and inspect free memory and competing processes; rerun with correctness enabled.'),
 ('cuda','CUDA / GPU failure',r'unhandled cuda error|illegal memory access|Xid\s*[:=]?\s*\d+|CUDA error|GPU has fallen off','Capture nvidia-smi -q and kernel Xid evidence; isolate the GPU and test locally.'),
 ('timeout','Timeout / stalled rank',r'Watchdog.*(?:timeout|timed out)|DUE TO TIME LIMIT|collective.*timed out|NCCL (?:WARN|ERROR).*timeout','Find the first failing rank and preceding errors; check fabric and allocation before increasing timeout.'),
 ('imex','NVLink / IMEX domain failure',r'(?:IMEX|MNNVL|NVLS).*(?:failed|error|unavailable)','Verify allocation membership, NVLink fabric health and required IMEX access for the selected GB system.'),
 ('runtime','Runtime / library mismatch',r'undefined symbol|error while loading shared libraries|libnccl.*not found|libcuda.*not found','Check container driver compatibility, MPI and loaded NCCL library versions across all nodes.'),
 ('generic','Job failure',r'NCCL (?:WARN|ERROR).*?(?:error|failed)|FATAL|Traceback \(most recent call last\)|Segmentation fault|TEST_END.*rc=[1-9]','Inspect the earliest error and correlated rank logs. This signature does not establish a root cause.'),
]

def analyze(entries,recipe_id):
    p=profile(recipe_id)
    if not p:raise ValueError('Unknown GPU/fabric recipe')
    checks=[]; groups=[]; runs=[]; env={}; all_text='\n'.join(t for _,t in entries)
    for name,text in entries:
        r=parse_log(text.splitlines(),name);r['metadata']['recipe_id']=recipe_id
        if p['fabric']=='spectrum-x' and r.get('transport')=='InfiniBand + NVLink':
            r['transport']='RoCE + NVLink' if re.search(r'Link layer\s*:\s*Ethernet|NET/IB.*RoCE',text,re.I) else None
        # Names are display-only and can never become filesystem paths.
        runs.append(dict(filename=name,size_bytes=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest(),parsed=r))
        for key,val in re.findall(r'\b(NCCL_[A-Z0-9_]+)\s*(?:=|set to\s+|set by environment to\s+)\s*["\x27]?([^\s"\x27;]+)',text):env.setdefault(key,set()).add(val)
    def check(key,title,state,expected,observed=None):checks.append(dict(id=key,title=title,status=state,expected=expected,observed=observed))
    for key,val in p['expected_env'].items():
        seen=env.get(key,set());check(key,key,'unknown' if not seen else 'pass' if seen=={val} else 'fail',val,sorted(seen))
    for key in ('NCCL_IB_HCA','NCCL_SOCKET_IFNAME'):
        seen=env.get(key,set());check(key,key,'review' if seen else 'unknown','Confirm selection against GPU-facing ports / bootstrap network',sorted(seen))
    for key in ('NCCL_P2P_DISABLE','NCCL_SHM_DISABLE'):
        seen=env.get(key,set());check(key,key,'review' if '1' in seen else 'pass' if seen else 'unknown','Usually unset or 0; investigate forced disable',sorted(seen))
    measured=[r['parsed'] for r in runs]
    models=sorted({r['hardware'] for r in measured if r.get('hardware')})
    check('gpu','GPU model','unknown' if not models else 'pass' if all(p['gpu'] in m.upper() for m in models) else 'fail',p['gpu'],models)
    geometries=[{'nodes':r['nodes'],'gpus':r['gpus']} for r in measured if r.get('nodes') and r.get('gpus')]
    check('geometry','GPUs per node','unknown' if not geometries else 'pass' if all(g['gpus']==g['nodes']*p['gpus_per_node'] for g in geometries) else 'review',p['gpus_per_node'],geometries)
    observed_rdma=bool(re.search(r'Using network (?:IB|[\w-]*IB[\w-]*)|NET/IB.*(?:Using|mlx5)',all_text,re.I))
    socket=bool(re.search(r'Using network Socket',all_text,re.I))
    check('rdma','Observed data transport','fail' if socket else 'pass' if observed_rdma else 'unknown','RDMA selected; NET/IB also names the RoCE verbs transport', 'RDMA observed' if observed_rdma else 'Socket observed' if socket else None)
    for key,title,pattern,advice in SIGNATURES:
        hits=[];count=0
        for name,text in entries:
            for line_no,line in enumerate(text.splitlines(),1):
                if re.search(pattern,line,re.I):
                    count+=1
                    if len(hits)<8:hits.append(dict(file=name,line=line_no,excerpt=scrub(line[:700])))
        if count:groups.append(dict(id=key,title=title,count=count,evidence=hits,next_action=advice,confidence='signature match; cause requires verification'))
    correctness=[r['metadata'].get('correctness_errors') for r in measured]
    failed=any(r['status']=='failed' for r in measured) or any(any(m.get('errors') for m in r['metrics']) for r in measured)
    check('correctness','NCCL correctness','fail' if failed else 'pass' if measured and all(r['status']=='passed' for r in measured) else 'unknown','Complete test and zero correctness errors',correctness)
    check('site','Site fabric policy','review','Verify firmware, link speed, topology, QoS and approved NCCL/container versions')
    if p['fabric']=='spectrum-x':
        check('roce','RoCE link layer','pass' if re.search(r'Link layer\s*:\s*Ethernet',all_text,re.I) else 'unknown','ibstat reports Ethernet; verify switch QoS / PFC / ECN separately')
        check('ece','Enhanced connection establishment','review' if env.get('NCCL_IB_ECE_ENABLE')=={'0'} else 'unknown','Confirm ECE and congestion control with the approved Spectrum-X deployment recipe',sorted(env.get('NCCL_IB_ECE_ENABLE',set())))
    else:check('ib_link','InfiniBand link layer','pass' if re.search(r'Link layer\s*:\s*InfiniBand',all_text,re.I) else 'unknown','ibstat reports InfiniBand')
    if p['gpu'].startswith('GB'):check('nvlink','NVLink / IMEX domain','review','Verify fabric membership and IMEX access for this allocation; do not force MNNVL globally')
    # Host diagnostics are risk indicators, never proof of GPU-path correctness.
    if re.search(r'ACSCtl:.*(?:ReqRedir\+|CmpltRedir\+)',all_text):check('acs','PCIe ACS redirect','review','Inspect the specific GPU/NIC PCIe path before changing BIOS settings')
    if re.search(r'\bpowersave\b',all_text):check('cpu','CPU power governor','review','Measure CPU clocks under load and review power/NUMA policy')
    if groups:check('crash','Crash evidence','fail','Resolve observed crash signatures before accepting results',sum(g['count'] for g in groups))
    tests=followups(p,groups,checks)
    verdict='fail' if any(c['status']=='fail' for c in checks) else 'needs_evidence'
    return scrub(dict(recipe=p,policy_version=VERSION,verdict=verdict,checks=checks,crashes=groups,follow_up_tests=tests,
                      files=[dict(filename=r['filename'],sha256=r['sha256'],status=r['parsed']['status'],metrics=len(r['parsed']['metrics'])) for r in runs],
                      runs=runs,environment={k:sorted(v) for k,v in env.items()},
                      limitations=['Checks apply to the uploaded evidence only. Missing settings remain unknown.',
                                   'Crash signatures suggest investigations; they are not a confirmed root cause.',
                                   'Recipe defaults do not certify site firmware, switch policy or complete allocation health.']))

def followups(p,groups=(),checks=()):
    ids={g['id'] for g in groups}
    tests=[dict(id='collect',title='Collect system and GPU/NIC topology',priority=1,reason='Complete missing configuration evidence before tuning.',command='nvidia-smi -q\nnvidia-smi topo -m\nlscpu\nnumactl -H\nibstat\nibdev2netdev',success='GPU, NIC, driver and topology agree with the selected recipe.'),
           dict(id='scale',title='Isolate local, pair and scale performance',priority=2,reason='Find where bandwidth or latency first degrades.',command='Use the NCCL Test Script generator: 1 node, 2 nodes, then target scale; repeat each run 3 times.',success='Correctness clean; compare identical sizes, versions, GPU counts and placement.'),
           dict(id='split',title='AllGather rail isolation',priority=3,reason='Compare all-GPU traffic with one GPU per node per group.',command=f'NCCL_TESTS_SPLIT_MASK=0x0\nNCCL_TESTS_SPLIT_MASK={hex(p["gpus_per_node"]-1)}',success='Compare each mask with its own baseline; bandwidth is per group, not aggregate.'),
           dict(id='tune',title='Controlled NCCL tuning',priority=4,reason='After correctness and fabric health pass, change one variable per run.',command='Keep default NCCL_ALGO/NCCL_PROTO as control. Compare Ring vs Tree only where supported, then restore defaults.',success='At least 3 repeats, same topology and size; retain median and spread. Accept only stable gains.')]
    if p['fabric']=='infiniband':tests.append(dict(id='fabric',title='InfiniBand fabric diagnostics',priority=1 if 'rdma' in ids else 3,reason='Check link health before changing NCCL timeouts.',command='ibstat\nibdiagnet',success='Fabric operator reviews unhealthy links, routing and increasing error counters.'))
    else:tests.append(dict(id='fabric',title='Spectrum-X RoCE and congestion review',priority=1 if 'rdma' in ids else 3,reason='Verify Ethernet/RDMA selection and site congestion policy.',command='rdma link show\nibdev2netdev\n# Ask fabric operator for per-port PFC, ECN, drops and congestion counters.',success='Correct GPU-facing ports, approved QoS and firmware; no unexplained drops.'))
    for g in groups:tests.append(dict(id='crash-'+g['id'],title=g['title'],priority=1,reason='Triggered by crash evidence.',command=g['next_action'],success='Reproduce and verify the fix on the same allocation.'))
    if any(c['id']=='cpu' for c in checks):tests.append(dict(id='cpu',title='CPU launch overhead',priority=2,reason='Power governor evidence warrants a CPU-sensitive follow-up.',command='USE_CASES="kernel_launch" llmb-run submit -w microbenchmark_cpu_overhead --dtype mxfp4 --scale 1',success='Review launch latency and clocks under load against a matching CPU reference.'))
    return sorted(tests,key=lambda t:t['priority'])
