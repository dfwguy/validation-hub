"""Evidence parsers: retain measured geometry and unknowns; never infer racks from nodes."""
import re
import statistics
import math
import shlex

COLLECTIVES = ['all_reduce', 'all_gather', 'reduce_scatter', 'alltoall', 'sendrecv', 'broadcast', 'reduce']

def parse_log(lines, name='', hints=None):
    result = dict(workload='Unknown', collective=None, nodes=None, gpus=None, racks=None, sus=None,
                  hardware=None, transport=None, nccl_version=None, status='unknown',
                  avg_bus_bw_gbps=None, peak_bus_bw_gbps=None, tflops_gpu=None, iterations=None,
                  metadata={}, metrics=[], job_id=None)
    hosts, ranks, averages, correctness, completed = set(), set(), [], [], False
    collectives_seen=set(); observed_ib=False; observed_nvlink=False
    iterations = {}
    for line in lines:
        if isinstance(line, bytes): line = line.decode('utf-8', errors='replace')
        job=re.search(r'\b(?:SLURM_JOB_ID|job_id)\s*=\s*(\d+)\b',line) if 'job_id' in line.lower() else None
        if job:result['job_id']=job[1]
        lower=line.lower()
        if 'NCCL_' in line:
            for key,value in re.findall(r'\b(NCCL_[A-Z0-9_]+)\s*(?:=|set by environment to\s+|set to\s+)\s*["\x27]?([^\s"\x27;]+)',line):
                result['metadata'].setdefault('nccl_environment',{})[key]=value[:512]
        # Preserve explicit experiment protocol, never reconstruct it from a
        # single result point or guess the installed test version.
        for key in ('min_bytes','max_bytes','iterations','warmup','registration'):
            m=re.search(r'\btest_'+key+r'=([^\s]+)',line) if 'test_' in line else None
            if m:result['metadata'].setdefault('test_protocol',{})[key]=int(m[1]) if m[1].isdigit() else m[1]
        if '_perf' in line and re.search(r'(?:all_reduce|all_gather|reduce_scatter|alltoall|sendrecv|broadcast|reduce)_perf',line):
            try:
                argv=shlex.split(line);start=next((i for i,v in enumerate(argv) if re.search(r'(?:all_reduce|all_gather|reduce_scatter|alltoall|sendrecv|broadcast|reduce)_perf(?:_mpi)?$',v)),None)
                if start is not None:
                    protocol=result['metadata'].setdefault('test_protocol',{})
                    for i in range(start+1,len(argv)-1):
                        if argv[i] not in ('-b','-e','-n','-w','-R'):continue
                        key={'-b':'min_bytes','-e':'max_bytes','-n':'iterations','-w':'warmup','-R':'registration'}[argv[i]]
                        value=re.fullmatch(r'(\d+)([kKmMgG]?)',argv[i+1])
                        if value:protocol[key]=int(value[1])*(1024**({'k':1,'m':2,'g':3}.get(value[2].lower(),0)))
            except ValueError:pass
        for key in ('seq_length','sequence_length','global_batch_size','micro_batch_size','tensor_model_parallel_size','pipeline_model_parallel_size','context_parallel_size','virtual_pipeline_model_parallel_size','expert_model_parallel_size','expert_tensor_parallel_size'):
            if key in lower:
                m=re.search(r'\b'+key+r'\s*(?:=|:|\.{2,}|\s)\s*(\d+|None)\b',line,re.I)
                if m:result['metadata'][key]=int(m[1]) if m[1].isdigit() else 'none'
        for field in ('nodes','gpus','racks'):
            if field not in lower:continue
            m = re.search(r'\b'+field+r'\s*[=:]\s*(\d+)', line, re.I)
            if m: result[field] = int(m[1])
        m = re.search(r'#\s*Rank\s+(\d+).*?on\s+(\S+)\s+device.*?NVIDIA\s+(.+)',line) if 'NVIDIA' in line and 'Rank' in line else None
        if m: ranks.add(m[1]); hosts.add(m[2]); result['hardware'] = m[3].strip()
        m = re.search(r'NCCL (?:INFO )?(?:version|VERSION)[ =]+([\d.]+)',line) if 'version' in lower and 'NCCL' in line else None
        if not m and 'NCCL=' in line: m = re.search(r'\bNCCL=([\d.]+)',line)
        if m: result['nccl_version'] = m[1].split('+')[0]
        m = re.search(r'nccl-library=(\d+)|ncclGetVersion=(\d+)',line) if 'nccl-library=' in line or 'ncclGetVersion=' in line else None
        if m:
            n = int(m[1] or m[2]); result['nccl_version'] = f'{n//10000}.{n//100%100}.{n%100}'
        m = re.search(r'Collective test starting:\s*(\S+)|\bcollective=(\w+)',line) if 'collective' in lower else None
        if m:
            token = m[1] or m[2]; result['collective'] = token.replace('_perf_mpi','').replace('_perf','').replace('all_to_all','alltoall').replace('send_recv','sendrecv'); collectives_seen.add(result['collective']); result['workload']='NCCL'
        for key in ('NCCL_RUN_DIAGNOSTICS','NCCL_RUN_RAS_DIAGNOSTICS','NCCL_NVLS_ENABLE','NCCL_MNNVL_ENABLE','NCCL_IB_HCA','NCCL_TESTS_SPLIT_MASK'):
            if key not in line:continue
            m = re.search(re.escape(key)+r'=([^\s]+)',line)
            if m:
                if key=='NCCL_TESTS_SPLIT_MASK' and key in result['metadata'] and result['metadata'][key]!=m[1]:result['metadata']['mixed_split_masks']=True
                result['metadata'][key] = m[1]
        if 'NET/IB' in line or 'Using network IB' in line: observed_ib=True
        if 'NVLS multicast support is available' in line or ('NVLS' in line and re.search(r'\bNVLS\b.*(?:channels|nChannels|comm|support)',line)) or 'NVLink' in line: observed_nvlink=True
        if observed_ib and observed_nvlink: result['transport']='InfiniBand + NVLink'
        if 'NCCL_TRANSPORT_CONTRACT=MULTIRACK_HYBRID' in line or 'LLAMA_TRANSPORT_CONTRACT' in line: result['transport']='InfiniBand + NVLink'
        m = re.search(r'Avg bus bandwidth\s*:\s*([\d.]+)',line) if 'Avg bus bandwidth' in line else None
        if m: averages.append(float(m[1])); completed = True
        m = re.search(r'Out of bounds values\s*:\s*(\d+)',line) if 'Out of bounds values' in line else None
        if m: correctness.append(int(m[1]))
        m = re.search(r'(?:TEST_END.*?\brc=|Exited with exit code\s+)(\d+)',line) if 'TEST_END' in line or 'Exited with exit code' in line else None
        if m and int(m[1]) > 0: result['metadata']['exit_failure']=True
        tokens = line.strip().split() if line.lstrip()[:1].isdigit() else []
        if len(tokens) >= 13 and tokens[0].isdigit() and tokens[1].isdigit() and tokens[2] in ('float','half','double','int','int8','uint8','bfloat16','float8_e4m3','float8_e5m2','uint32','int32','uint64','int64'):
            try:
                for offset, placement in ((5,'out_of_place'),(9,'in_place')):
                    errors = int(tokens[offset+3]) if tokens[offset+3] != 'N/A' else None
                    if not all(math.isfinite(float(tokens[offset+i])) and float(tokens[offset+i])>=0 for i in range(3)):continue
                    result['metrics'].append(dict(message_bytes=int(tokens[0]),datatype=tokens[2],operation=tokens[3],placement=placement,collective=result['collective'],time_us=float(tokens[offset]),algbw_gbps=float(tokens[offset+1]),busbw_gbps=float(tokens[offset+2]),errors=errors))
                result['workload']='NCCL'
            except (ValueError,IndexError): pass
        for key in ('ACTIVE_DIAG','RAS_DIAG','global_batch','steps'):
            if key not in line:continue
            found=re.search(r'\b'+key+r'=(\d+)',line)
            if found: result['metadata'][key]=int(found[1])
        m = re.search(r'global batch size:\s*(\d+)',line) if 'global batch size:' in line else None
        if m:result['metadata']['global_batch_size']=int(m[1])
        m = re.search(r'iteration\s+(\d+)\s*/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+)',line) if 'elapsed time per iteration' in line else None
        if m:
            result['workload']='Llama'; n,total,time = int(m[1]),int(m[2]),float(m[3]); iterations[n] = dict(iteration=n,time_ms=time)
            result['metadata']['expected_iterations']=total
            if n==total: completed=True
        m = re.search(r'GPU utilization:\s*([\d.]+)\s*MODEL_TFLOP/s/GPU',line) if 'GPU utilization:' in line else None
        if m:
            result['workload']='Llama'; result['metrics'].append(dict(tflops_gpu=float(m[1])))
        if 'GB300' in line: result['hardware']='GB300'
        if any(token in lower for token in ('traceback','out of memory','nccl warn','nccl error','fatal','time limit')) and re.search(r'(?:Traceback \(most recent call last\)|CUDA out of memory|NCCL (?:WARN|ERROR).*?(?:error|failed)|(?:^|[^A-Za-z_-])FATAL(?:$|[^A-Za-z_-])|DUE TO TIME LIMIT)',line,re.I): result['metadata']['exit_failure']=True
    if not result['job_id']:
        job=re.search(r'(?:^|[_/-])job(\d+)(?=[_.\/-]|$)',name,re.I)
        if job:result['job_id']=job[1]
    result['metadata']['observed_hosts']=len(hosts)
    result['metadata']['observed_ranks']=len(ranks)
    result['metadata']['inferred_fields']=[k for k in ('nodes','gpus') if result[k] is None]
    if result['nodes'] is None: result['nodes']=(hints or {}).get('nodes') or (len(hosts) if hosts else None)
    if result['gpus'] is None: result['gpus']=(hints or {}).get('gpus') or (len(ranks) if ranks else None)
    if len(collectives_seen)>1:
        result['collective']='mixed'; result['metadata']['collectives']=sorted(collectives_seen)
    if not result['collective'] and result['workload']!='Llama':
        for c in COLLECTIVES:
            if c in name.lower(): result['collective']=c; result['workload']='NCCL'; break
    if result['workload']=='Unknown' and ('llama' in name.lower() or 'llm-pretrain' in name.lower()): result['workload']='Llama'
    result['metrics'].extend(iterations.values())
    result['iterations']=len(iterations) or None
    bws=[m['busbw_gbps'] for m in result['metrics'] if 'busbw_gbps' in m]
    if averages: result['avg_bus_bw_gbps']=statistics.mean(averages)
    elif bws: result['avg_bus_bw_gbps']=statistics.mean(bws)
    if bws: result['peak_bus_bw_gbps']=max(bws)
    flops=[m['tflops_gpu'] for m in result['metrics'] if 'tflops_gpu' in m]
    if flops: result['tflops_gpu']=statistics.mean(flops)
    if correctness and sum(correctness)>0 or any(m.get('errors',0) for m in result['metrics']) or result['metadata'].get('exit_failure'): result['status']='failed'
    elif completed and (result['workload']=='Llama' or correctness and not sum(correctness)): result['status']='passed'
    elif result['metrics']: result['status']='incomplete'
    result['metadata']['correctness_errors']=sum(correctness) if correctness else None
    result['metadata']['hosts']=sorted(hosts)
    if hints:
        if hints.get('metadata'):result['metadata'].update(hints['metadata'])
        for k,v in hints.items():
            if k in result and result[k] is None: result[k]=v
    return result
