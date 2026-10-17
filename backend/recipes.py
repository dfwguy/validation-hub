"""Versioned baseline recipes; site-specific fabric policy is never guessed."""
import shlex
from pydantic import BaseModel, Field, field_validator

VERSION = '2026-10-17.1'
SOURCES = [
    'https://exemplar.nvidia.com/resources/performance-troubleshooting.json',
    'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html',
    'https://github.com/NVIDIA/nccl-tests',
    'https://github.com/NVIDIA/exemplar-performance/blob/8e9a5c8b51c377a7109d3001c17925ba810fd388/microbenchmarks/nccl/config/systems.toml',
]
GPUS = {'H100':8, 'H200':8, 'B200':8, 'B300':8, 'GB200':4, 'GB300':4}

def profiles():
    return [dict(id=f'{gpu.lower()}-{fabric}',gpu=gpu,fabric=fabric,gpus_per_node=count,
                 version=VERSION,scope='Baseline checks; site fabric policy requires review',
                 expected_env={'NCCL_IB_DISABLE':'0', **({'NCCL_IB_ROCE_VERSION_NUM':'2'} if fabric=='spectrum-x' else {'NCCL_IB_SL':'1'} if gpu.startswith('GB') else {'NCCL_NET':'IB'})},
                 sources=SOURCES,notes=['Confirm GPU-facing HCA ports, bootstrap interface, container and NCCL versions.',
                 'Spectrum-X additionally requires verified RoCE, QoS, firmware and congestion-control policy.' if fabric=='spectrum-x' else 'Confirm subnet, routing, link state and site service level.',
                 'GB systems: verify NVLink/NVSwitch and IMEX domain where required; MNNVL settings depend on allocation.'])
            for gpu,count in GPUS.items() for fabric in ('infiniband','spectrum-x')]

def profile(recipe_id):
    return next((p for p in profiles() if p['id']==recipe_id),None)

class ScriptRequest(BaseModel):
    recipe_id: str
    nodes: int = Field(2, ge=1, le=16384)
    gpus_per_node: int = Field(8, ge=1, le=8)
    collective: str = 'all_reduce'
    partition: str = Field('batch',pattern=r'^[A-Za-z0-9_.-]{1,80}$')
    image: str = Field(min_length=1,max_length=512)
    binary_dir: str = Field('/opt/nccl-tests/build',min_length=1,max_length=512)
    hca: str = Field(min_length=1,max_length=512,pattern=r'^=?[A-Za-z0-9_:,.-]+$')
    interface: str = Field(min_length=1,max_length=128,pattern=r'^=?[A-Za-z0-9_,.:-]+$')
    min_bytes: int = Field(8,ge=8,le=1073741824)
    max_bytes: int = Field(1073741824,ge=8,le=17179869184)
    iterations: int = Field(20,ge=1,le=1000)
    repeats: int = Field(3,ge=1,le=20)
    split: bool = True
    @field_validator('image','binary_dir')
    @classmethod
    def no_control(cls,v):
        if any(ord(c)<32 or ord(c)==127 for c in v):raise ValueError('Control characters are not allowed')
        return v

def generate(req):
    p=profile(req.recipe_id)
    if not p:raise ValueError('Unknown recipe')
    if req.collective not in ('all_reduce','all_gather','reduce_scatter','alltoall','sendrecv'):raise ValueError('Unsupported collective')
    if req.gpus_per_node not in (1,2,4,8):raise ValueError('GPUs per node must be 1, 2, 4 or 8')
    if req.max_bytes<req.min_bytes:raise ValueError('Maximum size must be at least minimum size')
    q=shlex.quote
    masks=['0x0']+([hex(req.gpus_per_node-1)] if req.split and req.nodes>1 and req.gpus_per_node>1 else [])
    env=dict(p['expected_env'],NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,NET,ENV',NCCL_IB_HCA=req.hca,NCCL_SOCKET_IFNAME=req.interface)
    lines=['#!/usr/bin/env bash',f'# Recipe {p["id"]} / {VERSION}',
           '# Requires Slurm + Pyxis/Enroot, PMIx, and MPI-built nccl-tests in the image.',
           '# Review fabric QoS, firmware, HCA and NVLink/IMEX policy before submitting.',
           f'#SBATCH --job-name=nccl-{p["gpu"].lower()}',f'#SBATCH --nodes={req.nodes}',
           f'#SBATCH --ntasks-per-node={req.gpus_per_node}',f'#SBATCH --gpus-per-node={req.gpus_per_node}',
           f'#SBATCH --partition={req.partition}','#SBATCH --time=01:00:00','#SBATCH --exclusive',
           'set -euo pipefail','umask 077',f'IMAGE={q(req.image)}',f'BIN={q(req.binary_dir.rstrip("/")+"/"+req.collective+"_perf_mpi")}',
           'OUT="$PWD/nccl-${SLURM_JOB_ID:?Submit with sbatch}"','mkdir -p "$OUT"']
    for key,value in env.items():lines.append(f'export {key}={q(value)}')
    lines += [f'for mask in {" ".join(masks)}; do','  export NCCL_TESTS_SPLIT_MASK="$mask"',
              f'  for repeat in $(seq 1 {req.repeats}); do',
              f'    LOG="$OUT/{req.collective}-mask${{mask}}-repeat${{repeat}}.log"',
              '    {',f'      echo "# Collective test starting: {req.collective}_perf_mpi"',
              f'      echo "nodes={req.nodes} gpus={req.nodes*req.gpus_per_node} recipe_id={p["id"]} recipe_version={VERSION}"',
              '      for key in NCCL_IB_DISABLE NCCL_IB_ROCE_VERSION_NUM NCCL_IB_SL NCCL_NET NCCL_DEBUG NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_TESTS_SPLIT_MASK; do',
              '        if [[ -v "$key" ]]; then printf "%s=%s\\n" "$key" "${!key}"; fi','      done',
              '      srun --mpi=pmix --distribution=block:block --cpu-bind=none --container-image="$IMAGE" \\',
              f'        "$BIN" -b {req.min_bytes} -e {req.max_bytes} -f 2 -g 1 -c 1 -w 20 -n {req.iterations}',
              '    } 2>&1 | tee "$LOG"','  done','done','']
    return dict(script='\n'.join(lines),filename=f'nccl-{p["id"]}.sbatch',recipe=p,
                warnings=['Confirm the MPI binary path in your image. Generation does not execute any test.',
                          'Split runs report bandwidth per group; compare only identical split masks.',
                          'GPU count differs from the profile default.' if req.gpus_per_node!=p['gpus_per_node'] else 'Keep algorithms and protocols on NCCL defaults for the first measurement.'])
