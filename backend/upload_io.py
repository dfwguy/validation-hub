"""Disk-backed large uploads; original bytes survive analysis errors."""
import codecs, hashlib, io, json, os, shutil, tarfile, uuid, zipfile, re
from . import config
from pathlib import Path, PurePosixPath
from contextlib import ExitStack
from datetime import datetime, timezone
import zstandard
import pyarrow as pa
import pyarrow.compute as pc
from itertools import islice

UPLOAD_LIMIT=1_000_000_000
EXPANDED_LIMIT=32*1024**3
MEMBER_LIMIT=25000
CHUNK=4*1024*1024

# Preserve every parser/signature/system-check input while discarding unrelated
# log chatter from the in-memory analysis index. Original files remain intact.
_INTEREST = re.compile(r"SLURM_JOB_ID|job_id|steps=|CANCELLED|CANCELED|grad(?:ient)?[ _-]*norm|illegal memory access|Bootstrap.*no socket|NCCL_[A-Z0-9_]+|NCCL (?:INFO )?(?:version|VERSION)|NCCL (?:WARN|ERROR)|NCCL=|nccl-library=|ncclGetVersion=|NET/IB|Using network|NVLS|NVLink|TRANSPORT_CONTRACT|GB300|Collective test starting|collective=|\b(?:nodes|gpus|racks)\s*[=:]|#\s*Rank|Avg bus bandwidth|Out of bounds values|TEST_END|Exited with exit code|ACTIVE_DIAG|RAS_DIAG|global_batch|global batch size|iteration.*elapsed time|GPU utilization|Link layer|ACSCtl|powersave|error|failed|FATAL|timeout|timed out|refused|unreachable|No route|out of memory|OOM-kill|Traceback|Segmentation|Xid|IMEX|MNNVL|fallen off|Got completion|retry exceeded|TIME LIMIT|undefined symbol|not found",re.I)
_METRIC = re.compile(r"^\s*\d+\s+\d+\s+(?:float|half|double|int|uint|bfloat)")

class DiskLog:
    def __init__(self,path,size,sha256,preview):
        self.path=path;self.size=size;self.sha256=sha256;self.preview=preview;self._lines=None;self._hits={};self._array=None
    def _index(self):
        if self._lines is None:
            self._lines=[]
            with self.path.open(encoding='utf-8') as f:
                offset=0
                pattern='(?:'+_INTEREST.pattern+')|(?:'+_METRIC.pattern+')|(?:test_(?:min_bytes|max_bytes|iterations|warmup|registration)=|(?:all_reduce|all_gather|reduce_scatter|alltoall|sendrecv)_perf|seq_length|sequence_length|micro_batch_size|tensor_model_parallel_size|pipeline_model_parallel_size|context_parallel_size|virtual_pipeline_model_parallel_size|expert_model_parallel_size|expert_tensor_parallel_size)'
                while lines:=list(islice(f,4096)):
                    selected=pc.match_substring_regex(pa.array(lines),pattern=pattern,ignore_case=True).to_pylist()
                    self._lines.extend((offset+i+1,line.rstrip('\r\n')) for i,(line,keep) in enumerate(zip(lines,selected)) if keep)
                    offset+=len(lines)
        return self._lines
    def splitlines(self):return (line for _,line in self._index())
    def parser_lines(self):
        # Data-plane debug chatter is useful for diagnosis but not measurement
        # parsing. Preserve all measurement rows and parser metadata in order.
        pattern=r"\b(?:nodes|gpus|racks)\s*[=:]|\b(?:SLURM_JOB_ID|job_id)\s*=|#\s*Rank.*NVIDIA|NCCL (?:INFO )?(?:version|VERSION)|NCCL=|nccl-library=|ncclGetVersion=|Collective test starting|collective=|NCCL_(?:RUN_DIAGNOSTICS|RUN_RAS_DIAGNOSTICS|NVLS_ENABLE|MNNVL_ENABLE|IB_HCA|TESTS_SPLIT_MASK)=|TRANSPORT_CONTRACT|Avg bus bandwidth|Out of bounds values|TEST_END|Exited with exit code|ACTIVE_DIAG|RAS_DIAG|steps=|global_batch|global batch size|elapsed time per iteration|GPU utilization|^\s*\d+\s+\d+\s+(?:float|half|double|int|uint|bfloat)"
        pattern+='|test_(?:min_bytes|max_bytes|iterations|warmup|registration)=|(?:all_reduce|all_gather|reduce_scatter|alltoall|sendrecv)_perf|seq_length|sequence_length|micro_batch_size|tensor_model_parallel_size|pipeline_model_parallel_size|context_parallel_size|virtual_pipeline_model_parallel_size|expert_model_parallel_size|expert_tensor_parallel_size'
        for _,line in self.matches(pattern):yield line
        if next(self.matches(r'NET/IB|Using network IB'),None):yield 'NCCL INFO NET/IB network observed'
        if next(self.matches(r'NVLink|NVLS.*(?:channels|nChannels|comm|support)'),None):yield 'NVLink observed'
        if next(self.matches(r'GB300'),None):yield 'GB300'
    def numbered_lines(self):return iter(self._index())
    def matches(self,pattern,flags=0):
        key=(pattern,flags)
        lines=self._index()
        if key not in self._hits:
            if self._array is None:self._array=pa.array([line for _,line in lines],type=pa.string())
            mask=pc.match_substring_regex(self._array,pattern=pattern,ignore_case=bool(flags & re.I))
            self._hits[key]=pc.indices_nonzero(mask).to_pylist()
        return (lines[i] for i in self._hits[key])
    def release_array(self):self._array=None


def receive(fileobj,filename):
    from fastapi import HTTPException
    root=config.uploads_root()/'received'
    root.mkdir(parents=True,exist_ok=True)
    tmp=root/(uuid.uuid4().hex+'.partial');digest=hashlib.sha256();size=0
    try:
        with tmp.open('xb') as out:
            while block:=fileobj.read(CHUNK):
                size+=len(block)
                if size>UPLOAD_LIMIT:raise HTTPException(413,'Upload limit is 1 GB per file')
                out.write(block);digest.update(block)
            out.flush();os.fsync(out.fileno())
        if not size:raise HTTPException(422,'Upload is empty')
        sha=digest.hexdigest();path=root/(sha+'.upload')
        if path.exists():tmp.unlink()
        else:tmp.replace(path)
        path.with_suffix('.receipt.json').write_text(json.dumps(dict(filename=filename,sha256=sha,size_bytes=size,received_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
        return path
    finally:
        if tmp.exists():tmp.unlink()


def disk_texts(path,name):
    """Extract bounded members to content-addressed paths, never archive paths."""
    path=Path(path);root=path.parent/'members';root.mkdir(exist_ok=True)
    manifest=path.with_suffix('.members.json')
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved.get('member_count',MEMBER_LIMIT+1)<=MEMBER_LIMIT and saved.get('expanded_bytes',EXPANDED_LIMIT+1)<=EXPANDED_LIMIT and all(Path(m['path']).is_file() and Path(m['path']).stat().st_size==m['size'] for m in saved['members']):
            return [(m['name'],DiskLog(Path(m['path']),m['size'],m['sha256'],'')) for m in saved['members']]
    entries=[];expanded=0;count=0;preview_total=0
    def add(member,stream,size):
        nonlocal preview_total
        preview_limit=min(1024*1024,max(0,8*1024*1024-preview_total))
        digest=hashlib.sha256();decoder=codecs.getincrementaldecoder('utf-8')();written=0;preview=[];preview_size=0
        temp=root/(uuid.uuid4().hex+'.partial')
        try:
            with temp.open('xb') as out:
                while block:=stream.read(CHUNK):
                    written+=len(block)
                    if written>size or written>EXPANDED_LIMIT:raise ValueError('Expanded archive exceeds 32 GiB')
                    if b'\0' in block:raise ValueError('Binary content found; upload UTF-8 logs')
                    text=decoder.decode(block)
                    if preview_size<preview_limit:
                        preview.append(text[:preview_limit-preview_size]);preview_size+=len(preview[-1]);preview_total+=len(preview[-1])
                    digest.update(block);out.write(block)
                decoder.decode(b'',final=True)
            sha=digest.hexdigest();target=root/(sha+'.log')
            if target.exists():temp.unlink()
            else:temp.replace(target)
            entries.append((member,DiskLog(target,written,sha,''.join(preview))))
        except UnicodeDecodeError:raise ValueError('Upload UTF-8 text logs')
        finally:
            if temp.exists():temp.unlink()
    def member(pathname,size,regular,opener):
        nonlocal expanded,count
        count+=1;expanded+=size
        if count>MEMBER_LIMIT:raise ValueError('Archive limit is 25,000 members')
        if expanded>EXPANDED_LIMIT:raise ValueError('Expanded archive exceeds 32 GiB')
        if regular and PurePosixPath(pathname).suffix.lower() in ('.log','.txt','.out','.err'):
            with opener() as f:add(pathname,f,size)
    with path.open('rb') as f:magic=f.read(4)
    gzip_content=magic.startswith(b'\x1f\x8b');zstd_content=magic==b'\x28\xb5\x2f\xfd'
    if name.lower().endswith('.tar.zg') and not gzip_content:raise ValueError('The .tar.zg extension is usually a typo for .tar.gz; upload a valid gzip-compressed tar archive')
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            if len(z.infolist())>MEMBER_LIMIT:raise ValueError('Archive limit is 25,000 members')
            for m in z.infolist():member(m.filename,m.file_size,not m.is_dir(),lambda m=m:z.open(m))
    elif name.lower().endswith(('.tar','.tar.gz','.tgz','.tar.zg','.tar.zst','.tzst')) or gzip_content or zstd_content:
        with ExitStack() as stack:
            stream=stack.enter_context(path.open('rb'))
            if zstd_content or name.lower().endswith(('.tar.zst','.tzst')):stream=stack.enter_context(zstandard.ZstdDecompressor().stream_reader(stream))
            t=stack.enter_context(tarfile.open(fileobj=stream,mode='r|' if zstd_content or name.lower().endswith(('.tar.zst','.tzst')) else 'r|*'))
            for m in t:member(m.name,m.size,m.isfile(),lambda m=m:t.extractfile(m))
    else:
        with path.open('rb') as f:add(name,f,path.stat().st_size)
    if not entries:raise ValueError('No .log, .txt, .out or .err text files found. Original upload is retained.')
    result={'member_count':count,'expanded_bytes':expanded,'members':[dict(name=name,path=str(text.path),size=text.size,sha256=text.sha256) for name,text in entries]}
    temp_manifest=manifest.with_suffix('.'+uuid.uuid4().hex+'.partial')
    temp_manifest.write_text(json.dumps(result));temp_manifest.replace(manifest)
    return entries


def search(pattern,text,flags=0):
    import re
    if isinstance(text,str):return re.search(pattern,text,flags)
    return next(text.matches(pattern,flags),None)


def findall(pattern,text):
    import re
    if isinstance(text,str):return re.findall(pattern,text)
    regex=re.compile(pattern)
    return [match for _,line in text.matches(pattern) for match in regex.findall(line)]


def digest_file(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
