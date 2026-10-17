import hashlib,io,json,sqlite3,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
import zstandard

class ArchiveExtractionTest(unittest.TestCase):
    def test_safe_bounded_idempotent_provenance(self):
        with tempfile.TemporaryDirectory(dir='/data/validation-hub') as tmp:
            root=Path(tmp);(root/'catalog').mkdir();base=root/'raw/nvftp/cluster-east/Validation_Logs';base.mkdir(parents=True)
            buffer=io.BytesIO()
            with tarfile.open(fileobj=buffer,mode='w') as tar:
                for name,data in [('job_123/result-all_reduce.txt',b'Avg bus bandwidth : 600\npassword=secret\n'),('../result-escape.txt',b'escape'),('job_123/config.txt',b'password=other'),('job_123/rank0_host/stdout.txt',b'Avg bus bandwidth : 610\n'),('job_123/rank1_host/stdout.txt',b'omit duplicate'),('job_123/rank0_host/nccl_debug_file.host.1.txt',b'NCCL INFO NCCL version 2.30.7+cuda13.0\nNCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB\nNCCL INFO Using network NCCL RDMA Plugin v11\nNCCL INFO MNNVL 1 cliqueId 1 cliqueSize 72\nNCCL_IB_HCA=mlx5_0 password=secret\n'+b'x'*(33*2**20))]:
                    info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
                link=tarfile.TarInfo('job_123/result-link.txt');link.type=tarfile.SYMTYPE;link.linkname='/etc/passwd';tar.addfile(link)
            archive=base/'job.tar.zst';data=zstandard.ZstdCompressor().compress(buffer.getvalue());archive.write_bytes(data)
            archive.with_name(archive.name+'.mirror.json').write_text(json.dumps({'remote_path':'/nvidia-space/shared/cluster-east/Validation_Logs/job.tar.zst','verified_size':True,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}))
            script=Path(__file__).resolve().parents[1]/'tools/extract_evidence.py'
            for _ in range(2):subprocess.run([sys.executable,str(script),'--root',str(root)],check=True,capture_output=True)
            index=json.loads((root/'catalog/extracted-evidence.json').read_text());self.assertEqual(index['total_files'],2);e=next(x for x in index['evidence'] if x['member'].endswith('result-all_reduce.txt'))
            self.assertEqual(e['member'],'job_123/result-all_reduce.txt');self.assertEqual(e['redacted_lines'],1)
            self.assertNotIn('secret',Path(e['local_path']).read_text());self.assertFalse((root/'selected/result-escape.txt').exists())
            self.assertEqual(index['this_scan']['files'],0)
            contexts=list((root/'selected').rglob('*.context.json'));self.assertEqual(len(contexts),1)
            context=json.loads(contexts[0].read_text());self.assertEqual(context['transport'],'InfiniBand + NVLink');self.assertEqual(context['nccl_version'],'2.30.7');self.assertNotIn('secret',contexts[0].read_text());self.assertEqual(context['prefix_bytes'],2*2**20)

    def test_configuration_is_not_active_transport_evidence(self):
        import importlib.util
        script=Path(__file__).resolve().parents[1]/'tools/extract_evidence.py'
        spec=importlib.util.spec_from_file_location('archive_extract',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context=module.transport_context(b'NCCL_NVLS_ENABLE set by environment to 1\nNVLS multicast support is available\nNET/IB : Using [0]mlx5_0:1/RoCE\n','debug','remote','hash')
        self.assertIsNone(context['transport'])

if __name__=='__main__':unittest.main()
