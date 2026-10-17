import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from backend import beacon_nccl, nccl_dashboard

PATH='b300_cx8_roce/2.31.2/allreduce/legacy/nvls_on/64nvld/allreduce_64nvld_64nvl8_observed.csv'
CSV=b'bytes_transferred,busbw_GB_s\n8,0.0\n2147483648,607.0233\n17179869184,758.09\n'
META=dict(sol_GBs=1000,sol_basis='theoretical roofline',observed_GBs=758.09,
 collection_notes='3 successful runs, 64 nodes / 64 NVL domains / 512 GPUs, out-of-place busbw only, averaged per size. RoCE SpectrumX-SN5600 CX8. Job 779330. The source tarball was labeled FAILED; this selected group completed 3/3 repeats with errors=0, complete result footers, and zero out-of-place #wrong.',
 observed_cluster='bia',observed_cmd='srun -N 64 env NCCL_IB_TC=96 env NCCL_DEBUG=INFO bash -c run_test all_reduce_perf -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0 -G10')
PLATFORM=dict(id='b300_cx8_roce',tag='B300',label='B300 CX8 RoCE')


def test_observed_measurements_preserve_spectrum_context_and_archive_caveat():
 row=nccl_dashboard.parse_curve(PATH,CSV,META,PLATFORM)
 assert row['gpu']=='B300' and row['gpus']==512 and row['nodes']==64
 assert row['fabric']=='Spectrum-X' and row['site_fabric']=='RoCE'
 assert row['metrics'][-1]['busbw_gbps']==758.09 and row['metrics'][-1]['message_bytes']==17179869184
 assert row['metrics'][0]['busbw_gbps']==0
 assert row['metrics'][1]['placement']=='out_of_place' and row['metrics'][1]['errors']==0
 assert row['source_archive_status']=='failed_label_in_source_notes' and row['selected_group_status']=='passed'
 assert row['status']=='passed' and row['repeat_count']==3 and row['reference_statistic']=='mean_per_message_size'
 assert row['test_protocol']['registration']==0 and row['test_protocol']['max_bytes']==16*1024**3
 assert row['test_protocol']['dashboard_kernel']=='legacy' and row['test_protocol']['nvlink_domains']==64
 assert row['nccl_environment']==dict(NCCL_IB_TC='96',NCCL_DEBUG='INFO')
 assert row['theoretical_sol']['value']==1000 and row['theoretical_sol']['used_as_measurement'] is False
 assert not any(point['busbw_gbps']==1000 for point in row['metrics'])


def test_native_domain_differs_from_cross_domain_and_no_spx_inference():
 native=PATH.replace('/64nvld/','/1nvld/').replace('_64nvld_64nvl8','_1nvld_1nvl8')
 row=nccl_dashboard.parse_curve(native,CSV,dict(observed_cmd='srun -N1 all_reduce_perf -R2 -x2'),PLATFORM)
 assert row['gpus']==8 and row['fabric']=='NVLink' and row['placement_scope']=='single_nvlink_domain'
 assert row['test_protocol']['registration']==2 and row['test_protocol']['device_implementation']==2
 assert row['test_protocol'].get('cuda_graph_launches') is None
 cross=nccl_dashboard.parse_curve(PATH,CSV,{},PLATFORM)
 assert cross['fabric']=='RoCE' and cross['status'] is None and cross['metrics'][0]['errors'] is None
 assert cross['metrics'][0]['datatype'] is None and cross['metrics'][0]['operation'] is None
 assert cross['nodes'] is None and cross['link_speed_gbps'] is None


def test_targets_and_invalid_points_are_not_imported_as_measured():
 with pytest.raises(ValueError):nccl_dashboard.parse_curve(PATH.replace('_observed','_target'),CSV,META,PLATFORM)
 with pytest.raises(ValueError):nccl_dashboard.parse_curve(PATH,b'bytes_transferred,busbw_GB_s\n8,nan\n',META,PLATFORM)
 with pytest.raises(ValueError):nccl_dashboard.parse_curve('../'+PATH,CSV,META,PLATFORM)


def test_collect_follows_metadata_fallback_and_import_preserves_source(tmp_path):
 assets={'catalog.json':json.dumps(dict(observed=[PATH])).encode(),
         'platforms/manifest.json':json.dumps(dict(platforms=[PLATFORM])).encode(),
         'data/'+PATH:CSV,'data/'+PATH.removesuffix('_observed.csv')+'.meta.json':json.dumps(META).encode()}
 requested=[]
 def get(path,optional=False):
  requested.append(path)
  if path in assets:return assets[path]
  if optional:return None
  raise AssertionError(path)
 rows,path,manifest=nccl_dashboard.collect(getter=get,snapshot_root=tmp_path/'snapshots')
 assert manifest['complete'] and len(rows)==1 and manifest['metric_points']==3
 assert not any(p.endswith('_target.csv') for p in requested)
 assert 'data/'+PATH.removesuffix('_observed.csv')+'_target.meta.json' in requested
 result=beacon_nccl.import_records(rows,source_url=nccl_dashboard.BASE_URL,db_path=tmp_path/'nccl.sqlite')
 refs=beacon_nccl.references(tmp_path/'nccl.sqlite')
 assert result['imported']==1 and len(refs)==1
 ref=refs[0]
 assert ref['source_record_id']==PATH and ref['source']=='NVIDIA NCCL Performance Dashboard'
 assert ref['test_protocol']['dashboard_kernel']=='legacy' and ref['test_protocol']['registration']==0
 assert ref['source_archive_status']=='failed_label_in_source_notes' and ref['selected_group_status']=='passed'
 assert ref['source_provenance']['csv_url'].endswith(PATH)
 assert ref['row_action'].endswith(PATH)
 assert ref['metrics'][-1]['busbw_gbps']==758.09 and ref['metrics'][-1]['errors']==0
 assert ref['raw_asset_sha256']==hashlib.sha256(CSV).hexdigest()


def test_snapshot_fetcher_stores_source_bytes_with_hash(tmp_path,monkeypatch):
 class Response:
  headers={'Content-Type':'text/csv'}
  url=nccl_dashboard.BASE_URL+'data/'+PATH
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def read(self,limit):return CSV
 monkeypatch.setattr(nccl_dashboard,'urlopen',lambda *a,**k:Response())
 fetch=nccl_dashboard.SnapshotFetcher(tmp_path)
 assert fetch('data/'+PATH)==CSV
 asset=fetch.assets['data/'+PATH]
 assert asset['sha256']==hashlib.sha256(CSV).hexdigest()
 assert gzip.decompress(Path(asset['gzip_path']).read_bytes())==CSV


def test_quoted_shell_command_flags_are_read_without_execution():
 parsed=nccl_dashboard.command_context("srun -N 2 bash -c 'run_test all_reduce_perf -R2 -x2 -n50 -w5 -b8 -e16G -dfloat'")
 assert parsed['nodes']==2 and parsed['datatype']=='float'
 assert parsed['test_protocol']['registration']==2 and parsed['test_protocol']['iterations']==50
 assert parsed['test_protocol']['max_bytes']==16*1024**3
