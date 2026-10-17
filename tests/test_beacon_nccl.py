import gzip,json
from pathlib import Path
import pytest
from backend import beacon_nccl as nccl


def row(**updates):
 data={'id':42,'job_id':'51','gpu':'b300','scale':256,'network':'Spectrum-X','collective':'allgather','nccl_version':'2.29.3',
       'protocol':'Simple','split_mask':'0x7','message_bytes':1048576,'datatype':'float','operation':'sum','placement':'in_place','busbw':800,'bandwidth_unit':'Gb/s','errors':0}
 return {**data,**updates}


def test_nccl_units_context_and_raw_evidence_are_preserved():
 raw=row();r=nccl.normalize_record(raw,'https://example.test/actual-nccl-export')
 assert r['gpu']=='B300' and r['gpus']==256 and r['fabric']=='spectrum-x'
 assert r['collective']=='all_gather' and r['protocol']=='Simple' and r['split_mask']=='0x7'
 assert r['metrics'][0]['busbw_gbps']==100 and r['metrics'][0]['bandwidth_unit']=='GB/s'
 assert r['metrics'][0]['message_bytes']==1048576 and r['metrics'][0]['errors']==0
 assert r['raw_json']==raw and r['source_url']=='https://example.test/actual-nccl-export'


def test_missing_context_stays_unknown_and_roce_is_not_spectrumx():
 r=nccl.normalize_record({'busbw_gbps':100,'network':'NET/IB'})
 assert r['fabric'] is None and r['gpus'] is None and r['nccl_version'] is None
 assert r['metrics'][0]['message_bytes'] is None and r['metrics'][0]['protocol'] is None
 assert nccl.normalize_record(row(network='RoCE'))['fabric']=='roce'
 assert nccl.normalize_record(row(network='InfiniBand + NVLink'))['fabric']=='infiniband+nvlink'


def test_ambiguous_bandwidth_unit_cannot_create_reference(tmp_path):
 with pytest.raises(ValueError,match='known units'):
  nccl.import_records([{'busbw':100}],db_path=tmp_path/'refs.sqlite')
 assert nccl.status(tmp_path/'refs.sqlite')['records']==0


def test_import_retains_raw_gzip_and_filters_actual_measurements(tmp_path):
 db=tmp_path/'refs.sqlite';receipt=nccl.import_records([row()],db_path=db)
 assert json.loads(gzip.decompress(Path(receipt['path']).read_bytes()))==row()
 assert nccl.list_references({'fabric':'spectrum-x'},db_path=db)['total']==1
 assert nccl.list_references({'fabric':'infiniband'},db_path=db)['total']==0
 assert nccl.status(db)['records']==1


def test_nccl_connector_refuses_training_endpoint_and_other_domains(tmp_path):
 for url in ('https://api.llmb.nvidia.com/v2/training/runs','https://other.example/nccl'):
  with pytest.raises(ValueError):nccl.fetch_all(url,db_path=tmp_path/'refs.sqlite')
 assert nccl.status(tmp_path/'refs.sqlite')['records']==0


def test_verified_gpucomms_native_schema_and_matcher_projection(tmp_path):
 raw={'id':'native-source-1','tool':'nccl-tests','tool_binary':'all_gather_perf','library_version':'2.26.5','num_nodes':1,'num_processes':8,
  'system_info':json.dumps({'gpu_arch_type':'NVIDIA B200'}),
  'network_info':json.dumps({'network_name':'IB','nics':'ConnectX-7','switch_type':'Quantum2'}),
  'test_config':json.dumps({'nthreads':1,'ngpus':1,'minbytes':8,'maxbytes':1048576,'iters':20,'check':True}),
  'test_env_vars':json.dumps({'NCCL_TESTS_SPLIT_MASK':'0x0'}),
  'test_result':json.dumps([{'size':1048576,'type':'float','redop':'none','in_place.bus_bw':100,'out_of_place.bus_bw':99,'in_place.time':10,'out_of_place.time':11,'in_place.nwrong':'0','out_of_place.nwrong':'0'}])}
 r=nccl.normalize_record(raw,'https://gpucomms.nvidia.com/')
 assert r['source']=='GPUComms' and r['gpu']=='B200' and r['gpus']==8
 assert r['fabric']=='infiniband' and r['nic_model']=='ConnectX-7'
 assert r['link_speed_gbps'] is None and r['placement_scope'] is None
 assert r['metrics'][0]['busbw_gbps']==100 and r['metrics'][0]['latency_us']==10
 assert r['metrics'][1]['placement']=='out_of_place'
 assert r['raw_json']==raw
 db=tmp_path/'refs.sqlite';nccl.import_records([raw],'https://gpucomms.nvidia.com/',db)
 public=nccl.list_references(db_path=db)['items'][0]
 assert 'raw_json' not in public and 'metrics' not in public and public['metric_points']==2
 matcher=nccl.references(db)[0]
 assert 'raw_json' not in matcher and len(matcher['metrics'])==2
 assert 'source_mapping' not in matcher['metrics'][0]


def test_public_gpucomms_connector_refuses_other_site_credentials():
 with pytest.raises(ValueError,match='Beacon authorization'):
  nccl.fetch_gpucomms({'headers':{'Authorization':'Bearer forbidden'}})
 with pytest.raises(ValueError,match='Only the public GPUComms'):
  nccl.fetch_gpucomms({'headers':{'Cookie':'__Secure-next-auth.session-token=forbidden'}})


@pytest.mark.parametrize('native,canonical',[('disabled',0),('LOCAL_REGISTER_ALL',1),('SYMMETRIC_REGISTER_ALL',2)])
def test_native_registration_modes_and_protocol_names_stay_distinct(native,canonical):
 raw={'id':'native','tool':'nccl-tests','test_config':json.dumps({'minbytes':8,'maxbytes':2147483648,'iters':20,'warmup_iters':5,'local_register':native}),
      'test_result':json.dumps([{'size':8,'type':'float','redop':'sum','in_place.bus_bw':10}])}
 protocol=nccl.normalize_record(raw)['test_protocol']
 assert protocol['min_bytes']==8 and protocol['max_bytes']==2147483648
 assert protocol['iterations']==20 and protocol['warmup']==5 and protocol['registration']==canonical
 assert protocol['local_register']==native


def test_protocol_upgrade_preserves_native_values_and_raw_snapshot(tmp_path):
 db=tmp_path/'refs.sqlite'
 raw={'id':'native','tool':'nccl-tests','test_config':json.dumps({'minbytes':8,'maxbytes':2147483648,'iters':20,'warmup_iters':5,'local_register':'SYMMETRIC_REGISTER_ALL'}),
      'network_info':json.dumps({'network_name':'EFA'}),'test_result':json.dumps([{'size':8,'type':'float','redop':'sum','in_place.bus_bw':10}])}
 receipt=nccl.import_records([raw],db_path=db);original=Path(receipt['path']).read_bytes()
 with nccl._db(db) as connection:
  connection.execute("UPDATE nccl_references SET normalized_json=json_remove(normalized_json,'$.test_protocol.min_bytes','$.test_protocol.max_bytes','$.test_protocol.iterations','$.test_protocol.warmup','$.test_protocol.registration')")
 nccl.normalize_protocol_aliases(db)
 r=nccl.references(db)[0]
 assert r['test_protocol']['min_bytes']==8 and r['test_protocol']['registration']==2
 assert r['test_protocol']['local_register']=='SYMMETRIC_REGISTER_ALL'
 assert r['fabric']=='efa' and Path(receipt['path']).read_bytes()==original


def test_metric_preview_prioritizes_script_two_gib_point(tmp_path):
 raw=row(metrics=[{'message_bytes':16*1024**3,'busbw_gbps':100},
                  {'message_bytes':2*1024**3,'busbw_gbps':98},
                  {'message_bytes':8*1024**3,'busbw_gbps':99}])
 db=tmp_path/'refs.sqlite';nccl.import_records([raw],db_path=db)
 compact=nccl.list_references(db_path=db)['items'][0]
 assert compact['metric_points']==3
 assert compact['metric_preview'][0]['message_bytes']==2*1024**3
 assert compact['metric_preview'][1]['message_bytes']==16*1024**3


def test_global_source_counts_do_not_use_only_the_last_snapshot(tmp_path):
 db=tmp_path/'refs.sqlite'
 nccl.import_records([row(id=1),row(id=2)],'https://gpucomms.nvidia.com/',db)
 nccl.import_records([row(id=3)],'https://nvidia.atlassian.net/wiki/pages/42',db)
 state=nccl.status(db)
 assert state['records']==3 and state['measured_records']==3 and state['metric_points']==3
 assert sorted(s['records'] for s in state['sources'])==[1,2]
 assert state['snapshot']['records']==1
