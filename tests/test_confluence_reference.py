from backend import confluence_reference as wiki
from backend import beacon_nccl


def fixture():
 headers=['#','Nodes','Placement','TP','OP','Collective','algBW','busBW','Baseline busBW','SOL busBW','Assessment','Execution']
 cells=['1','14N','cross-pod','2','Default','AllReduce','200','300','250','99999','max size 2147483520 B (nccl-tests rounds 2G to a rank multiple)','COMPLETED']
 context='<table><tr><th>Field</th></tr><tr><td>GPU: B200; NCCL 2.30.0; Fabric: RoCE; 8 GPUs/node; max_bytes=2 GiB;</td></tr></table><p>Values are the out-of-place final/max-size row in GB/s.</p>'
 return context+'<table><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr><tr>'+''.join('<td>'+x+'</td>' for x in cells)+'</tr></table>'


def test_actual_current_columns_exact_rounded_size_and_unknown_correctness():
 result=wiki.extract_page({'id':'42','title':'Measured table','version':{'number':5}},fixture(),'https://nvidia.atlassian.net/wiki/pages/42')
 assert len(result['references'])==1
 raw=result['references'][0];ref=beacon_nccl.normalize_record(raw,result['source_url'])
 assert ref['gpu']=='B200' and ref['gpus']==112 and ref['nodes']==14 and ref['fabric']=='roce'
 assert ref['metrics'][0]['busbw_gbps']==300 and ref['metrics'][0]['message_bytes']==2147483520
 assert ref['metrics'][0]['placement']=='out_of_place' and ref['metrics'][0]['datatype'] is None
 assert ref['status'] is None and ref['execution_status']=='COMPLETED'
 assert ref['source']=='NVIDIA Confluence' and ref['page_version']==5
 assert ref['source_test_profile']=='2' and ref['split_mask'] is None
 assert ref['source_record_id']=='42:table1:row1'
 assert ref['source_provenance']['source_row']['SOL busBW']=='99999'
 assert ref['metrics'][0]['busbw_gbps']!=99999


def test_missing_explicit_units_or_recipe_never_creates_measurements():
 for html in [fixture().replace('in GB/s','in unknown'),fixture().replace('max_bytes=2 GiB','max_bytes=unknown').replace('max size 2147483520 B','unknown size')]:
  result=wiki.extract_page({'id':'42'},html,'https://nvidia.atlassian.net/wiki/pages/42')
  assert result['references']==[] and len(result['tables'])==2


def test_expectations_and_images_remain_evidence_only():
 html='<table><tr><th>Expected bandwidth</th></tr><tr><td>400 GB/s<ac:image><ri:attachment ri:filename="plot.png" /></ac:image></td></tr></table>'
 result=wiki.extract_page({'id':'42'},html,'https://nvidia.atlassian.net/wiki/pages/42')
 assert result['references']==[] and result['tables'][0]['image_names']==['plot.png']


def test_canonical_dashboard_context_and_source_identity_survive_projection(tmp_path):
 source='https://nccl-perf-dashboard-example.nvidia.com/'
 raw={'source_record_id':'csv/path','gpu':'B300','gpus':512,'collective':'allreduce','busbw_gbps':300,
      'row_action':source+'data/actual_observed.csv','source_csv_url':source+'data/actual_observed.csv','nic_model':'ConnectX-8','test_protocol':{'iterations':50},'source_archive_status':'FAILED',
      'selected_group_status':'passed','is_measured':True,'reference_kind':'measured','source_provenance':{'repeats':3}}
 beacon_nccl.import_records([raw],source,tmp_path/'refs.sqlite')
 ref=beacon_nccl.references(tmp_path/'refs.sqlite')[0]
 assert ref['source_record_id']=='csv/path' and ref['nic_model']=='ConnectX-8'
 assert ref['row_action'].endswith('/actual_observed.csv') and ref['source_csv_url']==ref['row_action']
 assert ref['test_protocol']['iterations']==50 and ref['source_archive_status']=='FAILED'
 assert ref['selected_group_status']=='passed' and ref['source_provenance']['repeats']==3
 assert 'raw_json' not in ref


def test_training_result_is_separate_from_published_baseline_and_precision_unknown():
 html='<p>GPU: B200; Model: example-model; Model size: 12b; Workload: training/example-model;</p><table><tr><th>Scale</th><th>Placement</th><th>Result TFLOPS/GPU</th><th>Baseline TFLOPS/GPU</th></tr><tr><td>8 GPUs</td><td>same rack</td><td>200</td><td>250</td></tr></table>'
 result=wiki.extract_page({'id':'42','version':{'number':6}},html,'https://nvidia.atlassian.net/wiki/pages/42')
 assert result['references']==[]
 row=result['training_references'][0]
 assert row['tflops_gpu']==200 and row['gpus']==8
 assert row['model']=='example-model' and row['model_size']=='12b'
 assert row['precision'] is None and row['release'] is None
 assert row['source_provenance']['source_row']['Baseline TFLOPS/GPU']=='250'


def test_unknown_hardware_recipe_stays_unknown_without_fixed_customer_defaults():
 html=fixture().replace('GPU: B200; NCCL 2.30.0; Fabric: RoCE; 8 GPUs/node;','')
 result=wiki.extract_page({'id':'42'},html,'https://wiki.example.test/page/42')
 row=result['references'][0]
 assert row['gpu'] is None and row['gpus'] is None and row['fabric'] is None
 assert row['nic_model'] is None and row['nccl_version'] is None
 assert row['test_protocol']=={'max_bytes':2147483648}
 assert row['metrics'][0]['message_bytes']==2147483520


def test_final_row_endpoint_is_explicit_and_never_defaults_to_sixteen_gib():
 html=fixture().replace('max size 2147483520 B (nccl-tests rounds 2G to a rank multiple)','Measured')
 result=wiki.extract_page({'id':'42'},html,'https://wiki.example.test/page/42')
 assert result['references'][0]['metrics'][0]['message_bytes']==2147483648
 assert wiki.extract_page({'id':'42'},html.replace('max_bytes=2 GiB','no endpoint supplied'),'https://wiki.example.test/page/42')['references']==[]
