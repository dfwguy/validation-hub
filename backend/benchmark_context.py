"""Validate explicit user declarations separately from measured log evidence."""
import json
from .performance import CONTEXT_FIELDS, training_context, canonical

EXTRA = {'test_protocol','rack_ids','su_ids','nvlink_domains','nccl_tests_version','congestion_control','nccl_environment','placement_scope','rail_topology','nic_model','link_speed_gbps'}


def parse(value):
    if not value:return {}
    if len(value.encode())>16384:raise ValueError('Benchmark context JSON must be at most 16 KiB')
    def nonfinite(value):raise ValueError('Non-finite JSON values are not allowed')
    try:context=json.loads(value,parse_constant=nonfinite)
    except ValueError:raise ValueError('Benchmark context must be valid JSON')
    if not isinstance(context,dict):raise ValueError('Benchmark context must be a JSON object')
    allowed=set(CONTEXT_FIELDS)-{'gpu','gpus'}|EXTRA
    if set(context)-allowed:raise ValueError('Unsupported benchmark context fields: '+', '.join(sorted(set(context)-allowed)))
    text={'model','model_size','precision','release','nccl_tests_version','congestion_control','placement_scope','rail_topology','nic_model'}
    for key,val in context.items():
        if key in text:
            if not isinstance(val,(str,int,float)) or isinstance(val,bool) or not str(val).strip() or len(str(val))>256:raise ValueError(key+' must be a short nonempty value')
            context[key]=str(val)
        elif key in ('rack_ids','su_ids','nvlink_domains'):
            if not isinstance(val,list) or not val or len(val)>4096 or any(not isinstance(x,str) or not x or len(x)>128 for x in val):raise ValueError(key+' must be an array of nonempty identifiers')
            context[key]=sorted(set(val))
        elif key=='test_protocol':
            if not isinstance(val,dict) or set(val)-{'min_bytes','max_bytes','iterations','warmup','registration'}:raise ValueError('Unsupported test_protocol field')
            for field,number in val.items():
                if field=='registration' and number=='default':continue
                if type(number) is not int or number<0 or field not in ('warmup','registration') and number==0:raise ValueError('Invalid test_protocol.'+field)
                if field=='registration' and number>2:raise ValueError('Registration mode must be 0, 1 or 2')
            if val.get('min_bytes',0)>val.get('max_bytes',2**64):raise ValueError('Minimum message size exceeds maximum')
        elif key=='nccl_environment':
            import re
            if not isinstance(val,dict) or len(val)>200 or any(not re.fullmatch(r'NCCL_[A-Z0-9_]+',x) or not isinstance(y,str) or len(y)>512 for x,y in val.items()):raise ValueError('nccl_environment must map NCCL variable names to short string values')
        elif val=='none' and key in ('virtual_pipeline','expert_tensor_parallel'):continue
        elif type(val) is not int or val<=0:raise ValueError(key+' must be a positive integer (or explicit none for a disabled optional parallel mode)')
    return context


def attach(parsed,context):
    metadata=parsed['metadata']
    extracted=training_context(parsed)
    for key,value in context.items():
        observed=metadata.get(key)
        if observed is not None and observed!=value:raise ValueError(key+' contradicts log context')
        if key in CONTEXT_FIELDS and extracted['context'].get(key) is not None and canonical(key,value)!=extracted['context'][key]:raise ValueError(key+' contradicts parsed or experiment-path context')
        metadata[key]=value
    if context:metadata['user_declared_benchmark_fields']=sorted(set(metadata.get('user_declared_benchmark_fields',[]))|set(context))
