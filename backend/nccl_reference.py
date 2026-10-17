"""Source-backed NCCL point comparisons with reusable, cached reference cohorts."""
import json
import math
import re
import statistics
from collections import defaultdict, OrderedDict
from urllib.parse import urlsplit

CORE = ('gpu', 'gpus', 'collective')
FIELDS = CORE + ('fabric', 'network', 'nccl_version', 'message_bytes', 'datatype', 'operation', 'placement', 'protocol', 'split_mask',
                 'nodes','placement_scope','rail_topology','nic_model','link_speed_gbps','congestion_control','nccl_tests_version','nccl_environment')
HARD_POINT = ('message_bytes','datatype','operation','placement','split_mask')
PROTOCOL = ('min_bytes', 'max_bytes', 'iterations', 'warmup', 'registration')
ALIASES = {
    'gpu': ('gpu','gpu_model','hardware'), 'gpus': ('gpus','scale','gpu_count','num_gpus'),
    'collective': ('collective','collective_name','operation_type'),
    'fabric': ('fabric','fabric_type','transport'), 'network': ('network','network_type','network_name'),
    'nccl_version': ('nccl_version','ncclVersion'), 'message_bytes': ('message_bytes','size_bytes','message_size_bytes'),
    'datatype': ('datatype','data_type','dtype'), 'operation': ('operation','reduction_op','reduction'),
    'placement': ('placement','buffer_placement'), 'protocol': ('protocol','nccl_protocol','NCCL_PROTO'),
    'split_mask': ('split_mask','NCCL_TESTS_SPLIT_MASK'),
    'nodes': ('nodes','num_nodes','node_count'), 'placement_scope': ('placement_scope','data_domain','communication_domain'),
    'rail_topology': ('rail_topology','railTopology'), 'nic_model': ('nic_model','nic_type','nicType'),
    'link_speed_gbps': ('link_speed_gbps','linkSpeedGbps'), 'congestion_control': ('congestion_control','congestionControl'),
    'nccl_tests_version': ('nccl_tests_version','ncclTestsVersion'), 'nccl_environment': ('nccl_environment','ncclEnvironment'),
}
METRICS = {'busbw_gbps': ('busbw_gbps','bus_bandwidth_gbps','bus_bandwidth'),
           'algbw_gbps': ('algbw_gbps','algorithm_bandwidth_gbps','algorithm_bandwidth'),
           'time_us': ('time_us','latency_us')}


def _number(value, zero=False):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) and (parsed >= 0 if zero else parsed > 0) else None
    except (ValueError, TypeError):
        return None


def _canonical(field, value):
    if value is None or value == '': return None
    token = str(value).strip().lower()
    compact = re.sub(r'[^a-z0-9]+','',token)
    if field == 'nccl_environment':
        return json.dumps(value,sort_keys=True,separators=(',',':')) if isinstance(value,dict) else token
    if field == 'link_speed_gbps': return _number(value)
    if field in ('gpus','message_bytes','nodes'):
        value = _number(value)
        return int(value) if value is not None and value.is_integer() else None
    if field == 'split_mask':
        try: return int(token,0)
        except ValueError: return token
    if field == 'gpu':
        match = re.search(r'\b(gb300|gb200|b300|b200|b100|h200|h100|h800|a100)\b', token)
        return match[1].upper() if match else token.upper()
    if field == 'collective':
        return compact.replace('perfmpi','').removesuffix('perf')
    if field == 'fabric':
        if any(term in compact for term in ('spectrumx','roce','ethernet')): return 'roce'
        if 'infiniband' in compact or compact in ('ib','quantum','quantum2','quantum3'): return 'infiniband'
        if 'nvlink' in compact or 'nvswitch' in compact: return 'nvlink'
        if 'socket' in compact or compact == 'tcp': return 'socket'
        return compact
    if field == 'network':
        if 'spectrumx' in compact: return 'spectrum-x'
        if compact in ('ib','infiniband'): return 'infiniband'
        if compact in ('roce','ethernet'): return 'roce'
        if 'nvlink' in compact or 'nvswitch' in compact: return 'nvlink'
        return compact
    if field == 'nccl_version':
        match = re.search(r'\d+\.\d+(?:\.\d+)?', token)
        return match[0] if match else token.removeprefix('v')
    if field == 'datatype': return {'fp32':'float','float32':'float','fp16':'half','float16':'half','bf16':'bfloat16'}.get(compact,compact)
    if field == 'placement': return {'outofplace':'out_of_place','inplace':'in_place','oop':'out_of_place','ip':'in_place'}.get(compact,compact)
    return token


def _raw(record):
    value = record.get('raw_json') or {}
    if isinstance(value,str):
        try: value = json.loads(value)
        except (ValueError,TypeError): return {}
    return value if isinstance(value,dict) else {}


def _pick(layers, names):
    for label, layer in layers:
        if not isinstance(layer,dict): continue
        for name in names:
            if layer.get(name) not in (None,''): return layer[name],label
    return None,None


def nccl_context(run, point=None):
    meta = run.get('metadata') or {}; raw = _raw(run)
    layers = [('point',point or {}),('record',run),('metadata',meta),('raw',raw)]
    context = {}; provenance = {}; unverified = []
    for field,names in ALIASES.items():
        value,source = _pick(layers,names)
        context[field] = _canonical(field,value)
        if context[field] is not None: provenance[field] = source
    explicit_fabric,_ = _pick(layers,('fabric','fabric_type'))
    link_layer,_ = _pick(layers,('link_layer','linkLayer'))
    if link_layer is not None:
        context['fabric'] = _canonical('fabric',link_layer);provenance['fabric'] = 'link_layer_evidence'
    elif explicit_fabric is None and str(run.get('transport') or '').lower() == 'infiniband + nvlink':
        # Older parsers inferred this from NET/IB verbs plus NVLS. Verbs do
        # not distinguish InfiniBand from RoCE; preserve the uncertainty.
        context['fabric'] = None;provenance['fabric'] = 'legacy_verbs_transport_unverified'
    if context['fabric'] is None and context['network'] is not None:
        context['fabric'] = _canonical('fabric',context['network']);provenance['fabric'] = 'network_declaration'
    recipe = str(meta.get('recipe_id') or run.get('recipe_id') or '')
    if context['fabric'] is None and recipe:
        if 'spectrum-x' in recipe or 'infiniband' in recipe:
            context['fabric'] = 'roce' if 'spectrum-x' in recipe else 'infiniband'
            provenance['fabric'] = 'recipe_declaration_unverified';unverified.append('fabric')
    if context['network'] is None and context['fabric'] == 'nvlink':
        context['network'] = 'nvlink';provenance['network'] = 'native_fabric'
    if context['gpu'] is None and meta.get('declared_gpu_model'):
        context['gpu'] = _canonical('gpu',meta['declared_gpu_model'])
        provenance['gpu'] = 'recipe_declaration_unverified';unverified.append('gpu')
    test_protocol,source = _pick(layers,('test_protocol',))
    test_protocol = test_protocol if isinstance(test_protocol,dict) else {}
    for field in PROTOCOL:
        value = test_protocol.get(field)
        context['test.'+field] = value
        if value is not None: provenance['test.'+field] = source
    primary_keys = set(PROTOCOL) | {'minbytes','maxbytes','iters','warmup_iters','local_register'}
    extra_aliases = {'agg_iters':'aggregation_iterations','cudagraph':'cuda_graph_launches'}
    extras = {extra_aliases.get(key,key):value for key,value in test_protocol.items()
              if key not in primary_keys and value is not None}
    if extras:
        context['test.additional_protocol'] = json.dumps(extras,sort_keys=True,separators=(',',':'))
        provenance['test.additional_protocol'] = source
    return dict(context=context,provenance=provenance,unverified=unverified)


def _identity(ref):
    if ref.get('job_id') is not None:
        return (urlsplit(ref.get('source_url') or '').netloc,ref.get('cluster'),str(ref['job_id']))
    return (ref.get('source_url'),ref.get('source_record_id') or ref.get('ref_id'))


def _metric_value(point,metric,zero=False):
    for alias in METRICS[metric]:
        if point.get(alias) is not None:
            return _number(point[alias],zero=zero)
    return None


def _points(run):
    rows = [point for point in run.get('metrics') or [] if any(_metric_value(point,key,zero=key!='time_us') is not None for key in METRICS)]
    if rows: return rows
    row = {}
    for key,aliases in METRICS.items():
        value,_ = _pick([('record',run),('raw',_raw(run))],aliases)
        if _number(value): row[key] = float(value)
    if not row and _number(run.get('avg_bus_bw_gbps')):
        row['busbw_gbps'] = float(run['avg_bus_bw_gbps']);row['metric_scope'] = 'summary_average'
    return [row] if row else []


def _eligible(ref):
    raw = _raw(ref)
    if not ref.get('source_url') or not (ref.get('ref_id') or ref.get('source_record_id')): return False
    metadata = ref.get('metadata') or {}
    if ref.get('status') in ('failed','incomplete','unknown') or metadata.get('parent_job_failed') or metadata.get('exit_failure') or metadata.get('mixed_split_masks'): return False
    if metadata.get('correctness_errors') not in (None,0): return False
    for layer in (ref,raw):
        kind = str(layer.get('kind') or layer.get('reference_kind') or 'measured').lower()
        if layer.get('theoretical') is True or layer.get('is_measured') is False or 'theoretical' in kind or kind == 'sol': return False
    return True


def from_run_reference(run):
    """Adapt a correctness-clean catalog run without adding missing context.

    Returns None for incomplete, failed, mixed-mask, or unreviewed upload runs.
    Raw logs and explicitly reviewed uploaded measurements retain source IDs.
    """
    metadata = run.get('metadata') or {}
    if run.get('workload') != 'NCCL' or run.get('status') != 'passed': return None
    if metadata.get('parent_job_failed') or metadata.get('exit_failure') or metadata.get('mixed_split_masks'): return None
    if metadata.get('correctness_errors') != 0: return None
    if run.get('collective') in (None,'mixed'): return None
    if run.get('source_kind') == 'upload' and not run.get('_reference_review',{}).get('active'): return None
    points = [point for point in run.get('metrics') or [] if _number(point.get('busbw_gbps'))]
    if not points or any(point.get('errors') != 0 for point in points): return None
    identifier = run.get('id')
    if not identifier: return None
    result = dict(run)
    result.update(ref_id=identifier,source_record_id=identifier,source_url='/api/runs/'+str(identifier),row_action='/api/runs/'+str(identifier),
                  source_sha256=run.get('source',{}).get('sha256'),reference_origin='catalog_measured_log')
    return result


def _source_hash(record):
    source = record.get('source') or {}
    return record.get('source_sha256') or record.get('sha256') or (source.get('sha256') if isinstance(source,dict) else None)


def _job_summary(entries,details_cache=None):
    metrics = defaultdict(list);references = {};hashes = set();ids = set();unverified = set();best = {};best_references = {}
    for ref,point,extracted in entries:
        for metric in METRICS:
            value = _metric_value(point,metric)
            if value is not None:
                metrics[metric].append(value)
                if metric not in best or (value < best[metric] if metric == 'time_us' else value > best[metric]):
                    best[metric] = value;best_references[metric] = {ref.get('ref_id') or ref.get('source_record_id')}
                elif value == best[metric]: best_references[metric].add(ref.get('ref_id') or ref.get('source_record_id'))
        identifier = ref.get('ref_id') or ref.get('source_record_id')
        ids.add(identifier);hashes.add(_source_hash(ref));unverified.update(extracted['unverified'])
        detail_key = (identifier,ref.get('source_url'),_source_hash(ref))
        detail = details_cache.get(detail_key) if details_cache is not None else None
        if detail is None:
            detail = {key:ref.get(key) for key in ('ref_id','source_record_id','source_url','row_action','job_id','cluster','date_utc','source_mapping','raw_sha256','source_path','source_sha256','reference_origin',
                'source_page_id','source_page_version','source_page_title','source_section','source_table_index','source_csv_url','source_metadata_url','source_archive_status','archive_status_caveat','selected_group_status','repeat_count','reference_statistic')}
            if details_cache is not None: details_cache[detail_key] = detail
        references[identifier] = detail
    first = entries[0][0]
    return dict(entries=entries,values={metric:statistics.median(values) for metric,values in metrics.items()},
                counts={metric:len(values) for metric,values in metrics.items()},best=best,best_references=best_references,references=references,hashes=hashes,ids=ids,
                job=(first.get('cluster'),str(first.get('job_id'))),unverified=unverified,details_cache=details_cache)


def _cohort_summary(entries,details_cache=None):
    jobs = defaultdict(list)
    for entry in entries: jobs[_identity(entry[0])].append(entry)
    return dict(context=entries[0][2]['context'],jobs={key:_job_summary(value,details_cache) for key,value in jobs.items()})


def _filter_cohort(cohort,excluded_job,excluded_hash,excluded_id,reference_statistic='median',required_cluster=None):
    values = defaultdict(list);counts = defaultdict(int);details = {};ref_count = 0;identities = set();unverified = set();best_values = {};best_details = defaultdict(dict)
    for identity,job in cohort['jobs'].items():
        if required_cluster is not None and job['job'][0] != required_cluster: continue
        if excluded_job and job['job'] == excluded_job: continue
        if (excluded_hash and excluded_hash in job['hashes']) or (excluded_id and excluded_id in job['ids']):
            entries = [(ref,point,e) for ref,point,e in job['entries']
                       if not(excluded_hash and _source_hash(ref) == excluded_hash)
                       and not(excluded_id and ref.get('ref_id') == excluded_id)]
            if not entries: continue
            job = _job_summary(entries,job.get('details_cache'))
        identities.add(identity);unverified.update(job['unverified']);ref_count += len(job['references'])
        for metric,value in job['best' if reference_statistic == 'best' else 'values'].items():
            values[metric].append(value);counts[metric] += job['counts'][metric]
            if metric not in best_values or (value < best_values[metric] if metric == 'time_us' else value > best_values[metric]):
                best_values[metric] = value;best_details[metric] = {}
            if reference_statistic == 'best' and value == best_values[metric]:
                for identifier in sorted(job['best_references'].get(metric,()),key=str):
                    if len(best_details[metric]) < 20: best_details[metric][identifier] = dict(job['references'][identifier],reference_metric=metric,reference_value=value,message_bytes=cohort['context']['message_bytes'])
        if len(details)<20:
            for identifier,ref in job['references'].items():
                if reference_statistic == 'best' and not any(identifier in winners for winners in job['best_references'].values()): continue
                details.setdefault(identifier,ref)
                if len(details)>=20: break
    return dict(values=values,counts=counts,references=details,reference_count=ref_count,jobs=identities,unverified=unverified,best_references={metric:list(refs.values()) for metric,refs in best_details.items()})


class NCCLReferenceIndex:
    """Build once per reference snapshot; matching results are reused per context."""
    def __init__(self, references, max_cache=4096):
        self.buckets = defaultdict(lambda:defaultdict(list));self.point_buckets = defaultdict(lambda:defaultdict(dict));self.cache = OrderedDict();self.records = 0
        self.max_cache = max(1,int(max_cache));self.contexts = {};self.summaries = {};self.details_cache = {}
        self.hash_owners = defaultdict(set);self.id_owners = defaultdict(set)
        for ref in references:
            if not ref or not _eligible(ref): continue
            owner = (ref.get('cluster'),str(ref.get('job_id')))
            if _source_hash(ref): self.hash_owners[_source_hash(ref)].add(owner)
            if ref.get('ref_id'): self.id_owners[ref['ref_id']].add(owner)
            for point in _points(ref):
                if point.get('errors') not in (None,0): continue
                extracted = nccl_context(ref,point);ctx = extracted['context']
                if ref.get('status') != 'passed' or point.get('errors') != 0 and ref.get('metadata',{}).get('correctness_errors') != 0:
                    extracted['unverified'].append('reference_correctness')
                if str(ref.get('source_archive_status','')).startswith('failed'):
                    extracted['unverified'].append('reference_archive_caveat')
                core = tuple(ctx[field] for field in CORE)
                if any(value is None for value in core): continue
                key = json.dumps(ctx,sort_keys=True,default=str)
                extracted = self.contexts.setdefault((key,tuple(extracted['unverified'])),extracted)
                self.buckets[core][key].append((ref,point,extracted));self.records += 1
                hard_key = tuple(ctx[field] for field in HARD_POINT)
                self.point_buckets[core][hard_key][key] = self.buckets[core][key]
        for cohorts in self.buckets.values():
            for key,entries in cohorts.items(): self.summaries[key] = _cohort_summary(entries,self.details_cache)

    def match(self, extracted, excluded_job=None, excluded_hash=None, excluded_id=None, reference_statistic='median',allow_unknown_fabric_cluster=None,best_metric='busbw_gbps'):
        if excluded_job:
            if excluded_hash and self.hash_owners.get(excluded_hash) == {excluded_job}: excluded_hash = None
            if excluded_id and self.id_owners.get(excluded_id) == {excluded_job}: excluded_id = None
        context = extracted['context'];key = json.dumps([context,sorted(extracted['unverified']),excluded_job,excluded_hash,excluded_id,reference_statistic,allow_unknown_fabric_cluster,best_metric],sort_keys=True,default=str)
        # Message size is an identity, never a wildcard. Summary averages and
        # malformed parsed points cannot establish pointwise performance.
        same_cluster_fallback = bool(reference_statistic == 'best' and allow_unknown_fabric_cluster and context.get('fabric') is None)
        required = ['message_bytes'] + (['nccl_version'] + ([] if same_cluster_fallback else ['fabric']) if reference_statistic == 'best' else [])
        absent = [field for field in required if context.get(field) is None or field in extracted['unverified']]
        if absent:
            return dict(aggregate=dict(values={},counts={},references={},reference_count=0,jobs=set()),tier='none',missing=absent,
                        mismatches=[],alternative_cohorts=0,context=None)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        point_buckets = self.point_buckets.get(tuple(context[field] for field in CORE),{})
        buckets = {}
        for hard_key,cohorts in point_buckets.items():
            if hard_key[HARD_POINT.index('message_bytes')] != context['message_bytes']: continue
            if any(context[field] is not None and value is not None and context[field] != value
                   for field,value in zip(HARD_POINT,hard_key)): continue
            buckets.update(cohorts)
        candidates = []
        for canonical,entries in buckets.items():
            ref_context = self.summaries[canonical]['context']
            if reference_statistic == 'best' and (ref_context['nccl_version'] != context['nccl_version'] or ref_context['fabric'] != context['fabric']): continue
            if (context['fabric'] is None) != (ref_context['fabric'] is None): continue
            if ('fabric' in extracted['unverified']) and ref_context['fabric'] is not None: continue
            if any(context[field] is not None and ref_context[field] is not None and context[field] != ref_context[field]
                   for field in ('fabric','network','nodes','placement_scope','nic_model','link_speed_gbps','rail_topology')): continue
            filtered = _filter_cohort(self.summaries[canonical],excluded_job,excluded_hash,excluded_id,reference_statistic,
                                      required_cluster=allow_unknown_fabric_cluster if same_cluster_fallback else None)
            if not filtered['jobs']: continue
            fields = list(dict.fromkeys([*context,*ref_context]))
            conflicts = [field for field in fields if context.get(field) is not None and ref_context.get(field) is not None and context[field] != ref_context[field]]
            missing = [field for field in fields if context.get(field) is None or ref_context.get(field) is None]
            unverified = sorted(set(extracted['unverified']) | filtered['unverified'])
            exact = not missing and not conflicts and not unverified
            matches = sum(context.get(field) is not None and context[field] == ref_context.get(field) for field in fields)
            ranking = (not exact,len(conflicts),-matches,-len(filtered['jobs']),canonical)
            if reference_statistic == 'best':
                recorded = filtered['values'].get(best_metric,[])
                if not recorded: continue
                # Best means maximum recorded bandwidth across hard-compatible
                # cohorts, never the cohort with most jobs or closest tuning.
                # Keep one winning cohort; disclose its tuning differences.
                value = min(recorded) if best_metric == 'time_us' else max(recorded)
                ranking = (value if best_metric == 'time_us' else -value,) + ranking
            candidates.append((ranking,filtered,missing,conflicts,unverified,ref_context))
        if candidates:
            _,filtered,missing,conflicts,unverified,ref_context = min(candidates,key=lambda candidate:candidate[0])
            result = dict(aggregate=filtered,tier='exact' if not missing and not conflicts and not unverified else 'contextual',
                          missing=sorted(set(missing+unverified)),mismatches=conflicts,alternative_cohorts=len(candidates)-1,
                          context=ref_context)
        else:
            result = dict(aggregate=dict(values={},counts={},references={},reference_count=0,jobs=set()),tier='none',missing=[field for field,value in context.items() if value is None],
                          mismatches=[],alternative_cohorts=0,context=None)
        self.cache[key] = result
        if len(self.cache) > self.max_cache: self.cache.popitem(last=False)
        return result


    def best_point(self, run, message_bytes, metric='busbw_gbps', point=None,allow_same_cluster_unknown_fabric=False):
        """Best recorded value across hard-compatible, exact-version/size cohorts.

        Missing GPU, scale, collective, fabric, version or bytes returns no
        expectation. Physical or point-context conflicts never match. Remaining
        missing context and source caveats remain explicit and conditional.
        """
        if metric not in METRICS: raise ValueError('Unsupported NCCL metric')
        supplied = dict(point or {},message_bytes=message_bytes)
        extracted = nccl_context(run,supplied)
        match = self.match(extracted,
            excluded_job=(run.get('cluster'),str(run['job_id'])) if run.get('cluster') and run.get('job_id') is not None else None,
            excluded_hash=_source_hash(run),excluded_id=run.get('id'),reference_statistic='best',
            allow_unknown_fabric_cluster=run.get('cluster') if allow_same_cluster_unknown_fabric else None,best_metric=metric)
        aggregate = match['aggregate'];values = aggregate['values'].get(metric,[])
        expected = (min(values) if metric == 'time_us' else max(values)) if values else None
        return dict(metric=metric,message_bytes=extracted['context']['message_bytes'],expected=expected,
                    unit='us' if metric == 'time_us' else 'GB/s',reference_statistic='best',
                    reference_jobs=len(values),reference_records=aggregate['counts'].get(metric,0),
                    references=aggregate.get('best_references',{}).get(metric,[]),tier=match['tier'],
                    missing_context=match['missing'],mismatches=match['mismatches'],
                    reference_context=match['context'],alternative_cohorts=match['alternative_cohorts'],
                    context_note='Best recorded on this cluster; fabric metadata incomplete' if expected is not None and extracted['context']['fabric'] is None else None)


def compare_nccl(run, references, *, good_pct=5, bad_pct=10, reference_statistic='median'):
    if not (math.isfinite(good_pct) and math.isfinite(bad_pct) and 0 <= good_pct <= bad_pct < 100): raise ValueError('Invalid deficit thresholds')
    if reference_statistic not in ('median','best'): raise ValueError('Unsupported reference statistic')
    index = references if isinstance(references,NCCLReferenceIndex) else NCCLReferenceIndex(references)
    result = dict(verdict='insufficient_evidence',classification='unknown',candidate_band=None,tier='none',metrics=[],references=[],
                  missing_context=[],mismatches=[],provenance=dict(reference_jobs=0,reference_records=0,point_count=0),
                  context=nccl_context(run)['context'],policy=dict(good_pct=good_pct,bad_pct=bad_pct,reference_statistic=reference_statistic,
                  repeat_aggregation=('Best recorded per source job before best across independent jobs' if reference_statistic == 'best' else 'Median per source job before aggregation across jobs'),
                  cohort_selection=('Highest recorded bus bandwidth across hard-compatible cohorts; tuning differences stay conditional. Cohorts are never pooled.' if reference_statistic == 'best' else 'One compatible execution cohort; nearest declared context, then most independent source jobs. Incompatible cohorts are never pooled.'),
                  excluded_mismatch_fields=['gpu','gpus','collective','fabric','network','message_bytes','datatype','operation','placement','split_mask','nodes','placement_scope','nic_model','link_speed_gbps','rail_topology'],required_context=list(FIELDS)+['test.'+f for f in PROTOCOL],
                  description='Explicit hub performance comparison policy, not certification. Conditional ratios are unscored; known network/fabric differences never match.'),recommendations=[])
    refs = {};jobs = set();missing = set();conflicts = set();tiers = [];observed_count = 0;reference_total = 0
    for point in _points(run):
        observed_context = nccl_context(run,point)
        excluded_job = (run.get('cluster'),str(run['job_id'])) if run.get('cluster') and run.get('job_id') is not None else None
        match = index.match(observed_context,excluded_job=excluded_job,excluded_hash=_source_hash(run),excluded_id=run.get('id'),reference_statistic=reference_statistic)
        missing.update(match['missing']);conflicts.update(match['mismatches']);tiers.append(match['tier'])
        for metric in METRICS:
            measured = _metric_value(point,metric,zero=metric!='time_us')
            if measured is None:
                if any(alias in point for alias in METRICS[metric]):
                    observed_count += 1;missing.add('valid.'+metric)
                continue
            observed_count += 1
            aggregate = match['aggregate']
            medians = aggregate['values'].get(metric,[])
            if not medians: continue
            jobs.update(aggregate['jobs']);refs.update(aggregate['references']);reference_total = max(reference_total,aggregate['reference_count'])
            higher = metric != 'time_us'
            expected = statistics.median(medians) if reference_statistic == 'median' else (max(medians) if higher else min(medians))
            if higher and measured == 0 and expected <= 0.01:
                # Native NCCL bandwidth columns report two decimals. A printed
                # zero versus 0.01 does not establish a numeric 100% deficit.
                missing.add('bandwidth_reporting_resolution')
                result['provenance']['rounding_limited_measurements'] = result['provenance'].get('rounding_limited_measurements',0) + 1
                continue
            percent = 100*(measured/expected if higher else expected/measured)
            rating = 'good' if 100-percent <= good_pct+1e-9 else 'bad' if 100-percent <= bad_pct+1e-9 else 'worst'
            ctx = observed_context['context']
            result['metrics'].append(dict(metric=metric,observed=measured,expected=expected,expected_pct=percent,delta_pct=percent-100,deficit_pct=100-percent,
                performance_band=rating,classification='conditional',scored=False,unit='us' if metric=='time_us' else 'GB/s',
                direction='lower_better' if metric=='time_us' else 'higher_better',message_bytes=ctx['message_bytes'],datatype=ctx['datatype'],operation=ctx['operation'],placement=ctx['placement'],
                protocol=ctx['protocol'],tier=match['tier'],reference_jobs=len(medians),reference_records=aggregate['counts'][metric],
                reference_min=min(medians),reference_max=max(medians),reference_context=match['context'],alternative_cohorts=match['alternative_cohorts'],
                reference_statistic=reference_statistic,reference_sources=aggregate.get('best_references',{}).get(metric,[]) if reference_statistic == 'best' else []))
    result['missing_context'] = sorted(missing);result['mismatches'] = sorted(conflicts)
    result['references'] = list(refs.values())[:20];result['provenance']['reference_details_total'] = max(reference_total,len(refs))
    result['provenance']['reference_details_truncated'] = max(reference_total,len(refs)) > 20
    result['provenance'].update(reference_jobs=len(jobs),reference_records=len(refs),point_count=len(result['metrics']),observed_measurements=observed_count)
    nominal = max((metric['performance_band'] for metric in result['metrics']),key=lambda band:('good','bad','worst').index(band),default=None)
    result['candidate_band'] = nominal
    complete = bool(result['metrics']) and len(result['metrics']) == observed_count
    exact = complete and all(tier=='exact' for tier in tiers)
    meta = run.get('metadata') or {}
    if meta.get('mixed_split_masks'):
        exact = False
        result['missing_context'] = sorted(set(result['missing_context'] + ['single_split_mask']))
    failed = run.get('status') == 'failed' or meta.get('correctness_errors') not in (None,0) or meta.get('exit_failure') or meta.get('parent_job_failed') or any(point.get('errors') for point in run.get('metrics') or [])
    result['tier'] = 'exact' if exact else 'contextual' if refs else 'none'
    if failed: result.update(verdict='fail',classification='failed')
    elif exact and run.get('status') == 'passed': result.update(verdict='pass' if nominal=='good' else 'fail',classification=nominal)
    elif result['metrics']: result['classification'] = 'conditional'
    for metric in result['metrics']:
        metric.update(scored=bool(exact and not failed and run.get('status')=='passed'),classification=result['classification'])
    if not exact:
        result['recommendations'].append(dict(title='Reproduce the source NCCL protocol',reason='Missing or different context makes these ratios conditional.',
            command='Record GPU count, physical fabric/network, NCCL version, message bytes, datatype, reduction, placement, NCCL protocol, warmups, iterations and registration; repeat the same test at least 3 times.',
            expected='Zero correctness errors and a reproducible median against the same source-backed configuration.',rollback='Restore the recorded control environment after each isolated change.'))
    return result


assess_nccl = compare_nccl
