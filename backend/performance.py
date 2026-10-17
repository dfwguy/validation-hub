"""Evidence-based training comparisons; pure functions, no reference fabrication.

Beacon measurements are references, not a certification. Exact scoring requires
all declared workload context; partial matches remain explicitly conditional.
"""
import json
import math
import re
import statistics
from collections import defaultdict
from urllib.parse import urlsplit

CONTEXT_FIELDS = ('model','model_size','gpu','gpus','precision','release',
                  'global_batch_size','sequence_length','tensor_parallel','pipeline_parallel',
                  'context_parallel','virtual_pipeline','expert_parallel','expert_tensor_parallel','micro_batch_size')
CORE_FIELDS = ('model','model_size','gpu','gpus','precision')
ALIASES = {
    'model': ('model','model_name'), 'model_size': ('model_size','model_size_b','parameters_b','size'),
    'gpu': ('gpu','gpu_model','hardware'), 'gpus': ('gpus','scale','num_gpus','gpu_count'),
    'precision': ('precision','dtype'), 'release': ('release','llmb_release','framework_release','container_release'),
    'global_batch_size': ('global_batch_size','global_batch','gbs'),
    'sequence_length': ('sequence_length','seq_length','seq_len','seqlen'),
    'tensor_parallel': ('tensor_parallel','tensor_model_parallel_size','tp'),
    'pipeline_parallel': ('pipeline_parallel','pipeline_model_parallel_size','pp'),
    'context_parallel': ('context_parallel','context_parallel_size','cp'),
    'virtual_pipeline': ('virtual_pipeline','virtual_pipeline_model_parallel_size','vp'),
    'expert_parallel': ('expert_parallel','expert_model_parallel_size','ep'),
    'expert_tensor_parallel': ('expert_tensor_parallel','expert_tensor_parallel_size','etp'),
    'micro_batch_size': ('micro_batch_size','mbs'),
}


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed > 0 else None
    except (TypeError, ValueError):
        return None


def canonical(field, value):
    if value is None or value == '':
        return None
    text = str(value).strip().lower()
    if field == 'gpu':
        match = re.search(r'\b(gb300|gb200|b300|b200|h200|h100|a100)\b', text)
        return match[1].upper() if match else text.upper()
    if field == 'model':
        text = re.sub(r'^(?:pretrain|training)[_ -]+', '', text)
        text = re.sub(r'[_ |]+', '-', text)
        text = text.replace('llama31','llama3.1').replace('llama-3.1','llama3.1').replace('llama-3','llama3')
        text = text.replace('nemotronh','nemotron-h').replace('deepseek-v-3','deepseek-v3')
        text = re.sub(r'(?<![a-z0-9])\d+(?:\.\d+)?b(?![a-z0-9])', '', text)
        return re.sub(r'-+', '-', text).strip('-')
    if field == 'model_size':
        compact = re.fullmatch(r'(\d+(?:\.\d+)?)([bt])(?:[-_]?a\d+(?:\.\d+)?[bt])?', text)
        if compact:
            return str(float(compact[1])).removesuffix('.0') + compact[2]
        text = re.sub(r'\s*(?:billion|b)\s*$', '', text)
        return str(float(text)).removesuffix('.0') + 'b' if number(text) else text
    if field in ('precision','release'):
        if field == 'release':
            text = re.sub(r'^v(?=\d)', '', text)
        return re.sub(r'[ -]+', '_', text)
    if text in ('none','null','disabled'):
        return 'none'
    value = number(text)
    return int(value) if value and value.is_integer() else value


def training_context(run):
    """Extract explicit context and filename declarations, retaining provenance."""
    context = dict.fromkeys(CONTEXT_FIELDS)
    provenance = {}
    metadata = run.get('metadata') or {}
    raw = run.get('raw_json') or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = {}
    layers = [('record', run), ('metadata', metadata), ('config', metadata.get('config', {})),
              ('raw', raw), ('raw.config', raw.get('config', {}) if isinstance(raw, dict) else {})]
    for field, aliases in ALIASES.items():
        for source, layer in layers:
            if not isinstance(layer, dict):
                continue
            value = next((layer[key] for key in aliases if layer.get(key) not in (None, '')), None)
            if value is not None:
                context[field] = canonical(field, value)
                provenance[field] = source
                break
    if context['release'] is None:
        upload_name = str(metadata.get('upload_filename') or '')
        release_match = re.search(r'llmb[-_]archive[_-]v?(\d{2}\.\d{2}(?:\.\d{1,2})?)(?=[_./-]|$)', upload_name, re.I)
        if release_match:
            context['release'] = canonical('release', release_match[1])
            provenance['release'] = 'archive_filename_declared'
    name = str(metadata.get('archive_member') or run.get('filename') or run.get('job_dir') or
               (raw.get('job_dir') if isinstance(raw, dict) else None) or run.get('title') or run.get('source_path') or '')
    config_match = re.search(r'(?:pretrain[_-])(.+?)(?:_(?:bf16|fp16|fp8(?:_[a-z]+)?|nvfp4|fp4))(?=_|/|\.)', name, re.I)
    if config_match:
        model_spec = config_match[1].rsplit('/', 1)[-1]
        size = re.search(r'(?:^|[_-])(\d+(?:\.\d+)?b)(?:[_-]|$)', model_spec, re.I)
        path_model = canonical('model', model_spec)
        if context['model'] is None:
            context['model'] = path_model
            provenance['model'] = 'experiment_path'
        elif path_model.startswith(context['model'] + '-'):
            # A source recipe may explicitly carry an active-size or named
            # variant omitted from the display family. Retain that declaration.
            context['model'] = path_model
            provenance['model'] = 'experiment_path_variant'
        elif path_model != context['model']:
            # Conflicting explicit identities are not silently aliased.
            provenance['model'] = 'conflicting_model_declarations_unverified'
        if context['model_size'] is None and size:
            context['model_size'] = canonical('model_size', size[1])
            provenance['model_size'] = 'experiment_path'
    if context['precision'] is None:
        precision = re.search(r'(?:^|[_/])(bf16|fp16|fp8(?:_(?:mx|cs))?|nvfp4|fp4)(?=[_/\.]|$)', name, re.I)
        if precision:
            context['precision'] = canonical('precision', precision[1])
            provenance['precision'] = 'experiment_path'
    tokens = {'gpus':'gpus','tp':'tensor_parallel','pp':'pipeline_parallel','cp':'context_parallel',
              'vp':'virtual_pipeline','ep':'expert_parallel','etp':'expert_tensor_parallel',
              'mbs':'micro_batch_size','gbs':'global_batch_size','seq':'sequence_length'}
    for token, field in tokens.items():
        match = re.search(r'(?:^|[_/])' + token + r'(\d+|None)(?=[_/\.]|$)', name, re.I)
        if match and context[field] is None:
            context[field] = canonical(field, match[1])
            provenance[field] = 'experiment_path'
    if context['model_size'] is None:
        model_text = str(run.get('model') or run.get('model_name') or run.get('workload') or '')
        size = re.search(r'(?:^|[ _|/-])(\d+(?:\.\d+)?b)(?:[ _|/-]|$)', model_text, re.I)
        if size:
            context['model_size'] = canonical('model_size', size[1])
            provenance['model_size'] = 'model_identifier'
    for _, layer in layers:
        if not isinstance(layer, dict):
            continue
        size_spec = next((layer[key] for key in ALIASES['model_size'] if layer.get(key) not in (None, '')), None)
        active = re.fullmatch(r'\d+(?:\.\d+)?[bt][-_]?(a\d+(?:\.\d+)?[bt])', str(size_spec).lower()) if size_spec is not None else None
        if active and context['model'] is not None and not context['model'].endswith('-' + active[1]):
            context['model'] += '-' + active[1]
            provenance['model'] = 'model_size_variant_declaration'
            break
    # A workload family is usable context, but does not supply a missing size.
    if context['model'] is None and run.get('workload') not in (None,'Unknown','Training','NCCL'):
        context['model'] = canonical('model', run['workload'])
        provenance['model'] = 'workload_family'
    if context['gpu'] is None:
        filename_gpu = re.search(r'(?:^|[^a-z0-9])(gb300|gb200|b300|b200|h200|h100|a100)(?=[^a-z0-9]|$)',
                                 name + '/' + str(metadata.get('upload_filename') or ''), re.I)
        if filename_gpu:
            context['gpu'] = canonical('gpu', filename_gpu[1])
            provenance['gpu'] = 'filename_declaration_unverified'
    if context['gpu'] is None and metadata.get('declared_gpu_model'):
        context['gpu'] = canonical('gpu', metadata['declared_gpu_model'])
        provenance['gpu'] = 'recipe_declaration_unverified'
    return dict(context=context, provenance=provenance,
                missing=[field for field in CONTEXT_FIELDS if context[field] is None],
                unverified=[field for field, source in provenance.items() if source.endswith('_unverified')])


def _observed(run, statistic):
    samples = defaultdict(list)
    for metric in run.get('metrics') or []:
        for key in ('tflops_gpu', 'seconds_per_step'):
            value = number(metric.get(key))
            if value:
                samples[key].append(value)
        time = number(metric.get('time_ms'))
        if time:
            samples['seconds_per_step'].append(time / 1000)
    for key in ('tflops_gpu','seconds_per_step'):
        if not samples[key] and number(run.get(key)):
            samples[key].append(float(run[key]))
    fn = statistics.median if statistic == 'median' else statistics.mean
    return {key: dict(value=fn(values), samples=len(values), minimum=min(values), maximum=max(values))
            for key, values in samples.items() if values}


def _is_measured(reference):
    raw = reference.get('raw_json') or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = {}
    for record in (reference, raw):
        if not isinstance(record, dict):
            continue
        kind = str(record.get('reference_kind') or record.get('kind') or record.get('basis') or
                   record.get('measurement_type') or 'measured').lower()
        if record.get('is_measured') is False:
            return False
        if any(token in kind for token in ('theoretical','peak_spec','datasheet')) or re.search(r'(?:^|[_ -])sol(?:$|[_ -])', kind):
            return False
        if record.get('theoretical') is True or record.get('is_theoretical') is True or record.get('is_sol') is True:
            return False
    return bool(reference.get('source_url') and (reference.get('ref_id') or reference.get('source_record_id')))


def _reference_identity(reference):
    source = reference.get('source_url') or ''
    if reference.get('job_id'):
        return (urlsplit(source).netloc, reference.get('cluster'), str(reference['job_id']))
    return (source, reference.get('source_record_id') or reference.get('ref_id'))


def _suggestions(result):
    missing = result['missing_context']
    suggestions = []
    if result['verdict'] == 'fail' and result['classification'] == 'failed':
        suggestions.append(dict(title='Resolve failed training before tuning', reason='A crash, invalid result, or failed parent job prevents performance acceptance.',
            command='Review the earliest failing rank and scheduler reason; reproduce the same job after fixing the observed failure.',
            expected='Complete iterations with finite losses and gradients and no failed worker.', rollback='Restore the original recipe after each isolated diagnostic change.'))
    if missing or result['tier'] != 'exact':
        suggestions.append(dict(title='Complete matching training context', reason='Conditional reference comparisons cannot establish a passing result. Missing: ' + ', '.join(missing),
            command='Retain the exact model variant, precision, release/container, global and micro batch, sequence length, GPU count and TP/PP/CP/VP/EP/ETP configuration with both runs.',
            expected='Source-backed references and uploaded run agree on every required context field.', rollback='Do not change production settings to make a partial reference match.'))
    if result['classification'] in ('bad','worst'):
        suggestions.append(dict(title='Repeat and isolate the measured training regression', reason='An exact reference shows a performance deficit beyond the configured threshold.',
            command='Repeat the same training configuration at least 3 times; compare medians and spread. Check GPU clocks, launch/NUMA placement and communication trace before changing one setting.',
            expected='Correctness-clean repeat medians fall within the selected deficit threshold.', rollback='Restore the recorded control container and configuration after each experiment.'))
    return suggestions


def compare_training(run, references, *, good_pct=5, bad_pct=10,
                     reference_statistic='median', observed_statistic='median'):
    """Compare training metrics with exact or explicitly conditional references.

    Percent expected is observed/reference for TFLOPS and reference/observed
    for seconds per step. A 5% deficit is therefore always 95% of expected.
    Reference repeats are collapsed per source job before aggregating jobs.
    """
    if not (math.isfinite(good_pct) and math.isfinite(bad_pct) and 0 <= good_pct <= bad_pct < 100):
        raise ValueError('Thresholds must satisfy 0 <= good_pct <= bad_pct < 100')
    if reference_statistic not in ('median','best') or observed_statistic not in ('median','mean'):
        raise ValueError('Unsupported aggregation policy')
    observed = _observed(run, observed_statistic)
    extracted = training_context(run)
    context = extracted['context']
    exact, contextual = [], []
    mismatches = []
    for ref in references:
        if not _is_measured(ref) or ref.get('status') in ('failed','incomplete','unknown') or not any(
                number(ref.get(metric)) for metric in ('tflops_gpu','seconds_per_step')):
            continue
        candidate = training_context(ref)
        conflicts = [field for field in CONTEXT_FIELDS if context[field] is not None and candidate['context'][field] is not None
                     and context[field] != candidate['context'][field]]
        if context['model'] is None or candidate['context']['model'] is None or context['gpu'] is None or candidate['context']['gpu'] is None:
            continue
        # Different hardware/model/scale/precision are not useful expectations.
        if any(field in conflicts for field in CORE_FIELDS):
            continue
        unresolved = sorted(set(extracted['missing'] + candidate['missing'] + extracted['unverified'] + candidate['unverified']))
        entry = (ref, candidate, unresolved, conflicts)
        if not unresolved and not conflicts:
            exact.append(entry)
        else:
            contextual.append(entry)
            mismatches.extend(conflicts)
    cohorts = defaultdict(list)
    for entry in contextual:
        key = json.dumps(entry[1]['context'], sort_keys=True)
        cohorts[key].append(entry)
    selected_cohort = None
    if not exact and cohorts:
        def cohort_rank(item):
            key, entries = item
            ref_context = entries[0][1]['context']
            conflicts = len(entries[0][3])
            matched = sum(context[field] is not None and context[field] == ref_context[field] for field in CONTEXT_FIELDS)
            jobs = len({_reference_identity(entry[0]) for entry in entries})
            return (conflicts, -matched, -jobs, key)
        selected_cohort, cohort_entries = min(cohorts.items(), key=cohort_rank)
        selected = cohort_entries
    else:
        selected = exact
    tier = 'exact' if exact else 'contextual' if contextual else 'none'
    missing = sorted(set(extracted['missing'] + extracted['unverified'] + [field for _, _, fields, _ in selected for field in fields]))
    policy = dict(good_pct=good_pct, bad_pct=bad_pct, reference_statistic=reference_statistic,
                  observed_statistic=observed_statistic, repeat_aggregation='median per source job, then selected statistic across jobs',
                  threshold_basis='deficit from 100% expected; <= good_pct good, <= bad_pct bad, otherwise worst',
                  acceptance='Explicit comparison policy; not an NVIDIA certification. Contextual ratios are unscored.',
                  required_context=list(CONTEXT_FIELDS))
    result = dict(verdict='insufficient_evidence', classification='unknown', tier=tier, context=context,
                  context_provenance=extracted['provenance'], missing_context=missing,
                  mismatches=sorted({field for _, _, _, fields in selected for field in fields}) if not exact else [], metrics=[], references=[], policy=policy,
                  provenance=dict(reference_jobs=0, reference_records=len(selected), observed=observed,
                      contextual_cohorts=len(cohorts), alternative_cohorts=max(0, len(cohorts) - 1) if not exact else len(cohorts),
                      selected_reference_context=json.loads(selected_cohort) if selected_cohort else context if exact else None,
                      cohort_selection='Fewest explicit conflicts, then most known matches, then most distinct source jobs, deterministic canonical context tie-break; incompatible contexts never pooled'), recommendations=[])
    job_counts = set()
    for ref, _, _, _ in selected:
        identity = _reference_identity(ref)
        job_counts.add(identity)
        result['references'].append({key: ref.get(key) for key in ('ref_id','source_url','source_record_id','row_action','job_id','date_utc','cluster','release','refreshed_at','raw_sha256','source_mapping')})
    result['provenance']['reference_jobs'] = len(job_counts)
    for metric, measured in observed.items():
        jobs = defaultdict(list)
        for ref, _, _, _ in selected:
            value = number(ref.get(metric))
            if value:
                identity = _reference_identity(ref)
                jobs[identity].append(value)
        if not jobs:
            continue
        medians = [statistics.median(values) for values in jobs.values()]
        higher = metric == 'tflops_gpu'
        expected = statistics.median(medians) if reference_statistic == 'median' else (max(medians) if higher else min(medians))
        percent = 100 * (measured['value'] / expected if higher else expected / measured['value'])
        deficit = 100 - percent
        rating = 'good' if deficit <= good_pct + 1e-9 else 'bad' if deficit <= bad_pct + 1e-9 else 'worst'
        result['metrics'].append(dict(metric=metric, observed=measured['value'], expected=expected,
            expected_pct=percent, deficit_pct=deficit, delta_pct=percent - 100, performance_band=rating,
            classification=rating if exact else 'conditional',
            scored=bool(exact), unit='TFLOP/s/GPU' if higher else 's/step', direction='higher_better' if higher else 'lower_better',
            reference_jobs=len(jobs), reference_records=sum(len(values) for values in jobs.values()),
            reference_min=min(medians), reference_max=max(medians), observed_samples=measured['samples']))
    result['candidate_band'] = (max((metric['performance_band'] for metric in result['metrics']),
        key=lambda band: ('good','bad','worst').index(band)) if result['metrics'] else None)
    metadata = run.get('metadata') or {}
    failed = run.get('status') == 'failed' or metadata.get('parent_job_failed') or metadata.get('exit_failure')
    if failed:
        result.update(verdict='fail', classification='failed')
        for metric in result['metrics']:
            metric['scored'] = False
            metric['classification'] = 'failed'
    elif result['metrics'] and exact and run.get('status') == 'passed':
        ratings = [metric['classification'] for metric in result['metrics']]
        rating = max(ratings, key=lambda r: ('good','bad','worst').index(r))
        result.update(classification=rating, verdict='pass' if rating == 'good' else 'fail')
    elif result['metrics']:
        result['classification'] = 'conditional'
        for metric in result['metrics']:
            metric.update(scored=False, classification='conditional')
    result['recommendations'] = _suggestions(result)
    return result
