"""Evidence-triggered investigation plans; recommendations never execute changes."""
from datetime import datetime, timezone
import json
import re

VERSION = '2026-10-17.3'
NCCL_DOC = 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html'
TEST_DOC = 'https://github.com/NVIDIA/nccl-tests'
PERF_DOC = TEST_DOC + '/blob/master/doc/PERFORMANCE.md'


def build(report):
    recipe = report.get('recipe', {})
    fabric = recipe.get('fabric')
    comparisons = report.get('comparisons', [])
    measured = [c for c in comparisons if c.get('run', {}).get('metrics')]
    nccl = [c for c in measured if c['run'].get('workload') == 'NCCL']
    training = [c for c in measured if c['run'].get('workload') != 'NCCL']
    groups = {g['id']: g for g in report.get('crashes', [])}
    findings = []

    def add(key, title, severity, trigger, why, commands, expected, rollback, applicability, sources=None, evidence=None, cautions=None):
        findings.append(dict(id=key, title=title, severity=severity, trigger=trigger,
            why=why, commands=commands, expected=expected, rollback=rollback,
            applicability=applicability, source_urls=sources or [NCCL_DOC, TEST_DOC],
            evidence=evidence or [], cautions=cautions or []))

    if groups:
        for key, group in groups.items():
            add('failure-' + key, group['title'], 'critical', f"{group['count']} signature matches in uploaded evidence",
                'A failed job cannot establish acceptable performance, even if some iterations or bandwidth points are fast.',
                [group.get('next_action', 'Inspect the first failing rank and correlate its timestamp with scheduler and fabric logs.'),
                 'nvidia-smi -q', 'rdma link show', 'ibstat'],
                'Same workload completes with zero data errors and no failed rank; record the first error before retrying.',
                'Restore the recorded control configuration; preserve original logs and the scheduler cancellation reason.',
                'Only allocations and devices identified in the failing evidence.', evidence=group.get('evidence', []))

    missing = [c for c in measured if c.get('performance', {}).get('tier') != 'exact']
    if missing or not measured:
        add('reference-context', 'Establish a comparable performance target', 'high',
            f'{len(missing)} measured artifacts lack a fully compatible scored reference' if measured else 'No measured training or nccl-tests result was found',
            'GPU scale alone does not make two results comparable. Model, precision, release and workload configuration matter; NCCL also depends on placement and test protocol.',
            ['Collect the LLMB release/container digest, exact model variant, precision, sequence length, global/micro batch and TP/PP/CP/VP/EP/ETP settings.',
             'For NCCL retain collective, GPU and node count, NVLink domain/rack/SU placement, NCCL and nccl-tests versions, message range, datatype, reduction, in/out placement, warmups, iterations, registration and all environment overrides.'],
            'A source-linked target with explicit matching fields, target value and percentage of expected performance.',
            'Keep the original evidence; enrich missing metadata without overwriting measured values.',
            'All performance acceptance decisions.', [PERF_DOC, 'https://llmb.nvidia.com/beacon-v2/runs'])

    if training:
        add('training-isolation', 'Separate compute, launch and communication limits', 'high' if any(c.get('performance', {}).get('classification') in ('bad', 'worst') for c in training) else 'normal',
            f'{len(training)} training artifacts contain measured results',
            'An Exemplar throughput gap can originate in compute, CPU launch/NUMA placement, or communication. A bandwidth-only result cannot identify which one.',
            ['Repeat the same training configuration at least three times; retain steady-state step time, TFLOP/s/GPU and spread with the same warmup policy.',
             'nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv',
             'nvidia-smi topo -m', 'numactl -H',
             'Use the installed OVA/LLMB collection framework to capture a short training profile on the same allocation. Compare compute time, collective time and CPU gaps before changing one setting.'],
            'A reproducible deficit associated with a measured bottleneck, followed by an A/B result at identical model, precision, release, sequence, batch and parallelism.',
            'Restore the original container and training configuration after each A/B experiment.',
            'Completed training runs with matching references; profiling commands depend on the installed collection framework.',
            ['https://llmb.nvidia.com/beacon-v2/runs', TEST_DOC])

    gaps=[]
    for comparison in measured:
        evaluation=comparison.get('performance',{})
        if evaluation.get('classification')=='failed':continue
        metrics=[m for m in evaluation.get('metrics',[]) if m.get('performance_band',m.get('classification')) in ('bad','worst') and m.get('expected_pct') is not None]
        if metrics:gaps.append((min(m['expected_pct'] for m in metrics),comparison,evaluation,metrics))
    for percent,comparison,evaluation,metrics in sorted(gaps,key=lambda x:x[0])[:5]:
        run=comparison['run'];model=str(evaluation.get('context',{}).get('model') or run.get('workload',''))
        moe=any(token in model.lower() for token in ('deepseek','qwen','nemotron','kimi'))
        add('gap-'+comparison['run_id'],f'{run.get("workload","Workload")}: {percent:.1f}% of selected reference performance','high',
            ('Conditional comparison: confirm the listed context differences before accepting this deficit.' if evaluation.get('tier')!='exact' else 'Matched performance target is below the configured acceptance band.'),
            f'{comparison["filename"]}: the lowest normalized metric is {100-percent:.1f}% below expected. This indicates where to investigate; it does not identify a root cause.',
            ['Retain the source reference job, model, release and parallelism listed in this assessment. Repeat the original saved LLMB/sbatch command with those same settings at least three times.',
             'srun --ntasks-per-node=1 nvidia-smi --query-gpu=name,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv',
             ('For this MoE workload, compare expert-dispatch/all-to-all time with the reference profile. Run alltoall NCCL controls on the same allocation before changing EP, QP or channel settings.' if moe else 'Compare GPU compute time, CPU launch gaps and all-reduce/all-gather/reduce-scatter time with the reference profile.'),
             'Use the fixed-size NCCL defaults control first, then one isolated candidate from the applicable fabric plan.'],
            'A repeatable source-matched result that closes the measured gap, with finite training and zero NCCL correctness errors.',
            'Restore the original container, parallelism and site-approved NCCL environment after each experiment.',
            'This uploaded run and the selected reference cohort only; a conditional ratio is not an acceptance failure.',
            sources=[r['source_url'] for r in evaluation.get('references',[])[:3] if r.get('source_url')]+[TEST_DOC],
            evidence=[dict(file=comparison['filename'],run_id=comparison['run_id'],metrics=metrics,missing_context=evaluation.get('missing_context',[]),mismatches=evaluation.get('mismatches',[]))])

    if not nccl:
        add('collect-nccl', 'Measure NCCL before attributing training gaps to the fabric', 'high',
            'No nccl-tests bandwidth table is present in this evidence',
            'NCCL debug logs describe connections and failures; they do not supply a collective bandwidth acceptance result.',
            ['Generate a defaults control script for the selected GPU/fabric recipe.',
             'Run all_reduce, all_gather, reduce_scatter and alltoall separately: one node/NVLink domain, two nodes, then target scale.',
             'For a 2 GiB target use -b 2147483648 -e 2147483648 -n 100 -w 50 -c 1, with at least three independent repeats. Record whether the reference uses GiB or decimal GB.'],
            'Source-backed correctness-clean NCCL tables at the exact reference size and scale; upload OVA outputs and run manifests together.',
            'Collect evidence using site-approved defaults; no fabric policy change is needed for the control.',
            'Training uploads without NCCL performance tables.', [TEST_DOC, PERF_DOC])

    add('repeatability', 'Compare identical points and report repeatability', 'normal',
        'Every performance acceptance or tuning experiment',
        'A best point, a median repeat, a sweep point and a fixed-size test answer different questions. An isolated best run is not production repeatability.',
        ['Keep collective, message bytes, datatype, placement and registration fixed.',
         'Run a fixed-size test and a sweep as separate experiments, with three or more repeats each.',
         'Report median, minimum, maximum and spread. Compare best-to-best only when the target explicitly uses best-run policy.'],
        'A stable correctness-clean median with explicit protocol and percentage of the selected reference.',
        'Restore the control configuration and keep both failed and successful repeats.',
        'NCCL and training; never pool different message sizes, scales or workload configurations.', [PERF_DOC, TEST_DOC])

    if fabric == 'spectrum-x':
        add('rail-health', 'Check every GPU-facing rail and rack/SU placement', 'high' if 'rdma' in groups else 'normal',
            'Spectrum-X recipe selected; inspect before all-to-all tuning',
            'A link can report up while carrying no useful traffic. Rail health and NVLink-domain/rack/SU placement can limit all-to-all independently of NCCL tuning.',
            ['nvidia-smi topo -m', 'rdma link show', 'ibdev2netdev', 'ibstat',
             'Capture GPU-facing port traffic, errors, drops, PFC pause and ECN counters before and after the same test on each rail.',
             'With the fabric operator, compare equal healthy whole-domain allocations within and across SUs; record every rack, SU, GPU count and rail.'],
            'All selected rails carry traffic and remain error-free; placement effects reproduce at identical scale and protocol.',
            'Restore the original allocation. Exclude unhealthy paths only as a labelled diagnostic experiment, not as a passing target-scale result.',
            'Spectrum-X Ethernet/RDMA allocations; verify the observed topology before selecting a follow-up test.',
            evidence=[], cautions=['Do not compare a single NVLink-domain result to a cross-rack Ethernet result.'])
        add('alltoall-ab', 'Tune all-to-all with isolated A/B experiments', 'normal',
            'Only after correctness, rail health and compatible all-to-all reference are established',
            'QP splitting, channel caps, buffer/chunk sizes, PXN and registration interact with topology and library version. Improvements in one collective do not imply improvements in another.',
            ['Control: NCCL defaults with the recorded site HCA, bootstrap and QoS configuration.',
             'A/B: NCCL_IB_QPS_PER_CONNECTION=4 NCCL_IB_SPLIT_DATA_ON_QPS=1, then restore the control.',
             'All-to-all only: compare NCCL_MAX_NCHANNELS=16 and 24 as separate experiments; do not carry the cap into all-reduce/all-gather/reduce-scatter.',
             'Compare NCCL_BUFFSIZE=8388608 and NCCL_P2P_NET_CHUNKSIZE=262144 as a labelled paired experiment, after checking the installed NCCL version and effective chunk/buffer limits.',
             'Where supported by the installed nccl-tests build, compare registration modes -R 0, -R 1 and -R 2 separately.',
             'Record the control PXN/CROSS_NIC settings and verify rail topology before any PXN experiment.'],
            'A repeatable median gain on the same all-to-all target, zero data errors, and no regression in the other three collectives.',
            'Unset each experimental override and rerun the saved defaults control; retain the versioned experiment manifest.',
            'Conditional test candidates; verify GPU, topology and software context against the selected source reference before testing.',
            cautions=['HCA names, interface names, traffic class, GID and service level must come from the site recipe.',
                      'NIC congestion control, PFC and ECN remain at the approved production policy. An operator-controlled experiment must record and verify restoration.',
                      'QP outcomes depend on test method and topology; compare each candidate with the saved control.'])
    else:
        add('ib-health', 'Verify InfiniBand path health before timeout tuning', 'normal',
            'InfiniBand recipe selected', 'Increasing timeout cannot repair an unhealthy link or inconsistent allocation.',
            ['ibstat', 'ibdiagnet', 'Inspect affected port counters, routing and site service level with the fabric operator.'],
            'Healthy paths, consistent allocation and reproducible correctness-clean bandwidth.',
            'Restore site timeouts and routing policy after any controlled diagnostic.', 'InfiniBand allocations only.')

    # Reviewed wiki guidance remains separate from measured acceptance targets.
    # Only site-independent diagnostics, or an observed matching crash signature,
    # are surfaced automatically. Site-specific recipes remain browsable evidence.
    from . import wiki_sources
    reviewed=wiki_sources.knowledge(limit=200)['items']
    def xid_signatures(text):
        return set(re.findall(r'\bxid\s*(?:\([^)]*\)\s*)?[:=]?\s*(\d+)\b',text,re.I))
    observed_xids=xid_signatures(json.dumps(report.get('crashes',[])))
    fabric_terms={'spectrum-x':('spectrum-x','spectrum x','roce','ethernet'),
                  'infiniband':('infiniband',)}.get(fabric,())
    selected=[]
    for finding in reviewed:
        if finding.get('kind') not in ('troubleshooting','configuration_check','crash_signature_guidance','source_conflict'):continue
        applicability=finding.get('applicability',{})
        if not isinstance(applicability,dict):continue
        # A different site's recipe is browsable evidence, never an automatic plan.
        if finding.get('site_specific') or any(applicability.get(k) for k in ('site','sites','cluster','clusters','customer','customers','hostname','hostnames')):continue
        if finding.get('requires_mutation') or finding.get('mutating_commands') or finding.get('automatic_remediation') is True:continue
        declared_gpu=recipe.get('gpu')
        supported_gpu=applicability.get('gpu',[])
        if isinstance(supported_gpu,str):supported_gpu=[supported_gpu]
        if declared_gpu and supported_gpu and not any(re.search(r'\b'+re.escape(str(declared_gpu))+r'\b',str(v),re.I) for v in supported_gpu):continue
        signature_text=' '.join(str(finding.get(k,'')) for k in ('title','summary'))+' '+str(applicability.get('signature',''))
        source_xids=xid_signatures(signature_text)
        if source_xids:
            if not source_xids.intersection(observed_xids):continue
        else:
            if finding.get('kind') in ('crash_signature_guidance','source_conflict'):continue
            supported_fabric=applicability.get('fabric',[])
            if isinstance(supported_fabric,str):supported_fabric=[supported_fabric]
            if not any(term in str(value).lower() for value in supported_fabric for term in fabric_terms):continue
            if not any(finding.get(k) for k in ('read_only_commands','diagnostic_branches','required_context')):continue
        selected.append(finding)
    for finding in selected[:3]:
        key=finding['id']
        evidence=finding.get('evidence',[])
        commands=finding.get('read_only_commands',[])
        if not commands:commands=[b['condition']+': '+b['investigate'] for b in finding.get('diagnostic_branches',[])]
        if not commands:commands=['Record the exact test scope and environment, then compare with the cited versioned source section before selecting a follow-up test.']
        add('wiki-'+key,finding['title'],'normal','Relevant recipe or observed crash signature; verify the documented applicability.',
            finding['summary'],commands,
            finding.get('expected','Time-correlated evidence that supports or rules out the documented diagnostic branch.'),
            'These are read-only diagnostic suggestions. Preserve the control configuration and site-approved fabric policy.',
            json.dumps(finding.get('applicability',{}),sort_keys=True),
            sources=list(dict.fromkeys(e['source_url'] for e in evidence)),
            evidence=[{k:e.get(k) for k in ('page_id','page_version','source_sha256','section_path','table_index','block_index')} for e in evidence],
            cautions=finding.get('cautions',[])+finding.get('do_not_generalize',[]))

    return dict(version=VERSION, generated_at=datetime.now(timezone.utc).isoformat(),
        evidence_state='measured' if measured else 'awaiting_measured_evidence',
        summary='Resolve failures first, establish an exact target, isolate the bottleneck, then accept only repeatable A/B improvements.',
        findings=findings, customer_evidence_note='Assessments use only the evidence attached to this report and source-linked references. Diagnostic examples do not establish measured customer results.')
