import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

def quantile(values, q):
    if not values:
        return None
    x = sorted(values); pos = (len(x)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return x[lo]+(x[hi]-x[lo])*(pos-lo)

def load_runs(root):
    root = Path(root)
    runs = []
    for file in sorted(root.glob('*/events.jsonl')):
        state = json.loads((file.parent/'status.json').read_text())
        rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        runs.append((state,rows))
    return runs

def write_csv(path, rows):
    path = Path(path)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys or ['no_data'])
        w.writeheader(); w.writerows(rows)

def summarize(args):
    root = Path(args.runs)
    groups = defaultdict(list)
    traces = defaultdict(list)
    failures = []
    for f in root.glob('*/status.json'):
        s = json.loads(f.read_text())
        if s['status'] != 'ok':
            failures.append({'cell':f.parent.name, **s})
    for state, rows in load_runs(root):
        for r in rows:
            key = (r['kind'], r['manifest_sha256'], r['policy'], r['budget'])
            if r['type']=='request':
                groups[key].append(r)
            if r['type']=='trace':
                traces[key].append(r)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    result = []
    for key, rows in sorted(groups.items()):
        kind, manifest, policy, budget = key
        ok = [r for r in rows if r['status']=='ok']
        ttft = [r['ttft_ms'] for r in ok if r.get('ttft_ms') is not None]
        switch = [r['ttft_ms'] for r in ok if r['is_switch'] and r.get('ttft_ms') is not None]
        ts = traces[key]
        all_ok = all(t['status']=='ok' for t in ts) and len(ok)==len(rows)
        wall = sum(t['trace_wall_ms'] for t in ts)/1000
        cached_known = bool(ok) and all(type(r.get('cached_tokens')) is int for r in ok)
        result.append({'kind':kind,'manifest_sha256':manifest,'policy':policy,'budget_tokens':budget,
            'observed_requests':len(rows), 'failed_requests':len(rows)-len(ok),
            'completed_traces':sum(t['status']=='ok' for t in ts),
            'failed_traces':sum(t['status']!='ok' for t in ts),
            'ttft_sample_n':len(ttft),'ttft_p50_ms':quantile(ttft,.5),'ttft_p95_ms':quantile(ttft,.95),
            'switch_ttft_sample_n':len(switch),'switch_ttft_p50_ms':quantile(switch,.5),
            'switch_ttft_p95_ms':quantile(switch,.95),
            'ttft_unavailable_n':len(ok)-len(ttft), 'measured_trace_wall_s':wall,
            'requests_per_s_if_complete':len(ok)/wall if all_ok and wall else None,
            'native_cached_fraction':sum(r['cached_tokens'] for r in ok)/sum(r['prompt_tokens'] for r in ok)
                 if cached_known else None})
    write_csv(out/'summary.csv',result)
    paired = paired_speedups(root)
    write_csv(out/'paired_speedups.csv',paired)
    plan = json.loads((root/'plan.json').read_text())
    statuses = list(root.glob('*/status.json'))
    complete = len(statuses)==len(plan['cells']) and not failures
    md = ['# Observed midterm measurements','',
          f'Full planned matrix completed: **{complete}**. Planned cells: {len(plan["cells"])}; observed cells: {len(statuses)}.',
          '', 'Missing values are unknown, not zero. TTFT requires the first stream event to contain exactly one generated token.',
          'Request quantiles pool measured requests/repeats. Paired confidence intervals resample whole tasks, not requests.',
          'Synthetic task IDs are synthetic samples and do not support real-agent generalization.', '',
          '| Policy | KV token cap | Requests | Failed | Switch p95 ms |',
          '|---|---:|---:|---:|---:|']
    for r in result:
        v=r['switch_ttft_p95_ms']; v='N/A' if v is None else f'{v:.3f}'
        md.append(f'| {r["policy"]} | {r["budget_tokens"]} | {r["observed_requests"]} | {r["failed_requests"]} | {v} |')
    md += ['', '## Failure records', '', '```json',json.dumps(failures,indent=2),'```','',
           'No physical recomputation, allocator occupancy, GPU memory, or logits are inferred from client timing.',
           'Use engine instrumentation for those fields; native cached_tokens is preserved without renaming it as physical GPU savings.']
    (out/'summary.md').write_text('\n'.join(md)+'\n')
    return {'out':str(out),'matrix_complete':complete,'groups':len(result),'failed_cells':len(failures)}

def paired_speedups(root):
    config = json.loads((root/'config.json').read_text())
    expected = set(range(config['repeats']))
    data = defaultdict(dict)
    for state, rows in load_runs(root):
        if state['status']!='ok':
            continue
        for r in rows:
            if r['type']=='trace' and r['status']=='ok':
                key = (r['kind'],r['manifest_sha256'],r['budget'],r['policy'],r['task_id'])
                data[key][r['repeat']]=r['trace_wall_ms']
    output=[]
    for kind,manifest,budget,policy in sorted(set(k[:4] for k in data)):
        if policy in ('B0','NC'):
            continue
        tasks=sorted(k[4] for k in data if k[:4]==(kind,manifest,budget,policy))
        ratios=[]
        for task in tasks:
            a=data.get((kind,manifest,budget,policy,task),{})
            b=data.get((kind,manifest,budget,'B0',task),{})
            if set(a)==expected and set(b)==expected:
                ratios.append(statistics.mean(b.values())/statistics.mean(a.values()))
        if not ratios:
            continue
        logs=[math.log(x) for x in ratios]
        rng=random.Random(42)
        draws=[math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(2000)]
        output.append({'kind':kind,'manifest_sha256':manifest,'budget_tokens':budget,'policy':policy,
                       'baseline':'B0','paired_tasks':len(ratios),'excluded_unpaired_tasks':len(tasks)-len(ratios),
                       'geometric_mean_wall_speedup':math.exp(statistics.mean(logs)),
                       'ci95_low':quantile(draws,.025) if len(ratios)>1 else None,
                       'ci95_high':quantile(draws,.975) if len(ratios)>1 else None})
    return output

def compare(args):
    def collect(root, policy):
        d={}
        for state,rows in load_runs(root):
            if state['status']!='ok':
                raise ValueError('Correctness comparison requires completed runs')
            for r in rows:
                if r['type']=='request' and r['policy']==policy:
                    k=(r['manifest_sha256'],r['model_revision'],r['budget'],r['repeat'],r['task_id'],r['event_id'])
                    if k in d:
                        raise ValueError('Duplicate comparison key')
                    d[k]=r
        if not d:
            raise ValueError('No requests for '+policy)
        return d
    ref=collect(args.reference,'NC'); candidate=collect(args.candidate,args.policy)
    if set(ref)!=set(candidate):
        raise ValueError('Comparison requires identical manifest, model, budgets, repeats, tasks and events')
    mismatches=[]; text_only=0
    for k,a in ref.items():
        b=candidate[k]
        if a['prompt_sha256']!=b['prompt_sha256']:
            raise ValueError('Prompt hash mismatch')
        if a.get('output_ids') is not None and b.get('output_ids') is not None:
            equal=a['output_ids']==b['output_ids']
        else:
            if a.get('text') is None or b.get('text') is None:
                raise ValueError('Neither comparable output IDs nor text present')
            equal=a['text']==b['text']; text_only+=1
        if not equal:
            mismatches.append({'key':k,'reference':a.get('text'),'candidate':b.get('text')})
    result={'requests':len(ref),'matching':len(ref)-len(mismatches),'text_only_comparisons':text_only,
            'mismatches':mismatches,'passed':not mismatches,
            'limitation':'Output smoke check only. Full-logit equivalence and allocator/copy invariants require engine-side tests.'}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result,indent=2))
    if mismatches:
        raise RuntimeError(f'Output mismatch: see {args.out}; do not claim correctness')
    return result
