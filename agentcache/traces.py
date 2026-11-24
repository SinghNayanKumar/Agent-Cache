import hashlib
import json
import random
from pathlib import Path

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read(path):
    obj = json.loads(Path(path).read_text())
    validate(obj)
    return obj

def validate(obj):
    if obj.get('schema') != 1:
        raise ValueError('Expected schema=1')
    meta = obj['meta']
    if meta['kind'] not in ('synthetic_branching', 'synthetic_linear', 'real'):
        raise ValueError('Unknown workload kind')
    if meta['split'] not in ('dev', 'heldout'):
        raise ValueError('Unknown split')
    if not obj['traces']:
        raise ValueError('Empty dataset')
    ids = set()
    max_context = requests = switches = 0
    for trace in obj['traces']:
        task = trace['task_id']
        if task in ids:
            raise ValueError('Duplicate task_id')
        ids.add(task)
        branches, closed = {}, set()
        seen_events = set()
        last_requested = None
        last_event_id = -1
        for e in trace['events']:
            eid, branch, op = e['id'], e['branch'], e['op']
            if type(eid) is not int or eid <= last_event_id:
                raise ValueError('Event IDs must be monotonically increasing nonnegative integers')
            last_event_id = eid
            if eid in seen_events or not set(e.get('depends_on', [])) <= seen_events:
                raise ValueError('Duplicate event or forward/missing dependency')
            seen_events.add(eid)
            tokens = e.get('input_ids', [])
            if any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError('Token IDs must be nonnegative integers')
            if op in ('open', 'fork', 'request') and not tokens:
                raise ValueError('Missing exact token prefix')
            if op == 'open':
                if branch in branches or branch in closed:
                    raise ValueError('Branch ID reused')
                branches[branch] = {'tokens': tokens, 'state': 'active'}
            elif op == 'fork':
                if branch in branches or branch in closed or e['parent'] not in branches:
                    raise ValueError('Invalid fork handle')
                if branches[e['parent']]['tokens'][:len(tokens)] != tokens:
                    raise ValueError('Fork prefix mismatch')
                branches[branch] = {'tokens': tokens, 'state': 'active'}
            elif op == 'request':
                if branches[branch]['state'] != 'active':
                    raise ValueError('Resume suspended branch before requesting')
                old = branches[branch]['tokens']
                if tokens[:len(old)] != old:
                    raise ValueError('Request is not an append; use a fork checkpoint')
                branches[branch]['tokens'] = tokens
                n = e['max_new_tokens']
                if type(n) is not int or n < 1:
                    raise ValueError('Invalid output length')
                max_context = max(max_context, len(tokens) + n)
                requests += 1
                switches += last_requested is not None and branch != last_requested
                last_requested = branch
            elif op in ('suspend', 'resume'):
                branches[branch]['state'] = 'suspended' if op == 'suspend' else 'active'
            elif op == 'abandon':
                if branch not in closed:
                    branches.pop(branch)
                    closed.add(branch)
            else:
                raise ValueError('Unknown event: '+op)
        if branches:
            raise ValueError('Trace must explicitly close all live handles')
    return {'tasks': len(ids), 'requests': requests, 'branch_switches': switches,
            'max_active_tokens': max_context, 'sha256': digest(obj)}

def generate(args):
    if min(args.tasks, args.prefix, args.payload, args.branches, args.output_tokens) < 1:
        raise ValueError('Sizes must be positive')
    if args.toy:
        encode = lambda text: list(text.encode())
        tokenizer = 'toy-byte-ids-NOT-FOR-GPU'
        revision = None
    else:
        if not args.tokenizer or not args.revision:
            raise ValueError('Use --tokenizer and --revision (immutable model SHA), or --toy')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
        encode = lambda text: tok.encode(text, add_special_tokens=False)
        tokenizer, revision = args.tokenizer, args.revision
    def fill(text, count):
        t = encode(text)
        if not t:
            raise ValueError('Empty tokenization')
        return (t * (count // len(t)+1))[:count]
    traces = []
    for i in range(args.tasks):
        # Distinct task prefixes. Artificial strings are only a serving workload.
        prefix = encode(f'Task {args.seed}-{i}. ') + fill('Inspect the repository and repair the failing test. ', args.prefix)
        events = []
        def add(op, branch, **kw):
            j = len(events)
            events.append({'id': j, 'depends_on': [j-1] if j else [],
                           'op': op, 'branch': branch, **kw})
        add('open', 'root', input_ids=prefix)
        add('request', 'root', input_ids=prefix, max_new_tokens=args.output_tokens)
        if args.linear:
            current = list(prefix)
            for j in range(args.branches):
                current += encode(f'Attempt {j}: test failed. ') + fill('Traceback and diagnostic output. ', args.payload)
                add('request', 'root', input_ids=current[:], max_new_tokens=args.output_tokens)
            add('abandon', 'root')
        else:
            add('suspend', 'root')
            paths = {}
            for j in range(args.branches):
                b = f'child{j}'
                paths[b] = prefix + encode(f'Alternative {j}. ') + fill('Tool output: test failure details. ', args.payload)
                add('fork', b, parent='root', input_ids=prefix)
                add('request', b, input_ids=paths[b], max_new_tokens=args.output_tokens)
                add('suspend', b)
            # Revisit before closure. No future-aware signals are used by the runner.
            order = list(paths)
            random.Random(args.seed+i).shuffle(order)
            for b in order:
                add('resume', b)
                add('request', b, input_ids=paths[b], max_new_tokens=args.output_tokens)
                add('abandon', b)
            add('resume', 'root')
            add('request', 'root', input_ids=prefix, max_new_tokens=args.output_tokens)
            add('abandon', 'root')
        traces.append({'task_id': f'synthetic-{args.seed}-{i:03d}', 'events': events})
    result = {'schema': 1, 'meta': {'kind': 'synthetic_linear' if args.linear else 'synthetic_branching',
              'split': args.split, 'seed': args.seed, 'tokenizer': tokenizer,
              'tokenizer_revision': revision, 'toy': args.toy,
              'notes': 'Raw-token serving microbenchmark; not a live agent or task-quality evaluation.'},
              'traces': traces}
    validate(result)
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2)+'\n')
    return validate(result)
