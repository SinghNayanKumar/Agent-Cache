import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
import urllib.error
from .http import Client
from .traces import digest, read, validate

POLICIES = {'B0', 'B1', 'B1_branch', 'A0', 'NC'}

def load_config(path):
    c = json.loads(Path(path).read_text())
    if not set(c['policies']) <= POLICIES or not c['policies']:
        raise ValueError('Unsupported or empty policy list')
    if len(set(c['policies'])) != len(c['policies']):
        raise ValueError('Duplicate policies')
    if not c['budgets'] or min(c['budgets']) < 1 or c['repeats'] < 1:
        raise ValueError('Positive budgets and repetitions required')
    reserved = {'--model-path','--host','--port','--dtype','--context-length',
                '--max-total-tokens','--max-running-requests','--stream-interval',
                '--mem-fraction-static','--radix-eviction-policy','--disable-radix-cache',
                '--enable-session-radix-cache','--enable-hierarchical-cache'}
    for arg in c.get('extra_server_args', []) + c.get('agentcache_server_args', []):
        if arg.split('=')[0] in reserved:
            raise ValueError('Do not override controlled flag '+arg)
    return c

def plan(c, workload):
    cells = [{'policy': p, 'budget': b, 'repeat': r}
             for p in c['policies'] for b in c['budgets'] for r in range(c['repeats'])]
    random.Random(c['seed']).shuffle(cells)
    return {'audit': validate(workload), 'cells': cells,
            'note': 'Concurrency=1. Each cell restarts the server. Each task is isolated by cache reset.'}

def command(c, cell):
    args = [sys.executable, '-m', 'sglang.launch_server', '--model-path', c['model_path'],
        '--host', '127.0.0.1', '--port', str(c['port']), '--dtype', c['dtype'],
        '--context-length', str(c['context_length']), '--max-total-tokens', str(cell['budget']),
        '--max-running-requests', '1', '--stream-interval', '1',
        '--mem-fraction-static', str(c['mem_fraction_static']), '--radix-eviction-policy', 'lru']
    if cell['policy'] in ('B1', 'B1_branch'):
        args += ['--enable-session-radix-cache']
    if cell['policy'] == 'NC':
        args += ['--disable-radix-cache']
    args += c.get('extra_server_args', [])
    if cell['policy'] == 'A0':
        args += c.get('agentcache_server_args', [])
    return args

def values_for(obj, key):
    out = []
    if isinstance(obj, dict):
        if key in obj:
            out.append(obj[key])
        for value in obj.values():
            out.extend(values_for(value, key))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(values_for(value, key))
    return out

def verify_server(info, c, cell):
    # Fail closed if the pinned version changes its schema. Adapt and document it.
    actual = values_for(info, 'max_total_num_tokens')
    if not actual or any(type(x) is not int or x != cell['budget'] for x in actual):
        raise RuntimeError(f'Effective KV capacity must equal requested cap; got {actual}. Inspect server_info.')
    for field, expected in [('disable_radix_cache', cell['policy']=='NC'),
                            ('enable_session_radix_cache', cell['policy'] in ('B1','B1_branch'))]:
        vals = values_for(info, field)
        if not vals or any(x != expected for x in vals):
            raise RuntimeError(f'Cannot verify {field}={expected}; got {vals}')
    if any(values_for(info, 'enable_hierarchical_cache')):
        raise RuntimeError('This midterm harness is GPU-only; HiCache is not supported here')

def capability(client, c):
    # Project-specific endpoint. It is NOT provided by unmodified SGLang.
    x = client.request('/agentcache/capabilities')
    required = {'schema': 1, 'physical_reclamation': True, 'exact_context': True,
                'patch_commit': c['agentcache_patch_commit']}
    if any(x.get(k) != v for k, v in required.items()):
        raise RuntimeError('AgentCache engine capability check failed; no A0 measurement allowed')

def reset(client, policy):
    if policy == 'A0':
        client.request('/agentcache/reset', {}, 'POST')
    client.request('/flush_cache?timeout=30', {}, 'POST')

def lifecycle(client, policy, namespace, event):
    if policy == 'A0':
        body = dict(event, session_id=namespace)
        response = client.request('/agentcache/event', body, 'POST')
        if response.get('success') is not True or response.get('ack_event_id') != event['id']:
            raise RuntimeError('AgentCache event was not acknowledged')
        return response
    if policy == 'B1_branch' and event['op'] == 'abandon':
        return client.request('/close_session', {'session_id': namespace+'/'+event['branch']}, 'POST')
    return None

def replay(client, trace, policy, run_id, metadata, log):
    namespace = run_id+'/'+trace['task_id']
    last_branch = None
    start = time.perf_counter_ns()
    n = 0
    # Stream to disk on every event; failures cannot silently disappear.
    def write(row):
        log.write(json.dumps({**metadata, **row})+'\n'); log.flush()
    try:
        for e in trace['events']:
            if e['op'] != 'request':
                t = time.perf_counter_ns()
                response = lifecycle(client, policy, namespace, e)
                write({'type': 'event', 'task_id': trace['task_id'], 'event_id': e['id'],
                       'op': e['op'], 'branch': e['branch'],
                       'client_event_ms': (time.perf_counter_ns()-t)/1e6,
                       'response': response})
                continue
            extra = {}
            if policy == 'B1':
                extra['session_id'] = namespace
            elif policy == 'B1_branch':
                extra['session_id'] = namespace+'/'+e['branch']
            elif policy == 'A0':
                extra['agentcache'] = {'session_id': namespace, 'branch_id': e['branch']}
            row = {'type':'request', 'task_id':trace['task_id'], 'event_id':e['id'],
                   'branch':e['branch'], 'prompt_sha256':digest(e['input_ids']),
                   'is_switch': last_branch is not None and last_branch != e['branch'],
                   'status':'ok'}
            try:
                result = client.generate(e['input_ids'], e['max_new_tokens'], extra)
            except Exception as exc:
                write({**row, 'status':'error', 'error_class':type(exc).__name__, 'error':str(exc)})
                raise
            write({**row, **result})
            last_branch = e['branch']; n += 1
        if policy == 'B1':
            client.request('/close_session', {'session_id':namespace}, 'POST')
        write({'type':'trace', 'task_id':trace['task_id'], 'status':'ok', 'requests':n,
               'trace_wall_ms':(time.perf_counter_ns()-start)/1e6})
    except Exception as exc:
        write({'type':'trace', 'task_id':trace['task_id'], 'status':'error', 'requests':n,
               'trace_wall_ms':(time.perf_counter_ns()-start)/1e6, 'error':str(exc)})
        raise  # Restart/reinspect; a timeout can leave a request active on the server.

def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)

def run(args):
    c, w = load_config(args.config), read(args.traces)
    if w['meta'].get('toy'):
        raise ValueError('Toy tokenizer traces are CPU-only. Generate real token IDs first.')
    for key in ('model_path', 'model_revision', 'sglang_commit'):
        if 'EDIT_ME' in c[key]:
            raise ValueError('Set '+key+' in config before GPU execution')
    if not Path(c['model_path']).is_dir():
        raise ValueError('model_path must be an immutable local snapshot directory')
    if w['meta']['tokenizer_revision'] != c['model_revision'] or w['meta']['tokenizer'] != c['tokenizer_id']:
        raise ValueError('Trace tokenizer identity/revision does not match config')
    if 'A0' in c['policies'] and 'EDIT_ME' in c['agentcache_patch_commit']:
        raise ValueError('A0 requires an implemented and pinned engine patch')
    audit = validate(w)
    if audit['max_active_tokens'] > min(min(c['budgets']), c['context_length']):
        raise ValueError('A requested budget/context cannot fit the largest active request')
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)  # Never overwrite a prior run.
    specification = plan(c, w)
    (root/'plan.json').write_text(json.dumps(specification, indent=2))
    (root/'config.json').write_text(json.dumps(c, indent=2))
    (root/'trace_manifest.json').write_text(json.dumps(w, indent=2))
    help_run = subprocess.run([sys.executable,'-m','sglang.launch_server','--help'],
                              capture_output=True,text=True,timeout=120)
    if help_run.returncode:
        raise RuntimeError('Cannot import SGLang server: '+help_run.stderr[-2000:])
    (root/'server-help.txt').write_text(help_run.stdout)
    for cell in specification['cells']:
        for flag in command(c,cell):
            if flag.startswith('--') and flag not in help_run.stdout:
                raise RuntimeError('Pinned engine does not advertise flag '+flag)
    env = {'python':sys.version, 'platform':platform.platform(), 'config_sha256':digest(c),
           'trace_sha256':digest(w), 'packages':{}}
    for pkg in ('sglang','torch','transformers'):
        try:
            env['packages'][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            env['packages'][pkg] = None
    for label, cmd in [('nvidia_smi',['nvidia-smi']), ('pip_freeze',[sys.executable,'-m','pip','freeze'])]:
        try:
            r = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            env[label] = r.stdout or r.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            env[label] = str(exc)
    (root/'environment.json').write_text(json.dumps(env,indent=2))
    base = f"http://127.0.0.1:{c['port']}"
    client = Client(base,c['request_timeout_s'])
    # Do not attach to or terminate an unrelated server on this port.
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',c['port'])) == 0:
            raise RuntimeError('Port already occupied; choose a free port')
    for idx, cell in enumerate(specification['cells']):
        name = f"{idx:03d}_{cell['policy']}_{cell['budget']}_r{cell['repeat']}"
        folder = root/name; folder.mkdir()
        argv = command(c,cell)
        (folder/'command.json').write_text(json.dumps(argv,indent=2))
        state = {'status':'starting', **cell}
        (folder/'status.json').write_text(json.dumps(state))
        print('Starting', name, flush=True)
        with (folder/'server.log').open('w') as server_log:
            proc = subprocess.Popen(argv,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline = time.monotonic()+c['startup_timeout_s']
                health = Client(base,2)
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError('Server exited; inspect server.log')
                    try:
                        health.request('/health'); break
                    except Exception:
                        if time.monotonic()>deadline:
                            raise TimeoutError('Server startup timeout')
                        time.sleep(1)
                info = client.request('/server_info')
                (folder/'server_info.json').write_text(json.dumps(info,indent=2))
                verify_server(info,c,cell)
                if cell['policy']=='A0':
                    capability(client,c)
                first = next(e for e in w['traces'][0]['events'] if e['op']=='request')
                for _ in range(c['warmup_requests']):
                    client.generate(first['input_ids'],first['max_new_tokens'])
                reset(client,cell['policy'])
                # Same ordering seed for each policy/budget at the same repetition.
                traces = list(w['traces'])
                random.Random(c['seed']+cell['repeat']).shuffle(traces)
                metadata = {**cell, 'run_id':name, 'kind':w['meta']['kind'], 'split':w['meta']['split'],
                            'manifest_sha256':digest(w), 'model_revision':c['model_revision']}
                with (folder/'events.jsonl').open('w') as events:
                    for trace in traces:
                        reset(client,cell['policy'])
                        replay(client,trace,cell['policy'],name,metadata,events)
                state['status']='ok'
            except Exception as exc:
                state.update(status='error', error_class=type(exc).__name__,error=str(exc))
                raise
            finally:
                stop(proc)
                (folder/'status.json').write_text(json.dumps(state,indent=2))
        print('Completed', name, flush=True)
    return {'status':'ok','out':str(root)}
