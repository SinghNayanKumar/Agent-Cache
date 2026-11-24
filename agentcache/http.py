"""Standard-library native HTTP adapter; records raw responses for version audits."""
import json
import time
import urllib.request

class Client:
    def __init__(self, base, timeout=600):
        self.base, self.timeout = base.rstrip('/'), timeout

    def request(self, path, data=None, method=None):
        req = urllib.request.Request(self.base+path,
            data=None if data is None else json.dumps(data).encode(),
            headers={'Content-Type': 'application/json'}, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            raw = response.read().decode()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = {'raw': raw}
        if isinstance(value, dict) and value.get('success') is False:
            raise RuntimeError(f'{path} returned success=false: {value}')
        return value

    def generate(self, tokens, output_tokens, extra=None):
        payload = {'input_ids': tokens, 'stream': True,
                   'sampling_params': {'temperature': 0, 'max_new_tokens': output_tokens,
                                       'ignore_eos': True}}
        payload.update(extra or {})
        req = urllib.request.Request(self.base+'/generate', data=json.dumps(payload).encode(),
                                     headers={'Content-Type':'application/json'})
        start = time.perf_counter_ns()
        first_ns = first_count = None
        final = None
        done = False
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            for item in sse(response):
                now = time.perf_counter_ns()
                if item == '[DONE]':
                    done = True
                    break
                value = json.loads(item)
                if 'error' in value:
                    raise RuntimeError(str(value['error']))
                final = value
                count = value.get('meta_info', {}).get('completion_tokens', 0)
                if count > 0 and first_ns is None:
                    first_ns, first_count = now, count
        end = time.perf_counter_ns()
        if final is None or first_ns is None or not done:
            raise RuntimeError('Incomplete SSE response or missing completion-token metadata')
        meta = final.get('meta_info', {})
        if meta.get('completion_tokens') != output_tokens:
            raise RuntimeError('Generation length differs from fixed replay length')
        if meta.get('prompt_tokens') != len(tokens):
            raise RuntimeError('Server prompt length differs from supplied token IDs')
        # If the first chunk includes >1 token, it is NOT exact first-token latency.
        return {'ttft_ms': (first_ns-start)/1e6 if first_count == 1 else None,
                'first_chunk_ms': (first_ns-start)/1e6, 'first_chunk_tokens': first_count,
                'latency_ms': (end-start)/1e6, 'prompt_tokens': len(tokens),
                'completion_tokens': meta['completion_tokens'],
                'cached_tokens': meta.get('cached_tokens'),
                'output_ids': final.get('output_ids'), 'text': final.get('text'),
                'raw_final': final}

def sse(lines):
    """Parse SSE across multiline data events and CRLF; ignore keepalive comments."""
    parts = []
    for raw in lines:
        line = raw.decode('utf-8').rstrip('\r\n') if isinstance(raw, bytes) else raw.rstrip('\r\n')
        if not line:
            if parts:
                yield '\n'.join(parts)
                parts = []
        elif line.startswith('data:'):
            parts.append(line[5:].lstrip(' '))
    if parts:
        yield '\n'.join(parts)
