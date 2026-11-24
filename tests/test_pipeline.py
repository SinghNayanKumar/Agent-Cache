import argparse
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from agentcache.registry import Registry
from agentcache.http import Client, sse
from agentcache.traces import generate, read, validate
from agentcache.runner import replay, verify_server
from agentcache.analysis import quantile

class Ownership(unittest.TestCase):
    def test_shared_prefix(self):
        r=Registry();r.open('r',[1,2]);r.open('a',[1,2],parent='r');r.open('b',[1,2],parent='r')
        r.update('a',[1,2,3]);r.update('b',[1,2,4]);r.abandon('r')
        self.assertEqual(r.abandon('a'),1);self.assertIn((1,2),r.nodes)
        r.assert_valid();self.assertEqual(r.abandon('b'),3);self.assertEqual(len(r.nodes),0)
        self.assertEqual(r.abandon('b'),0)
    def test_lock_and_copy(self):
        r=Registry();r.open('r',[1,2]);r.lock([1,2]);r.nodes[(1,2)].copies=1
        self.assertEqual(r.abandon('r'),0);r.lock([1,2],-1)
        self.assertEqual(r.collect(),0);r.nodes[(1,2)].copies=0
        self.assertEqual(r.collect(),2)
    def test_history_not_suffix(self):
        r=Registry();r.open('a',[1,7]);r.open('b',[2,7]);r.abandon('a')
        self.assertIn((2,7),r.nodes);self.assertNotIn((1,7),r.nodes)
    def test_no_rewind(self):
        r=Registry();r.open('r',[1,2,3])
        with self.assertRaises(ValueError):r.update('r',[1,4])

class FixtureTests(unittest.TestCase):
    def fixture(self,path,linear=False):
        return generate(argparse.Namespace(tasks=2,prefix=8,payload=10,branches=2,output_tokens=2,
            seed=42,toy=True,tokenizer=None,revision=None,split='dev',linear=linear,out=str(path)))
    def test_validation_linear_and_branch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';a=self.fixture(p);self.assertGreater(a['branch_switches'],0)
            b=self.fixture(p,True);self.assertEqual(b['branch_switches'],0)
    def test_bad_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';self.fixture(p);w=read(p)
            fork=next(e for e in w['traces'][0]['events'] if e['op']=='fork');fork['input_ids']=[999]
            with self.assertRaises(ValueError):validate(w)
    def test_dependency(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';self.fixture(p);w=read(p)
            w['traces'][0]['events'][0]['depends_on']=[999]
            with self.assertRaises(ValueError):validate(w)
    def test_quantile(self):
        self.assertEqual(quantile([1,2,3],.5),2);self.assertIsNone(quantile([],.95))
    def test_sse_multiline(self):
        self.assertEqual(list(sse([b': ping\n',b'data: {"a":\r\n',b'data: 1}\r\n',b'\r\n',b'data: [DONE]\n',b'\n'])),['{"a":\n1}','[DONE]'])
    def test_effective_budget_not_label(self):
        c={};cell={'policy':'B0','budget':100}
        good={'max_total_num_tokens':100,'disable_radix_cache':False,'enable_session_radix_cache':False}
        verify_server(good,c,cell)
        with self.assertRaises(RuntimeError):verify_server({**good,'max_total_num_tokens':200},c,cell)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        if self.path=='/generate':
            for n in range(1,body['sampling_params']['max_new_tokens']+1):
                result={'text':'x'*n,'meta_info':{'prompt_tokens':len(body['input_ids']),
                        'completion_tokens':n,'cached_tokens':0}}
                self.wfile.write(('data: '+json.dumps(result)+'\n\n').encode());self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
        else:self.wfile.write(b'{"success":true}')

class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.client=Client('http://127.0.0.1:'+str(self.server.server_port),5)
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
    def test_actual_stream_adapter(self):
        r=self.client.generate([1,2],3);self.assertEqual(r['completion_tokens'],3)
        self.assertIsNotNone(r['ttft_ms']);self.assertEqual(r['text'],'xxx')
    def test_replay_does_not_append_generated_text(self):
        trace={'task_id':'t','events':[{'id':0,'op':'open','branch':'r','input_ids':[1]},
               {'id':1,'op':'request','branch':'r','input_ids':[1], 'max_new_tokens':2},
               {'id':2,'op':'request','branch':'r','input_ids':[1,2], 'max_new_tokens':2},
               {'id':3,'op':'abandon','branch':'r'}]}
        log=io.StringIO();replay(self.client,trace,'B0','run',{},log)
        rows=[json.loads(x) for x in log.getvalue().splitlines()]
        self.assertEqual([x['prompt_tokens'] for x in rows if x['type']=='request'],[1,2])
        self.assertEqual(rows[-1]['status'],'ok')


class AnalysisTests(unittest.TestCase):
    def test_paired_bootstrap_and_summary(self):
        from agentcache.analysis import summarize, paired_speedups
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'runs';root.mkdir()
            (root/'config.json').write_text(json.dumps({'repeats':2}))
            cells=[]
            for policy,wall in [('B0',100),('A0',50)]:
                for repeat in range(2):
                    cell={'policy':policy,'budget':100,'repeat':repeat};cells.append(cell)
                    folder=root/f'{policy}_{repeat}';folder.mkdir()
                    (folder/'status.json').write_text(json.dumps({**cell,'status':'ok'}))
                    rows=[]
                    for task in ['a','b']:
                        base={**cell,'kind':'synthetic_branching','manifest_sha256':'testhash','task_id':task}
                        rows.append({**base,'type':'request','status':'ok','ttft_ms':1,'is_switch':True,
                                     'cached_tokens':3,'prompt_tokens':5})
                        rows.append({**base,'type':'trace','status':'ok','trace_wall_ms':wall})
                    (folder/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
            (root/'plan.json').write_text(json.dumps({'cells':cells}))
            result=summarize(argparse.Namespace(runs=str(root),out=str(Path(d)/'summary')))
            self.assertTrue(result['matrix_complete'])
            pair=paired_speedups(root)[0]
            self.assertAlmostEqual(pair['geometric_mean_wall_speedup'],2)
            self.assertEqual(pair['paired_tasks'],2)
            # An incomplete repeat must exclude the task pairing, not improve it.
            (root/'A0_1'/'status.json').write_text(json.dumps({'status':'error'}))
            self.assertEqual(paired_speedups(root),[])
    def test_compare_fails_on_missing_requests(self):
        from agentcache.analysis import compare
        with tempfile.TemporaryDirectory() as d:
            roots=[]
            for policy in ['NC','B0']:
                root=Path(d)/policy;folder=root/'cell';folder.mkdir(parents=True)
                (folder/'status.json').write_text('{"status":"ok"}')
                row={'type':'request','policy':policy,'manifest_sha256':'same','model_revision':'model',
                     'budget':100,'repeat':0,'task_id':'t','event_id':0 if policy=='NC' else 1,
                     'prompt_sha256':'p','text':'x'}
                (folder/'events.jsonl').write_text(json.dumps(row)+'\n');roots.append(root)
            with self.assertRaises(ValueError):
                compare(argparse.Namespace(reference=str(roots[0]),candidate=str(roots[1]),policy='B0',out=str(Path(d)/'check.json')))

class CoalescedHandler(Handler):
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        result={'text':'xx','meta_info':{'prompt_tokens':len(body['input_ids']),'completion_tokens':2}}
        self.send_response(200);self.end_headers()
        self.wfile.write(('data: '+json.dumps(result)+'\n\ndata: [DONE]\n\n').encode())

class TimingTests(unittest.TestCase):
    def test_coalesced_chunk_is_not_ttft(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),CoalescedHandler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            result=Client('http://127.0.0.1:'+str(server.server_port),5).generate([1],2)
            self.assertIsNone(result['ttft_ms'])
            self.assertIsNotNone(result['first_chunk_ms'])
        finally:
            server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
