import argparse
import json
from pathlib import Path
from . import traces, runner, analysis

def main():
    p=argparse.ArgumentParser(description='AgentCache midterm pipeline; GPU-only serial replay')
    s=p.add_subparsers(dest='command',required=True)
    g=s.add_parser('make-traces')
    for key,default in [('tasks',10),('prefix',2048),('payload',2048),('branches',4),('output-tokens',128),('seed',42)]:
        g.add_argument('--'+key,type=int,default=default)
    g.add_argument('--linear',action='store_true');g.add_argument('--toy',action='store_true')
    g.add_argument('--tokenizer');g.add_argument('--revision')
    g.add_argument('--split',choices=['dev','heldout'],default='dev')
    g.add_argument('--out',required=True)
    v=s.add_parser('validate');v.add_argument('--traces',required=True)
    for name in ('plan','run'):
        q=s.add_parser(name);q.add_argument('--config',required=True);q.add_argument('--traces',required=True)
        if name=='run':q.add_argument('--out',required=True)
    a=s.add_parser('summarize');a.add_argument('--runs',required=True);a.add_argument('--out',required=True)
    c=s.add_parser('compare');c.add_argument('--reference',required=True);c.add_argument('--candidate',required=True)
    c.add_argument('--policy',choices=['B0','B1','B1_branch','A0'],required=True);c.add_argument('--out',required=True)
    args=p.parse_args()
    if args.command=='make-traces': result=traces.generate(args)
    elif args.command=='validate': result=traces.validate(traces.read(args.traces))
    elif args.command=='plan': result=runner.plan(runner.load_config(args.config),traces.read(args.traces))
    elif args.command=='run': result=runner.run(args)
    elif args.command=='summarize': result=analysis.summarize(args)
    else: result=analysis.compare(args)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
