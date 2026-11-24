"""Inside container: download one immutable model snapshot and write AWS configs.

Existing model-pin/config files are not overwritten. Model download uses disk and
network but does not load GPU weights. Run on the bind-mounted /work directory.
"""
import json
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root=Path('/work')
if not (root/'agentcache').is_dir():
    raise SystemExit('Run this inside the provided container with /work mounted')
for name in ('aws_pilot','aws_baselines','aws_reference'):
    if (root/'configs'/f'{name}.json').exists():
        raise SystemExit('AWS config already exists; inspect it instead of overwriting')
model='Qwen/Qwen2.5-Coder-7B-Instruct'
provenance=root/'provenance';provenance.mkdir(exist_ok=True)
pin=provenance/'model.json'
if pin.exists():
    model_data=json.loads(pin.read_text())
    if model_data['repo_id'] != model: raise SystemExit('Model pin mismatch')
else:
    sha=HfApi().model_info(model).sha
    if not sha: raise SystemExit('No immutable model revision returned')
    model_data={'repo_id':model,'revision':sha}
    pin.write_text(json.dumps(model_data,indent=2))
path=snapshot_download(repo_id=model,revision=model_data['revision'])
model_data['snapshot_path']=path
pin.write_text(json.dumps(model_data,indent=2))
image=(provenance/'image.txt').read_text().strip()
base=json.loads((root/'configs/midterm.json').read_text())
# Historical key name: here provenance is an immutable image digest, not a git SHA.
base.update(model_path=path,model_revision=model_data['revision'],tokenizer_id=model,
            sglang_commit=image,engine_image_digest=image,
            engine_provenance_kind='docker_repo_digest',context_length=16384,
            budgets=[8192,16384],policies=['B0','B1'],repeats=3)
pilot=dict(base,context_length=8192,budgets=[4096],policies=['B0'],repeats=1)
reference=dict(base,policies=['NC'])
for name,value in [('aws_pilot',pilot),('aws_baselines',base),('aws_reference',reference)]:
    (root/'configs'/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
freeze=subprocess.run(['python3','-m','pip','freeze'],capture_output=True,text=True,check=True)
(provenance/'pip-freeze.txt').write_text(freeze.stdout)
print('Pinned model revision:',model_data['revision'])
print('Snapshot:',path)
print('Created configs/aws_pilot.json, configs/aws_baselines.json, configs/aws_reference.json')
print('Docker digest identifies the engine image; it is not a SGLang source commit.')
