"""Validate the allocated GPU and the legacy server entry point used by the harness."""
import json, subprocess
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'No CUDA GPU visible inside container'
assert torch.cuda.device_count()==1, 'Expected exactly one allocated GPU'
name=torch.cuda.get_device_name(0)
assert 'L40S' in name, f'Expected L40S, got {name}'
assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'
x=torch.ones((256,256),device='cuda',dtype=torch.bfloat16)
y=x@x
torch.cuda.synchronize()
assert y[0,0].item()==256
info={'gpu':name,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
      'compute_capability':torch.cuda.get_device_capability(0),
      'vram_bytes':torch.cuda.get_device_properties(0).total_memory}
Path('provenance/gpu.json').write_text(json.dumps(info,indent=2))
r=subprocess.run(['python3','-m','sglang.launch_server','--help'],capture_output=True,text=True)
Path('provenance/server-help.txt').write_text(r.stdout+r.stderr)
if r.returncode: raise SystemExit('Server entry point failed; inspect provenance/server-help.txt')
for flag in ['--max-total-tokens','--enable-session-radix-cache','--disable-radix-cache',
             '--stream-interval','--radix-eviction-policy']:
    if flag not in r.stdout: raise SystemExit('Missing harness flag: '+flag)
print(json.dumps(info,indent=2))
print('GPU and CLI preflight passed. Native API compatibility still requires the pilot.')
