#!/usr/bin/env bash
# Run INSIDE the container after python3 aws/prepare.py.
set -euo pipefail
cd /work
model_revision="$(python3 -c 'import json;print(json.load(open("provenance/model.json"))["revision"])')"
for trace in traces/aws_pilot.json traces/aws_branching.json traces/aws_linear.json; do
  if [[ -e "$trace" ]]; then echo "Refusing to overwrite $trace" >&2; exit 1; fi
done
python3 -m agentcache make-traces --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
  --revision "$model_revision" --tasks 2 --prefix 1024 --payload 512 --branches 2 \
  --output-tokens 32 --split dev --out traces/aws_pilot.json
python3 -m agentcache make-traces --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
  --revision "$model_revision" --tasks 10 --seed 100 --prefix 2048 --payload 2048 \
  --branches 4 --output-tokens 128 --split heldout --out traces/aws_branching.json
python3 -m agentcache make-traces --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
  --revision "$model_revision" --tasks 10 --seed 100 --prefix 2048 --payload 512 \
  --branches 4 --output-tokens 128 --split heldout --linear --out traces/aws_linear.json
