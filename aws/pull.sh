#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p provenance hf_cache traces runs analysis logs cache
[[ ! -e provenance/image.txt ]] || { echo 'Image already pinned; reuse it.'; exit 0; }
image="${1:-lmsysorg/sglang:latest}"
sudo docker pull "$image"
sudo docker image inspect "$image" > provenance/image-inspect.json
sudo docker image inspect --format '{{index .RepoDigests 0}}' "$image" > provenance/image.txt
grep -Eq '^.+@sha256:[a-f0-9]{64}$' provenance/image.txt
nvidia-smi > provenance/host-gpu.txt
uname -a > provenance/host.txt
lscpu >> provenance/host.txt
free -h >> provenance/host.txt
echo 'Image pinned. All subsequent commands reuse this digest.'
