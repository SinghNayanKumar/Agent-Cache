#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
image="$(cat provenance/image.txt)"
grep -Eq '^.+@sha256:[a-f0-9]{64}$' provenance/image.txt
args=(--gpus all)
if [[ "${CPU_ONLY:-0}" == 1 ]]; then args=(); fi
exec sudo docker run --rm --init "${args[@]}" --shm-size=8g \
  --user "$(id -u):$(id -g)" --env HOME=/work/cache \
  --env HF_HOME=/work/hf_cache --env XDG_CACHE_HOME=/work/cache \
  --env TRITON_CACHE_DIR=/work/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/work/cache/torch \
  --volume "$root:/work" --workdir /work --entrypoint /usr/bin/env "$image" "$@"
