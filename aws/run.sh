#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-pilot}"
case "$mode" in
 pilot) config=aws_pilot; traces=aws_pilot ;;
 baselines) config=aws_baselines; traces=aws_branching ;;
 reference) config=aws_reference; traces=aws_branching ;;
 linear) config=aws_baselines; traces=aws_linear ;;
 *) echo 'Choose pilot, baselines, reference, or linear' >&2; exit 2 ;;
esac
mkdir -p logs runs analysis
# Lock covers all modes. Never share the GPU with another experiment.
exec 9>logs/gpu.lock
flock -n 9 || { echo 'Another experiment holds the GPU lock'; exit 1; }
if [[ "$mode" == pilot ]]; then rm -f provenance/pilot.ok; fi
run="aws_${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
start=$SECONDS
set +e
bash aws/container.sh python3 -m agentcache run --config "configs/$config.json" \
 --traces "traces/$traces.json" --out "runs/$run" 2>&1 | tee "logs/$run.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$((SECONDS-start))" > "logs/$run.elapsed_seconds"
if [[ -f "runs/$run/plan.json" ]]; then
 CPU_ONLY=1 bash aws/container.sh python3 -m agentcache summarize --runs "runs/$run" --out "analysis/$run" || rc=1
fi
if [[ $rc == 0 && "$mode" == pilot ]]; then
 sha256sum provenance/image.txt configs/aws_pilot.json traces/aws_pilot.json > provenance/pilot.ok
fi
exit "$rc"
