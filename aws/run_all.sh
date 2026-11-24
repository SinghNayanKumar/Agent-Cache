#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f provenance/pilot.ok ]] || { echo 'Run and pass the pilot first'; exit 1; }
sha256sum -c provenance/pilot.ok
for mode in baselines reference linear; do
 bash aws/run.sh "$mode"
done
