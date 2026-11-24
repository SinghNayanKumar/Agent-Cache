#!/usr/bin/env bash
# A hard time cap, not a billing guarantee. Re-arm after every EC2 start.
set -euo pipefail
minutes="${1:-240}"
[[ "$minutes" =~ ^[0-9]+$ && "$minutes" -ge 10 && "$minutes" -le 1440 ]] || { echo 'Minutes must be 10..1440'; exit 2; }
sudo shutdown -h "+$minutes"
echo 'Scheduled host shutdown. Confirm EC2 shutdown behavior is STOP. EBS remains billed.'
