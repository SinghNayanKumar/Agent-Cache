#!/usr/bin/env bash
# Run in AWS CloudShell, before launching anything.
set -euo pipefail
region="${1:-us-east-1}"
aws sts get-caller-identity
aws service-quotas get-service-quota --region "$region" --service-code ec2 \
 --quota-code L-DB2E81BA --query 'Quota.{Name:QuotaName,AvailableQuota:Value}'
aws ec2 describe-instance-type-offerings --region "$region" --location-type availability-zone \
 --filters Name=instance-type,Values=g6e.2xlarge --query 'InstanceTypeOfferings[].Location'
echo 'Offerings are not live capacity. Need 8 unused G/VT quota vCPUs for g6e.2xlarge.'
aws pricing get-products --region us-east-1 --service-code AmazonEC2 \
 --filters Type=TERM_MATCH,Field=instanceType,Value=g6e.2xlarge \
 Type=TERM_MATCH,Field=regionCode,Value="$region" Type=TERM_MATCH,Field=operatingSystem,Value=Linux \
 Type=TERM_MATCH,Field=tenancy,Value=Shared Type=TERM_MATCH,Field=preInstalledSw,Value=NA \
 Type=TERM_MATCH,Field=capacitystatus,Value=Used --output json > /tmp/agentcache-price.json
python3 - <<'PYCOST'
import json
for raw in json.load(open('/tmp/agentcache-price.json'))['PriceList']:
 p=json.loads(raw)
 for term in p['terms'].get('OnDemand',{}).values():
  for d in term['priceDimensions'].values():
   if d['unit']=='Hrs': print('USD/hour:',d['pricePerUnit']['USD'],d['description'])
PYCOST
