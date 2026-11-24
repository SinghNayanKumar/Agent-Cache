# AgentCache midterm experiments on AWS — L40S edition

Prepared 4 November 2026. This package adapts the PSC experiment harness to EC2 Docker. It does not provision or charge an AWS account automatically.

## 1. Decision: use AWS if quota and capacity are available

Use **one On-Demand g6e.2xlarge**: one NVIDIA L40S (48 GB marketed capacity), 8 vCPUs, and 64 GiB host RAM. This keeps the GPU model close to PSC and gives model loading more host RAM than g6e.xlarge (32 GiB). Do not choose G6 by mistake: G6 and G6e are different families. Avoid Spot for these first deadline-sensitive runs; interruption recovery is not implemented.

AWS can avoid the PSC queue, but credits do not guarantee a GPU quota or available capacity. Check this BEFORE spending time on installation. Leave the existing PSC submission intact until AWS is working. If you later cancel PSC baselines, remember that its afterok children will not become runnable.

### Credits, quota, and price

1. Open Billing → Credits. Open the university credit and verify that EC2 and the storage charges you plan to use are eligible, that it is active, and that the remaining credit is still $150. A screenshot showing the balance alone does not establish eligibility.
2. Select a region, initially `us-east-1` (N. Virginia).
3. In Service Quotas → Amazon EC2, inspect **Running On-Demand G and VT instances**. You need **8 available vCPUs**, after any other G/VT usage, for g6e.2xlarge. If the quota is zero, request 8. Approval is not guaranteed within your deadline. Credits do not bypass this.
4. Check the current Linux On-Demand hourly price in the launch console. No Reserved Instance or Savings Plan is needed. Confirm the chosen AMI has no additional software charge.
5. Set billing alerts, for example at $25, $50, and $100. Make sure the budget cost settings expose usage before credits; a net-cost budget can hide consumption until credits run out. Alerts are delayed notifications, not a spending cap.

Optional: upload `aws/check_account.sh` to CloudShell and run:

```bash
bash check_account.sh us-east-1
```

This retrieves the quota, supported zones, and live On-Demand price through AWS APIs. It needs read permissions for EC2, Service Quotas, and Pricing. A supported zone is not a capacity guarantee. If launch reports insufficient capacity, try another offered zone in the same region; verify no instance launched before trying again. A quota error requires a quota change, not another zone.

### Budget worksheet

Let **R** be the hourly instance rate shown by AWS. Keep $30 of the $150 aside for storage, public IPv4, transfers, and mistakes; the remaining compute allowance is approximately **120 / R hours**. This is a planning limit, not a guaranteed bill.

| Total instance-on time | Compute estimate |
|---|---:|
| 4 hours | 4 × R |
| 8 hours | 8 × R |
| 12 hours | 12 × R |

For illustration only, if the console says $2.50/hour, those are $10, $20, and $30; $120 covers 48 hours. That is not a verified instance quote. Setup, image download, model download, and idle time all count while EC2 is running. Experiment runtime is still unmeasured: the successful PSC pilot does not establish the duration of the full matrix.

## 2. Launch one EC2 instance

Use the EC2 console:

- Name: `agentcache-midterm`.
- AMI: a current **AWS-published Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04, x86_64**, with support for G6e. Verify the publisher and AMI details; do not select an unrelated Marketplace image. Use a recent driver because the SGLang container may require a recent CUDA driver.
- Instance: `g6e.2xlarge`, On-Demand, quantity 1.
- Key pair: create/download a `.pem` SSH key and keep it private.
- Networking: a public subnet with Internet access and a public IPv4 address. Allow inbound SSH TCP 22 from **My IP** only. No inbound 30000, 8000, or Jupyter port is needed. The model server stays inside Docker on loopback.
- Storage: **200 GiB gp3 EBS root volume**, encrypted. This holds Docker layers, model cache, code, and outputs. Check free space during setup. Do not use instance-store NVMe for the only copy of your results.
- Advanced details: set **shutdown behavior = Stop**, not Terminate. Require IMDSv2. No broad IAM role or AWS credentials inside the instance are needed for this workflow.

Wait for status checks, then use the SSH example shown under Connect. If capacity or quota blocks launch, AWS is not an immediate solution yet.

## 3. Upload and connect

Download and unzip this package on your Windows machine. In PowerShell, replace the key path and public IP:

```powershell
scp -i "C:\path\agentcache.pem" -r .\AgentCache_AWS_L40S ubuntu@PUBLIC_IP:~/
ssh -i "C:\path\agentcache.pem" ubuntu@PUBLIC_IP
```

On EC2:

```bash
cd ~/AgentCache_AWS_L40S
sudo apt-get update
sudo apt-get install -y tmux unzip
sudo systemctl start docker
nvidia-smi
sudo docker info
bash aws/arm_stop.sh 240
```

The last command schedules a shutdown in **four hours**, including setup time. Confirm the instance's shutdown behavior is Stop first. It can interrupt unfinished work. It is a safety timer, not a billing guarantee; verify the instance reaches Stopped in the console. Re-arm after every start. To replace the timer deliberately:

```bash
sudo shutdown -c
bash aws/arm_stop.sh 240
```

Files on EBS survive stopping. Running processes do not. EBS still incurs storage charges while stopped.

## 4. Install the engine once; pin the container

```bash
bash aws/pull.sh lmsysorg/sglang:latest
cat provenance/image.txt
bash aws/container.sh python3 aws/preflight.py
```

The first command resolves `latest` once and writes its immutable repository digest. Later commands use that digest, not a moving tag. There is no separate Python venv: Python, PyTorch, and SGLang run inside Docker. Unlike PSC's Apptainer workflow, Docker does not need to convert the image to SIF.

If you know the exact original PSC OCI repository digest, you may supply it instead. A SIF file checksum is NOT an OCI image digest. Using the same `latest` tag on another day does not prove identical software.

Preflight checks one L40S, CUDA access, BF16 matrix multiplication, and the server flags the harness uses. It does not prove native API compatibility; the pilot does that. If the container needs a newer driver, stop and choose a compatible current DLAMI/container pair; do not repeatedly change versions during an experiment matrix. Preserve the error and pinned digest.

Check available disk space with `df -h /` and Docker usage with `sudo docker system df`. Do not delete the pinned image or model cache while runs are active.

## 5. Download the model and make traces

For continuity, you can copy PSC's `provenance/model.json` here **before** preparation; preparation re-downloads that immutable revision and updates its local path. Without that file it resolves a model revision once and records it. The AWS and PSC revisions may then differ and must be reported separately.

```bash
CPU_ONLY=1 bash aws/container.sh python3 aws/prepare.py
CPU_ONLY=1 bash aws/container.sh bash aws/make_traces.sh
CPU_ONLY=1 bash aws/container.sh python3 -m agentcache plan \
  --config configs/aws_baselines.json --traces traces/aws_branching.json
```

These tasks do not need a GPU, although the EC2 instance is still billed. Preparation refuses to overwrite existing generated configs. If the model download is interrupted before configs are written, rerun preparation to reuse the download cache. Inspect partially generated files before retrying other setup steps.

Defaults match the PSC scaffold: Qwen2.5-Coder-7B-Instruct, BF16, serial requests, KV token caps 8192 and 16384, and three repetitions. No quantization change is introduced.

| Mode | Policies | KV caps | Repeats | Tasks per cell | Cells |
|---|---|---|---|---|---|
| pilot | B0 | 4096 | 1 | 2 | 1 |
| baselines | B0, B1 | 8192, 16384 | 3 | 10 branching | 12 |
| reference | NC, caching disabled | 8192, 16384 | 3 | 10 branching | 6 |
| linear | B0, B1 | 8192, 16384 | 3 | 10 linear | 12 |

B0 is ordinary radix LRU. B1 retains an agent session until completion. NC supplies an uncached comparison. The workload is synthetic; “heldout” is a generator split label, not evidence of real-agent evaluation. Each cell restarts the server and isolates tasks by cache reset.

## 6. Run and inspect the pilot

Use tmux so an SSH disconnect does not stop the work:

```bash
tmux new -s agentcache
bash aws/run.sh pilot
ls -td analysis/aws_pilot_*
cat analysis/aws_pilot_*/summary.md
cat logs/aws_pilot_*.elapsed_seconds
```

Look for a completed matrix, zero failed cells/requests, and valid timings. A successful pilot creates `provenance/pilot.ok`. If it fails, inspect its `logs/*.log`, `runs/*/*/server.log`, and `status.json` before spending more GPU time. Unsupported engine features are unavailable measurements, not improvements or zero-cost outcomes.

Detach with Ctrl+B, then D. Reattach with `tmux attach -t agentcache`. tmux survives SSH loss, not EC2 stop/reboot.

## 7. Run the full matrix

Check the remaining shutdown timer and your budget first. The following sequential run keeps one GPU instance throughout; there is no Slurm queue between modes:

```bash
bash aws/run_all.sh
```

Or run modes individually:

```bash
bash aws/run.sh baselines
bash aws/run.sh reference
bash aws/run.sh linear
```

Use either approach, not both. A file lock prevents simultaneous GPU runs through these scripts. Each invocation creates a fresh timestamped output directory; it does not overwrite a prior run or resume one. `run_all.sh` stops on failure. After a failure, do not restart all completed modes unnecessarily: rerun only the affected mode into a new directory after diagnosing it. Do not cherry-pick successful requests from failed runs.

The wrapper logs elapsed seconds including server startup and summarization preparation, and attempts a summary after ordinary run errors. A forced poweroff may prevent final cleanup; summarize those directories later and label them incomplete. There is no automatic checkpoint/resume implementation.

Monitor from a second SSH session:

```bash
cd ~/AgentCache_AWS_L40S
nvidia-smi
ls -t logs/*.log | head
# Substitute a specific log from the command above:
tail -f logs/ACTUAL_RUN_NAME.log
```

After baselines complete, use its measured elapsed time to budget the remaining modes. Do not extrapolate a full run from the pilot's per-request latency alone: model/server startup is repeated for every cell.

## 8. Collect results and stop billing for compute

After runs have finished, on EC2:

```bash
cd ~/AgentCache_AWS_L40S
tar -czf ../agentcache-aws-results.tgz runs analysis logs provenance configs traces
```

In local PowerShell:

```powershell
scp -i "C:\path\agentcache.pem" ubuntu@PUBLIC_IP:~/agentcache-aws-results.tgz .
```

Verify the archive downloaded and opens. Then stop the instance in EC2 and verify its state is **Stopped**. Closing SSH does not stop billing. Keep the EBS volume only as long as needed; it continues billing. After results are backed up and you no longer need the environment, terminate the instance and check for retained EBS volumes, snapshots, and allocated public addresses. Stopping/restarting may change the public IP; reconnect using the current one and re-arm the timer.

Keep AWS results under AWS labels. The same L40S does not make PSC and AWS latency directly interchangeable: record host RAM/CPU, driver, image digest, model revision, configuration, and trace hashes. Compare B0/B1 within the same AWS setup; do not combine the PSC pilot and AWS full runs into one statistical sample.

## 9. Scope and validation

This is the existing midterm measurement scaffold with AWS execution scripts. It does **not** implement physical AgentCache KV reclamation, HiCache, or A0 engine hooks. The CPU registry is a reference ownership model, not a GPU implementation. Client timings and native cached-token fields do not establish physical GPU memory savings. Real-agent traces and logit-level correctness checks remain separate work.

Local checks: Python unit tests and shell syntax checks can run without a GPU. EC2 quota, Docker/GPU compatibility, engine native APIs, and full runtime must be validated on your instance. Start with the pilot.

```bash
python3 -m unittest discover -s tests -v
```

## Sources checked for this adaptation

- GPU specifications: https://aws.amazon.com/ec2/instance-types/accelerated-computing/
- Regional EC2 quotas: https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html
- DLAMI release example (choose a current compatible release): https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpubaseoss-ul2204-2026-08-19.html
- Credit terms: https://aws.amazon.com/awscredits/
- Pricing: https://aws.amazon.com/ec2/pricing/on-demand/
- Shutdown behavior: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html
- Stop and storage billing: https://aws.amazon.com/ec2/faqs/
- Official SGLang Docker image instructions: https://github.com/sgl-project/sglang
