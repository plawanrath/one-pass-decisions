#!/usr/bin/env bash
# One x86 rental session on AWS: launch the metal host, sync code, weights and pools, run the CPU chain, fetch results,
# terminate. Everything the host needs travels by rsync; no repository credentials leave this machine.
#
#   bash scripts/x86_session.sh <ami-id> [hours=7]
#
# Environment: AWS_REGION (default us-east-1), X86_KEY (default ~/.ssh/ss-bench.pem), X86_KEY_NAME (default ss-bench),
# X86_SG (security group id), X86_TAG_PROJECT (the tag value the IAM policy requires), X86_TYPE (default c7i.metal-24xl).
set -euo pipefail
cd "$(dirname "$0")/.."
AMI=${1:?ami id}; HOURS=${2:-7}
REGION=${AWS_REGION:-us-east-1}; KEY=${X86_KEY:-$HOME/.ssh/ss-bench.pem}; KEY_NAME=${X86_KEY_NAME:-ss-bench}
SG=${X86_SG:?security group id}; TAG=${X86_TAG_PROJECT:?project tag value}; TYPE=${X86_TYPE:-c7i.metal-24xl}
log() { echo "[x86] $(date '+%F %T') $*"; }
IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY_NAME" \
      --security-group-ids "$SG" --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$TAG},{Key=Name,Value=onepass-x86}]" \
      --instance-initiated-shutdown-behavior terminate \
      --query 'Instances[0].InstanceId' --output text)
log "launched $IID"
mkdir -p logs; echo "$(date -u +%FT%TZ) launch $IID $TYPE" >> logs/x86_session.log
terminate() { log "terminating $IID"; aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null || true
              echo "$(date -u +%FT%TZ) terminate $IID" >> logs/x86_session.log; }
trap terminate EXIT
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
log "public ip $IP"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=10 "ubuntu@$IP")
for i in $(seq 1 60); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
# Ubuntu 24.04 ships Python 3.12; the project pins 3.11, which comes from the deadsnakes archive
"${SSH[@]}" 'sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq software-properties-common build-essential rsync >/dev/null && sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1 && sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.11 python3.11-venv python3.11-dev >/dev/null; python3.11 --version; nproc; free -g | head -2'
RS=(rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" --exclude=.venv --exclude=.git --exclude=docs --exclude=.claude
    --exclude=.research-kit --exclude=logs --exclude=results --exclude=__pycache__ --exclude='*.onnx'
    --exclude=.env --exclude=CLAUDE.md --exclude='*.pem')  # never ship secrets or private notes to a rented host
"${RS[@]}" ./ "ubuntu@$IP:onepass/"
"${SSH[@]}" 'cd onepass && MODELS=0 bash scripts/env/setup.sh && .venv/bin/python scripts/env/pin_models.py --from-lock --no-download'
# the kill switch: the host shuts itself down (and terminates) when the budgeted hours run out, even if this shell dies
"${SSH[@]}" "sudo shutdown -h +$(( HOURS * 60 )) >/dev/null 2>&1 || true"
"${SSH[@]}" 'cd onepass && mkdir -p logs && bash scripts/host_chain.sh cpu 8 2>&1 | tee -a logs/host_chain.log' || true
mkdir -p results "logs/x86_$IID"
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" "ubuntu@$IP:onepass/results/" results/
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" "ubuntu@$IP:onepass/logs/" "logs/x86_$IID/"
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" "ubuntu@$IP:onepass/scripts/env/requirements.lock.linux-x86_64.txt" scripts/env/ || true
log "results fetched"
