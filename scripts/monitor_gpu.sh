#!/bin/bash
# Live monitoring of GPUs 0-4 utilization, with rolling averages.
# Use this during a training run to verify the data pipeline is keeping GPUs fed.
#
# Usage:
#   bash scripts/monitor_gpu.sh                 # default 60s, 60 samples
#   INTERVAL=10 SAMPLES=180 bash scripts/monitor_gpu.sh   # 10s sampling for 30 min
#
# Healthy targets:
#   sustained util  >= 85 %  -> compute-bound, optimal
#                  60-85 %  -> mild dataloader lag, acceptable
#                  <  60 %  -> data pipeline bottleneck, fix needed

set -euo pipefail

GPUS=${GPUS:-0,1,2,3,4}
INTERVAL=${INTERVAL:-60}
SAMPLES=${SAMPLES:-60}

echo "Monitoring GPUs $GPUS  (interval=${INTERVAL}s, samples=${SAMPLES})"
echo "================================================================"

# Per-GPU running sum and count for rolling mean.
declare -A SUM
declare -A COUNT
for g in $(echo "$GPUS" | tr ',' ' '); do
  SUM[$g]=0
  COUNT[$g]=0
done

printf "%-10s" "time"
for g in $(echo "$GPUS" | tr ',' ' '); do
  printf "    GPU%s     " "$g"
done
printf "    load    free\n"

for i in $(seq 1 "$SAMPLES"); do
  ts=$(date '+%H:%M:%S')
  printf "%-10s" "$ts"

  # nvidia-smi: query gpu, util, mem
  while IFS=, read -r idx util mem_used; do
    util=$(echo "$util" | tr -dc '0-9')
    if [ -n "${SUM[$idx]:-}" ]; then
      SUM[$idx]=$(( ${SUM[$idx]} + util ))
      COUNT[$idx]=$(( ${COUNT[$idx]} + 1 ))
      avg=$(( ${SUM[$idx]} / ${COUNT[$idx]} ))
      printf "  %3d%% (avg %3d%%)" "$util" "$avg"
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
                      --format=csv,noheader,nounits | tr -d ' ')

  load1=$(awk '{print $1}' /proc/loadavg)
  free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  printf "  %5s   %sG" "$load1" "$free_gb"
  echo
  sleep "$INTERVAL"
done

echo "================================================================"
echo "Final rolling averages:"
for g in $(echo "$GPUS" | tr ',' ' '); do
  if [ "${COUNT[$g]:-0}" -gt 0 ]; then
    avg=$(( ${SUM[$g]} / ${COUNT[$g]} ))
    bar=""
    for ((i=0; i<avg/5; i++)); do bar="${bar}#"; done
    printf "  GPU%s: %3d%%  %s\n" "$g" "$avg" "$bar"
  fi
done
