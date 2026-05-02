#!/usr/bin/env bash
RES=$(ls -td /home/irteam/local-node-d/hbkimi/benchmark/results/*ecg_fm_hb 2>/dev/null | head -1)
[[ -z "$RES" ]] && { echo "no results dir yet"; exit 1; }
echo "=== $RES ==="
n_done=$(ls -d "$RES"/*/test_metrics.txt 2>/dev/null | wc -l)
n_running=$(ls -d "$RES"/*/val_metrics.txt 2>/dev/null | wc -l)
echo "completed: $n_done / 240   in-progress: $((n_running - n_done))"
echo
echo "=== 5 GPU 진행 ==="
for i in 0 1 2 3 4; do
  log=$(ls -t /home/irteam/local-node-d/hbkimi/ecg-fm/logs/benchmark/gpu${i}_*.log 2>/dev/null | head -1)
  [[ -z "$log" ]] && continue
  cur=$(tr '\r' '\n' < "$log" | grep -E "^\[run\]|Train [0-9]+:|Eval (val|test):|Best val" | tail -2 | tr '\n' '|')
  printf "GPU %d: %.180s\n" "$i" "$cur"
done
echo
echo "=== GPU util ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader | head -5
echo
echo "=== 최근 완료 5개 ==="
[[ -f "$RES/results_all.csv" ]] && tail -5 "$RES/results_all.csv" | column -t -s,
