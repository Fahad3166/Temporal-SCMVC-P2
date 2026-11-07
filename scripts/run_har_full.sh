#!/bin/bash
# HAR full controlled comparison (30+50 epochs, kmeans_seed=args.seed)
# Expected: 9 runs, ~60 min total

set -e
cd "$(dirname "$0")/../temporal_scmvc"

RESULTS_DIR="../p2_results/full_runs"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "HAR FULL RUN — MLP, TCN, LSTM × 3 seeds"
echo "Started: $(date)"
echo "=========================================="

for METHOD in mlp tcn lstm; do
  echo ""
  METHOD_UPPER=$(echo "$METHOD" | tr '[:lower:]' '[:upper:]')
  echo "===== HAR $METHOD_UPPER ====="
  for SEED in 0 1 2; do
    echo "--- seed $SEED  ($(date '+%H:%M:%S')) ---"
    python train_temporal_${METHOD}.py \
      --dataset har \
      --max_samples 5000 \
      --sample_strategy stratified \
      --seed $SEED \
      --pre_epochs 30 \
      --con_epochs 50 \
      --no_save \
      2>&1 | tee "$RESULTS_DIR/har_${METHOD}_seed${SEED}.log"
  done
done

echo ""
echo "=========================================="
echo "HAR DONE: $(date)"
echo "=========================================="
echo "Summary of best ACCs:"
grep -H "Best ACC" "$RESULTS_DIR"/har_*.log | sort