#!/bin/bash
set -euo pipefail

# Base parameters
CLASSIFIER="MLP"
GENERATOR="CTGAN"
FILTER="synthetic"

# Dataset "tuples": DATASET|BUDGET
DATASETS=(
  "cover|1452"
  "shuttle|145"
  "poker|1281"
  "diabetes|634"
)

# AL methods
AL_METHODS=("DA+ALFA")

# Random states
RS=("42")

# AL functions
AL_FUNCTIONS=("random" "entropy" "powermargin")

# Create logs directory
mkdir -p logs


# =========================
# 2) Loop over all combinations (your second block)
#    Uses per-dataset budget from the tuple list (no $BUDGET bug)
# =========================
for method in "${AL_METHODS[@]}"; do
  for func in "${AL_FUNCTIONS[@]}"; do
    for rand in "${RS[@]}"; do
      for ds in "${DATASETS[@]}"; do
        IFS="|" read -r data budget <<< "$ds"

        echo "Running experiment with al_method=$method and al_function=$func dataset=$data budget=$budget rand=$rand"

        LOGFILE="logs/${method}_${func}_${data}_rs${rand}.log"

        if python main.py \
            --al_method "$method" \
            --al_function "$func" \
            --classifier "$CLASSIFIER" \
            --budget "$budget" \
            --dataset "$data" \
            --generator "$GENERATOR" \
            --random_state "$rand" \
            --filter_synthetic

        then
          echo "✅ Success: al_method=$method, al_function=$func dataset=$data"
        else
          echo "❌ Failed: al_method=$method, al_function=$func dataset=$data (check $LOGFILE)"
          continue
        fi
      done
    done
  done
done
