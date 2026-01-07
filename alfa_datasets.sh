#!/bin/bash
set -euo pipefail

# Base parameters
CLASSIFIER="MLP"
GENERATOR="CTGAN"
FILTER="synthetic"

# Dataset "tuples": DATASET|BUDGET
DATASETS=(
  "cover_quantile_shift|1452"
  "cover_cluster_shift|1452"

  "shuttle_quantile_shift|145"
  "shuttle_cluster_shift|145"

  "poker_quantile_shift|1281"
  "poker_cluster_shift|1281"

  "diabetes_quantile_shift|634"
  "diabetes_cluster_shift|634"
)


# AL methods
AL_METHODS=("DA+ALFA" "base")

# Random states
RS=("42")

# AL functions
AL_FUNCTIONS=("random" "entropy" "powermargin")

# Create logs directory
mkdir -p logs

# =========================
# 1) Run BASE with anchormal pooling (your first block intent)
# =========================
for func in "${AL_FUNCTIONS[@]}"; do
  for rand in "${RS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      IFS="|" read -r data budget <<< "$ds"

      echo "Running experiment with al_method=base and al_function=$func dataset=$data budget=$budget rand=$rand"

      LOGFILE="logs/base_${func}_${data}_rs${rand}.log"

      if python main.py \
          --al_method "base" \
          --al_function "$func" \
          --classifier "$CLASSIFIER" \
          --budget "$budget" \
          --dataset "$data" \
          --generator "$GENERATOR" \
          --random_state "$rand" \
          --pooling_method "anchoral" \

      then
        echo "✅ Success: al_method=base, al_function=$func dataset=$data"
      else
        echo "❌ Failed: al_method=base, al_function=$func dataset=$data (check $LOGFILE)"
        continue
      fi
    done
  done
done

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
            --minority \

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
