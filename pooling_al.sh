#!/bin/bash

# Base parameters
CLASSIFIER="MLP"
BUDGET=500
DATASET="cic-ids-17-18"
AL_METHOD="base"

# Pooling methods
POOLING_METHODS=("anchoral" "randsub" "seals")

# Active learning functions
AL_FUNCTIONS=("random" "entropy" "margin" "powermargin" "coreset" "galaxy" "clue" "density")

# Create logs directory
mkdir -p logs

# Loop over all combinations
for pool in "${POOLING_METHODS[@]}"; do
  for func in "${AL_FUNCTIONS[@]}"; do
    echo "Running experiment with pooling=$pool and al_function=$func"

    LOGFILE="logs/${pool}_${func}.log"

    # Run experiment and capture success/failure
    if python main.py \
        --al_method "$AL_METHOD" \
        --al_function "$func" \
        --pooling_method "$pool" \
        --classifier "$CLASSIFIER" \
        --budget "$BUDGET" \
        --dataset "$DATASET" >"$LOGFILE" 2>&1; then
      echo "✅ Success: pooling=$pool, al_function=$func"
    else
      echo "❌ Failed: pooling=$pool, al_function=$func (check $LOGFILE)"
      continue
    fi

  done
done
