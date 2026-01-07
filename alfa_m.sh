#!/bin/bash

# Base parameters
CLASSIFIER="MLP"
BUDGET=700
DATASETS=("cover", "shuttle", "poker", "diabetes" ) #"cic-ids-17-18-70-S2" "cic-ids-17-18-70-S3"
GENERATOR="CTGAN"
FILTER="synthetic"

# AL methods
AL_METHODS=("DA+ALFA", "base")

RS=("42")

# AL functions
AL_FUNCTIONS=("random" "entropy"  "powermargin")

# Create logs directory
mkdir -p logs

# Loop over all combinations
for method in "${AL_METHODS[@]}"; do
  for func in "${AL_FUNCTIONS[@]}"; do
    for rand in "${RS[@]}"; do
      for data in "${DATASETS[@]}"; do 
        echo "Running experiment with al_method=$method and al_function=$func"

        LOGFILE="logs/${method}_${func}.log"

        # Run experiment and capture success/failure
        if python main.py \
            --al_method "$method" \
            --al_function "$func" \
            --classifier "$CLASSIFIER" \
            --budget "$BUDGET" \
            --dataset "$data" \
            --generator "$GENERATOR" \
            --random_state "$rand" \
            --minority 
        then 
          echo "✅ Success: al_method=$method, al_function=$func"
        else
          echo "❌ Failed: al_method=$method, al_function=$func (check $LOGFILE)"
          continue
        fi
        done
    done
  done
done
