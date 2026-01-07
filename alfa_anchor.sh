#!/bin/bash

# Base parameters
SUB_POOLING="anchoral"
CLASSIFIER="MLP"
BUDGET=700
DATASET="cic-ids-17-18-70"
GENERATOR="CTGAN"
FILTER="synthetic"


# AL methods
AL_METHODS=("base")

# AL functions
AL_FUNCTIONS=("random" "entropy" "margin" "powermargin" "coreset" "galaxy" "clue" "density")

# Create logs directory
mkdir -p logs

# Loop over all combinations
for method in "${AL_METHODS[@]}"; do
  for func in "${AL_FUNCTIONS[@]}"; do
    echo "Running experiment with al_method=$method and al_function=$func"

    LOGFILE="logs/${method}_${func}.log"

    # Run experiment and capture success/failure
    if python main.py \
        --al_method "$method" \
        --al_function "$func" \
        --classifier "$CLASSIFIER" \
        --budget "$BUDGET" \
        --dataset "$DATASET" \
        --generator "$GENERATOR" \
        --pooling_method "$SUB_POOLING" \
        --minority 
    
    then 
      echo "✅ Success: al_method=$method, al_function=$func"
    else
      echo "❌ Failed: al_method=$method, al_function=$func (check $LOGFILE)"
      continue
    fi

  done
done
