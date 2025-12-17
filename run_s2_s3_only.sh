#!/bin/bash
CLASSIFIER="MLP"
BUDGET=700
GENERATOR="CTGAN"
AL_FUNCTIONS=("random" "powermargin" "entropy")
DATASETS_S=("cic-ids-17-18-S2" "cic-ids-17-18-S3")
SEED=42

mkdir -p logs

echo "Re-running 12 S2 and S3 experiments"

counter=0

for dataset in "${DATASETS_S[@]}"; do
  for func in "${AL_FUNCTIONS[@]}"; do
    # ALFA-ALL
    counter=$((counter + 1))
    echo ">>> ${counter}/12: ${dataset} ALFA-ALL ${func}"
    
    python main.py \
        --al_method "DA+ALFA" \
        --al_function "$func" \
        --classifier "$CLASSIFIER" \
        --budget "$BUDGET" \
        --dataset "$dataset" \
        --generator "$GENERATOR" \
        --random_state "$SEED" \
        > "logs/${dataset}_ALFA-ALL_${func}_seed${SEED}.log" 2>&1
    
    echo "✅ Done"
    
    # ALFA-M
    counter=$((counter + 1))
    echo ">>> ${counter}/12: ${dataset} ALFA-M ${func}"
    
    python main.py \
        --al_method "DA+ALFA" \
        --al_function "$func" \
        --classifier "$CLASSIFIER" \
        --budget "$BUDGET" \
        --dataset "$dataset" \
        --generator "$GENERATOR" \
        --random_state "$SEED" \
        --minority \
        > "logs/${dataset}_ALFA-M_${func}_seed${SEED}.log" 2>&1
    
    echo "✅ Done"
  done
done

echo "All 12 experiments completed!"