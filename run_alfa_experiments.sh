#!/bin/bash
# Run ALFA experiments
# - S2 and S3 with seed 42 (default)
# - Original dataset with seeds 123 and 456

CLASSIFIER="MLP"
BUDGET=700
GENERATOR="CTGAN"
AL_FUNCTIONS=("random" "powermargin" "entropy")

mkdir -p logs

echo "Starting 24 ALFA experiments"
echo "=================================="

counter=0

# ===== GROUP 1: S2 and S3 with seed 42 =====
echo ""
echo "===== GROUP 1: S2 and S3 datasets (seed 42) ====="

DATASETS_S=("cic-ids-17-18-S2" "cic-ids-17-18-S3")
SEED=42

for dataset in "${DATASETS_S[@]}"; do
  for func in "${AL_FUNCTIONS[@]}"; do
    # ALFA-ALL (no minority)
    counter=$((counter + 1))
    echo ""
    echo ">>> Experiment ${counter}/24: ${dataset} ALFA-ALL ${func} seed${SEED}"
    echo ">>> Started at: $(date)"
    
    python main.py \
        --al_method "DA+ALFA" \
        --al_function "$func" \
        --classifier "$CLASSIFIER" \
        --budget "$BUDGET" \
        --dataset "$dataset" \
        --generator "$GENERATOR" \
        --random_state "$SEED" \
        > "logs/${dataset}_ALFA-ALL_${func}_seed${SEED}.log" 2>&1
    
    echo "✅ Completed: ${dataset} ALFA-ALL ${func}"
    
    # ALFA-M (with minority)
    counter=$((counter + 1))
    echo ""
    echo ">>> Experiment ${counter}/24: ${dataset} ALFA-M ${func} seed${SEED}"
    echo ">>> Started at: $(date)"
    
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
    
    echo "✅ Completed: ${dataset} ALFA-M ${func}"
  done
done

# ===== GROUP 2: Original dataset with seeds 123 and 456 =====
echo ""
echo "===== GROUP 2: Original dataset (seeds 123, 456) ====="

DATASET="cic-ids-17-18"
SEEDS=("123" "456")

for seed in "${SEEDS[@]}"; do
  for func in "${AL_FUNCTIONS[@]}"; do
    # ALFA-ALL (no minority)
    counter=$((counter + 1))
    echo ""
    echo ">>> Experiment ${counter}/24: ${DATASET} ALFA-ALL ${func} seed${seed}"
    echo ">>> Started at: $(date)"
    
    python main.py \
        --al_method "DA+ALFA" \
        --al_function "$func" \
        --classifier "$CLASSIFIER" \
        --budget "$BUDGET" \
        --dataset "$DATASET" \
        --generator "$GENERATOR" \
        --random_state "$seed" \
        > "logs/${DATASET}_ALFA-ALL_${func}_seed${seed}.log" 2>&1
    
    echo "✅ Completed: ${DATASET} ALFA-ALL ${func} seed${seed}"
    
    # ALFA-M (with minority)
    counter=$((counter + 1))
    echo ""
    echo ">>> Experiment ${counter}/24: ${DATASET} ALFA-M ${func} seed${seed}"
    echo ">>> Started at: $(date)"
    
    python main.py \
        --al_method "DA+ALFA" \
        --al_function "$func" \
        --classifier "$CLASSIFIER" \
        --budget "$BUDGET" \
        --dataset "$DATASET" \
        --generator "$GENERATOR" \
        --random_state "$seed" \
        --minority \
        > "logs/${DATASET}_ALFA-M_${func}_seed${seed}.log" 2>&1
    
    echo "✅ Completed: ${DATASET} ALFA-M ${func} seed${seed}"
  done
done

echo ""
echo "=================================="
echo "All 24 experiments completed!"
echo "Finished at: $(date)"
