#!/usr/bin/env python3
# augmented_AL.py

import os
import sys
# ensure parent directory is on PYTHONPATH for calcd_AL and config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from active_learning_framework.al import METHOD_DICT
from active_learning_framework.config import get_config
from realtabformer import REaLTabFormer
import re
import transformers
import math
from active_learning_framework.mlp import TorchMLPClassifier

def compute_anchor_fraction(f_c, min_frac=0.05, center=0.1, steepness=20):
    raw = 1 / (1 + math.exp(steepness * (f_c - center)))
    return max(min_frac, raw)

def print_dist(name, y):
    uniq, cnts = np.unique(y, return_counts=True)
    print(f"{name}:")
    for u, c in zip(uniq, cnts): print(f"  Class {u}: {c} ({c/len(y):.2%})")
    print()

def load_data(cfg):
    """
    Load train, validation, and test splits from NPZ files.
    """
    def _load(path):
        with np.load(path, allow_pickle=True) as data:
            X = data['feature']
            y = data['label']
        return X, y

    base = cfg.DATASET.DATA_DIR
    train_path = os.path.join(base, cfg.DATASET.TRAIN_FILE)
    val_path   = os.path.join(base, cfg.DATASET.VAL_FILE)
    test_path  = os.path.join(base, cfg.DATASET.TEST_FILE)

    X_train, y_train = _load(train_path)
    X_val,   y_val   = _load(val_path)
    X_test,  y_test  = _load(test_path)
    return X_train, y_train, X_val, y_val, X_test, y_test


def print_label_distribution(name, labels):
    unique, counts = np.unique(labels, return_counts=True)
    props = counts / labels.shape[0]
    print(f"{name} distribution:")
    for cls, cnt, prop in zip(unique, counts, props):
        print(f"  Class {cls}: {cnt} ({prop:.2%})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Augmented Active Learning with MLP + TVAE")
    parser.add_argument("--dataset", required=True, help="Dataset key for config")
    parser.add_argument("--method",  required=True, help="Active Learning method key in METHOD_DICT")
    parser.add_argument("--budget",  type=int, default=50, help="Number of AL samples to query")
    parser.add_argument("--num_synthetic", type=float, default=1000,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--filter_synthetic", action='store_true',
                        help="If set, filter synthetic to binary class-1 using initial classifier")
    parser.add_argument("--max_iter", type=int, default=300, help="MLP max iterations")
    parser.add_argument("--layers", nargs='+', type=int, default=[100], help="MLP hidden layers")
    parser.add_argument("--random_state", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    # Load config & data
    cfg = get_config(args.dataset, 'mlp')
    cfg.EXPERIMENT.BUDGET    = args.budget
    cfg.EXPERIMENT.AL_METHOD = args.method

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg)

    # Optional standardization
    scaler = None
    if cfg.DATASET.STANDARDIZE:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

    # 1) Baseline MLP on original train
    base_params = dict(
        hidden_layer_sizes=tuple(args.layers),
        max_iter=args.max_iter,
        random_state=args.random_state
    )

    if args.method == "galaxy" or args.method == "clue":
 
        clf = TorchMLPClassifier(
           cfg,
           hidden_layer_sizes=(100,100),
           max_iter=100,
           batch_size=64,
           lr=1e-3,
           random_state=42
         )
        clf.fit(X_train, y_train)
    else:
        clf = MLPClassifier(**base_params)
        clf.fit(X_train, y_train)

    print_label_distribution("Original train set", y_train)

    # 2) Active Learning sampling on validation
    al = METHOD_DICT[args.method]()
    idx_al = al.sample(X_val, args.budget, clf, X_train)
    if idx_al is not None and len(idx_al) > 0:
        X_al = np.vstack([X_train, X_val[idx_al]])
        y_al = np.concatenate([y_train, y_val[idx_al]])
    else:
        X_al, y_al = X_train.copy(), y_train.copy()
    print_label_distribution("Post-AL train set", y_al)

    # 3) Underrepresented sampling (<5% across all classes)
    uniq, cnts = np.unique(y_al, return_counts=True)
    freqs = cnts / len(y_al)
    maj = uniq[np.argmax(freqs)]
    X_min, y_min = [], []
    X_min_full, y_min_full = [],[]
    for u, f in zip(uniq, freqs):
        if u==maj: continue
        idx = np.where(y_al==u)[0]
        frac = compute_anchor_fraction(f)
        n = max(1, int(frac*len(idx)))
        choice = np.random.RandomState(args.random_state).choice(idx, size=n, replace=False)
        X_min.append(X_al[choice]); y_min.append(y_al[choice])
        X_min_full.append(X_al[idx]); y_min_full.append(y_al[idx])
    X_min = np.vstack(X_min)
    X_under = np.vstack(X_min_full); y_under = np.hstack(y_min_full)
    print_dist("Adaptation anchors", y_under)

    # 4) Create DF for TVAE
    fnames = cfg.DATASET.FEATURE_NAMES
    discrete = cfg.DATASET.DISCRETE_FEATURES
    df_tvae = pd.DataFrame(X_under, columns=fnames)
    df_tvae['Label'] = y_under

    # 5) Train TVAE
    rtf = REaLTabFormer(
              model_type="tabular",
              gradient_accumulation_steps=1,
              epochs=10,
              logging_steps=100,
              numeric_max_len=12
    )
    rtf.fit(df_tvae)
    n_samples = int(X_under.shape[0] * args.num_synthetic)
    # 6) Generate synthetic
    df_syn = rtf.sample(n_samples=n_samples)
    print_label_distribution("Synthetic raw", df_syn['Label'].values)
    print("synthetic count", df_syn.shape[0])
    # 7) Optional filter: binary prediction
    X_syn = df_syn[fnames].values; y_syn = df_syn['Label'].values
    if args.filter_synthetic:
        X_in = scaler.transform(X_syn) if scaler else X_syn
        y_pred_bin = clf.predict(X_in)
        mask_bin = np.where(y_pred_bin == 0, False, True)
        X_syn = X_syn[mask_bin]; y_syn = y_syn[mask_bin]
        print_label_distribution("Synthetic filtered", y_syn)

    # 8) Final augment & retrain with NEW model
    X_final = np.vstack([X_al, X_syn]); y_final = np.concatenate([y_al, y_syn])
    print_label_distribution("Final train set", y_final)
    clf_final = MLPClassifier(**base_params)
    clf_final.fit(X_final, y_final)

    # 9) Metrics on test
    #if scaler: X_test = scaler.transform(X_test)
    y_pred = clf_final.predict(X_test)
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    # 1. Compute multiclass metrics
    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro    = recall_score(y_test,    y_pred, average='macro')
    f1_macro        = f1_score(y_test,        y_pred, average='macro')
    precision_micro = precision_score(y_test,  y_pred, average='micro')
    recall_micro    = recall_score(y_test,     y_pred, average='micro')
    f1_micro        = f1_score(y_test,         y_pred, average='micro')
    report_mc       = classification_report(y_test, y_pred, digits=4)

    # 2. Compute binary metrics (0 vs rest)
    y_true_bin = np.where(y_test == 0, 0, 1)
    y_pred_bin = np.where(y_pred == 0, 0, 1)
    precision_bin = precision_score(y_true_bin, y_pred_bin)
    recall_bin    = recall_score(   y_true_bin, y_pred_bin)
    f1_bin        = f1_score(       y_true_bin, y_pred_bin)

    # 3. Save to results/<script>_<dataset>_<method>_<budget>.txt
    import os

    os.makedirs('results', exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    fname = f"rft_{script_name}_{args.dataset}_{args.method}_{args.budget}.txt"
    path = os.path.join('results', fname)

    with open(path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Method:  {args.method}\n")
        f.write(f"Budget:  {args.budget}\n\n")

        f.write("Multiclass Metrics:\n")
        f.write(f"Accuracy      : {acc:.4f}\n")
        f.write(f"Precision (M) : {precision_macro:.4f}\n")
        f.write(f"Recall    (M) : {recall_macro:.4f}\n")
        f.write(f"F1-score  (M) : {f1_macro:.4f}\n")
        f.write(f"Precision (m) : {precision_micro:.4f}\n")
        f.write(f"Recall    (m) : {recall_micro:.4f}\n")
        f.write(f"F1-score  (m) : {f1_micro:.4f}\n\n")

        f.write("Binary Metrics (0 vs rest):\n")
        f.write(f"Precision: {precision_bin:.4f}\n")
        f.write(f"Recall   : {recall_bin:.4f}\n")
        f.write(f"F1-score : {f1_bin:.4f}\n\n")

        f.write("Classification Report (multiclass):\n")
        f.write(report_mc)
if __name__ == '__main__':
    main()
