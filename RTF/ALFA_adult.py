#!/usr/bin/env python3
# augmented_adapt_knn_al_multi_gen.py

import os
import sys
# ensure parent directory is on PYTHONPATH for calcd_AL and config

import argparse
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from  active_learning_framework.al import METHOD_DICT
from active_learning_framework.config import get_config
from realtabformer import REaLTabFormer
import transformers
import re
from active_learning_framework.mlp import TorchMLPClassifier
import torch

def compute_anchor_fraction(f_c, min_frac=0.05, alpha=1, steepness=1):
    raw = alpha*math.exp(-(steepness * (f_c)))
    return raw

def knn_analysis_per_class(
        X_under: np.ndarray,
        y_under: np.ndarray,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        X_full: np.ndarray,
        y_full: np.ndarray,
        feature_names: list[str]
    ) -> (pd.DataFrame, dict[int, pd.DataFrame]):
    """
    Expand underrepresented samples via 1-NN per class, return combined_df and per_class dict
    """
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(X_pool)
    combined_rows = []
    per_class = {}
    for cls in np.unique(y_under):
        mask = (y_under == cls)
        mask_full = (y_full == cls)
        X_cls = X_under[mask]
        X_cls_full = X_full[mask_full]
        _, idxs = nn.kneighbors(X_cls)
        X_neigh = X_pool[idxs.flatten()]
        # Relabel neighbors
        y_neigh = np.full(idxs.shape[0], cls, dtype=type(cls))

        df_cls = pd.DataFrame(X_cls_full, columns=feature_names)
        
        df_neigh = pd.DataFrame(X_neigh, columns=feature_names)
        

        df_combo = pd.concat([df_cls, df_neigh], ignore_index=True)
        per_class[cls] = df_combo
        combined_rows.append(df_combo)
    combined_df = pd.concat(combined_rows, ignore_index=True)
    return combined_df, per_class

def load_data(cfg):
    def _load(path):
        with np.load(path, allow_pickle=True) as data:
            return data['feature'], data['label']
    base = cfg.DATASET.DATA_DIR
    return (*_load(os.path.join(base, cfg.DATASET.TRAIN_FILE)),
            *_load(os.path.join(base, cfg.DATASET.VAL_FILE)),
            *_load(os.path.join(base, cfg.DATASET.TEST_FILE)))

def print_dist(name, y):
    uniq, cnts = np.unique(y, return_counts=True)
    print(f"{name} distribution:")
    for u, c in zip(uniq, cnts):
        print(f"  Class {u}: {c} ({c/len(y):.2%})")
    print()

def main():
    parser = argparse.ArgumentParser(description="Adapt+KNN AL with multi-generator TVAE augmentation")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--num_synthetic", type=float, default=1000)
    parser.add_argument("--filter_synthetic", action='store_true')
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--layers", nargs='+', type=int, default=[100])
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    cfg = get_config(args.dataset, 'mlp')
    cfg.EXPERIMENT.BUDGET = args.budget
    cfg.EXPERIMENT.AL_METHOD = args.method
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg)

    scaler = None
    if cfg.DATASET.STANDARDIZE:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

    # Baseline classifier
    params = dict(hidden_layer_sizes=tuple(args.layers), max_iter=args.max_iter, random_state=args.random_state)
    
    if args.method == "galaxy" or args.method == "clue":
        clf = TorchMLPClassifier(
           cfg,
           hidden_layer_sizes=(100,100),
           max_iter=100,
           batch_size=64,
           lr=1e-3,
           random_state=42,
           device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
         )
        clf.fit(X_train, y_train)
    else:
        clf = MLPClassifier(**params)
        clf.fit(X_train, y_train)
   
    print_dist("Original train", y_train)

    # Active Learning sampling
    al = METHOD_DICT[args.method]()
    sel = al.sample(X_val, args.budget, clf, X_train)
    if sel is not None and len(sel)>0:
        X_al = np.vstack([X_train, X_val[sel]]); y_al = np.hstack([y_train, y_val[sel]])
    else:
        X_al, y_al = X_train.copy(), y_train.copy()
    print_dist("Post-AL train", y_al)

    count_per_cat = {}
    uniq, cnts = np.unique(y_al, return_counts=True)
    for cat, cnts in zip(uniq, cnts):
        count_per_cat[cat] = cnts

    # Adaptation sampling (exclude majority)
    '''
    uniq_full, cnts_full = np.unique(y_al, return_counts=True)
    freqs_full = cnts_full/len(y_al)
    
    maj = uniq_full[np.argmax(freqs_full)]
    y_al_wo_majority_index = np.where(y_al!=maj)
    y_al_wo_majority = y_al[y_al_wo_majority_index]
    '''
    uniq, cnts = np.unique(y_al, return_counts=True)
    freqs = cnts/len(y_al)
    print("Unique", uniq, "frequencies", freqs)
    maj = uniq[np.argmax(freqs)]
    anchors_X, anchors_y = [], []
    full_X, full_y = [],[]
    freqs_min = freqs.min()
    rng = np.random.RandomState(args.random_state)
    
    for u, f in zip(uniq, freqs):
        if u==maj: continue
        idx = np.where(y_al==u)[0]
        frac = compute_anchor_fraction(f)
        print("bear",u, f, frac)
        n = int(frac * len(idx))
        chosen = rng.choice(idx, size=n, replace=True)
        anchors_X.append(X_al[chosen]); anchors_y.append(y_al[chosen])
        full_X.append(X_al[idx]); full_y.append(y_al[idx])


    X_under_anch = np.vstack(anchors_X); y_under_anch = np.hstack(anchors_y)
    X_under_full = np.vstack(full_X); y_under_full = np.hstack(full_y)

    print_dist("Adaptation anchors", y_under_anch)

    # KNN expansion
    mask_pool = np.ones(len(y_val), bool)
    if sel is not None: mask_pool[sel] = False
    X_pool, y_pool = X_val[mask_pool], y_val[mask_pool]
    fnames = cfg.DATASET.FEATURE_NAMES
    combined_df, per_class = knn_analysis_per_class(X_under = X_under_anch, y_under=  y_under_anch,X_pool =  X_pool, y_pool = y_pool, X_full = X_under_full, y_full = y_under_full, feature_names = fnames)

    # Multi-generator augmentation
    discrete = cfg.DATASET.DISCRETE_FEATURES
    syn_frames = []
    for cls, df_cls in per_class.items():
        rtf = REaLTabFormer(
              model_type="tabular",
              checkpoints_dir = "rtf_checkpoints_"+args.dataset,
              samples_save_dir = "rtf_samples_"+args.dataset,
              gradient_accumulation_steps=1,
              epochs=10,
              logging_steps=100,
              numeric_max_len=12
        )
        rtf.fit(df_cls)
        # sample proportional to class_df size
        n_samples = int((count_per_cat[cls]) * args.num_synthetic)
        df_syn = rtf.sample(n_samples=n_samples)
        df_syn['Label'] = cls
        syn_frames.append(df_syn)
    synthetic_df = pd.concat(syn_frames, ignore_index=True)
    print_dist("Synthetic raw (multi-gen)", synthetic_df['Label'].values)
    print("synthetic count", synthetic_df.shape[0] )

    # Optional filtering
    X_syn = synthetic_df[fnames].values; y_syn = synthetic_df['Label'].values
    if args.filter_synthetic:
        inp = X_syn
        y_pred = clf.predict(inp)
        mask_keep = y_pred != 0
        X_syn, y_syn = X_syn[mask_keep], y_syn[mask_keep]
        print_dist("Synthetic filtered", y_syn)

    # Final augment & retrain new model
    X_final = np.vstack([X_al, X_syn]); y_final = np.hstack([y_al, y_syn])
    # X_final = np.vstack([X_train, X_syn]); y_final = np.concatenate([y_train, y_syn])

    print_dist("Final train", y_final)
    clf_final = MLPClassifier(**params)
    #if scaler: X_final = scaler.transform(X_final)
    clf_final.fit(X_final, y_final)

    # Evaluate on test
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


    # Capture classification report
    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report (multiclass):")
    print(report)

    # 3. Save to results/<script>_<dataset>_<method>_<budget>.txt
    import os

    save_dir = os.path.join('results', args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    fname = f"{args.method}+RTF+ALFA.txt"
    path = os.path.join(save_dir, fname)

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
