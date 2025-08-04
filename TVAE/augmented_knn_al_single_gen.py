#!/usr/bin/env python3
# augmented_knn_al_single_gen.py

import os
import sys
# ensure parent directory is on PYTHONPATH for calcd_AL, config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from active_learning_framework.al import METHOD_DICT
from active_learning_framework.config import get_config
from ctgan import TVAE
from typing import List, Dict, Tuple
from sklearn.neighbors import NearestNeighbors


def knn_analysis_per_class(
        X_under: np.ndarray,
        y_under: np.ndarray,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        *,
        metric: str = "euclidean",
        feature_names: List[str] | None = None
    ) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Expand each class in X_under/y_under with its 1-NN from pool and return
    combined_df (all samples) and per_class dict (class-specific df)
    """
    cols = feature_names
    nn = NearestNeighbors(n_neighbors=1, metric=metric)
    nn.fit(X_pool)

    combined_rows = []
    per_class = {}
    for cls in np.unique(y_under):
        mask_cls = (y_under == cls)
        X_cls = X_under[mask_cls]
        distances, indices = nn.kneighbors(X_cls)
        neigh_idx = indices.flatten()

        X_neigh = X_pool[neigh_idx]
        # relabel neighbours to class
        y_neigh = np.full(neigh_idx.shape, cls)

        df_cls = pd.DataFrame(X_cls, columns=cols)
        df_cls['Label'] = cls
        df_neigh = pd.DataFrame(X_neigh, columns=cols)
        df_neigh['Label'] = y_neigh

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
    return (
        *_load(os.path.join(base, cfg.DATASET.TRAIN_FILE)),
        *_load(os.path.join(base, cfg.DATASET.VAL_FILE)),
        *_load(os.path.join(base, cfg.DATASET.TEST_FILE))
    )


def print_label_distribution(name, labels):
    unique, counts = np.unique(labels, return_counts=True)
    props = counts / len(labels)
    print(f"{name} distribution:")
    for cls, cnt, prop in zip(unique, counts, props):
        print(f"  Class {cls}: {cnt} ({prop:.2%})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Single-Gen KNN-augmented AL with TVAE")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method",  required=True)
    parser.add_argument("--budget",  type=int, default=50)
    parser.add_argument("--num_synthetic", type=int, default=1000)
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

    # 1) Baseline classifier
    params = dict(hidden_layer_sizes=tuple(args.layers), max_iter=args.max_iter, random_state=args.random_state)
    clf = MLPClassifier(**params)
    clf.fit(X_train, y_train)
    print_label_distribution("Original train set", y_train)

    # 2) Active Learning on validation
    al = METHOD_DICT[args.method]()
    idx_al = al.sample(X_val, args.budget, clf, X_train)
    if idx_al is not None and len(idx_al)>0:
        X_al = np.vstack([X_train, X_val[idx_al]])
        y_al = np.concatenate([y_train, y_val[idx_al]])
    else:
        X_al, y_al = X_train.copy(), y_train.copy()
    print_label_distribution("Post-AL train set", y_al)

    # 3) Underrepresented sampling
    uniq, cnts = np.unique(y_al, return_counts=True)
    props = cnts/len(y_al)
    under = uniq[props<0.05]
    mask = np.isin(y_al, under)
    X_under, y_under = X_al[mask], y_al[mask]
    print_label_distribution("Underrep samples", y_under)

    # prepare pool
    mask_pool = np.ones(len(y_val), bool)
    if idx_al is not None: mask_pool[idx_al]=False
    X_pool, y_pool = X_val[mask_pool], y_val[mask_pool]

    # 4) KNN combined_df
    fnames = cfg.DATASET.FEATURE_NAMES
    combined_df, _ = knn_analysis_per_class(X_under, y_under, X_pool, y_pool, feature_names=fnames)
    print_label_distribution("KNN combined samples", combined_df['Label'].values)

    # 5) Single TVAE on combined_df
    discrete = cfg.DATASET.DISCRETE_FEATURES
    df_tvae = combined_df.copy()
    tvae = TVAE(epochs=100,
                batch_size=60,
                )
    tvae.fit(df_tvae, discrete)

    # 6) Generate synthetic total
    df_syn = tvae.sample(samples=args.num_synthetic)
    print_label_distribution("Synthetic raw", df_syn['Label'].values)

    # 7) Optional filter
    X_syn = df_syn[fnames].values; y_syn = df_syn['Label'].values
    if args.filter_synthetic:
        X_in = X_syn
        y_pred = clf.predict(X_in)
        mask_keep = y_pred!=0
        X_syn, y_syn = X_syn[mask_keep], y_syn[mask_keep]
        print_label_distribution("Synthetic filtered", y_syn)

    # 8) Final augment & retrain new model
    X_final = np.vstack([X_al, X_syn]); y_final = np.concatenate([y_al, y_syn])
    print_label_distribution("Final train set", y_final)
    clf_final = MLPClassifier(**params)
    clf_final.fit(X_final, y_final)

    # 9) Evaluate
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
    report_mc       = classification_report(y_test, y_pred)

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
    fname = f"{script_name}_{args.dataset}_{args.method}_{args.budget}.txt"
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
