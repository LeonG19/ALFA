#!/usr/bin/env python3
# augmented_knn_al.py

import os
import sys
# ensure parent directory is on PYTHONPATH for calcd_AL, config, and knn module
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
# KNN expansion function inlined from active_learning_single_mult.py
from typing import List, Dict, Tuple
from sklearn.neighbors import NearestNeighbors


def knn_analysis_per_class(
        X_under: np.ndarray,
        y_under: np.ndarray,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        *,
        metric: str = "euclidean",
        self_match_safe: bool = False,
        feature_names: List[str] | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Expand each class in (X_under, y_under) with 1‑NN samples from
    (X_pool, y_pool), relabel those neighbours to the query class and
    return:

        1. combined_df  – one big DataFrame with *all* rows
        2. per_class_df – dict[class_value, DataFrame] per class
    """
    cols = feature_names
    # Build NN on pool
    nn = NearestNeighbors(n_neighbors=1, metric=metric)
    nn.fit(X_pool)

    combined_rows = []
    per_class = {}
    for cls in np.unique(y_under):
        mask_cls = (y_under == cls)
        X_cls = X_under[mask_cls]

        distances, indices = nn.kneighbors(X_cls)
        neigh_idx = indices.flatten()

        X_neigh_raw = X_pool[neigh_idx]
        y_neigh_raw = y_pool[neigh_idx]

        print(f"\nKNN neighbours for class {cls}:")
        unique_vals, cnts = np.unique(y_neigh_raw, return_counts=True)
        total = cnts.sum()
        for val, cnt in zip(unique_vals, cnts):
            print(f"  {val}: {cnt} ({cnt/total:.2%})")

        y_neigh = np.full_like(y_neigh_raw, fill_value=cls)

        df_cls = pd.DataFrame(X_cls, columns=cols)
        df_neigh = pd.DataFrame(X_neigh_raw, columns=cols)
        
        

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
    parser = argparse.ArgumentParser(description="Augmented AL with KNN‑guided CTGAN augmentation")
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
    cfg.EXPERIMENT.BUDGET    = args.budget
    cfg.EXPERIMENT.AL_METHOD = args.method

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg)

    scaler = None
    if cfg.DATASET.STANDARDIZE:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

    params = dict(hidden_layer_sizes=tuple(args.layers), max_iter=args.max_iter, random_state=args.random_state)
    clf = MLPClassifier(**params)
    clf.fit(X_train, y_train)
    print_label_distribution("Original train set", y_train)

    al = METHOD_DICT[args.method]()
    idx_al = al.sample(X_val, args.budget, clf, X_train)
    if idx_al is not None and len(idx_al) > 0:
        X_al = np.vstack([X_train, X_val[idx_al]])
        y_al = np.concatenate([y_train, y_val[idx_al]])
    else:
        X_al, y_al = X_train.copy(), y_train.copy()
    print_label_distribution("Post-AL train set", y_al)

    uniq, cnts = np.unique(y_al, return_counts=True)
    props = cnts / y_al.shape[0]
    under = uniq[props < 0.05]
    mask_under = np.isin(y_al, under)
    X_under = X_al[mask_under]; y_under = y_al[mask_under]
    print_label_distribution("Underrep. for KNN augmentation", y_under)

    mask_pool = np.ones(len(y_val), dtype=bool)
    if idx_al is not None:
        mask_pool[idx_al] = False
    X_pool = X_val[mask_pool]; y_pool = y_val[mask_pool]

    fnames = cfg.DATASET.FEATURE_NAMES
    combined_df, per_class_df = knn_analysis_per_class(
        X_under, y_under, X_pool, y_pool, feature_names=fnames
    )

    discrete = cfg.DATASET.DISCRETE_FEATURES
    syn_frames = []
    for cls, df_cls in per_class_df.items():
        rtf = REaLTabFormer(
              model_type="tabular",
              gradient_accumulation_steps=1,
              epochs=100,
              logging_steps=100,
              numeric_max_len=12
        )
        rtf.fit(df_cls)
        df_syn = rtf.sample(n_samples=df_cls.shape[0] * 3)
        df_syn['Label'] = cls
        syn_frames.append(df_syn)

    synthetic_df = pd.concat(syn_frames, ignore_index=True)
    print_label_distribution("Synthetic raw (per-class)", synthetic_df['Label'].values)

    X_syn = synthetic_df[fnames].values; y_syn = synthetic_df['Label'].values
    if args.filter_synthetic:
        X_in = X_syn
        y_pred_bin = clf.predict(X_in)
        mask_bin = y_pred_bin != 0
        X_syn = X_syn[mask_bin]; y_syn = y_syn[mask_bin]
        print_label_distribution("Synthetic filtered", y_syn)

    X_final = np.vstack([X_al, X_syn]); y_final = np.concatenate([y_al, y_syn])
    print_label_distribution("Final train set", y_final)
    clf_final = MLPClassifier(**params)
    clf_final.fit(X_final, y_final)

    #if scaler: X_test = scaler.transform(X_test)
    y_pred = clf_final.predict(X_test)
    print("\nTest set evaluation:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
