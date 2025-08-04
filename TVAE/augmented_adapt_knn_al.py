#!/usr/bin/env python3
# augmented_adapt_knn_al.py

import os
import sys
# ensure parent directory is on PYTHONPATH for calcd_AL and config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from active_learning_framework.al import METHOD_DICT
from active_learning_framework.config import get_config
from ctgan import TVAE
from active_learning_framework.mlp import TorchMLPClassifier


def compute_anchor_fraction(f_c, min_frac=0.05, alpha=1, steepness=100):
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
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(X_pool)
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
        df_cls["Label"] = cls
        df_neigh = pd.DataFrame(X_neigh, columns=feature_names)
        df_neigh["Label"] = cls

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
    print(f"{name}:")
    for u, c in zip(uniq, cnts): print(f"  Class {u}: {c} ({c/len(y):.2%})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Adapt+KNN Augmented AL with single TVAE")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--num_synthetic", type=float, default=1000)
    parser.add_argument("--filter_synthetic", action='store_true')
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--layers", nargs='+', type=int, default=[100])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1, help="Seed for reproducibility")
    parser.add_argument("--steepness", type=float, default=20, help="Seed for reproducibility")
    args = parser.parse_args()

    # load config and data
    cfg = get_config(args.dataset, 'mlp')
    cfg.EXPERIMENT.BUDGET = args.budget
    cfg.EXPERIMENT.AL_METHOD = args.method
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg)

    # standardize
    scaler = None
    if cfg.DATASET.STANDARDIZE:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # baseline model
    params = dict(hidden_layer_sizes=tuple(args.layers), max_iter=args.max_iter, random_state=args.random_state)
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
        clf = MLPClassifier(**params)
        clf.fit(X_train, y_train)
    print_dist("Original train", y_train)

    # active learning
    al = METHOD_DICT[args.method]()
    sel = al.sample(X_val, args.budget, clf, X_train)
    if sel is not None and len(sel)>0:
        X_al = np.vstack([X_train, X_val[sel]]); y_al = np.hstack([y_train, y_val[sel]])
    else:
        X_al, y_al = X_train.copy(), y_train.copy()
    print_dist("Post-AL train", y_al)

    # adaptation sampling: exclude majority
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
        frac = compute_anchor_fraction(f, alpha = args.alpha, steepness= args.steepness)
        print("bear",u, f, frac)
        n = int(frac * len(idx))
        chosen = rng.choice(idx, size=n, replace=True)
        anchors_X.append(X_al[chosen]); anchors_y.append(y_al[chosen])
        full_X.append(X_al[idx]); full_y.append(y_al[idx])


    X_under_anch = np.vstack(anchors_X); y_under_anch = np.hstack(anchors_y)
    X_under_full = np.vstack(full_X); y_under_full = np.hstack(full_y)

    print_dist("Adaptation anchors", y_under_anch)
    

    # knn combined
    feature_names = cfg.DATASET.FEATURE_NAMES
    # pool for knn: exclude AL sel from val
    mask_pool = np.ones(len(y_val),bool); mask_pool[sel]=False if sel is not None else None
    X_pool, y_pool = X_val[mask_pool], y_val[mask_pool]
    combined_df, per_class = knn_analysis_per_class(X_under = X_under_anch, y_under=  y_under_anch,X_pool =  X_pool, y_pool = y_pool, X_full = X_under_full, y_full = y_under_full, feature_names = feature_names)
    print_dist("KNN combined", combined_df['Label'].values)

    # single TVAE on combined
    discrete = cfg.DATASET.DISCRETE_FEATURES
    tvae = TVAE(epochs=50, batch_size=60, )
    tvae.fit(combined_df, discrete)
    n_samples = int(X_under_full.shape[0] * args.num_synthetic)

    df_syn = tvae.sample(samples=n_samples)
    print_dist("Synthetic raw", df_syn['Label'].values)
    print("synthetic count", df_syn.shape[0] )

    # optional filter
    X_syn = df_syn[feature_names].values; y_syn = df_syn['Label'].values
    if args.filter_synthetic:
        inp = X_syn
        y_pred = clf.predict(inp)
        mask = y_pred!=0
        X_syn, y_syn = X_syn[mask], y_syn[mask]
        print_dist("Synthetic filtered", y_syn)

    # final augment & retrain
    X_final = np.vstack([X_al, X_syn]); y_final = np.hstack([y_al, y_syn])
    print_dist("Final train", y_final)
    clf_final = MLPClassifier(**params)
    clf_final.fit(X_final, y_final)

    # evaluate
    #if scaler: X_test = scaler.transform(X_test)
    print("\nTest eval:")
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
