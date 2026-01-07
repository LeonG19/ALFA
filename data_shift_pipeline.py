#!/usr/bin/env python3
"""
split_and_save_npz.py

Creates train/val/test NPZ splits for AL experiments.

DEFAULT (no shift):
  - train: 50% random
  - val:   25% random
  - test:  25% random

CIC-IDS-LIKE SHIFT (domain shift):
  - train is sampled mostly from Domain A (labeled distribution)
  - val + test are sampled mostly from Domain B (unlabeled/test distribution)
  - then Domain B is split into val and test

Shift modes:
  --shift_mode none     : random split (no shift)
  --shift_mode cluster  : subpopulation drift via KMeans clusters
  --shift_mode quantile : covariate drift via feature quantiles

Also computes drift metrics between train and test:
  - PSI (per-feature + aggregate)
  - JS divergence (per-feature + aggregate)
  - MMD-RBF (global, multivariate)

Outputs:
  data/<output_dir>/train.npz, val.npz, test.npz
  data/<output_dir>/label2id.json
  data/<output_dir>/drift_report.json
  data/<output_dir>/drift_features.csv
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from yacs.config import CfgNode as CN
from config import get_dataset_config
from ucimlrepo import fetch_ucirepo


# ----------------------------
# Discrete transform (your code)
# ----------------------------

def transform_discrete(df: pd.DataFrame, discrete_cols: list) -> pd.DataFrame:
    df_transformed = df.copy()
    le = LabelEncoder()
    for col in discrete_cols:
        if col in df_transformed.columns:
            df_transformed[col] = le.fit_transform(df_transformed[col].astype(str))
    return df_transformed


# ----------------------------
# Preprocess features (FIX: keep rows aligned with labels)
# ----------------------------

def preprocess_aligned(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, List[str]]:
    """
    Returns:
      df_clean (DataFrame): cleaned + aligned + reset_index
      X (ndarray): feature matrix aligned with df_clean
      y (Series): labels aligned with df_clean (index matches df_clean)
      feature_names (list): feature column names
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame columns: {df.columns.tolist()}")

    df_work = df.copy()

    # Ensure index is sane and unique for consistent positional indexing
    df_work = df_work.reset_index(drop=True)

    y = df_work[label_col].astype(str)
    Xdf = df_work.drop(columns=[label_col]).copy()
    feature_names = list(Xdf.columns)

    Xdf = Xdf.apply(pd.to_numeric, errors='coerce')
    Xdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    Xdf.fillna(Xdf.mean(numeric_only=True), inplace=True)

    # Drop remaining NaNs consistently
    mask = ~Xdf.isna().any(axis=1)
    Xdf = Xdf.loc[mask].copy()
    y = y.loc[mask].copy()

    # Build cleaned df with aligned rows only, and reset index again
    df_clean = pd.concat([Xdf, y.rename(label_col)], axis=1).reset_index(drop=True)
    X = df_clean.drop(columns=[label_col]).values
    y = df_clean[label_col].astype(str)

    return df_clean, X, y, feature_names


# ----------------------------
# Drift metrics
# ----------------------------

def _safe_hist_prob(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=bins)
    p = counts.astype(float)
    p = p / max(p.sum(), 1.0)
    eps = 1e-12
    return np.clip(p, eps, 1.0)

def psi_feature(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if len(train) < 5 or len(test) < 5:
        return 0.0

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(train, qs))
    if len(edges) < 3:
        edges = np.linspace(np.min(train), np.max(train) + 1e-9, n_bins + 1)

    p = _safe_hist_prob(train, edges)
    q = _safe_hist_prob(test, edges)
    return float(np.sum((q - p) * np.log(q / p)))

def js_divergence_1d(train: np.ndarray, test: np.ndarray, n_bins: int = 30) -> float:
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if len(train) < 5 or len(test) < 5:
        return 0.0

    lo = min(np.min(train), np.min(test))
    hi = max(np.max(train), np.max(test))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    p = _safe_hist_prob(train, edges)
    q = _safe_hist_prob(test, edges)
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None, max_points: int = 2000) -> float:
    rng = np.random.default_rng(42)

    if X.shape[0] > max_points:
        X = X[rng.choice(X.shape[0], max_points, replace=False)]
    if Y.shape[0] > max_points:
        Y = Y[rng.choice(Y.shape[0], max_points, replace=False)]

    Z = np.vstack([X, Y])

    if gamma is None:
        idx = rng.choice(Z.shape[0], min(800, Z.shape[0]), replace=False)
        Zs = Z[idx]
        dists = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, axis=2)
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / max(med, 1e-12)

    def k(a, b):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * d2)

    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    m = X.shape[0]
    n = Y.shape[0]
    if m < 2 or n < 2:
        return 0.0

    mmd2 = (Kxx.sum() / (m * (m - 1))) + (Kyy.sum() / (n * (n - 1))) - (2.0 * Kxy.mean())
    return float(np.sqrt(max(mmd2, 0.0)))

def compute_drift_report(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    psi_bins: int = 10,
    js_bins: int = 30,
    compute_mmd_flag: bool = True
) -> Tuple[Dict[str, float], pd.DataFrame]:
    psis, jss = [], []
    for j in range(X_train.shape[1]):
        psis.append(psi_feature(X_train[:, j], X_test[:, j], n_bins=psi_bins))
        jss.append(js_divergence_1d(X_train[:, j], X_test[:, j], n_bins=js_bins))

    feat_df = pd.DataFrame({
        "feature": feature_names[:X_train.shape[1]],
        "psi": psis,
        "js_divergence": jss
    }).sort_values("psi", ascending=False).reset_index(drop=True)

    report = {
        "psi_mean": float(np.mean(psis)) if len(psis) else 0.0,
        "psi_median": float(np.median(psis)) if len(psis) else 0.0,
        "psi_max": float(np.max(psis)) if len(psis) else 0.0,
        "js_mean": float(np.mean(jss)) if len(jss) else 0.0,
        "js_median": float(np.median(jss)) if len(jss) else 0.0,
        "js_max": float(np.max(jss)) if len(jss) else 0.0,
    }
    if compute_mmd_flag:
        report["mmd_rbf"] = mmd_rbf(X_train, X_test)
    return report, feat_df


# ----------------------------
# Sampling helper (preserve label proportions roughly)
# ----------------------------

def _stratified_sample_indices(
    y_int: np.ndarray,
    candidate_idx: np.ndarray,
    n_total: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample n_total indices from candidate_idx approximately preserving the class distribution
    found within candidate_idx.
    """
    if n_total <= 0 or len(candidate_idx) == 0:
        return np.array([], dtype=int)

    y_c = y_int[candidate_idx]
    classes, counts = np.unique(y_c, return_counts=True)
    probs = counts / counts.sum()

    chosen = []
    for c, p in zip(classes, probs):
        want = int(round(n_total * float(p)))
        c_idx = candidate_idx[y_c == c]
        if want <= 0 or len(c_idx) == 0:
            continue
        take = min(want, len(c_idx))
        chosen.extend(rng.choice(c_idx, take, replace=False).tolist())

    if len(chosen) < n_total:
        remaining = np.setdiff1d(candidate_idx, np.array(chosen, dtype=int), assume_unique=False)
        if len(remaining) > 0:
            extra = rng.choice(remaining, min(n_total - len(chosen), len(remaining)), replace=False)
            chosen.extend(extra.tolist())

    if len(chosen) > n_total:
        chosen = rng.choice(np.array(chosen), n_total, replace=False).tolist()

    return np.array(chosen, dtype=int)


# ----------------------------
# CIC-like Domain Shift Splitters
# ----------------------------

def split_domain_shift_cluster(
    df: pd.DataFrame,
    label_col: str,
    drift_strength: float,
    n_clusters: int,
    pca_dim: Optional[int],
    random_state: int,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
    test_frac: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CIC-IDS-like domain shift using clusters (new convention):
      - Clean + align once using preprocess_aligned -> df_clean, X_all, y_all_str, feature_names
      - Cluster in standardized feature space (optional PCA)
      - Domain A = half of clusters, Domain B = other half
      - train sampled mostly from Domain A (labeled)
      - future (val+test) sampled mostly from Domain B (unlabeled/test)
      - split future into val/test
    """
    rng = np.random.default_rng(random_state)

    # NEW convention: always work on cleaned df with reset positional index
    df_clean, X_all, y_all_str, _ = preprocess_aligned(df, label_col=label_col)
    y_all_int = LabelEncoder().fit_transform(y_all_str.to_numpy())

    # cluster on standardized features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)

    if pca_dim is not None and pca_dim > 0 and pca_dim < Xs.shape[1]:
        Z = PCA(n_components=pca_dim, random_state=random_state).fit_transform(Xs)
    else:
        Z = Xs

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = km.fit_predict(Z)

    idx_all = np.arange(len(df_clean))

    # split clusters into Domain A / Domain B
    all_c = np.unique(clusters)
    rng.shuffle(all_c)
    half = max(1, len(all_c) // 2)
    domainA = set(all_c[:half].tolist())
    domainB = set(all_c[half:].tolist()) if len(all_c) > 1 else set(all_c.tolist())

    idx_A = idx_all[np.isin(clusters, list(domainA))]
    idx_B = idx_all[np.isin(clusters, list(domainB))]

    # sizes
    n_total = len(idx_all)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    n_test = n_total - n_train - n_val

    if n_test < 1:
        n_test = max(1, int(round(n_total * test_frac)))
        n_val = max(1, n_total - n_train - n_test)

    # train mostly A
    n_train_A = int(round(n_train * drift_strength))
    n_train_B = n_train - n_train_A

    train_idx = np.concatenate([
        _stratified_sample_indices(y_all_int, idx_A, min(n_train_A, len(idx_A)), rng=rng),
        _stratified_sample_indices(y_all_int, idx_B, min(n_train_B, len(idx_B)), rng=rng),
    ])
    train_idx = np.unique(train_idx)

    remaining = np.setdiff1d(idx_all, train_idx, assume_unique=False)

    # future mostly B
    idx_A_rem = remaining[np.isin(clusters[remaining], list(domainA))]
    idx_B_rem = remaining[np.isin(clusters[remaining], list(domainB))]

    n_future = n_total - len(train_idx)
    n_future_B = int(round(n_future * drift_strength))
    n_future_A = n_future - n_future_B

    future_idx = np.concatenate([
        _stratified_sample_indices(y_all_int, idx_B_rem, min(n_future_B, len(idx_B_rem)), rng=rng),
        _stratified_sample_indices(y_all_int, idx_A_rem, min(n_future_A, len(idx_A_rem)), rng=rng),
    ])
    future_idx = np.unique(future_idx)

    # top up if needed
    if len(future_idx) < n_future:
        leftover = np.setdiff1d(remaining, future_idx, assume_unique=False)
        if len(leftover) > 0:
            extra = rng.choice(leftover, min(n_future - len(future_idx), len(leftover)), replace=False)
            future_idx = np.unique(np.concatenate([future_idx, extra]))

    # build splits from df_clean (positional indices)
    df_train = df_clean.iloc[train_idx].copy()
    df_future = df_clean.iloc[future_idx].copy()

    df_val, df_test = train_test_split(
        df_future,
        test_size=(n_test / max(n_val + n_test, 1)),
        random_state=random_state,
        shuffle=True
    )

    return df_train, df_val, df_test



def split_domain_shift_quantile(
    df: pd.DataFrame,
    label_col: str,
    train_qmax: float,
    test_qmin: float,
    drift_features: Optional[List[str]],
    random_state: int,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
    test_frac: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    df_clean, X_all, y_all_str, feature_names = preprocess_aligned(df, label_col=label_col)
    y_all_int = LabelEncoder().fit_transform(y_all_str.to_numpy())

    # pick drift features if none supplied: highest variance numeric column
    if drift_features is None or len(drift_features) == 0:
        var = np.nanvar(X_all, axis=0)
        j = int(np.nanargmax(var))
        drift_features = [feature_names[j]]

    # Domain masks on CLEAN df
    mask_A = np.ones(len(df_clean), dtype=bool)
    mask_B = np.ones(len(df_clean), dtype=bool)

    for fname in drift_features:
        if fname not in df_clean.columns:
            continue
        col = pd.to_numeric(df_clean[fname], errors="coerce").to_numpy()
        qA = np.nanquantile(col, train_qmax)
        qB = np.nanquantile(col, test_qmin)
        mask_A &= (col <= qA)
        mask_B &= (col >= qB)

    idx_all = np.arange(len(df_clean))
    idx_A = idx_all[mask_A]
    idx_B = idx_all[mask_B]

    # fallbacks if too strict
    if len(idx_A) < 100:
        idx_A = idx_all
    if len(idx_B) < 100:
        idx_B = idx_all

    n_total = len(idx_all)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = max(1, int(round(n_total * test_frac)))
        n_val = max(1, n_total - n_train - n_test)

    # Train from Domain A
    train_idx = _stratified_sample_indices(y_all_int, idx_A, min(n_train, len(idx_A)), rng=rng)
    train_idx = np.unique(train_idx)

    remaining = np.setdiff1d(idx_all, train_idx, assume_unique=False)

    # Future primarily from Domain B among remaining
    idx_B_rem = np.intersect1d(idx_B, remaining, assume_unique=False)
    future_need = n_total - len(train_idx)

    future_idx = _stratified_sample_indices(y_all_int, idx_B_rem, min(future_need, len(idx_B_rem)), rng=rng)
    future_idx = np.unique(future_idx)

    # Top up future if needed
    if len(future_idx) < future_need:
        leftover = np.setdiff1d(remaining, future_idx, assume_unique=False)
        if len(leftover) > 0:
            extra = rng.choice(leftover, min(future_need - len(future_idx), len(leftover)), replace=False)
            future_idx = np.unique(np.concatenate([future_idx, extra]))

    df_train = df_clean.iloc[train_idx].copy()
    df_future = df_clean.iloc[future_idx].copy()

    df_val, df_test = train_test_split(
        df_future,
        test_size=(n_test / max(n_val + n_test, 1)),
        random_state=random_state,
        shuffle=True
    )

    return df_train, df_val, df_test


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Split data and save NPZ with optional discrete transform + CIC-like drift.'
    )

    parser.add_argument('--uci_num', default=None, type=int)

    parser.add_argument('--input_csv', required=False,
                        help='Path to the input CSV file (must contain a label column).')
    parser.add_argument("--unlabeled_csv", required=False,
                        help="Optional separate csv for unlabeled (keeps your existing behavior).",
                        default=False)

    parser.add_argument('--output_dir', required=True,
                        help='Directory where the NPZ files and label2id.json will be saved.')

    parser.add_argument('--label-col', default='Label',
                        help="Name of the label column (default: 'Label').")

    parser.add_argument('--random-state', type=int, default=42)

    parser.add_argument('--discrete_to_label', action='store_true',
                        help='Transform discrete columns to numeric using label encoding.')

    # NEW: shift settings
    parser.add_argument('--shift_mode', type=str, default="none",
                        choices=["none", "cluster", "quantile"],
                        help="Artificial drift mode. Default: none")

    # cluster shift params
    parser.add_argument('--drift_strength', type=float, default=0.9,
                        help="Cluster shift purity (0..1). Higher = stronger domain split.")
    parser.add_argument('--n_clusters', type=int, default=12,
                        help="KMeans clusters for cluster shift.")
    parser.add_argument('--pca_dim', type=int, default=10,
                        help="PCA dimension for clustering (<=0 disables PCA).")

    # quantile shift params
    parser.add_argument('--train_qmax', type=float, default=0.6,
                        help="Domain A definition uses <= train_qmax quantile.")
    parser.add_argument('--test_qmin', type=float, default=0.6,
                        help="Domain B definition uses >= test_qmin quantile.")
    parser.add_argument('--drift_features', type=str, default="",
                        help="Comma-separated feature names to drive quantile shift.")

    args = parser.parse_args()

    out_dir = "data/" + str(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    discrete_cols = []
    if args.uci_num is None:
        cfg = CN()
        cfg.DATASET = get_dataset_config(args.output_dir)
        discrete_cols = cfg.DATASET.DISCRETE_FEATURES

    # ----------------------------
    # Load df
    # ----------------------------

    if args.uci_num is not None:
        dataset = fetch_ucirepo(id=args.uci_num)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)

        # ensure label column name exists
        if args.label_col not in df.columns:
            df.rename(columns={df.columns[-1]: args.label_col}, inplace=True)

        # keep your behavior of using half the data (optional)
        #df = df.sample(frac=0.5, random_state=args.random_state)

    else:
        df = pd.read_csv("raw_data/" + str(args.input_csv))

    # Optional discrete transform BEFORE splitting (recommended for stable drift)
    if args.discrete_to_label:
        if not discrete_cols:
            raise ValueError('No discrete features defined in cfg.DATASET.DISCRETE_FEATURES')
        df = transform_discrete(df, discrete_cols)

    # ----------------------------
    # Splitting logic
    # ----------------------------

    if args.unlabeled_csv is not False:
        # Your original "two-source" setting:
        # - train from labeled df
        # - val/test from df_unlabeled
        df_unlabeled = pd.read_csv("raw_data/" + str(args.unlabeled_csv))
        if args.discrete_to_label and discrete_cols:
            df_unlabeled = transform_discrete(df_unlabeled, discrete_cols)

        df_train, _ = train_test_split(
            df, test_size=0.3, random_state=args.random_state, shuffle=True
        )
        df_val, df_test = train_test_split(
            df_unlabeled, test_size=0.3, random_state=args.random_state, shuffle=True
        )

        shift_settings = {
            "shift_mode": "external_unlabeled_csv",
            "note": "val/test come from separate CSV; no artificial shift applied."
        }

    else:
        # Single-source UCI/CSV with optional artificial domain shift.
        if args.shift_mode == "none":
            df_train, df_remain = train_test_split(
                df, test_size=0.5, random_state=args.random_state, shuffle=True
            )
            df_val, df_test = train_test_split(
                df_remain, test_size=0.5, random_state=args.random_state, shuffle=True
            )
            shift_settings = {"shift_mode": "none"}

        elif args.shift_mode == "cluster":
            pca_dim = None if args.pca_dim is None or args.pca_dim <= 0 else args.pca_dim
            df_train, df_val, df_test = split_domain_shift_cluster(
                df=df,
                label_col=args.label_col,
                drift_strength=args.drift_strength,
                n_clusters=args.n_clusters,
                pca_dim=pca_dim,
                random_state=args.random_state
            )
            shift_settings = {
                "shift_mode": "cluster",
                "drift_strength": args.drift_strength,
                "n_clusters": args.n_clusters,
                "pca_dim": pca_dim
            }

        elif args.shift_mode == "quantile":
            drift_feats = [s.strip() for s in args.drift_features.split(",") if s.strip()] if args.drift_features else None
            df_train, df_val, df_test = split_domain_shift_quantile(
                df=df,
                label_col=args.label_col,
                train_qmax=args.train_qmax,
                test_qmin=args.test_qmin,
                drift_features=drift_feats,
                random_state=args.random_state
            )
            shift_settings = {
                "shift_mode": "quantile",
                "train_qmax": args.train_qmax,
                "test_qmin": args.test_qmin,
                "drift_features": args.drift_features
            }

        else:
            raise ValueError(f"Unknown shift_mode: {args.shift_mode}")

    # ----------------------------
    # Build label mapping
    # ----------------------------

    # Align + preprocess to keep shapes consistent
    _, X_train, y_train_series, feat_names = preprocess_aligned(df_train, label_col=args.label_col)
    _, X_val, y_val_series, _ = preprocess_aligned(df_val, label_col=args.label_col)
    _, X_test, y_test_series, _ = preprocess_aligned(df_test, label_col=args.label_col)


    all_labels = pd.concat([y_train_series, y_val_series, y_test_series], ignore_index=True)
    class_names = sorted(all_labels.unique())
    label2id = {label: idx for idx, label in enumerate(class_names)}

    mapping_path = os.path.join(out_dir, 'label2id.json')
    with open(mapping_path, 'w') as f:
        json.dump(label2id, f, indent=2)

    # Encode labels
    y_train = y_train_series.map(label2id).to_numpy(dtype=np.int64)
    y_val   = y_val_series.map(label2id).to_numpy(dtype=np.int64)
    y_test  = y_test_series.map(label2id).to_numpy(dtype=np.int64)

    # Dummy timestamps
    ts_train = np.arange(len(y_train), dtype=np.int64)
    ts_val   = np.arange(len(y_val), dtype=np.int64)
    ts_test  = np.arange(len(y_test), dtype=np.int64)

    # Save NPZ
    np.savez(os.path.join(out_dir, 'train.npz'), feature=X_train, label=y_train, timestamp=ts_train)
    np.savez(os.path.join(out_dir, 'val.npz'),   feature=X_val,   label=y_val,   timestamp=ts_val)
    np.savez(os.path.join(out_dir, 'test.npz'),  feature=X_test,  label=y_test,  timestamp=ts_test)

    # Drift report (train vs test)
    drift_report, drift_feat_df = compute_drift_report(
        X_train=X_train,
        X_test=X_test,
        feature_names=feat_names if len(feat_names) == X_train.shape[1] else [f"f{j}" for j in range(X_train.shape[1])]
    )

    drift_report_path = os.path.join(out_dir, "drift_report.json")
    with open(drift_report_path, "w") as f:
        json.dump({
            "split_definition": "CIC-like: train from Domain A (labeled), val/test from Domain B (future)",
            "settings": shift_settings,
            "drift_report_train_vs_test": drift_report,
            "sizes": {
                "train": int(X_train.shape[0]),
                "val": int(X_val.shape[0]),
                "test": int(X_test.shape[0])
            },
            "classes": {
                "n_classes": int(len(class_names)),
                "class_names": class_names
            }
        }, f, indent=2)

    drift_feat_path = os.path.join(out_dir, "drift_features.csv")
    drift_feat_df.to_csv(drift_feat_path, index=False)

    # Summary
    print(f"\nSaved splits in '{out_dir}':")
    print(f"  train.npz: {X_train.shape[0]} samples, classes={len(class_names)}")
    print(f"  val.npz:   {X_val.shape[0]} samples (used as unlabeled pool in AL)")
    print(f"  test.npz:  {X_test.shape[0]} samples")
    print(f"  label2id:  {mapping_path}")
    print(f"  drift_report: {drift_report_path}")
    print(f"  drift_features: {drift_feat_path}")

    print("\nDrift metrics (train vs test):")
    print(f"  PSI(mean/median/max)=({drift_report['psi_mean']:.4f}, {drift_report['psi_median']:.4f}, {drift_report['psi_max']:.4f})")
    print(f"  JS(mean/median/max)=({drift_report['js_mean']:.4f}, {drift_report['js_median']:.4f}, {drift_report['js_max']:.4f})")
    print(f"  MMD_RBF={drift_report.get('mmd_rbf', 0.0):.4f}")


if __name__ == '__main__':
    main()
