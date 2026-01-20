#!/usr/bin/env python3
"""
Unified Active Learning runner with pluggable generators:
- Active Learning methods (`--al_method`): base, augmented_al, ALFA
- Active Learning functions (`--al_function`): specific query functions (e.g. margin, powermargin)
- Generators (`--generator`): TVAE, CTGAN, RTF
- Classifiers (`--classifier`): MLP, RF, XGBC

This script creates a results directory (`results/{al_method}/{classifier}/{generator}/{dataset}_{budget}`)
and saves classification reports and detailed metrics (accuracy, precision, recall, F1 — micro & macro).
Supports custom anchor fraction via `--anchor_alpha` and `--anchor_steepness`.

FLAGS:
- --neighbor_only: in ALFA, add kNN neighbors directly (no generator / no synthetic data)
- --filter_bad_neighbors: remove incorrectly-labeled neighbors (uses pool ground-truth labels y_p)

NEW FLAGS (this patch):
- --gen_train_all_labeled: in ALFA, train generator on (all labeled so far) + (retrieved neighbors)
  instead of only (anchors + neighbors). Works in minority and base ALFA.
- --group_training: in ALFA, train ONE generator using the selected generator-training dataset
  (anchors+neighbors OR all-labeled+neighbors). Then sample and split per class.
- --no_anchor_knn: in ALFA, SKIP anchor selection + kNN. After active learning:
    * if --minority: train generator using ALL labeled samples of the two most underrepresented classes
    * else: train generator using ALL labeled samples from all classes except majority
  Supports --group_training, and is compatible with --gen_train_all_labeled (no-op in this mode).

UPDATED FLAG BEHAVIOR:
- --num_synthetic now supports a LIST of multipliers:
    * If 1 number: used for all augmentable classes
    * If k>1 numbers: applied to the k most underrepresented classes (ascending by labeled count),
      and the rest get multiplier 0 (no synthetic) by default.
  This also works in --minority mode (which uses the 2 most underrepresented classes).
"""

import argparse
import os
import math
import json

from pool_filters_cumulative import subpool_anchoral, subpool_randsub, subpool_seals
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors

from active_learning_functions import METHOD_DICT
from config import get_config
from ctgan import TVAE
from ctgan import CTGAN
from realtabformer import REaLTabFormer
from classifiers.mlp import TorchMLPClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Unified AL runner")
    parser.add_argument('--al_method', choices=['base','DA','DA+ALFA'], required=True)
    parser.add_argument('--pooling_method', choices=['anchoral', 'randsub', 'seals'], required=False, default=False)
    parser.add_argument('--al_function', choices=list(METHOD_DICT.keys()), required=True)
    parser.add_argument('--generator', choices=['TVAE','CTGAN','RTF'])
    parser.add_argument('--classifier', choices=['MLP','RF','XGBC'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--random_state', type=int, default=42)

    # UPDATED: accept list of multipliers (space and/or comma separated)
    # examples:
    #   --num_synthetic 3
    #   --num_synthetic 5 3
    #   --num_synthetic 6,4,2
    parser.add_argument('--num_synthetic', nargs='+', type=str, default=['3'])

    parser.add_argument('--filter_synthetic', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--steepness', type=float, default=100.0)
    parser.add_argument('--minority', action='store_true')
    parser.add_argument('--neighbor_only', action='store_true')
    parser.add_argument('--filter_bad_neighbors', action='store_true')
    parser.add_argument('--gen_train_all_labeled', action='store_true',
                        help="Train generator on (all labeled so far) + (retrieved neighbors) "
                             "instead of (anchors + neighbors).")
    parser.add_argument('--group_training', action='store_true',
                        help="Train a single generator (not per-class) using the generator-training dataset.")
    parser.add_argument('--no_anchor_knn', action='store_true',
                        help="Skip anchor selection and kNN. After active learning, train generator using "
                             "ALL labeled minority class (if --minority) OR ALL labeled non-majority classes.")
    return parser.parse_args()


def _parse_num_synthetic_list(raw_tokens):
    """
    Accepts:
      raw_tokens like ['3'] or ['5','3'] or ['6,4,2'] or ['6,4','2']
    Returns: list[float]
    """
    if raw_tokens is None:
        return [3.0]
    out = []
    for tok in raw_tokens:
        if tok is None:
            continue
        parts = [p.strip() for p in str(tok).split(',') if p.strip() != '']
        for p in parts:
            out.append(float(p))
    if len(out) == 0:
        out = [3.0]
    return out


def _rank_classes_by_underrep(y, classes):
    """
    Return list of classes sorted by ascending count in y (most underrepresented first).
    Only ranks within the provided `classes`.
    """
    classes = list(classes)
    counts = {cls: int(np.sum(y == cls)) for cls in classes}
    return sorted(classes, key=lambda cls: (counts[cls], str(cls)))


def _make_multiplier_map(y_labeled, classes_to_aug, multipliers):
    """
    multipliers:
      - len==1: apply to all classes_to_aug
      - len>1: apply to k most underrepresented classes (k=len(multipliers)), others -> 0.0
    """
    multipliers = list(multipliers)
    classes_to_aug = list(classes_to_aug)

    if len(classes_to_aug) == 0:
        return {}

    if len(multipliers) == 1:
        return {cls: float(multipliers[0]) for cls in classes_to_aug}

    ranked = _rank_classes_by_underrep(y_labeled, classes_to_aug)
    m = {cls: 0.0 for cls in classes_to_aug}
    for i, mult in enumerate(multipliers):
        if i >= len(ranked):
            break
        m[ranked[i]] = float(mult)
    return m


def ensure_results_dir(args):
    gen = args.generator if args.generator else ''
    clf = args.classifier if args.classifier else ''
    pool = args.pooling_method if args.pooling_method else ''
    minority = "minority" if args.minority else ''
    knn_only = "knnonly" if args.neighbor_only else ''
    knn_filter = "filter" if args.filter_bad_neighbors else ''

    # encode these modes in path so you can separate runs cleanly
    all_labeled = "alllabeled" if args.gen_train_all_labeled else ''
    group = "groupgen" if args.group_training else ''
    noak = "noanchorknn" if args.no_anchor_knn else ''

    path = os.path.join(
        'results', args.dataset, pool, args.al_method, clf, gen,
        minority, group, all_labeled, noak,
        f"{args.al_function}_{knn_only}_{knn_filter}"
    )
    os.makedirs(path, exist_ok=True)
    return path


def load_data(cfg):
    def _load(p):
        with np.load(p, allow_pickle=True) as d:
            return d['feature'], d['label']
    base = cfg.DATASET.DATA_DIR
    return (*_load(os.path.join(base, cfg.DATASET.TRAIN_FILE)),
            *_load(os.path.join(base, cfg.DATASET.VAL_FILE)),
            *_load(os.path.join(base, cfg.DATASET.TEST_FILE)))


def print_dist(title, y):
    u, c = np.unique(y, return_counts=True)
    print(f"{title}:")
    for ui, ci in zip(u, c):
        print(f"  Class {ui}: {ci} ({ci/len(y):.2%})")
    print()


def compute_anchor_fraction(f_c, min_frac=0.01, alpha=1, steepness=100):
    return min(1, (alpha * math.exp(-steepness * (f_c - min_frac))))


def init_pooling(name, Xl, yl, Xu, yu):
    if name == 'anchoral':
        return subpool_anchoral(Xl, yl, Xu, yu)
    elif name == 'randsub':
        return subpool_randsub(Xl, yl, Xu, yu)
    elif name == "seals":
        return subpool_seals(Xl, yl, Xu, yu)


def init_classifier(name, args, cfg):
    if name == 'MLP':
        return TorchMLPClassifier(cfg,
                                  hidden_layer_sizes=tuple([100]),
                                  max_iter=100,
                                  batch_size=64,
                                  lr=1e-3,
                                  random_state=args.random_state,
                                  device="cuda")
    if name == 'RF':
        return RandomForestClassifier(n_estimators=100,
                                      random_state=args.random_state,
                                      n_jobs=-1)
    if name == 'XGBC':
        return XGBClassifier(use_label_encoder=False, learning_rate=0.1, max_depth=6, n_estimators=100,
                             eval_metric='mlogloss',
                             random_state=args.random_state)
    raise ValueError(name)


def init_generator(name, cfg):
    if name == 'TVAE':
        return TVAE(epochs=100, batch_size=60)
    if name == 'CTGAN':
        return CTGAN(epochs=100, batch_size=60)
    if name == 'RTF':
        return REaLTabFormer(model_type='tabular',
                             epochs=100,
                             gradient_accumulation_steps=1,
                             logging_steps=100,
                             numeric_max_len=12)
    raise ValueError(name)


def compute_metrics(y_pred, y_true):
    rpt = classification_report(y_true, y_pred, digits=4)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro')
    }, rpt


def _discrete_cols_with_label(cfg, df):
    cols = list(getattr(cfg.DATASET, "DISCRETE_FEATURES", []))
    if 'Label' in df.columns and 'Label' not in cols:
        cols = cols + ['Label']
    return cols


def _fit_generator(gen, df, discrete_cols):
    """
    Normalize generator.fit(...) calls across TVAE/CTGAN/RTF, since signatures can differ.
    """
    try:
        gen.fit(df, discrete_cols)
        return
    except TypeError:
        pass
    try:
        gen.fit(df, discrete_columns=discrete_cols)
        return
    except TypeError:
        pass
    gen.fit(df)


def _sample_generator(gen, args, n):
    if n <= 0:
        return pd.DataFrame()
    if args.generator == "RTF":
        return gen.sample(n_samples=n)
    try:
        return gen.sample(samples=n)
    except TypeError:
        return gen.sample(n)


def _sample_and_split_by_label(gen, args, fn, yal_dtype, k_per_cls, gen_train_df_len, random_state):
    """
    Group-training helper:
      - samples a big batch, then slices/pads per class to meet k_per_cls exactly.
      - if generator doesn't output Label, we assign labels in the exact needed proportions.
    """
    need_total = int(sum(int(v) for v in k_per_cls.values()))
    if need_total <= 0:
        return pd.DataFrame(columns=fn + ['Label'])

    # oversample to reduce risk of missing classes
    gen_total = max(need_total * 3, need_total)

    dfs_all = _sample_generator(gen, args, gen_total)
    if len(dfs_all) == 0:
        return pd.DataFrame(columns=fn + ['Label'])

    if 'Label' not in dfs_all.columns:
        labels = []
        for cls, k in k_per_cls.items():
            labels.extend([cls] * int(k))
        if len(labels) == 0:
            labels = [list(k_per_cls.keys())[0]] * len(dfs_all)
        if len(labels) < len(dfs_all):
            labels = (labels * int(math.ceil(len(dfs_all) / max(1, len(labels)))))[:len(dfs_all)]
        else:
            labels = labels[:len(dfs_all)]
        dfs_all['Label'] = np.array(labels, dtype=yal_dtype)

    try:
        dfs_all['Label'] = dfs_all['Label'].astype(yal_dtype, copy=False)
    except Exception:
        pass

    parts = []
    for cls, k in k_per_cls.items():
        k = int(k)
        if k <= 0:
            continue
        df_cls = dfs_all[dfs_all['Label'] == cls]
        if len(df_cls) >= k:
            parts.append(df_cls.iloc[:k])
        else:
            if len(df_cls) > 0:
                pad = df_cls.sample(n=(k - len(df_cls)), replace=True, random_state=random_state)
                parts.append(pd.concat([df_cls, pad], ignore_index=True))
            else:
                fallback = dfs_all.sample(n=k, replace=True, random_state=random_state).copy()
                fallback['Label'] = cls
                parts.append(fallback)

    return pd.concat(parts, ignore_index=True) if len(parts) else pd.DataFrame(columns=fn + ['Label'])


def knn_retrieve_neighbors(X_u, y_u, X_p, y_p, filter_bad_neighbors=False, n_neighbors=1):
    """
    Return (X_neighbors, y_pseudo) where y_pseudo is the assumed class label (anchor class).

    filter_bad_neighbors=True removes neighbors whose true y_p != assumed class.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_p)

    X_neighbors_all = []
    y_pseudo_all = []

    total_correct = 0
    total_neighbors = 0
    total_kept = 0

    for cls in np.unique(y_u):
        mask = (y_u == cls)
        Xu = X_u[mask]

        _, idxs = nn.kneighbors(Xu)
        idxs = idxs.reshape(-1)

        y_nn_true = y_p[idxs]
        correct_mask = (y_nn_true == cls)

        correct = int(np.sum(correct_mask))
        total = int(len(idxs))
        pct = (correct / total * 100.0) if total > 0 else 0.0

        if filter_bad_neighbors:
            idxs = idxs[correct_mask]

        kept = int(len(idxs))
        kept_pct = (kept / total * 100.0) if total > 0 else 0.0

        print(f"KNN neighbors | assumed class={cls}: {correct}/{total} match ({pct:.2f}%). "
              f"Kept={kept}/{total} ({kept_pct:.2f}%) | filter_bad_neighbors={filter_bad_neighbors}")

        total_correct += correct
        total_neighbors += total
        total_kept += kept

        if kept > 0:
            Xn = X_p[idxs]
            X_neighbors_all.append(Xn)
            y_pseudo_all.append(np.full(kept, cls, dtype=y_u.dtype))

    if len(X_neighbors_all) == 0:
        X_neighbors = np.empty((0, X_p.shape[1]))
        y_pseudo = np.empty((0,), dtype=y_u.dtype)
    else:
        X_neighbors = np.vstack(X_neighbors_all)
        y_pseudo = np.hstack(y_pseudo_all)

    overall_pct = (total_correct / total_neighbors * 100.0) if total_neighbors > 0 else 0.0
    overall_kept_pct = (total_kept / total_neighbors * 100.0) if total_neighbors > 0 else 0.0
    print(f"KNN neighbors | overall: {total_correct}/{total_neighbors} match ({overall_pct:.2f}%). "
          f"Kept={total_kept}/{total_neighbors} ({overall_kept_pct:.2f}%) | filter_bad_neighbors={filter_bad_neighbors}")

    return X_neighbors, y_pseudo


def base_set_up(args):
    cfg = get_config(args.dataset, args.al_method)
    Xtr, ytr, Xv, yv, Xt, yt = load_data(cfg)
    print_dist("original label", ytr)
    print_dist("original unlabel", yv)
    print_dist("original test", yt)
    if cfg.DATASET.STANDARDIZE:
        s = StandardScaler().fit(Xtr)
        Xtr, Xv, Xt = s.transform(Xtr), s.transform(Xv), s.transform(Xt)
    clf = init_classifier(args.classifier, args, cfg)
    clf.fit(Xtr, ytr)
    return cfg, clf, Xtr, ytr, Xv, yv, Xt, yt


def active_function(args, Xtr, Xv, ytr, yv, clf):
    sel = METHOD_DICT[args.al_function]().sample(Xv, args.budget, clf, Xtr)
    u, c = np.unique(yv[sel], return_counts=True)
    print(u, c)
    if sel is not None and len(sel):
        Xtr = np.vstack([Xtr, Xv[sel]]); ytr = np.hstack([ytr, yv[sel]])
    return Xtr, ytr, sel


def run_base(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xtr, ytr, _ = active_function(args, Xtr, Xv, ytr, yv, clf)
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xt)
    return y_pred, yt


def run_base_pooling(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xv, yv = init_pooling(args.pooling_method, Xtr, ytr, Xv, yv)
    Xtr, ytr, _ = active_function(args, Xtr, Xv, ytr, yv, clf)
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xt)
    return y_pred, yt


def run_augmented(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xal, yal, _ = active_function(args, Xtr, Xv, ytr, yv, clf)

    # parse new multipliers
    multipliers = _parse_num_synthetic_list(args.num_synthetic)

    u, c = np.unique(yal, return_counts=True)
    maj = u[np.argmax(c)]
    fn = cfg.DATASET.FEATURE_NAMES

    # Underrepresented classes are everything except majority
    underrep_classes = [cls for cls in u if cls != maj]
    if len(underrep_classes) == 0:
        print("[WARN] No underrepresented classes found (only one class present). Skipping augmentation.")
        clf.fit(Xal, yal)
        y_pred = clf.predict(Xt)
        return y_pred, yt

    # multiplier map within underrep classes
    mult_map = _make_multiplier_map(yal, underrep_classes, multipliers)

    mask = np.isin(yal, underrep_classes)
    Xu = Xal[mask]; yu = yal[mask]
    dfu = pd.DataFrame(Xu, columns=fn); dfu['Label'] = yu
    print_dist('Underrepresented combined', yu)

    # how many synthetic per class
    k_per_cls = {}
    for cls in underrep_classes:
        cls_count = int(np.sum(yal == cls))
        k_per_cls[cls] = int(cls_count * float(mult_map.get(cls, 0.0)))

    print("DA k_per_cls (class -> synthetic count):", k_per_cls)

    # Train one generator on underrep combined (same as before), but sample per-class targets
    gen = init_generator(args.generator, cfg)
    disc = _discrete_cols_with_label(cfg, dfu)
    _fit_generator(gen, dfu, disc)

    comb = _sample_and_split_by_label(
        gen=gen,
        args=args,
        fn=fn,
        yal_dtype=yal.dtype,
        k_per_cls=k_per_cls,
        gen_train_df_len=len(dfu),
        random_state=args.random_state
    )

    if len(comb):
        print_dist('Synthetic', comb['Label'].values)
        Xs, ys = comb[fn].values, comb['Label'].values
    else:
        Xs = np.empty((0, Xal.shape[1]))
        ys = np.empty((0,), dtype=yal.dtype)

    if args.filter_synthetic and len(ys):
        msk = clf.predict(Xs) != 0
        Xs, ys = Xs[msk], ys[msk]

    Xf = np.vstack([Xal, Xs]) if len(ys) else Xal
    yf = np.hstack([yal, ys]) if len(ys) else yal

    clf.fit(Xf, yf)
    y_pred = clf.predict(Xt)
    return y_pred, yt


def run_alfa(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    if args.pooling_method:
        print("using pooling")
        Xv, yv = init_pooling(args.pooling_method, Xtr, ytr, Xv, yv)

    Xal, yal, sel = active_function(args, Xtr, Xv, ytr, yv, clf)

    # parse new multipliers
    multipliers = _parse_num_synthetic_list(args.num_synthetic)

    fn = cfg.DATASET.FEATURE_NAMES
    labeled_df = pd.DataFrame(Xal, columns=fn)
    labeled_df['Label'] = yal

    # ------------------------------------------------------------------
    # NEW MODE: skip anchor selection + kNN; train generator directly
    # ------------------------------------------------------------------
    if args.no_anchor_knn:
        print("ALFA: no_anchor_knn mode enabled (skipping anchors + kNN).")

        u, c = np.unique(yal, return_counts=True)
        freqs = c / len(yal)
        maj = u[np.argmax(freqs)]

        if args.minority:
            sorted_classes_index = np.argsort(freqs)[:2]
            sorted_classes = u[sorted_classes_index]
            print(f"no_anchor_knn: training generator with ALL labeled samples from minority classes={sorted_classes}")
            gen_train_df = labeled_df[labeled_df['Label'].isin(sorted_classes)].copy()
            print(gen_train_df['Label'].value_counts())
            classes_to_aug = np.array([cls for cls in sorted_classes])
        else:
            print(f"no_anchor_knn: training generator with ALL labeled samples from all classes except majority={maj}")
            gen_train_df = labeled_df[labeled_df['Label'] != maj].copy()
            classes_to_aug = np.array([cls for cls in u if cls != maj])

        if len(gen_train_df) == 0:
            print("[WARN] gen_train_df empty in no_anchor_knn mode; falling back to full labeled_df.")
            gen_train_df = labeled_df.copy()

        mult_map = _make_multiplier_map(yal, classes_to_aug, multipliers)

        # choose k per class (based on labeled counts in yal)
        k_per_cls = {}
        for cls in classes_to_aug:
            cls_count = int(np.sum(yal == cls))
            k_per_cls[cls] = int(cls_count * float(mult_map.get(cls, 0.0)))

        print("no_anchor_knn k_per_cls (class -> synthetic count):", k_per_cls)

        syn = []

        if args.group_training:
            gen = init_generator(args.generator, cfg)
            disc = _discrete_cols_with_label(cfg, gen_train_df)
            _fit_generator(gen, gen_train_df, disc)

            comb = _sample_and_split_by_label(
                gen=gen,
                args=args,
                fn=fn,
                yal_dtype=yal.dtype,
                k_per_cls=k_per_cls,
                gen_train_df_len=len(gen_train_df),
                random_state=args.random_state
            )
            syn.append(comb)
        else:
            for cls in classes_to_aug:
                train_df_cls = gen_train_df[gen_train_df['Label'] == cls].copy()
                if len(train_df_cls) < 2:
                    print(f"[WARN] Skipping generator for class={cls} (train_df_cls has {len(train_df_cls)} rows).")
                    continue

                gen = init_generator(args.generator, cfg)
                disc = _discrete_cols_with_label(cfg, train_df_cls)
                _fit_generator(gen, train_df_cls, disc)

                k = int(k_per_cls.get(cls, 0))
                dfs = _sample_generator(gen, args, k)
                if len(dfs) == 0:
                    continue
                dfs['Label'] = cls
                syn.append(dfs)
                print(syn[-1]['Label'].value_counts())

        comb = pd.concat(syn, ignore_index=True) if len(syn) else pd.DataFrame(columns=fn + ['Label'])

        if len(comb):
            Xs = comb[fn].values
            ys = comb['Label'].values
        else:
            Xs = np.empty((0, Xal.shape[1]))
            ys = np.empty((0,), dtype=yal.dtype)

        if args.filter_synthetic and len(ys):
            msk = clf.predict(Xs) != 0
            Xs, ys = Xs[msk], ys[msk]

        Xf = np.vstack([Xal, Xs]) if len(ys) else Xal
        yf = np.hstack([yal, ys]) if len(ys) else yal

        clf.fit(Xf, yf)
        y_pred = clf.predict(Xt)
        return y_pred, yt

    # ------------------------------------------------------------------
    # ORIGINAL ALFA MODE: anchor selection + kNN
    # ------------------------------------------------------------------
    u, c = np.unique(yal, return_counts=True)
    freqs = c / len(yal)
    rng = np.random.RandomState(args.random_state)
    all_choices = []

    if args.minority:
        # UPDATED: minority uses TWO most underrepresented classes
        sorted_classes_index = np.argsort(freqs)[:2]
        minority_classes = u[sorted_classes_index]
        print(f"using minority class alfa composition (2 classes): {minority_classes}")

        for cls in minority_classes:
            cls_idx = np.where(yal == cls)[0]
            cls_freq = freqs[u.tolist().index(cls)]
            frac = compute_anchor_fraction(cls_freq, args.alpha, args.steepness)
            n_cls = max(1, int(frac * len(cls_idx)))
            chosen = rng.choice(cls_idx, size=n_cls, replace=False)
            all_choices.extend(chosen)
    else:
        max_class = u[np.argmax(freqs)]
        for cls in u:
            if cls == max_class:
                continue
            cls_idx = np.where(yal == cls)[0]
            cls_freq = freqs[u.tolist().index(cls)]
            frac = compute_anchor_fraction(cls_freq, args.alpha, args.steepness)
            n_cls = max(1, int(frac * len(cls_idx)))
            chosen = rng.choice(cls_idx, size=n_cls, replace=False)
            all_choices.extend(chosen)

    choice = np.array(all_choices)
    Xu, yu = Xal[choice], yal[choice]
    print_dist('Anchors', yu)

    # remove AL-selected from pool (avoid retrieving neighbors from already-labeled points)
    mask = np.ones(len(yv), bool)
    if sel is not None:
        mask[sel] = False
    X_pool = Xv[mask]
    y_pool = yv[mask]

    # retrieve neighbors ONCE (used for neighbor-only and generator training)
    Xn, yn = knn_retrieve_neighbors(
        Xu, yu, X_pool, y_pool,
        filter_bad_neighbors=args.filter_bad_neighbors,
        n_neighbors=1
    )
    print_dist("Neighbors (pseudo-labeled)", yn)

    if args.neighbor_only:
        print("ALFA: neighbor-only mode enabled (skipping generator + synthetic data).")
        Xf = np.vstack([Xal, Xn]) if len(Xn) else Xal
        yf = np.hstack([yal, yn]) if len(yn) else yal
        clf.fit(Xf, yf)
        y_pred = clf.predict(Xt)
        return y_pred, yt

    anchors_df = pd.DataFrame(Xu, columns=fn)
    anchors_df['Label'] = yu

    neighbors_df = pd.DataFrame(Xn, columns=fn) if len(Xn) else pd.DataFrame(columns=fn)
    if len(Xn):
        neighbors_df['Label'] = yn
    else:
        neighbors_df['Label'] = pd.Series([], dtype=yal.dtype)

    base_gen_df = labeled_df if args.gen_train_all_labeled else anchors_df
    gen_train_df = pd.concat([base_gen_df, neighbors_df], ignore_index=True)

    # We will augment these classes (the ones represented by anchors)
    classes_to_aug = np.unique(yu)

    mult_map = _make_multiplier_map(yal, classes_to_aug, multipliers)

    # how many synthetic per class (based on class counts in yal)
    k_per_cls = {}
    for cls in classes_to_aug:
        cls_count_in_labeled = int(np.sum(yal == cls))
        k_per_cls[cls] = int(cls_count_in_labeled * float(mult_map.get(cls, 0.0)))

    print("ALFA k_per_cls (class -> synthetic count):", k_per_cls)

    syn = []

    if args.group_training:
        gen = init_generator(args.generator, cfg)
        disc = _discrete_cols_with_label(cfg, gen_train_df)
        _fit_generator(gen, gen_train_df, disc)

        comb = _sample_and_split_by_label(
            gen=gen,
            args=args,
            fn=fn,
            yal_dtype=yal.dtype,
            k_per_cls=k_per_cls,
            gen_train_df_len=len(gen_train_df),
            random_state=args.random_state
        )
        syn.append(comb)

    else:
        for cls in classes_to_aug:
            if args.gen_train_all_labeled:
                base_cls = labeled_df[labeled_df['Label'] == cls]
            else:
                base_cls = anchors_df[anchors_df['Label'] == cls]
            neigh_cls = neighbors_df[neighbors_df['Label'] == cls] if len(neighbors_df) else neighbors_df.iloc[0:0]

            train_df_cls = pd.concat([base_cls, neigh_cls], ignore_index=True)

            if len(train_df_cls) < 2:
                print(f"[WARN] Skipping generator for class={cls} (train_df_cls has {len(train_df_cls)} rows).")
                continue

            gen = init_generator(args.generator, cfg)
            disc = _discrete_cols_with_label(cfg, train_df_cls)
            _fit_generator(gen, train_df_cls, disc)

            k = int(k_per_cls.get(cls, 0))
            dfs = _sample_generator(gen, args, k)
            if len(dfs) == 0:
                continue

            dfs['Label'] = cls
            syn.append(dfs)

    comb = pd.concat(syn, ignore_index=True) if len(syn) else pd.DataFrame(columns=fn + ['Label'])

    if len(comb):
        Xs = comb[fn].values
        ys = comb['Label'].values
    else:
        Xs = np.empty((0, Xal.shape[1]))
        ys = np.empty((0,), dtype=yal.dtype)

    if args.filter_synthetic and len(ys):
        msk = clf.predict(Xs) != 0
        Xs, ys = Xs[msk], ys[msk]

    Xf = np.vstack([Xal, Xs]) if len(ys) else Xal
    yf = np.hstack([yal, ys]) if len(ys) else yal

    clf.fit(Xf, yf)
    y_pred = clf.predict(Xt)
    return y_pred, yt


def main():
    args = parse_args()
    res = ensure_results_dir(args)

    if args.al_method == 'base' and not args.pooling_method:
        y_pred, y_true = run_base(args, res)
    elif args.al_method == 'DA':
        y_pred, y_true = run_augmented(args, res)
    elif args.pooling_method and args.al_method == "base":
        y_pred, y_true = run_base_pooling(args, res)
    else:
        y_pred, y_true = run_alfa(args, res)

    m, rpt = compute_metrics(y_pred, y_true)
    with open(os.path.join(res, 'report.txt'), 'w') as f:
        f.write(rpt)
    with open(os.path.join(res, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=4)


if __name__ == '__main__':
    main()
