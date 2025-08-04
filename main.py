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
"""
import argparse
import os
import math
import json

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

from active_learning_framework.al import METHOD_DICT
from active_learning_framework.config import get_config
from TVAE.tvae import TVAE
from ctgan import CTGAN
from realtabformer import REaLTabFormer
from active_learning_framework.mlp import TorchMLPClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Unified AL runner")
    parser.add_argument('--al_method', choices=['base','DA','DA+ALFA'], required=True)
    parser.add_argument('--al_function', choices=list(METHOD_DICT.keys()), required=True)
    parser.add_argument('--generator', choices=['TVAE','CTGAN','RTF'])
    parser.add_argument('--classifier', choices=['MLP','RF','XGBC'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--num_synthetic', type=float, default=3)
    parser.add_argument('--filter_synthetic', action='store_true')
    parser.add_argument('--anchor_alpha', type=float, default=1.0)
    parser.add_argument('--anchor_steepness', type=float, default=100.0)
    return parser.parse_args()

def ensure_results_dir(args):
    gen = args.generator if args.generator else ''
    alpha = args.alpha if args.alpha else ''
    path = os.path.join('results', args.dataset, args.al_method, gen, f"{args.function}_{alpha}")
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
    return alpha * math.exp(-steepness * (f_c - min_frac))

def init_classifier(name, args, cfg):
    if name=='MLP':
        return TorchMLPClassifier(cfg,
                                  hidden_layer_sizes=tuple(100),
                                  max_iter=300,
                                  batch_size=64,
                                  lr=1e-3,
                                  random_state=args.random_state,
                                  device="cuda")
    if name=='RF':
        return RandomForestClassifier(n_estimators=100,
                                      random_state=args.random_state,
                                      n_jobs=-1)
    if name=='XGBC':
        return XGBClassifier(use_label_encoder=False,
                             eval_metric='mlogloss',
                             random_state=args.random_state)
    raise ValueError(name)

def init_generator(name, cfg):
    if name=='TVAE':
        return TVAE(epochs=100, batch_size=60)
    if name=='CTGAN':
        return CTGAN(epochs=100, batch_size=60)
    if name=='RTF':
        return REaLTabFormer(model_type='tabular',
                             epochs=100,
                             gradient_accumulation_steps=1,
                             logging_steps=100,
                             numeric_max_len=12)
    raise ValueError(name)

def compute_metrics(y_pred, y_true):

    rpt = classification_report(y_true,y_pred,digits=4)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro')
    }, rpt

def knn_expand(X_u, y_u, X_p, y_p, names):
    nn = NearestNeighbors(n_neighbors=1).fit(X_p)
    out = {}
    for cls in np.unique(y_u):
        mask = (y_u==cls)
        Xu = X_u[mask]
        _, idxs = nn.kneighbors(Xu)
        Xn = X_p[idxs.flatten()]
        dfu = pd.DataFrame(Xu, columns=names)
        dfn = pd.DataFrame(Xn, columns=names)
        out[cls] = pd.concat([dfu, dfn], ignore_index=True)
    return out



def base_set_up(args):
    cfg = get_config(args.dataset, args.al_method)
    Xtr, ytr, Xv, yv, Xt, yt = load_data(cfg)
    if cfg.DATASET.STANDARDIZE:
        s=StandardScaler().fit(Xtr)
        Xtr,Xv,Xt=s.transform(Xtr),s.transform(Xv),s.transform(Xt)
    clf=init_classifier(args.al_method, args, cfg)
    clf.fit(Xtr,ytr)
    return cfg, clf, Xtr, ytr, Xv, yv, Xt, yt

def active_function(args, Xtr, Xv, ytr, yv, clf):
    sel=METHOD_DICT[args.al_function]().sample(Xv,args.budget,clf,Xtr)
    if sel is not None and len(sel):
        Xtr=np.vstack([Xtr,Xv[sel]]); ytr=np.hstack([ytr,yv[sel]])
    return Xtr, ytr


def run_base(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xtr, ytr = active_function(args, Xtr, Xv, ytr, yv, clf)
    clf.fit(Xtr, ytr)
    y_pred=clf.predict(Xt)
    return y_pred, yt



def run_augmented(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xal, yal = active_function(args, Xtr, Xv, ytr, yv, clf)
    u,c=np.unique(yal,return_counts=True)
    maj=u[np.argmax(c)]
    fn=cfg.DATASET.FEATURE_NAMES
    mask=np.isin(yal,[x for x in u if x!=maj])
    Xu=Xal[mask]; yu=yal[mask]
    dfu=pd.DataFrame(Xu,columns=fn); dfu['Label']=yu
    print_dist('Underrepresented combined',yu)
    gen=init_generator(args.generator,cfg)
    gen.fit(dfu,cfg.DATASET.DISCRETE_FEATURES)
    n=int(Xu.shape[0]*args.num_synthetic)
    try: dfs=gen.sample(samples=n)
    except TypeError: dfs=gen.sample(n)
    if 'Label' not in dfs: dfs['Label']=dfu['Label'].values.repeat(int(math.ceil(n/len(yu))))[:n]
    print_dist('Synthetic',dfs['Label'].values)
    Xs,ys=dfs[fn].values,dfs['Label'].values
    if args.filter_synthetic:
        msk=clf.predict(Xs)!=0
        Xs,ys=Xs[msk],ys[msk]
    Xf=np.vstack([Xal,Xs]); yf=np.hstack([yal,ys])
    clf.fit(Xf, yf)
    y_pred=clf.predict(Xt)
    return y_pred, yt


def run_alfa(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xal, yal = active_function(args, Xtr, Xv, ytr, yv, clf)
    u,c=np.unique(yal,return_counts=True)
    freqs=c/len(yal)
    mc=u[np.argmin(freqs)]; idx=np.where(yal==mc)[0]
    frac=compute_anchor_fraction(freqs[u.tolist().index(mc)],args.anchor_alpha,args.anchor_steepness)
    n=max(1,int(frac*len(idx)))
    choice=np.random.RandomState(args.random_state).choice(idx,size=n,replace=False)
    Xu,yu=Xal[choice],yal[choice]
    print_dist('Anchors',yu)
    mask=np.ones(len(yv),bool)
    if sel is not None: mask[sel]=False
    per=knn_expand(Xu,yu,Xv[mask],yv[mask],cfg.DATASET.FEATURE_NAMES)
    gen=init_generator(args.generator,cfg)
    syn=[]
    discrete = cfg.DATASET.DISCRETE_FEATURES
    discrete = discrete[:-1]
    for cls,dfc in per.items():
        gen.fit(dfc,)
        k=int(c[u.tolist().index(cls)]*args.num_synthetic)
        try: dfs=gen.sample(samples=k)
        except TypeError: dfs=gen.sample(k)
        dfs['Label']=cls; syn.append(dfs)
    comb=pd.concat(syn,ignore_index=True)
    Xs,ys=comb[cfg.DATASET.FEATURE_NAMES].values,comb['Label'].values
    if args.filter_synthetic:
        msk=clf.predict(Xs)!=0
        Xs,ys=Xs[msk],ys[msk]
    Xf=np.vstack([Xal,Xs]); yf=np.hstack([yal,ys])
    clf.fit(Xf, yf)
    y_pred=clf.predict(Xt)
    return y_pred, yt


def main():
    args=parse_args()
    res=ensure_results_dir(args)
    if args.al_method=='base':
        y_pred, y_true = run_base(args,res)
    elif args.al_method=='augmented_al':
        y_pred, y_true =  run_augmented(args,res)
    else:
        y_pred, y_true =  run_alfa(args,res)
    m, rpt = compute_metrics(y_pred, y_true)
    with open(os.path.join(res,'report.txt'),'w') as f: f.write(rpt)
    with open(os.path.join(res,'metrics.json'),'w') as f: json.dump(m,f,indent=4)

if __name__=='__main__':
    main()
