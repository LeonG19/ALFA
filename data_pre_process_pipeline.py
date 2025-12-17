#!/usr/bin/env python3
"""
split_and_save_npz.py

This script:
1. Receives arguments: the dataset name, input CSV file (with a label column), and the output directory.
2. Optionally transforms discrete features into numeric labels.
3. Splits the dataset into train (50%) and remaining (50%) sets.
4. Uses one half as X_train, y_train.
5. Splits the other half into validation and test (each 25% of the original) sets.
6. Encodes labels to integer IDs, builds and saves a label2id.json.
7. Creates dummy timestamp arrays and saves each split as .npz files (feature, label, timestamp).
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from yacs.config import CfgNode as CN
from config import get_dataset_config


def transform_discrete(df: pd.DataFrame, discrete_cols: list) -> pd.DataFrame:
    """
    Label-encode each column in discrete_cols in-place and return the DataFrame.
    """
    df_transformed = df.copy()
    le = LabelEncoder()
    for col in discrete_cols:
        if col in df_transformed.columns:
            df_transformed[col] = le.fit_transform(df_transformed[col].astype(str))
        else:
            continue
            raise KeyError(f"Discrete column '{col}' not found in DataFrame")
    return df_transformed


def preprocess_data(df: pd.DataFrame, label_col: str = 'Label') -> np.ndarray:
    """
    Convert features to numeric, handle missing/infinite values.
    Returns:
        X (ndarray): Feature array
    """
    
    
    X = df.drop(columns=[label_col])
    print(X)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.mean(), inplace=True)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    X.dropna(inplace=True)
    return X.values


def main():
    parser = argparse.ArgumentParser(
        description='Split CSV and save features/labels to NPZ with optional discrete transformation.'
    )

    parser.add_argument(
        '--input_csv',
        required=True,
        help='Path to the input CSV file (must contain a label column).'
    )
    parser.add_argument(
        "--unlabeled_csv",
        required=False,
        help = "Path to the unlabeled csv file (in case we have different csv for unlabeled and labeled) ",
        default = False
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory where the NPZ files and label2id.json will be saved.'
        
    )
    parser.add_argument(
        '--label-col',
        default='Label',
        help="Name of the label column in the CSV (default: 'Label')."
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42).'
    )
    parser.add_argument(
        '--discrete_to_label',
        action='store_true',
        help='Transform discrete columns to numeric using label encoding.'
    )
    args = parser.parse_args()

    # Load dataset config for discrete feature list
    cfg = CN()
    cfg.DATASET = get_dataset_config(args.output_dir)
    discrete_cols = cfg.DATASET.DISCRETE_FEATURES

    # Load full dataset
    df = pd.read_csv("raw_data/" + str(args.input_csv))
    print("number of classes in labeled dataset", len(set(df["Label"])))
    # Optionally transform discrete features before splitting

    if args.unlabeled_csv == False:
        # First split: 50% train, 50% remainder
        df_train, df_remain = train_test_split(
            df,
            test_size=0.5,
            random_state=args.random_state,
            shuffle=True
        )

        # Second split: 50% of remainder for val, 50% for test
        df_val, df_test = train_test_split(
            df_remain,
            test_size=0.5,
            random_state=args.random_state,
            shuffle=True
        )
    else:
        df_unlabeled = pd.read_csv("raw_data/" + str(args.unlabeled_csv))
        df_train, df_remain = train_test_split(
            df,
            test_size=0.3,
            random_state=args.random_state,
            shuffle=True
        )

        df_val, df_test = train_test_split(
            df_unlabeled,
            test_size=0.3,
            random_state=args.random_state,
            shuffle=True
        )


    # Prepare output directory
    out_dir = "data/"+str(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Build label-to-id mapping
    y_train_series = df_train[args.label_col].astype(str)
    y_val_series   = df_val[args.label_col].astype(str)
    y_test_series  = df_test[args.label_col].astype(str)
    all_labels     = pd.concat([y_train_series, y_val_series, y_test_series], ignore_index=True)
    class_names    = sorted(all_labels.unique())
    label2id       = {label: idx for idx, label in enumerate(class_names)}

    # Save label mapping
    mapping_path = os.path.join(out_dir, 'label2id.json')
    with open(mapping_path, 'w') as f:
        json.dump(label2id, f, indent=2)

    if args.discrete_to_label:
        if not discrete_cols:
            raise ValueError('No discrete features defined in cfg.DATASET.DISCRETE_FEATURES')
        df_train = transform_discrete(df_train, discrete_cols)
        df_val = transform_discrete(df_val, discrete_cols)
        df_test = transform_discrete(df_test, discrete_cols)
        

    # Preprocess features for each split
    X_train = preprocess_data(df_train, label_col=args.label_col)
    X_val   = preprocess_data(df_val,   label_col=args.label_col)
    X_test  = preprocess_data(df_test,  label_col=args.label_col)

    # Encode labels
    y_train = y_train_series.map(label2id).to_numpy(dtype=np.int64)
    y_val   = y_val_series.map(label2id).to_numpy(dtype=np.int64)
    y_test  = y_test_series.map(label2id).to_numpy(dtype=np.int64)

    # Dummy timestamps
    ts_train = np.arange(len(y_train), dtype=np.int64)
    ts_val   = np.arange(len(y_val),   dtype=np.int64)
    ts_test  = np.arange(len(y_test),  dtype=np.int64)

    # Save splits as NPZ
    np.savez(os.path.join(out_dir, 'train.npz'), feature=X_train, label=y_train, timestamp=ts_train)
    np.savez(os.path.join(out_dir, 'val.npz'),   feature=X_val,   label=y_val,   timestamp=ts_val)
    np.savez(os.path.join(out_dir, 'test.npz'),  feature=X_test,  label=y_test,  timestamp=ts_test)

    # Summary
    print(f"Saved splits in '{out_dir}':")
    print(f"  train.npz: {X_train.shape[0]} samples, classes={len(class_names)}")
    print(f"  val.npz:   {X_val.shape[0]} samples")
    print(f"  test.npz:  {X_test.shape[0]} samples")
    print(f"Label mapping saved to: {mapping_path}")

if __name__ == '__main__':
    main()
