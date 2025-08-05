#!/usr/bin/env python3
"""
split_and_save_npz.py

This script:
1. Receives two arguments: the input CSV file (with a label column) and the output directory.
2. Splits the dataset into train (50%) and remaining (50%) sets.
3. Uses one half as X_train, y_train.
4. Splits the other half into validation and test (each 25% of the original) sets.
5. Encodes labels to integer IDs, builds and saves a label2id.json.
6. Creates dummy timestamp arrays and saves each split as .npz files (feature, label, timestamp).
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df, label_col='Label'):
    """
    Convert features to numeric, handle missing/infinite values, and separate label.
    Returns:
        X (ndarray): Feature array
    """
    X = df.drop(columns=[label_col])
    # Numeric conversion and cleanup
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.mean(), inplace=True)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    return X.values


def load_clean_sample_cic(
    directory: str,
    dataset_flag: str,  # '2017' or '2018'
    sampling_size: int = 1000,
    encoding: str = 'utf-8',
    label_col: str = "Label"
) -> pd.DataFrame:
    """
    Loads, harmonizes, filters, and samples from CSVs in a directory.

    Parameters:
    - directory (str): Path to CSV files.
    - dataset_flag (str): Either '2017' or '2018' to apply corresponding label harmonization.
    - sampling_size (int): Number of rows to sample from each file.
    - encoding (str): Encoding to use when reading CSVs.
    - label_col (str): Name of the label column.

    Returns:
    - pd.DataFrame: Merged and sampled DataFrame.
    """

    def clean_and_harmonize(df: pd.DataFrame, year: str) -> pd.DataFrame:
        df[label_col] = df[label_col].astype(str).str.strip()
        extra_2017 = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Protocol', 'Timestamp']
        extra_2018 = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Protocol', 'Timestamp']
        if year == '2017':
            # Collapse Web Attack variants
            web_attacks_17 = [
                "Web Attack - XSS", "Web Attack - Brute Force", "Web Attack - SQL Injection"
            ]
            df[label_col] = df[label_col].replace({lbl: "Web Attack" for lbl in web_attacks_17})
            df[label_col] = df[label_col].replace({
                "Infiltration - Portscan": "Infiltration",
                "SSH-Patator": "SSH-BruteForce"
            })
            # Drop unwanted rows
            df = df[~df[label_col].isin(["Heartbleed", "Portscan", "DoS Slowhttptest", "FTP-Patator"])]
            df = df.drop(columns=extra_2017, errors="ignore")

        elif year == '2018':
            DDos_18 = ["DDoS-HOIC", "DDoS-LOIC-HTTP", "DDoS-LOIC-UDP"]
            web_attacks_18 = ["Web Attack - XSS", "Web Attack - Brute Force", "Web Attack - SQL"]
            infiltration_18 = [
                "Infiltration - NMAP Portscan", "Infiltration - Dropbox Download",
                "Infiltration - Communication Victim Attacker"
            ]
            bot_18 = ["Botnet Ares"]

            df[label_col] = df[label_col].replace({lbl: "DDoS" for lbl in DDos_18})
            df[label_col] = df[label_col].replace({lbl: "Web Attack" for lbl in web_attacks_18})
            df[label_col] = df[label_col].replace({lbl: "Infiltration" for lbl in infiltration_18})
            df[label_col] = df[label_col].replace({lbl: "Botnet" for lbl in bot_18})
            df = df.drop(columns=extra_2018, errors="ignore")

        return df

    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    sampled_dfs = []

    for file in all_files:
        try:
            df = pd.read_csv(file, low_memory=False, encoding=encoding)

            if 'Attempted Category' not in df.columns or label_col not in df.columns:
                print(f"Skipping file (missing critical columns): {file}")
                continue

            df = clean_and_harmonize(df, dataset_flag)
            df = df[df["Attempted Category"] == -1]

            if df.empty:
                print(f"No usable rows in: {file}")
                continue

            sample = df.sample(n=min(sampling_size, len(df)), random_state=1)
            sampled_dfs.append(sample)

        except Exception as e:
            print(f"Failed to process {file}: {e}")

    final_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Final shape after cleaning and sampling: {final_df.shape}")
    return final_df



def main():
    
    parser = argparse.ArgumentParser(
        description='Split CSV and save features/labels to NPZ with label encoding.'
    )
    parser.add_argument(
        '--input_csv',
        help='Path to the input CSV file (must contain a label column).'
    )
    parser.add_argument(
        '--output_dir',
        help='Directory where the NPZ files (train.npz, val.npz, test.npz) and label2id.json will be saved.'
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
    args = parser.parse_args()

    # Load the full dataset
    df = pd.read_csv("unprocess_csv_files/" +str(args.input_csv))

    # First split: half train, half remain
    df_train, df_remain = train_test_split(
        df,
        test_size=0.5,
        random_state=args.random_state,
        shuffle=True
    )

    # Second split: half of remaining for validation, half for test
    df_val, df_test = train_test_split(
        df_remain,
        test_size=0.5,
        random_state=args.random_state,
        shuffle=True
    )


    final_dir_base = "data/" + str(args.output_dir)
    os.makedirs(final_dir_base, exist_ok=True)
    # Prepare output directory

    # Build label-to-id mapping from all splits
    y_train_series = df_train[args.label_col].astype(str)
    y_val_series   = df_val[args.label_col].astype(str)
    y_test_series  = df_test[args.label_col].astype(str)
    all_labels     = pd.concat([y_train_series, y_val_series, y_test_series], ignore_index=True)
    class_names    = sorted(all_labels.unique())
    label2id       = {label: idx for idx, label in enumerate(class_names)}

    # Save mapping
    mapping_path = os.path.join(final_dir_base, 'label2id.json')
    with open(mapping_path, 'w') as f:
        json.dump(label2id, f, indent=2)

    # Preprocess features
    X_train = preprocess_data(df_train, label_col=args.label_col)
    X_val   = preprocess_data(df_val,   label_col=args.label_col)
    X_test  = preprocess_data(df_test,  label_col=args.label_col)

    # Encode labels
    y_train = y_train_series.map(label2id).to_numpy(dtype=np.int64)
    y_val   = y_val_series.map(label2id).to_numpy(dtype=np.int64)
    y_test  = y_test_series.map(label2id).to_numpy(dtype=np.int64)

    # Create dummy timestamps
    ts_train = np.arange(len(y_train), dtype=np.int64)
    ts_val   = np.arange(len(y_val),   dtype=np.int64)
    ts_test  = np.arange(len(y_test),  dtype=np.int64)



    # Save NPZ files
    np.savez(
        os.path.join(final_dir_base,  'train.npz'),
        feature=X_train,
        label=y_train,
        timestamp=ts_train
    )
    np.savez(
        os.path.join(final_dir_base,'val.npz'),
        feature=X_val,
        label=y_val,
        timestamp=ts_val
    )
    np.savez(
        os.path.join(final_dir_base,  'test.npz'),
        feature=X_test,
        label=y_test,
        timestamp=ts_test
    )

    # Summary
    print(f"Saved splits in '{args.output_dir}':")
    print(f"  train.npz: {X_train.shape[0]} samples, classes={len(class_names)}")
    print(f"  val.npz:   {X_val.shape[0]} samples")
    print(f"  test.npz:  {X_test.shape[0]} samples")
    print(f"Label mapping saved to: {mapping_path}")



if __name__ == '__main__':
    main()
