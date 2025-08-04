import os
import sys
# ensure parent directory is on PYTHONPATH so we can import calcd_AL
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
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
from active_learning_framework.mlp import TorchMLPClassifier

def load_data(cfg):
    """
    Load train, validation, and test splits from NPZ files.
    """
    def _load(path):
        with np.load(path, allow_pickle=True) as data:
            features = data['feature']
            labels = data['label']
        return features, labels

    base = cfg.DATASET.DATA_DIR
    train_path = os.path.join(base, cfg.DATASET.TRAIN_FILE)
    val_path   = os.path.join(base, cfg.DATASET.VAL_FILE)
    test_path  = os.path.join(base, cfg.DATASET.TEST_FILE)

    X_train, y_train = _load(train_path)
    X_val,   y_val   = _load(val_path)
    X_test,  y_test  = _load(test_path)
    return X_train, y_train, X_val, y_val, X_test, y_test

def print_label_distribution(name, labels):
    """
    Print count and proportion of each class in labels.
    """
    unique, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(labels)
    print(f"{name} label distribution:")
    for cls, cnt, prop in zip(unique, counts, proportions):
        print(f"  Class {cls}: count={cnt}, proportion={prop:.4f}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Active Learning with MLP and NPZ datasets")
    parser.add_argument("--dataset", required=True, help="Name of the dataset (must match config)")
    parser.add_argument("--method",  required=True, help="Key of active learning method in METHOD_DICT")
    parser.add_argument("--budget",  type=int, default=50, help="Number of samples to query per iteration")
    parser.add_argument("--max_iter", type=int, default=300, help="Max iterations for MLP training")
    parser.add_argument("--layers", nargs='+', type=int, default=[100], help="Hidden layer sizes for MLP")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    print(args.method)

    # Load configuration and data
    cfg = get_config(args.dataset, 'mlp')
    cfg.EXPERIMENT.BUDGET    = args.budget
    cfg.EXPERIMENT.AL_METHOD = args.method

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg)

    # Standardize if required
    if cfg.DATASET.STANDARDIZE:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

    # Show original training set distribution
    print_label_distribution("Original training set", y_train)

    if args.method == "galaxy" or args.method == "clue":
 
        mlp = TorchMLPClassifier(
           cfg,
           hidden_layer_sizes=(100,100),
           max_iter=100,
           batch_size=64,
           lr=1e-3,
           random_state=42
         )
        mlp.fit(X_train, y_train)
    else:
        # Initial MLP training
        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(args.layers),
            max_iter=args.max_iter,
            random_state=args.random_state
        )
        mlp.fit(X_train, y_train)

    # Active Learning selection on validation set
    al_method = METHOD_DICT[args.method]()
    selected_indices = al_method.sample(
        X_val,
        cfg.EXPERIMENT.BUDGET,
        mlp,
        X_train
    )

    # Only retrain if new samples were selected
    if selected_indices is not None and len(selected_indices) > 0:
        # Selected samples distribution
        y_selected = y_val[selected_indices]
        print_label_distribution("Selected validation samples", y_selected)

        # Augment training set and show new distribution
        X_selected = X_val[selected_indices]
        X_aug = np.vstack([X_train, X_selected])
        y_aug = np.concatenate([y_train, y_selected])
        print_label_distribution("Augmented training set", y_aug)

        # Retrain MLP on augmented data
        mlp.fit(X_aug, y_aug)
    else:
        print("No new validation samples selected; using initial model without retraining.")

    # Final evaluation on test set
    y_pred = mlp.predict(X_test)
    y_true = y_test

    # Multiclass metrics
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    # Capture classification report
    report = classification_report(y_true, y_pred, digits=4)
    print("\nClassification Report (multiclass):")
    print(report)

    # Binary evaluation: map all non-zero labels to 1
    y_true_bin = np.where(y_true == 1, 0, 1)
    y_pred_bin = np.where(y_pred == 1, 0, 1)
    precision_bin = precision_score(y_true_bin, y_pred_bin)
    recall_bin = recall_score(y_true_bin, y_pred_bin)
    f1_bin = f1_score(y_true_bin, y_pred_bin)

    print("\nBinary Evaluation (0 vs rest):")
    print(f"Precision: {precision_bin:.4f}")
    print(f"Recall   : {recall_bin:.4f}")
    print(f"F1-score : {f1_bin:.4f}")

    # Save metrics to file
    os.makedirs('results', exist_ok=True)
    result_file = os.path.join(
        'results', f"{args.dataset}_{args.method}_{args.budget}.txt"
    )
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Budget: {args.budget}\n\n")
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
        f.write(report)

if __name__ == '__main__':
    main()
