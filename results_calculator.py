import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
def aggregate_f1_macro(
    root_dir,
    metric_key="f1_macro",
    dataset_mode=False,
    active_learning_method="",
    generator="",
    classifier="",
    save_csv=True,
    output_path="averaged_results/final_f1_macro_summary_base.csv"
):


    results = defaultdict(list)

    # ================================
    # MODE B: DATASET COMPARISON
    # ================================
    if dataset_mode:
        if active_learning_method is None or generator is None:
            raise ValueError(
                "dataset_mode=True requires active_learning_method and generator"
            )

        for dataset in sorted(os.listdir(root_dir)):
            dataset_path = os.path.join(root_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue

            # Construct fixed experiment path
            exp_path = os.path.join(
                dataset_path,
                active_learning_method,
                classifier,
                generator
            )


            for al_func in sorted(os.listdir(exp_path)):
                al_path = os.path.join(exp_path, al_func)
                metrics_path = os.path.join(al_path, "metrics.json")

                if not os.path.isfile(metrics_path):
                    print(f"[WARNING] Missing metrics.json: {metrics_path}")
                    continue

                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)

                    if metric_key not in metrics:
                        print(f"[WARNING] '{metric_key}' not found in {metrics_path}")
                        continue

                    results[al_func].append(metrics[metric_key])

                except Exception as e:
                    print(f"[ERROR] Failed reading {metrics_path}: {e}")

    # ================================
    # MODE A: RANDOM STATE AGGREGATION
    # ================================
    else:
        for seed_name in sorted(os.listdir(root_dir)):
            seed_path = os.path.join(root_dir, seed_name)
            if not os.path.isdir(seed_path):
                continue

            for al_func in sorted(os.listdir(seed_path)):
                al_path = os.path.join(seed_path, al_func)
                metrics_path = os.path.join(al_path, "metrics.json")

                if not os.path.isfile(metrics_path):
                    print(f"[WARNING] Missing metrics.json: {metrics_path}")
                    continue

                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)

                    if metric_key not in metrics:
                        print(f"[WARNING] '{metric_key}' not found in {metrics_path}")
                        continue

                    results[al_func].append(metrics[metric_key])

                except Exception as e:
                    print(f"[ERROR] Failed reading {metrics_path}: {e}")

    # ================================
    # BUILD FINAL TABLE
    # ================================
    rows = []
    for al_func, values in results.items():
        values = np.array(values)

        rows.append({
            "active_learning_function": al_func,
            "num_runs": len(values),
            "f1_macro_mean": values.mean(),
            "f1_macro_std": values.std(ddof=1)
        })

    df = pd.DataFrame(rows).sort_values("active_learning_function")

    df["f1_macro (mean ± std)"] = (
        df["f1_macro_mean"].round(4).astype(str)
        + " ± "
        + df["f1_macro_std"].round(4).astype(str)
    )

    if save_csv:
        df.to_csv(output_path, index=False)
        print(f"[INFO] Saved summary to {output_path}")

    return df



df = aggregate_f1_macro(
    root_dir="/home/lgarza3/projects/ctgan/28428_ALFA_Active_Learning_wit_Supplementary Material/supplementary_materials/source_code/results/cover_results",
    metric_key="f1_macro",
    dataset_mode = True,
    active_learning_method = "DA+ALFA",
    classifier = "MLP",
    generator = "CTGAN"

)

print(df)