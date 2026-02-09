
# DenseGAAL: Density-Aware Generative Augmentation for Active Learning

A research framework that improves active learning under low-label regimes by adaptively augmenting rare classes via generative modeling.

---

## About the Project

**ALFA** enhances standard active learning pipelines by:

- Selecting latent **anchors** from rare or complex classes each active learning round.
- Retrieving nearby unlabeled examples in latent space.
- Training class‑conditional generators (TVAE, CTGAN, or RTF) to synthesize data only for under‑represented classes.
- Filtering synthetic samples with the classifier to avoid oversampling dominant classes.

This leads to better generalization in imbalanced or shifting distributions—especially in cybersecurity datasets like intrusion detection, where ALFA yields **20–30% performance gains** over baselines.

---

## Motivation

When annotation budgets are tight, active learning often misses rare classes, leading to poor generalization. ALFA addresses this gap by selectively expanding under‑represented regions in the latent space using generative sampling.

---

## Built With

- **Python 3.x**  
- Generative modeling: TVAE, CTGAN, RTF  
- Classifiers: MLP, Random Forest (RF), XGBoost (XGBC)

---

## 🛠 Getting Started

### Prerequisites

- Python 3.x  
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Setup

1. Clone this repo:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```
2. Place your input CSV in `source_code/raw_data/`.
3. Edit `source_code/config.py` to define:
   - `LABEL_NAME`, `FEATURE_NAMES`, `DISCRETE_FEATURES`  
   - `NUM_FEATURE`, `STANDARDIZE`, `NUM_CLASS`, etc.
4. Example config snippet:

   [config screenshot placeholder]

---

## Usage

### 1. Data Preprocessing

From `source_code/`, run:
```bash
python -m data_pre_process_pipeline \
  --input_csv adult.csv \
  --output_dir adult \
  --label_col income \
  --discrete_to_label
```
- `--discrete_to_label`: label‑encodes discrete/categorical features. Use ONLY if your discrete features are not numerical
- `output_dir` must match the reference name set in `config.py`.

Produces: `data/adult/train.npz`, `val.npz`, `test.npz`, and `label2id.json`.

### 2. Main Experiment Pipeline

Run:
```bash
python -m main \
  --al_method <base|DA|DA+ALFA> \
  --al_function <random|entropy|lc|margin|coreset|galaxy|bald|powermargin|clue|diana|eada|upper|lower|density> \
  --classifier <MLP|RF|XGBC> \
  --dataset adult \
  --budget 5 \
  --random_state 42 \
  --generator CTGAN \
  --num_synthetic 3.0 \
  --filter_synthetic \
  --alpha 1.0 \
  --steepness 50.0
```

#### Brief argument descriptions

| Argument             | Description                                     |
|----------------------|-------------------------------------------------|
| `--al_method`         | Experiment type                                  |
| `--al_function`       | Active learning selection strategy               |
| `--classifier`        | Classification model                             |
| `--dataset`           | Reference dataset name                           |
| `--budget`            | Number of samples per AL round                  |
| `--random_state`      | Seed for reproducibility                         |
| `--generator`         | Generative model for augmentation                |
| `--num_synthetic`     | Synthetic sample multiplier                      |
| `--filter_synthetic`  | Only include filtered synthetic data             |
| `--alpha`, `--steepness` | Generation hyperparameters                   |

❗ Note: `galaxy` and `clue` AL methods work **only** with the **MLP classifier**.

#### Example Runs

**Base Active Learning (no augmentation):**
```bash
python -m main --al_method base --al_function random --classifier MLP --dataset adult --budget 5
```

**With data augmentation only:**
```bash
python -m main --al_method DA --al_function random --generator CTGAN --classifier XGBC --dataset adult --budget 5
```

**Full ALFA pipeline:**
```bash
python -m main --al_method DA+ALFA --al_function random --generator CTGAN --classifier XGBC --dataset adult --budget 5 --filter_synthetic --alpha 1 --steepness 50
```

---

## Project Structure

```
source_code/
├── active_learning_functions/
├── classifiers/
├── config.py
├── data_pre_process_pipeline.py
├── main.py
├── raw_data/
├── data/
└── results/
```

---

## Results & Benchmarks

Place outputs under `results/`. ALFA consistently outperforms standard methods, especially in class‑imbalanced and distribution‑shifted scenarios.

---
