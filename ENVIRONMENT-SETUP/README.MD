# RAVEN-ADE++ — Anaconda Environment Setup Guide (Windows)
## Full step-by-step from scratch

---

## STEP 1 — Install Anaconda (if not already installed)
Download from: https://www.anaconda.com/download
- Choose: Windows 64-bit
- During install: ✅ "Add Anaconda to PATH" (recommended)

---

## STEP 2 — Open Anaconda Prompt
Start Menu → search "Anaconda Prompt" → Open as Administrator

---

## STEP 3 — Create the Environment
```bash
conda create -n raven_env python=3.10 -y
```

---

## STEP 4 — Activate the Environment
```bash
conda activate raven_env
```
You should now see **(raven_env)** at the start of your prompt.

---

## STEP 5 — Install All Required Libraries

Run these commands **one by one**:

```bash
# Core scientific stack
conda install -c conda-forge numpy pandas scikit-learn matplotlib -y

# TensorFlow (CPU+GPU auto-detected)
pip install tensorflow

# XGBoost
pip install xgboost

# Joblib (model saving)
pip install joblib

# Jupyter (optional but recommended)
conda install -c conda-forge jupyter notebook -y
```

### ✅ Verify Installation
```bash
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
python -c "import xgboost as xgb; print('XGB:', xgb.__version__)"
python -c "import sklearn; print('SKLearn:', sklearn.__version__)"
```

---

## STEP 6 — Prepare Your Folder Structure

Create this folder layout on your machine:

```
D:\Raven\
│
├── data\
│   └── keystone_features_parsed_struct.csv   ← your CSV file goes here
│
├── outputs\
│   ├── tables\        ← all printed tables saved as CSV
│   ├── models\        ← all trained models saved here (.keras + .pkl)
│   ├── figures\       ← all PNG figures saved here
│   └── logs\          ← timing + run logs
│
└── raven_ade_local.py  ← the main script (provided separately)
```

Create it quickly with this command in Anaconda Prompt:
```bash
mkdir D:\Raven\data D:\Raven\outputs\tables D:\Raven\outputs\models D:\Raven\outputs\figures D:\Raven\outputs\logs
```

---

## STEP 7 — Place Your CSV File
Copy your CSV file to:
```
D:\Raven\data\keystone_features_parsed_struct.csv
```

---

## STEP 8 — Run the Script
```bash
cd D:\Raven
conda activate raven_env
python raven_ade_local.py
```

---

## STEP 9 — What Gets Saved (Automatically)

| Output | Location |
|--------|----------|
| Table 1 — Dataset Statistics | `D:\Raven\outputs\tables\table1_dataset_stats.csv` |
| Table 2 — Attack Scenario Breakdown | `D:\Raven\outputs\tables\table2_attack_scenarios.csv` |
| Table 3 — Main Performance Comparison | `D:\Raven\outputs\tables\table3_main_comparison.csv` |
| Table 4 — Architecture Ablation | `D:\Raven\outputs\tables\table4_ablation_arch.csv` |
| Table 5 — Feature Group Impact | `D:\Raven\outputs\tables\table5_feature_groups.csv` |
| Table 6 — Robustness Scenarios | `D:\Raven\outputs\tables\table6_robustness.csv` |
| Table 7 — Explainability | `D:\Raven\outputs\tables\table7_explainability.csv` |
| Table 8 — Energy & Latency | `D:\Raven\outputs\tables\table8_efficiency.csv` |
| Table 9 — Literature Baselines | `D:\Raven\outputs\tables\table9_literature_baselines.csv` |
| Figure 1 — Main Comparison | `D:\Raven\outputs\figures\fig1_main_comparison.png` |
| Figure 2 — Architecture Ablation | `D:\Raven\outputs\figures\fig2_ablation_arch.png` |
| Figure 3 — Feature Ablation | `D:\Raven\outputs\figures\fig3_feature_ablation.png` |
| Figure 4 — Confusion Matrix | `D:\Raven\outputs\figures\fig4_confusion_matrix.png` |
| Figure 5 — Training Curves | `D:\Raven\outputs\figures\fig5_training_curves.png` |
| Figure 6 — Feature Importance | `D:\Raven\outputs\figures\fig6_feature_importance.png` |
| RAVEN-ADE++ Full Model | `D:\Raven\outputs\models\raven_ade_full.keras` |
| Light Model | `D:\Raven\outputs\models\raven_light.keras` |
| CNN Baseline Model | `D:\Raven\outputs\models\cnn_baseline.keras` |
| Random Forest | `D:\Raven\outputs\models\random_forest.pkl` |
| XGBoost | `D:\Raven\outputs\models\xgboost.pkl` |
| SVM | `D:\Raven\outputs\models\svm.pkl` |
| Feature Scaler | `D:\Raven\outputs\models\scaler.pkl` |
| Run Log | `D:\Raven\outputs\logs\run_log.txt` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow` inside `(raven_env)` |
| `ModuleNotFoundError: xgboost` | Run `pip install xgboost` inside `(raven_env)` |
| TensorFlow GPU not detected | Install CUDA 11.8 + cuDNN 8.6 from nvidia.com, then `pip install tensorflow[and-cuda]` |
| CSV not found | Check path is exactly `D:\Raven\data\keystone_features_parsed_struct.csv` |
| Permission denied on D:\ | Run Anaconda Prompt as Administrator |
