# RAVEN-ADE++

Risk-Adaptive Variational Ensemble Network for Attack Detection and Explainability



### 1. OSDS-LogSec Dataset

The project introduces the OpenStack DevStack Log Security Dataset (OSDS-LogSec Dataset), a structured cloud-security dataset collected from a private OpenStack DevStack deployment.

The dataset contains:

| Property | Value |
|---|---:|
| Total Samples | 444,602 |
| Normal Samples | 222,302 |
| Attack Samples | 222,300 |
| Raw Features | 18 |
| HyperGraph-Augmented Features | 27 |
| Train Samples | 302,328 |
| Validation Samples | 53,353 |
| Test Samples | 88,921 |
| Class Balance | 50 / 50 |

The dataset contains five attack families:

1. DoS
2. Payload Injection
3. Header Flood
4. API Abuse
5. Insider Mimic

---

## OSDS-LogSec Dataset Workflow

The dataset workflow starts with a private OpenStack DevStack deployment where authentication, authorization, and resource-management events are logged. Audit logs and service logs are synchronized by user, request, session, and timestamp. Raw metadata, execution statistics, semantic context, and temporal information are extracted and converted into structured features.

The final dataset combines:

- Behavioral features
- System-level features
- Semantic features
- Temporal features
- Outcome features

---

The framework is composed of eight major modules.

| Module | Description |
|---|---|
| Module 1 | Data cleaning, encoding, normalization |
| Module 2 | Risk-Adaptive Attack Injection and Labeling |
| Module 3 | Train, validation, and test partitioning |
| Module 4 | HyperGraph contextual augmentation |
| Module 5 | Risk-Adaptive Encoder |
| Module 6 | Attention-Based Risk Weighting |
| Module 7 | Supervised Variational Latent Learning |
| Module 8 | Binary attack classification head |

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

Based on the proposed RAVEN-ADE++ framework and OSDS-LogSec dataset described in the accompanying paper. fileciteturn4file0

