# Level 1 (Basic)

The fundamentals: getting raw data ML-ready, and a first regression model.

## Task 1 — [Data Preprocessing](01_data_preprocessing.ipynb)
Raw social-media sentiment data (`data/sentiment.csv`) → model-ready matrices.

- whitespace stripping, duplicate dropping, median fill for missing numerics
- **~180 chaotic raw labels consolidated into 3 sentiment classes** (rule-based keyword mapping)
- one-hot encoding (`Platform`, `Country`) + label encoding (target)
- StandardScaler on continuous features
- stratified 80/20 split → train (568 × 41) / test (142 × 41)

## Task 2 — [Linear Regression](02_linear_regression_house_prices.ipynb)
Boston Housing (`data/house_prediction.csv`) → price prediction (`MEDV`, $1000s).

- space-separated file loaded with hand-supplied column names
- EDA: correlation with target (`RM` strongest positive, `LSTAT` strongest negative)
- simple LR (`RM` only, test R² 0.37) vs multiple LR (13 features, test **R² 0.67**, RMSE ≈ $4.9k)
- coefficient interpretation: each extra room ≈ **+$3.8k** value
