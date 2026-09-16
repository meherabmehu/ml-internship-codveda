# Machine Learning Internship — Codveda Technology

Six end-to-end machine learning projects completed for the **ML Internship at
[Codveda Technology](https://www.codveda.com)** — from raw messy data to tuned,
evaluated, and honestly-compared models. Every task lives in its own
fully-executed Jupyter notebook (0 errors on clean re-run).

> The program offers 3 tasks per level; **2 per level are required** — these are my picks.

## 🏆 Results at a glance

| Level | Task | Model(s) | Key result |
|---|---|---|---|
| 1 | [Data Preprocessing](level-1-basic/01_data_preprocessing.ipynb) | cleaning pipeline | 732 raw posts, ~180 junk labels → 3 clean classes; output (568×41) train / (142×41) test |
| 1 | [Linear Regression](level-1-basic/02_linear_regression_house_prices.ipynb) | simple + multiple OLS | test **R² = 0.67**, RMSE ≈ **$4.9k** on Boston Housing |
| 2 | [Logistic Regression](level-2-intermediate/01_logistic_regression_churn.ipynb) | logit (+ class-weighted) | **AUC 0.825**; intl plan = **7.7× churn odds**; balanced variant lifts churn recall 0.18 → 0.76 |
| 2 | [Decision Tree](level-2-intermediate/02_decision_tree_iris.ipynb) | tree + cost-complexity pruning | depth 5 → **3 leaves**, macro-F1 **0.89** on Iris |
| 3 | [Random Forest](level-3-advanced/01_random_forest_churn.ipynb) | RF, tuned + feature-engineered (400 trees) | **best of the internship**: acc **0.966**, churn F1 **0.870**, AUC **0.941** |
| 3 | [SVM](level-3-advanced/02_svm_churn.ipynb) | linear vs RBF kernels | **AUC 0.901** (RBF beats linear); decision-boundary visualised with support vectors |

## 📖 One dataset, three notebooks — the churn story

Levels 2 & 3 deliberately attack the **same telecom-churn problem** with four
different models, so the notebooks read as one honest model-selection story:

| Model | Accuracy | Churn F1 | Churn recall | Note |
|---|---|---|---|---|
| Logistic regression | 0.853 | 0.258 | 0.179 | most interpretable, misses churners at default threshold |
| Single decision tree | 0.909 | 0.690 | 0.716 | high-variance, memorises noise |
| **Random forest** | **0.966** | **0.870** | 0.811 | champion — feature engineering + bagging |
| SVM (RBF, tuned) | 0.916 | 0.678 | 0.621 | best AUC (0.901), great at ranking risk |

Shared finding across all models: churn is driven by **total day minutes**,
**customer service calls**, and the **international plan** — and with a 14.6%
churn rate, plain accuracy lying at ~85% is a trap every notebook addresses.

## 📁 Repository structure

```
├── data/                       # all datasets provided for the program
├── level-1-basic/              # preprocessing + linear regression
├── level-2-intermediate/       # logistic regression + decision tree
├── level-3-advanced/           # random forest + SVM
├── requirements.txt
└── LICENSE (MIT)
```

Each level folder has its own README with the task-by-task breakdown.

## 📊 Datasets

| File | Shape | Used in |
|---|---|---|
| `data/iris.csv` | 150 × 5 | L2 — decision tree |
| `data/house_prediction.csv` | 506 × 14 (Boston Housing, no header) | L1 — linear regression |
| `data/churn-bigml-80.csv` | 2,666 × 20 (train) | L2/L3 — logit, RF, SVM |
| `data/churn-bigml-20.csv` | 667 × 20 (held-out test) | L2/L3 evaluation |
| `data/sentiment.csv` | 732 × 15 | L1 — preprocessing |
| `data/stock_prices.csv` | 497,472 × 7 | provided by the program, not used by the selected tasks |

## 🛠️ Tools

Python · pandas · NumPy · scikit-learn · matplotlib · Jupyter

## 🚀 Run it yourself

```bash
git clone https://github.com/meherabmehu/ml-internship-codveda.git
cd ml-internship-codveda
pip install -r requirements.txt
jupyter notebook    # open any .ipynb — all paths are relative to data/
```

Every notebook executes top-to-bottom without errors (verified after the final
commit on the repo's main branch).
