# Level 3 (Advanced)

The same churn problem as Level 2 — attacked with ensemble and margin-based models,
then honestly ranked against each other.

## Task 1 — [Random Forest](01_random_forest_churn.ipynb)
- hyperparameter sweeps with 5-fold CV: `n_estimators` (plateau past ~200), `max_depth` (picked 14 via the "simplest good model" rule)
- cross-validation: accuracy 0.946 ± 0.009, F1 0.786 ± 0.041
- held-out test: **accuracy 0.966, churn F1 0.870, recall 0.811, AUC 0.941** — the internship's best model
- **extra-mile section**: engineered customer-behavior features (`total_minutes`, `day_share`, `avg_call_length`) + a wider randomized search + an F1-optimal decision threshold picked on train folds only — together these lifted churn F1 from 0.778 → **0.870**; a gradient-boosting challenger (F1 0.851) still loses
- feature importance: day minutes (24.5%) + service calls (12.4%) lead; top 5 features carry 64%

## Task 2 — [SVM](02_svm_churn.ipynb)
- kernels compared **fairly** (both tuned): linear collapses to F1 = 0 without `class_weight` on this imbalanced target
- best config — RBF, C=10, gamma=0.1: **AUC 0.901**, test F1 0.678
- decision boundaries visualised on the top-2 features, **support vectors marked**
- final verdict across all four churn models: random forest wins on F1, SVM wins on pure risk-ranking (AUC)
