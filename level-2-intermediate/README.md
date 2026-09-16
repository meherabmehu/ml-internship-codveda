# Level 2 (Intermediate)

Classification gets real: probabilistic models, odds ratios, and overfitting control.

## Task 1 — [Logistic Regression](01_logistic_regression_churn.ipynb)
Telecom churn (`data/churn-bigml-80/20.csv`, conveniently pre-split).

- preprocessing: binary maps, dropped collinear charge columns (corr = 1.0 with minutes, proven in-notebook), StandardScaler
- **odds-ratio interpretation**: international plan ≈ 7.7× churn odds, voice-mail plan ≈ 0.24×
- evaluation: accuracy 0.853, **AUC 0.825**, full ROC curve
- imbalance lesson: default threshold catches only 18% of churners; `class_weight='balanced'` lifts recall to **0.76**

## Task 2 — [Decision Tree](02_decision_tree_iris.ipynb)
Iris species (`data/iris.csv`).

- grew an unpruned tree first: depth 5, 8 leaves, 100% train accuracy = pure memorisation
- **cost-complexity pruning** with `ccp_alpha` chosen by 5-fold CV → depth 2, 3 leaves
- test macro-F1 **0.89**; both trees drawn with `plot_tree`, plus a decision-region plot
