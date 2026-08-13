# Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to
leave an organization ("attrition"), based on HR attributes such as age,
income, overtime, job satisfaction, and tenure.

## Project Structure

```
employee-attrition/
├── dataset.csv                # Raw dataset (not modified)
├── employee_attrition.ipynb   # Main notebook (run top to bottom, outputs included)
├── employee_attrition.py      # Equivalent standalone script
├── README.md
└── requirements.txt
```

Running either the notebook or the script also produces four chart images
in the project folder: `eda_overview.png`, `correlation_heatmap.png`,
`model_evaluation.png`, and `feature_importance.png`.

## Dataset

`dataset.csv` is the IBM HR Analytics Employee Attrition dataset: 1,470
employees, 35 columns, no missing values, no duplicate rows. The target
column is `Attrition` (`Yes` / `No`), and it is **imbalanced**: ~84% of
employees stayed and ~16% left.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook employee_attrition.ipynb   # run all cells
```

or, to run the equivalent script directly:

```bash
python employee_attrition.py
```

Both produce identical results (`random_state=42` throughout).

## Approach

1. **Data understanding** — shape, dtypes, missing values, duplicates, summary stats, target distribution.
2. **EDA** — 9 focused plots: attrition distribution, and attrition by age, department, job role, overtime, job satisfaction, monthly income, years at company, and work-life balance.
3. **Preprocessing**
   - Dropped `EmployeeNumber` (identifier), `EmployeeCount`/`StandardHours`/`Over18` (constant columns, zero variance).
   - Correlation analysis showed `JobLevel` is nearly a duplicate of `MonthlyIncome` (r ≈ 0.95) → dropped `JobLevel`.
   - Target encoded `Yes → 1`, `No → 0`.
   - Categorical features one-hot encoded (`drop='first'`); numeric features standard-scaled.
   - Train/test split (80/20, `stratify=y`, `random_state=42`) performed **before** fitting the encoder/scaler, and all preprocessing is fit only on the training data (inside an sklearn `Pipeline`) to avoid data leakage.
4. **Feature selection** — Random Forest feature importance and `SelectKBest` (ANOVA F-test) were computed on the training set; the top 20 features were kept for modeling.
5. **Models trained** — Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors, each wrapped in a `Pipeline` (preprocess → select → classify). `class_weight="balanced"` is used where supported, since the target is imbalanced.
6. **Evaluation** — Accuracy, Precision, Recall, F1-score for all models; Confusion Matrix, ROC Curve, and ROC-AUC for the best model.
7. **Feature importance** — top features from the best tree-based model, visualized as a bar chart.
8. **Prediction function** — `predict_attrition(employee_dict)` returns `"Likely to Leave"` / `"Likely to Stay"` plus the predicted probability.

## Results

| Model                | Accuracy | Precision | Recall | F1-Score |
|-----------------------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression   | 0.7653   | 0.3750    | 0.7021 | **0.4889** |
| Decision Tree         | 0.7381   | 0.3333    | 0.6383 | 0.4380   |
| Random Forest         | 0.8401   | 0.5000    | 0.1489 | 0.2295   |
| K-Nearest Neighbors   | 0.8537   | 0.8333    | 0.1064 | 0.1887   |

**Best model: Logistic Regression** (highest F1-score, ROC-AUC ≈ 0.78).

Random Forest and KNN reach higher raw *accuracy*, but their recall on the
"Leave" class is very low — they mostly predict the majority class ("No")
and miss most employees who actually leave. Since the dataset is
imbalanced (~84%/16%), accuracy alone is misleading; F1-score (which
balances precision and recall) is the fairer metric, and Logistic
Regression — helped by `class_weight="balanced"` — catches roughly 70% of
actual leavers, which is far more useful for an HR early-warning tool.

### Most important attrition factors

From the Decision Tree feature importance (cross-checked against
`SelectKBest` and the EDA): **Total Working Years**, **Monthly Income**,
**OverTime**, **Stock Option Level**, **Environment Satisfaction**, **Age**,
**Years In Current Role**, and **Distance From Home**.

## Key Insights

- Employees who work **overtime** leave at a much higher rate proportionally than those who don't.
- Attrition skews toward **younger, less-tenured, lower-income** employees.
- **Low job satisfaction** and **poor work-life balance** both correlate with higher attrition.
- **Sales Representatives** and **Laboratory Technicians** show comparatively higher attrition shares than other roles.

## Recommendations for Improving Retention

1. **Monitor and reduce overtime** — one of the strongest predictors of attrition; redistribute workload or hire additional staff in chronically over-worked teams.
2. **Review compensation for early-career employees** — attrition concentrates among lower-income, lower-tenure staff.
3. **Expand stock option / equity participation** — employees with no stock options are more likely to leave.
4. **Improve environment satisfaction and work-life balance**, especially in roles/teams that score low.
5. **Focus retention efforts on younger, newer employees** — mentorship and career development in the first two years.
6. **Use the prediction function proactively** to flag high-risk employees for a retention conversation before they decide to leave.

## Notes

- `dataset.csv` is never modified.
- No deep learning, external APIs, or downloaded datasets are used.
- All metrics and figures above are produced directly from `dataset.csv` by the notebook/script — nothing is hardcoded.
