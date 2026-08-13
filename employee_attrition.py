"""
Employee Attrition Prediction
==============================
A machine learning project that predicts whether an employee is likely to
leave a company, based on HR attributes (age, income, overtime, job
satisfaction, etc.). Built with pandas, numpy, matplotlib, seaborn and
scikit-learn only (no deep learning).

Run:  python employee_attrition.py
"""

import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save plots to file instead of popping up a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def clean_feature_name(name):
    """Strip the ColumnTransformer prefix (e.g. 'num__Age' -> 'Age') for readable output."""
    return name.split("__", 1)[-1] if "__" in name else name

# ---------------------------------------------------------------------------
# STEP 1: DATA LOADING & UNDERSTANDING
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: DATA LOADING & UNDERSTANDING")
print("=" * 70)

df = pd.read_csv("dataset.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nShape (rows, columns):", df.shape)

print("\nColumn names:")
print(list(df.columns))

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary (numeric columns):")
print(df.describe())

print("\nTarget variable distribution (Attrition):")
print(df["Attrition"].value_counts())
print(df["Attrition"].value_counts(normalize=True).round(3) * 100, "(%)")

# ---------------------------------------------------------------------------
# STEP 2: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 1. Attrition distribution
sns.countplot(data=df, x="Attrition", hue="Attrition", palette="Set2", legend=False, ax=axes[0, 0])
axes[0, 0].set_title("Attrition Distribution")

# 2. Attrition by age
sns.histplot(data=df, x="Age", hue="Attrition", multiple="stack", palette="Set2", ax=axes[0, 1])
axes[0, 1].set_title("Attrition by Age")

# 3. Attrition by department
sns.countplot(data=df, x="Department", hue="Attrition", palette="Set2", ax=axes[0, 2])
axes[0, 2].set_title("Attrition by Department")
axes[0, 2].tick_params(axis="x", rotation=20)

# 4. Attrition by job role
role_order = df["JobRole"].value_counts().index
sns.countplot(data=df, y="JobRole", hue="Attrition", palette="Set2", order=role_order, ax=axes[1, 0])
axes[1, 0].set_title("Attrition by Job Role")

# 5. Attrition by overtime
sns.countplot(data=df, x="OverTime", hue="Attrition", palette="Set2", ax=axes[1, 1])
axes[1, 1].set_title("Attrition by OverTime")

# 6. Attrition by job satisfaction
sns.countplot(data=df, x="JobSatisfaction", hue="Attrition", palette="Set2", ax=axes[1, 2])
axes[1, 2].set_title("Attrition by Job Satisfaction (1=Low, 4=High)")

# 7. Attrition by monthly income
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", hue="Attrition", palette="Set2", legend=False, ax=axes[2, 0])
axes[2, 0].set_title("Attrition by Monthly Income")

# 8. Attrition by years at company
sns.boxplot(data=df, x="Attrition", y="YearsAtCompany", hue="Attrition", palette="Set2", legend=False, ax=axes[2, 1])
axes[2, 1].set_title("Attrition by Years at Company")

# 9. Attrition by work-life balance
sns.countplot(data=df, x="WorkLifeBalance", hue="Attrition", palette="Set2", ax=axes[2, 2])
axes[2, 2].set_title("Attrition by Work-Life Balance (1=Bad, 4=Best)")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nSaved EDA visualizations to eda_overview.png")

# ---------------------------------------------------------------------------
# STEP 3: PREPROCESSING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: PREPROCESSING")
print("=" * 70)

data = df.copy()

# Columns that are identifiers or carry no predictive information:
# - EmployeeNumber: unique ID per employee, not a real attribute.
# - EmployeeCount and StandardHours: constant for every row (no variance -> no signal).
# - Over18: constant ('Y' for every employee) -> no signal.
cols_to_drop = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
print(f"\nDropping identifier/constant columns: {cols_to_drop}")
data = data.drop(columns=cols_to_drop)

# Check for redundant/highly correlated numeric features before encoding.
numeric_preview = data.select_dtypes(include="number")
corr_matrix = numeric_preview.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()

high_corr_pairs = (
    corr_matrix.abs()
    .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
)
print("\nHighly correlated numeric feature pairs (|r| > 0.7):")
print(high_corr_pairs[high_corr_pairs > 0.7])

# JobLevel is nearly a duplicate of MonthlyIncome (r ~ 0.95) and also
# strongly overlaps with TotalWorkingYears. We drop JobLevel and keep
# MonthlyIncome/TotalWorkingYears since income is more directly
# interpretable for an attrition/retention story.
print("\nDropping 'JobLevel' -> redundant with MonthlyIncome (r ~ 0.95).")
data = data.drop(columns=["JobLevel"])

# Missing values (checked again after column drops, should still be none)
print("\nMissing values after cleanup:", data.isnull().sum().sum())

# Encode target: Yes -> 1, No -> 0
data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0})

X = data.drop(columns=["Attrition"])
y = data["Attrition"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include="number").columns.tolist()
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Train/test split BEFORE fitting any encoder/scaler -> avoids data leakage.
# stratify=y keeps the same Yes/No ratio in both splits (important since
# the target is imbalanced: ~84% No vs ~16% Yes).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ColumnTransformer: one-hot encode categoricals, standard-scale numerics.
# drop='first' avoids the dummy-variable trap (perfect multicollinearity).
# Fitting happens only inside each model's Pipeline.fit(X_train, ...), so
# the encoder/scaler statistics never see the test set -> no leakage.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ]
)

# ---------------------------------------------------------------------------
# STEP 4: FEATURE SELECTION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: FEATURE SELECTION")
print("=" * 70)

# Fit the preprocessor on the TRAIN split only, to inspect features safely.
preprocessor_fitted = preprocessor.fit(X_train, y_train)
feature_names = preprocessor_fitted.get_feature_names_out()
X_train_proc = pd.DataFrame(preprocessor_fitted.transform(X_train), columns=feature_names)

# Method 1: Feature importance from a Random Forest (easy to explain: how
# much each feature reduces impurity across all trees).
rf_probe = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")
rf_probe.fit(X_train_proc, y_train)
rf_importance = pd.Series(rf_probe.feature_importances_, index=feature_names).sort_values(ascending=False)
rf_importance_clean = rf_importance.rename(index=clean_feature_name)
print("\nTop 15 features by Random Forest importance:")
print(rf_importance_clean.head(15))

# Method 2: SelectKBest with ANOVA F-value, a simple univariate statistical
# test that scores each feature's relationship with the target.
K_BEST = 20
selector = SelectKBest(score_func=f_classif, k=K_BEST)
selector.fit(X_train_proc, y_train)
selected_mask = selector.get_support()
selected_features = feature_names[selected_mask]
print(f"\nTop {K_BEST} features selected by SelectKBest (ANOVA F-test):")
print([clean_feature_name(f) for f in selected_features])

overlap = set(rf_importance.head(K_BEST).index) & set(selected_features)
print(f"\nOverlap between RF top-{K_BEST} and SelectKBest top-{K_BEST}: {len(overlap)}/{K_BEST} features")
print("These consistently-important features are the ones most predictive of attrition.")

# ---------------------------------------------------------------------------
# STEP 5: TRAIN MULTIPLE MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: TRAIN MULTIPLE MODELS")
print("=" * 70)

# Each model gets its own pipeline: preprocess -> select top-K features -> classify.
# Wrapping everything in one Pipeline means .fit() only ever learns from
# X_train, and the same fitted steps are reused for X_test/predictions
# (no leakage, and it makes the final prediction function trivial in Step 8).
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
}

pipelines = {}
results = []

for name, estimator in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("select", SelectKBest(score_func=f_classif, k=K_BEST)),
            ("model", estimator),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        }
    )
    pipelines[name] = pipe

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\nModel comparison (test set):")
print(results_df)

# Because the target is imbalanced (~84% No / ~16% Yes), we pick the best
# model by F1-score (balances precision and recall) rather than accuracy,
# which would be misleadingly high for a model that just predicts "No" often.
best_model_name = results_df["F1-Score"].idxmax()
best_pipeline = pipelines[best_model_name]
print(f"\nBest performing model (highest F1-score): {best_model_name}")

# ---------------------------------------------------------------------------
# STEP 6: MODEL EVALUATION (best model)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"STEP 6: MODEL EVALUATION - {best_model_name}")
print("=" * 70)

y_pred_best = best_pipeline.predict(X_test)
y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred_best)
prec = precision_score(y_test, y_pred_best, zero_division=0)
rec = recall_score(y_test, y_pred_best, zero_division=0)
f1 = f1_score(y_test, y_pred_best, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba_best)

print(f"\nAccuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred_best, target_names=["No (Stay)", "Yes (Leave)"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Leave"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False
)
axes[0].set_title(f"Confusion Matrix - {best_model_name}")

RocCurveDisplay.from_predictions(y_test, y_proba_best, ax=axes[1], name=best_model_name)
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
axes[1].set_title("ROC Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nSaved evaluation plots to model_evaluation.png")

# ---------------------------------------------------------------------------
# STEP 7: FEATURE IMPORTANCE (best tree-based model)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: FEATURE IMPORTANCE")
print("=" * 70)

tree_based_models = {k: v for k, v in results_df.iterrows() if k in ("Decision Tree", "Random Forest")}
best_tree_name = max(tree_based_models, key=lambda k: tree_based_models[k]["F1-Score"])
best_tree_pipeline = pipelines[best_tree_name]
print(f"\nBest tree-based model: {best_tree_name}")

fitted_selector = best_tree_pipeline.named_steps["select"]
fitted_model = best_tree_pipeline.named_steps["model"]
all_feature_names = best_tree_pipeline.named_steps["preprocess"].get_feature_names_out()
selected_names = all_feature_names[fitted_selector.get_support()]

importances = pd.Series(fitted_model.feature_importances_, index=selected_names).sort_values(ascending=False)
importances = importances[importances > 1e-6].rename(index=clean_feature_name)  # drop features the tree never actually used
top_n = min(15, len(importances))
top_importances = importances.head(top_n)

print(f"\nTop {top_n} features driving attrition (from {best_tree_name}):")
print(top_importances)

plt.figure(figsize=(9, 7))
sns.barplot(x=top_importances.values, y=top_importances.index, hue=top_importances.index, palette="viridis", legend=False)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(f"Top {top_n} Features Influencing Employee Attrition ({best_tree_name})")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nSaved feature importance chart to feature_importance.png")

# ---------------------------------------------------------------------------
# STEP 8: PREDICTION FUNCTION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: PREDICTION FUNCTION")
print("=" * 70)


def predict_attrition(employee_dict, pipeline=best_pipeline):
    """
    Predict whether a single employee is likely to leave.

    Parameters
    ----------
    employee_dict : dict
        Raw employee attributes using the SAME column names as the
        original dataset (minus Attrition, EmployeeNumber, EmployeeCount,
        StandardHours, Over18 and JobLevel, which are not used by the model).
    pipeline : sklearn Pipeline
        A fitted (preprocess -> select -> model) pipeline. Defaults to the
        overall best model found in Step 5.

    Returns
    -------
    dict with the predicted label and probability of leaving.
    """
    employee_df = pd.DataFrame([employee_dict])
    prediction = pipeline.predict(employee_df)[0]
    probability_leave = pipeline.predict_proba(employee_df)[0, 1]

    label = "Likely to Leave" if prediction == 1 else "Likely to Stay"
    return {
        "prediction": label,
        "probability_of_leaving": round(float(probability_leave), 4),
    }


# Example: a sample employee profile (values follow the same columns as X)
sample_employee = {
    "Age": 29,
    "BusinessTravel": "Travel_Frequently",
    "DailyRate": 800,
    "Department": "Sales",
    "DistanceFromHome": 15,
    "Education": 3,
    "EducationField": "Marketing",
    "EnvironmentSatisfaction": 2,
    "Gender": "Male",
    "HourlyRate": 60,
    "JobInvolvement": 2,
    "JobRole": "Sales Executive",
    "JobSatisfaction": 1,
    "MaritalStatus": "Single",
    "MonthlyIncome": 2800,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 3,
    "OverTime": "Yes",
    "PercentSalaryHike": 12,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 2,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 5,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 2,
    "YearsAtCompany": 2,
    "YearsInCurrentRole": 2,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 1,
}

result = predict_attrition(sample_employee)
print(f"\nSample employee prediction: {result['prediction']}")
print(f"Probability of leaving: {result['probability_of_leaving']:.2%}")

# ---------------------------------------------------------------------------
# STEP 9: FINAL INSIGHTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9: FINAL INSIGHTS")
print("=" * 70)

print(f"""
Best performing model : {best_model_name}
  Accuracy  : {results_df.loc[best_model_name, 'Accuracy']:.4f}
  Precision : {results_df.loc[best_model_name, 'Precision']:.4f}
  Recall    : {results_df.loc[best_model_name, 'Recall']:.4f}
  F1-Score  : {results_df.loc[best_model_name, 'F1-Score']:.4f}
  ROC-AUC   : {roc_auc:.4f}

Top attrition factors (from {best_tree_name} feature importance):
{chr(10).join(f"  {i+1}. {name}" for i, name in enumerate(top_importances.head(8).index))}
""")

print("Full script finished successfully.")
