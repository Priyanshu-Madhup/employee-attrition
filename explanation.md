# Employee Attrition Prediction — Project Explanation

*A plain-language walkthrough of what this project does, why each step was
taken, and how to explain it to your teacher. Read this before your
presentation/viva.*

---

## 1. The Problem, in One Sentence

We built a model that looks at an employee's profile (age, salary,
overtime, satisfaction scores, tenure, etc.) and predicts whether they are
**likely to leave the company or likely to stay** — so HR can step in
*before* a valuable employee resigns, instead of finding out after the fact.

This is a **binary classification** problem: the output is one of two
classes, `Leave` (1) or `Stay` (0).

---

## 2. The Dataset, at a Glance

| Property | Value |
|---|---|
| Source file | `dataset.csv` (IBM HR Analytics Employee Attrition dataset) |
| Rows (employees) | 1,470 |
| Columns | 35 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Target column | `Attrition` (`Yes` / `No`) |
| Class balance | 1,233 "No" (83.9%) vs. 237 "Yes" (16.1%) — **imbalanced** |

**Why "imbalanced" matters:** if a lazy model just predicted "No" for
*every* employee, it would still be right 83.9% of the time. That's why we
can't judge our model on accuracy alone — a more careful set of metrics is
needed (explained in Section 5).

---

## 3. The Pipeline — What We Did and Why

Think of this as a factory line: raw data goes in one end, a trained model
comes out the other. Each station does one job.

### Step 1 — Load & Understand the Data
We opened `dataset.csv` and checked its shape, column types, missing
values, duplicates, and basic statistics. **Why:** you can't clean or
model data you don't understand — this is always the first step.

### Step 2 — Exploratory Data Analysis (EDA)
We plotted 9 charts comparing each factor against attrition (age,
department, job role, overtime, job satisfaction, income, tenure,
work-life balance). **Why:** visualizing the data reveals patterns before
we build any model, and gives us a sanity check later — if the model's
conclusions don't match what we see in the charts, something is wrong.

**What the charts showed:**
- Employees who work **overtime** leave at a much higher *rate* than those who don't.
- **Younger** employees (roughly 18–30) leave more than mid-career/senior staff.
- **Low job satisfaction** and **poor work-life balance** both line up with more attrition.
- Employees who leave tend to have **lower income** and **fewer years at the company**.
- **Sales Representatives** and **Laboratory Technicians** show higher attrition shares than roles like Manager or Research Director.

### Step 3 — Preprocessing (Cleaning the Data for the Model)
Models can't read raw data the way humans can — everything must become
numbers, on comparable scales, with no useless or duplicate information.

| Action | Columns | Why |
|---|---|---|
| Dropped | `EmployeeNumber` | Just an ID number, not a real trait — carries zero predictive meaning. |
| Dropped | `EmployeeCount`, `StandardHours`, `Over18` | Same value for *every* employee (constant) — a column that never changes can't help distinguish who leaves. |
| Dropped | `JobLevel` | Correlated 0.95 with `MonthlyIncome` — basically saying the same thing twice. Keeping both adds redundancy without adding information, so we kept the more interpretable one (`MonthlyIncome`). |
| Encoded | `Attrition` | `Yes → 1`, `No → 0` — models need numbers, not text. |
| One-hot encoded | 7 categorical columns (`Department`, `JobRole`, `OverTime`, etc.) | Turns each category into a 0/1 column so the model doesn't wrongly assume categories have an order (e.g. that "Sales" > "HR"). |
| Scaled | All numeric columns | Puts features like `MonthlyIncome` (thousands) and `Age` (tens) on the same footing, so no feature dominates just because its numbers are bigger. |
| Split | 80% train / 20% test | The model learns only from the training set and is judged on data it has never seen — this is how we test whether it actually *generalizes* rather than memorizes. |

**Avoiding data leakage:** the split happens *before* we fit the scaler/encoder,
and the scaler/encoder are only ever fit on the training data. If we
scaled using statistics from the full dataset (including test data), the
model would get a sneak peek at the test set, and our evaluation would be
overly optimistic — a classic beginner mistake we deliberately avoided.

### Step 4 — Feature Selection
We used two simple, easy-to-explain techniques to find out which columns
actually matter:
1. **Random Forest importance** — trains a forest of trees and measures how much each feature helps the trees make correct splits.
2. **SelectKBest (ANOVA F-test)** — a statistical test scoring each feature's individual relationship with the target.

Both pointed to the same handful of features (`MonthlyIncome`, `Age`,
`TotalWorkingYears`, `OverTime`, tenure columns, `StockOptionLevel`) — so
we kept the top 20 out of 42 encoded features for training, dropping the
noisy, low-signal ones.

### Step 5 — Train Multiple Models
We trained four different classifiers so we could compare them fairly:

| Model | One-line idea |
|---|---|
| **Logistic Regression** | Draws a (weighted) line that separates "Leave" from "Stay" based on a weighted sum of features. Simple, fast, interpretable. |
| **Decision Tree** | Asks a series of yes/no questions (e.g. "Overtime = Yes?") to arrive at a prediction. Easy to visualize, prone to overfitting alone. |
| **Random Forest** | Trains *many* decision trees on random subsets of data/features and averages their votes. Usually more accurate and stable than one tree. |
| **K-Nearest Neighbors (KNN)** | Looks at the K most similar employees in the training data and predicts based on what most of them did. |

All four (except KNN) were told to treat the minority class ("Leave") as
more important via `class_weight="balanced"`, to counteract the 84/16
imbalance.

### Step 6 — Evaluate the Models
Each model made predictions on the untouched 20% test set, and we scored
them on 4 metrics (explained below) rather than accuracy alone.

### Step 7 — Feature Importance
For the winning tree-based model, we plotted which features it relied on
most — this doubles as our "why does the model think someone will leave"
explanation.

### Step 8 — Prediction Function
A simple Python function, `predict_attrition(employee_dict)`, that takes
one employee's details and returns `"Likely to Leave"` or `"Likely to
Stay"` plus a probability (e.g. "78% chance of leaving"). This is the
practical, usable output of the whole project.

### Step 9 — Insights & Recommendations
We tied the model's findings back to real HR actions (Section 7 below).

---

## 4. Understanding the Metrics (in case you're asked)

Imagine the test set of 294 employees, 247 of whom stayed and 47 of whom
left. A model makes predictions, and we compare them to what actually
happened:

| Metric | Question it answers | Formula (plain language) |
|---|---|---|
| **Accuracy** | "Overall, how often was the model right?" | (Correct predictions) / (Total predictions) |
| **Precision** | "When the model said 'Leave', how often was it actually right?" | (Correctly predicted leavers) / (All predicted leavers) |
| **Recall** | "Of everyone who actually left, how many did the model catch?" | (Correctly predicted leavers) / (All actual leavers) |
| **F1-Score** | "A single balanced score combining precision and recall." | Harmonic mean of precision and recall |
| **ROC-AUC** | "How well does the model separate leavers from stayers overall, across all thresholds?" | Area under the ROC curve; 0.5 = random guessing, 1.0 = perfect |

**Why we picked F1 instead of accuracy to choose the "best" model:**
Because "Leave" is the rare class (16%), a model can score high accuracy
by mostly predicting "Stay" and still be nearly useless at catching real
leavers. F1 forces the model to be good at *both* catching leavers
(recall) and not crying wolf too often (precision).

---

## 5. Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.7653 | 0.3750 | 0.7021 | **0.4889** ⭐ |
| Decision Tree | 0.7381 | 0.3333 | 0.6383 | 0.4380 |
| Random Forest | 0.8401 | 0.5000 | 0.1489 | 0.2295 |
| K-Nearest Neighbors | 0.8537 | 0.8333 | 0.1064 | 0.1887 |

**Best model: Logistic Regression** — chosen for the highest F1-score.

**How to explain this to your teacher (this is the most important insight
in the whole project):** Random Forest and KNN *look* better at first
glance because their accuracy is higher (84–85%). But look at their
recall — Random Forest only catches 15% of actual leavers, and KNN only
11%. They're both quietly betting on "everyone stays" and winning that bet
most of the time, because 84% of employees really do stay. Logistic
Regression has lower accuracy (77%) but catches **70% of actual leavers**
— which is the entire point of the project. For an HR early-warning tool,
missing an at-risk employee (a false negative) is worse than occasionally
flagging someone who was fine (a false positive), so we optimized for
recall/F1, not accuracy.

**Best model's full scorecard:**
- Accuracy: 76.5%
- Precision: 37.5% (about 1 in 3 flagged employees actually leaves — a real limitation, discussed below)
- Recall: 70.2%
- F1-Score: 48.9%
- ROC-AUC: 0.78 (well above the 0.5 "random guess" line)

**A limitation worth mentioning if asked:** low precision (37.5%) means
the model over-flags — roughly 6 in 10 employees it labels "at risk"
actually stay. That's the trade-off we accepted in exchange for high
recall. In practice this is reasonable for a low-cost intervention (e.g. a
retention check-in conversation), but would need improvement before being
used for anything higher-stakes (e.g. automated decisions).

---

## 6. What Actually Drives Attrition (Feature Importance)

From the best tree-based model (Decision Tree), cross-checked against
SelectKBest and the EDA charts, in order of importance:

1. **Total Working Years** — less overall work experience → higher attrition risk.
2. **Monthly Income** — lower pay → higher attrition risk.
3. **OverTime** — working overtime is one of the single strongest predictors of leaving.
4. **Stock Option Level** — no stock options → more likely to leave.
5. **Environment Satisfaction** — lower satisfaction with the work environment → higher risk.
6. **Age** — younger employees leave more often.
7. **Years In Current Role** — less time in the current role → higher risk.
8. **Distance From Home** — living farther from work → slightly higher risk.

Notice these line up closely with what we already saw in the EDA charts
(Section 3) — that agreement is a good sign the model learned real
patterns, not noise.

---

## 7. Recommendations for Improving Retention

1. **Monitor and reduce overtime** — the strongest lever; redistribute workload or hire more staff in chronically over-worked teams.
2. **Review compensation for early-career / lower-tenure employees** — attrition concentrates in this group.
3. **Expand stock option / equity participation** — makes staying financially more attractive.
4. **Improve environment satisfaction and work-life balance**, especially in roles like Sales Representative and Laboratory Technician.
5. **Invest in retention programs for younger employees** — mentorship and career development in the first 1–2 years.
6. **Use `predict_attrition()` proactively** — periodically score current employees and flag high-risk ones for a retention conversation before they resign.

---

## 8. Likely Questions from the Teacher (and Suggested Answers)

**Q: Why didn't you just use accuracy to pick the best model?**
A: The dataset is imbalanced (84% stay / 16% leave), so a model that
always predicts "stay" would already hit 84% accuracy while being useless.
F1-score forces the model to actually find the leavers, not just guess
the majority class.

**Q: How did you avoid data leakage?**
A: We split the data into train/test *before* fitting the scaler and
one-hot encoder, and wrapped preprocessing + model into a single sklearn
`Pipeline` so the scaler/encoder are only ever fit on training data. The
test set is only ever *transformed*, never used to fit anything.

**Q: Why did you drop `JobLevel`?**
A: It correlates 0.95 with `MonthlyIncome` — nearly duplicate
information. Keeping both adds redundancy without new signal, so we kept
the more interpretable one.

**Q: Why is precision so low (37.5%) for your best model?**
A: We deliberately traded precision for recall using `class_weight
="balanced"`, because in an HR context, missing a real leaver (false
negative) is usually more costly than a false alarm (false positive) — a
false alarm just costs one extra retention conversation.

**Q: Could you improve the model further?**
A: Yes — options include hyperparameter tuning (e.g. `GridSearchCV`),
trying resampling techniques like SMOTE for the minority class, or trying
gradient boosting models (e.g. XGBoost). We kept the project to
scikit-learn's core, easily-explainable models as required by the
assignment scope.

**Q: How do you know the model isn't just overfitting?**
A: All metrics reported are computed on the **held-out test set** (the 20%
never seen during training), not on the training data. We also capped
Decision Tree depth (`max_depth=6`) specifically to reduce overfitting.

**Q: What does the ROC-AUC of 0.78 mean?**
A: It measures how well the model ranks employees by risk of leaving,
across every possible decision threshold. 0.5 means no better than random
guessing, 1.0 means perfect separation. 0.78 means the model has a good,
meaningfully-better-than-random ability to distinguish leavers from
stayers.

---

## 9. File Guide

| File | What it is |
|---|---|
| `dataset.csv` | Original, untouched dataset |
| `employee_attrition.ipynb` | Main notebook — run this to show the full project, already pre-executed with outputs |
| `employee_attrition.py` | Same project as a plain Python script (for quick terminal runs) |
| `README.md` | Condensed project summary + results table |
| `explanation.md` | This file — the "explain it to the teacher" guide |
| `requirements.txt` | Python packages needed to run the project |
| `eda_overview.png` | The 9-panel EDA chart |
| `correlation_heatmap.png` | Numeric feature correlation heatmap |
| `model_evaluation.png` | Confusion matrix + ROC curve for the best model |
| `feature_importance.png` | Top features driving attrition |

**To run it yourself:**
```bash
pip install -r requirements.txt
jupyter notebook employee_attrition.ipynb
```
