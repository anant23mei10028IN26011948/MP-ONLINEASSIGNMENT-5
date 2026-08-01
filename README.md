# Employee Attrition Prediction using Decision Tree and Random Forest Classification

**Name:** ANANT YADAV

**Reg No:** 23MEI10028

**Email:** anant.23mei10028@vitbhopal.ac.in

**Application no:** IN26011948

## 🎯 Objective
Build and compare Decision Tree and Random Forest classification models to predict
whether an employee is likely to leave the organization, based on demographic,
professional, and work-related attributes.

## 📊 Dataset
**IBM HR Analytics Employee Attrition & Performance Dataset**
Kaggle link: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

- 1,470 records, 35 columns
- Numerical features: Age, DailyRate, DistanceFromHome, MonthlyIncome, TotalWorkingYears,
  YearsAtCompany, and other job/satisfaction-related numeric columns
- Categorical features: BusinessTravel, Department, EducationField, Gender, JobRole,
  MaritalStatus, OverTime (Over18 is constant and dropped)
- Target variable: `Attrition` (Yes / No) — imbalanced, with ~16% of employees leaving
- Dropped columns: `EmployeeCount`, `StandardHours`, `Over18` (constant for all rows) and
  `EmployeeNumber` (non-predictive ID)

## 🛠 Libraries Used
- `pandas` – data loading and manipulation
- `numpy` – numerical operations
- `matplotlib` / `seaborn` – data visualization
- `scikit-learn` – train/test split, `LabelEncoder`, `DecisionTreeClassifier`,
  `RandomForestClassifier`, evaluation metrics

## 🧭 Methodology
1. **Data Understanding** – Loaded the dataset with Pandas, inspected the first five
   records, identified numerical vs. categorical features and the `Attrition` target, and
   reviewed dataset info, summary statistics, and class balance.
2. **Data Preprocessing**
   - Checked for missing values (none found).
   - Removed constant/ID columns (`EmployeeCount`, `StandardHours`, `Over18`,
     `EmployeeNumber`).
   - Encoded categorical variables: label encoding for `Attrition`, `OverTime`, `Gender`;
     one-hot encoding for `BusinessTravel`, `Department`, `EducationField`, `JobRole`,
     `MaritalStatus`.
   - Split the data into 80% training and 20% testing sets (stratified, `random_state=42`).
3. **Model Development**
   - **Model 1:** `DecisionTreeClassifier` trained on the training set.
   - **Model 2:** `RandomForestClassifier` with **100 estimators** trained on the same
     training set.
   - Both models predicted attrition on the same test set for a fair comparison.
4. **Model Evaluation & Comparison** – Evaluated both models using Accuracy, Precision,
   Recall, and F1-Score, generated a confusion matrix for each, plotted Random Forest
   feature importances, and visualized a side-by-side metric comparison.

## 📈 Results

| Metric | Decision Tree | Random Forest |
|---|---|---|
| Accuracy  | ≈ 0.759 | ≈ 0.827 |
| Precision (Yes) | ≈ 0.293 | ≈ 0.300 |
| Recall (Yes)    | ≈ 0.362 | ≈ 0.064 |
| F1-Score (Yes)  | ≈ 0.324 | ≈ 0.105 |

*(Metrics for the "Yes"/attrition class, the class of real business interest.)*

## 🔍 Model Comparison
On **overall accuracy**, Random Forest clearly outperforms the Decision Tree (≈83% vs.
≈76%), consistent with the general expectation that ensembling many trees reduces
overfitting and variance. However, a closer look at the classification report reveals a
more nuanced picture: **the Decision Tree actually achieves noticeably better recall and
F1-score on the minority "attrition = Yes" class**. Random Forest's accuracy advantage
comes largely from being very good at predicting the majority "No" class — a direct effect
of the dataset's class imbalance (~16% attrition rate) biasing the ensemble toward the
majority outcome. In other words, if the real business goal is *catching as many at-risk
employees as possible*, accuracy alone is a misleading metric here — the "smarter"-looking
high-accuracy model can actually underperform on the class that matters most. Techniques
like class weighting (`class_weight='balanced'`), resampling (SMOTE/undersampling), or
adjusting the classification threshold would likely be needed to make Random Forest
genuinely useful for this specific business objective.

**Key observations:**
- Random Forest wins on accuracy; Decision Tree wins on recall/F1 for the attrition class —
  the two models are not simply "better vs. worse," they trade off differently under class
  imbalance.
- The feature importance plot shows `MonthlyIncome`, `Age`, `TotalWorkingYears`,
  `DailyRate`, `OverTime`, and `YearsAtCompany` among the most influential predictors,
  aligning with common HR intuition about compensation, tenure, and workload.
- Both models still miss a substantial share of true attrition cases, showing there's room
  for further tuning (class balancing, hyperparameter tuning, threshold adjustment) before
  either model would be reliable for real HR decision-making.

## ✅ Conclusion
In this project, both a Decision Tree and a Random Forest (100 estimators) classifier were
trained to predict employee attrition using demographic, professional, and work-related
attributes. On overall accuracy, Random Forest performed better (≈82.7% vs. ≈75.9% for the
Decision Tree), consistent with the general expectation that ensembling many trees reduces
the variance and overfitting a single Decision Tree is prone to. However, a closer look at
the classification report reveals a more nuanced picture: the Decision Tree actually
achieved better recall and F1-score on the minority "attrition = Yes" class, meaning it
caught more of the employees who actually left, while Random Forest's accuracy advantage
came mostly from being very good at predicting the majority "No" class — a direct effect of
the dataset's class imbalance (~16% attrition rate). This is a key reminder that **Random
Forest often outperforms Decision Trees** in general because it averages predictions across
many trees trained on bootstrapped samples and random feature subsets, reducing variance and
improving generalization — but this benefit is not guaranteed on every metric, especially
with imbalanced data. A core **limitation of Decision Trees** is their tendency to overfit
and their instability (small data changes can produce very different trees), while a core
**limitation of Random Forest** is reduced interpretability and a tendency to be biased
toward the majority class in imbalanced problems like this one, along with higher
computational cost. For this business problem, further work such as class weighting or
resampling would likely be needed before deploying either model.
