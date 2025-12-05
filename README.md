# Kyphosis Disease Classification (Decision Tree vs Random Forest)

This repository contains a machine learning workflow for predicting **post–operative kyphosis** (spinal deformity) in pediatric patients using classical tree-based models.  
The project walks through exploratory data analysis, preprocessing, model training, and evaluation, with a focus on comparing a **Decision Tree** and a **Random Forest** classifier.

The work is implemented in the notebook:

- `Kyphosis Disease Classification Modeling part 2.ipynb`

---

## 🩺 Problem Overview

**Kyphosis** is a spinal curvature condition that may or may not be present after corrective surgery.  
The goal of this project is to:

> Predict whether kyphosis is **present** or **absent** after surgery using pre-operative and intra-operative features.

This is a **binary classification** problem where:
- `Kyphosis = 0` → *Absent* (no kyphosis after surgery)  
- `Kyphosis = 1` → *Present* (kyphosis still present after surgery)

---

## 📊 Dataset

The project uses the classic **Kyphosis** dataset (often seen in R examples), loaded from:

- `kyphosis.csv`

### Main columns

- `Kyphosis` – Target variable (Present/Absent, later label-encoded to 1/0)
- `Age` – Age of the patient (in months)
- `Number` – Number of vertebrae involved in the operation
- `Start` – Starting vertebra of the operation

### Basic exploratory analysis performed

- `.head()`, `.tail()`, `.describe()` for basic inspection
- `df.info()` to check data types and missing values
- Class counts and percentages for **Kyphosis present vs absent**
- Count plot of the target:
  - Shows strong class imbalance (many more “Absent” than “Present” cases)

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook includes several visualizations to understand the data:

1. **Target distribution**
   - `sns.countplot(y='Kyphosis', data=df, hue='Kyphosis')`
   - Used to confirm class imbalance.

2. **Correlation matrix**
   - `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')`
   - Shows relationships between `Age`, `Number`, `Start`, and `Kyphosis`.

3. **Pairplot**
   - `sns.pairplot(df, hue='Kyphosis')`
   - Visual inspection of how classes separate across feature pairs.

These plots help assess whether features have reasonable signal for classification and whether there is potential collinearity.

---

## 🧹 Data Preprocessing

Key preprocessing steps:

1. **Label Encoding**
   - `Kyphosis` is originally categorical (“present/absent”).
   - Encoded using `LabelEncoder` so that:
     - 0 → Absent  
     - 1 → Present  

2. **Handling Missing Values**
   - `SimpleImputer(strategy='median')` is applied to numeric features.
   - Ensures the model can handle any missing entries.

3. **Feature Scaling**
   - `StandardScaler` is applied to the imputed feature matrix.
   - Important for tree ensembles when distance or thresholding can be sensitive to feature scale.

4. **Train–Test Split**
   - `train_test_split(X, y, test_size=0.2, stratify=y)`
   - Stratification preserves class proportions in both train and test sets.

---

## 🤖 Models

Two classical tree-based models are implemented and compared:

### 1. Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(class_weight='balanced')
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
