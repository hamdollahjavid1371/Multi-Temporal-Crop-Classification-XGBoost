# -*- coding: utf-8 -*-

"""
XGBoost Crop Classification
Multi-Temporal Sentinel-1 and Sentinel-2 Features

Input:
    train_50_features.xls
    test_50_features.xls

Features:
    16 Sentinel-2 features
    24 Sentinel-1 features
    10 temporal features

Total:
    50 features

Classes:
    Barley
    Melon
    Tomato
    Wheat
    Other
"""

# ============================================================
# 1. Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. Read Train and Test Data
# ============================================================

train = pd.read_excel(
    "train_50_features.xls",
    engine="xlrd"
)

test = pd.read_excel(
    "test_50_features.xls",
    engine="xlrd"
)


print("Train shape:", train.shape)
print("Test shape :", test.shape)


# ============================================================
# 3. Correct Class Names
# ============================================================

train["Name"] = (
    train["Name"]
    .astype(str)
    .str.strip()
    .replace({
        "Barey": "Barley"
    })
)

test["Name"] = (
    test["Name"]
    .astype(str)
    .str.strip()
    .replace({
        "Barey": "Barley"
    })
)


# ============================================================
# 4. Define Months
# ============================================================

months = [
    "Mehr",
    "Aban",
    "Azar",
    "Dey",
    "Bahman",
    "Esfand",
    "Farvardin",
    "Ordibehesht"
]


# ============================================================
# 5. Sentinel-2 Features
# ============================================================

s2_features = []

for month in months:

    s2_features.extend([
        f"NDVI_{month}",
        f"NDRE_{month}"
    ])


# ============================================================
# 6. Sentinel-1 Features
# ============================================================

s1_features = []

for month in months:

    s1_features.extend([
        f"VV_{month}",
        f"VH_{month}",
        f"VVVH_{month}"
    ])


# ============================================================
# 7. Temporal Features
# ============================================================

temporal_features = [

    "NDRE_Range",

    "NDVI_Diff_Mehr_Aban",

    "NDVI_Range",

    "NDRE_Diff_Bahman_Esfand",

    "NDVI_Diff_Farvardin_Ordibehesht",

    "NDRE_Diff_Mehr_Aban",

    "NDVI_Diff_Bahman_Esfand",

    "NDRE_Diff_Farvardin_Ordibehesht",

    "NDVI_Diff_Aban_Azar",

    "VV_Diff_Esfand_Farvardin"
]


# ============================================================
# 8. Final 50 Features
# ============================================================

features_50 = (
    s2_features
    +
    s1_features
    +
    temporal_features
)


print("\n========================================")
print("FEATURE INFORMATION")
print("========================================")

print(
    "Sentinel-2 features :",
    len(s2_features)
)

print(
    "Sentinel-1 features :",
    len(s1_features)
)

print(
    "Temporal features   :",
    len(temporal_features)
)

print(
    "Total features      :",
    len(features_50)
)


assert len(features_50) == 50


# ============================================================
# 9. Check Features
# ============================================================

missing_train = [
    feature
    for feature in features_50
    if feature not in train.columns
]

missing_test = [
    feature
    for feature in features_50
    if feature not in test.columns
]


if missing_train:

    print("\nMissing features in TRAIN:")
    print(missing_train)


if missing_test:

    print("\nMissing features in TEST:")
    print(missing_test)


assert len(missing_train) == 0
assert len(missing_test) == 0


# ============================================================
# 10. Define Classes
# ============================================================

class_order = [
    "Barley",
    "Melon",
    "Tomato",
    "Wheat",
    "Other"
]


class_to_id = {

    "Barley": 0,

    "Melon": 1,

    "Tomato": 2,

    "Wheat": 3,

    "Other": 4
}


# ============================================================
# 11. Prepare X and Y
# ============================================================

X_train = train[
    features_50
].copy()

X_test = test[
    features_50
].copy()


y_train = (
    train["Name"]
    .map(class_to_id)
)

y_test = (
    test["Name"]
    .map(class_to_id)
)


# ============================================================
# 12. Check Unknown Classes
# ============================================================

assert y_train.notnull().all(), \
    "Unknown class found in training data."

assert y_test.notnull().all(), \
    "Unknown class found in test data."


y_train = y_train.astype(int)
y_test = y_test.astype(int)


# ============================================================
# 13. Check Missing Values
# ============================================================

print(
    "\nMissing values in X_train:",
    X_train.isnull().sum().sum()
)

print(
    "Missing values in X_test :",
    X_test.isnull().sum().sum()
)


assert X_train.isnull().sum().sum() == 0
assert X_test.isnull().sum().sum() == 0


# ============================================================
# 14. XGBoost Model
# ============================================================

model = XGBClassifier(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    colsample_bytree=0.8,

    objective="multi:softprob",

    eval_metric="mlogloss",

    num_class=5,

    n_jobs=2,

    random_state=42
)


# ============================================================
# 15. Train Model
# ============================================================

print("\n========================================")
print("TRAINING XGBOOST")
print("========================================")


model.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# 16. Prediction
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 17. Overall Accuracy
# ============================================================

oa = accuracy_score(
    y_test,
    y_pred
)


print("\n========================================")
print("FINAL RESULT")
print("========================================")

print(
    "Train samples      :",
    len(y_train)
)

print(
    "Test samples       :",
    len(y_test)
)

print(
    "Number of features :",
    len(features_50)
)

print(
    f"Overall Accuracy   : {oa * 100:.2f}%"
)


# ============================================================
# 18. Confusion Matrix
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=[
        0,
        1,
        2,
        3,
        4
    ]
)


cm_df = pd.DataFrame(

    cm,

    index=class_order,

    columns=class_order
)


print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm_df)


# ============================================================
# 19. Classification Report
# ============================================================

report = classification_report(

    y_test,

    y_pred,

    labels=[
        0,
        1,
        2,
        3,
        4
    ],

    target_names=class_order,

    output_dict=True,

    zero_division=0
)


report_df = pd.DataFrame(
    report
).T


print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(report_df)


# ============================================================
# 20. Feature Importance
# ============================================================

importance_df = pd.DataFrame({

    "Feature":
        features_50,

    "Importance":
        model.feature_importances_

})


importance_df = (

    importance_df

    .sort_values(
        "Importance",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


importance_df["Rank"] = (
    importance_df.index + 1
)


importance_df = importance_df[
    [
        "Rank",
        "Feature",
        "Importance"
    ]
]


# ============================================================
# 21. Print Feature Importance
# ============================================================

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 22. Plot Feature Importance
# ============================================================

plt.figure(
    figsize=(20, 10)
)


plt.bar(

    importance_df["Feature"],

    importance_df["Importance"]

)


plt.xlabel(
    "Features",
    fontsize=14
)

plt.ylabel(
    "XGBoost Feature Importance",
    fontsize=14
)

plt.title(
    "Feature Importance of 50 Features",
    fontsize=16
)


plt.xticks(
    rotation=90,
    fontsize=8
)

plt.yticks(
    fontsize=10
)


plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    "XGBoost_Feature_Importance_50.png",

    dpi=600,

    bbox_inches="tight"

)


plt.show()


# ============================================================
# 23. Save Feature Importance Table
# ============================================================

importance_df.to_excel(

    "Feature_Importance_50.xlsx",

    index=False

)


# ============================================================
# 24. Save Confusion Matrix
# ============================================================

cm_df.to_excel(

    "Confusion_Matrix.xlsx"

)


# ============================================================
# 25. Save Classification Report
# ============================================================

report_df.to_excel(

    "Classification_Report.xlsx"

)


# ============================================================
# 26. Final Summary
# ============================================================

print("\n========================================")
print("PROCESS COMPLETED")
print("========================================")

print(
    f"Overall Accuracy: {oa * 100:.2f}%"
)

print(
    "50-feature XGBoost classification completed."
)

print(
    "Results saved successfully."
)
