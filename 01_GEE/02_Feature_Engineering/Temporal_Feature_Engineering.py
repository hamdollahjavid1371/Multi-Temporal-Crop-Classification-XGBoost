# -*- coding: utf-8 -*-

"""
Temporal Feature Engineering
Multi-Temporal Crop Classification

Input:
    Train/Test Excel files containing
    40 Sentinel-1 and Sentinel-2 features

Output:
    10 temporal features
    50-feature dataset
"""

import pandas as pd


# ============================================================
# 1. Read Train and Test Data
# ============================================================

train = pd.read_excel(
    "train.xls",
    engine="xlrd"
)

test = pd.read_excel(
    "test.xls",
    engine="xlrd"
)


print("Train shape:", train.shape)
print("Test shape :", test.shape)


# ============================================================
# 2. Correct Class Names
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
# 3. Months
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
# 4. Sentinel-2 Features
# NDVI + NDRE
# 8 months × 2 = 16 features
# ============================================================

s2_features = []

for month in months:

    s2_features.extend([
        f"NDVI_{month}",
        f"NDRE_{month}"
    ])


# ============================================================
# 5. Sentinel-1 Features
# VV + VH + VVVH
# 8 months × 3 = 24 features
# ============================================================

s1_features = []

for month in months:

    s1_features.extend([
        f"VV_{month}",
        f"VH_{month}",
        f"VVVH_{month}"
    ])


# ============================================================
# 6. Temporal Feature Engineering
# ============================================================

def add_temporal_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # NDVI and NDRE monthly columns
    # --------------------------------------------------------

    ndvi_cols = [
        f"NDVI_{m}"
        for m in months
    ]

    ndre_cols = [
        f"NDRE_{m}"
        for m in months
    ]

    # --------------------------------------------------------
    # 1. NDRE Range
    # Maximum NDRE - Minimum NDRE
    # --------------------------------------------------------

    data["NDRE_Range"] = (
        data[ndre_cols].max(axis=1)
        -
        data[ndre_cols].min(axis=1)
    )

    # --------------------------------------------------------
    # 2. NDVI Difference: Mehr → Aban
    # --------------------------------------------------------

    data["NDVI_Diff_Mehr_Aban"] = (
        data["NDVI_Aban"]
        -
        data["NDVI_Mehr"]
    )

    # --------------------------------------------------------
    # 3. NDVI Range
    # Maximum NDVI - Minimum NDVI
    # --------------------------------------------------------

    data["NDVI_Range"] = (
        data[ndvi_cols].max(axis=1)
        -
        data[ndvi_cols].min(axis=1)
    )

    # --------------------------------------------------------
    # 4. NDRE Difference: Bahman → Esfand
    # --------------------------------------------------------

    data["NDRE_Diff_Bahman_Esfand"] = (
        data["NDRE_Esfand"]
        -
        data["NDRE_Bahman"]
    )

    # --------------------------------------------------------
    # 5. NDVI Difference: Farvardin → Ordibehesht
    # --------------------------------------------------------

    data["NDVI_Diff_Farvardin_Ordibehesht"] = (
        data["NDVI_Ordibehesht"]
        -
        data["NDVI_Farvardin"]
    )

    # --------------------------------------------------------
    # 6. NDRE Difference: Mehr → Aban
    # --------------------------------------------------------

    data["NDRE_Diff_Mehr_Aban"] = (
        data["NDRE_Aban"]
        -
        data["NDRE_Mehr"]
    )

    # --------------------------------------------------------
    # 7. NDVI Difference: Bahman → Esfand
    # --------------------------------------------------------

    data["NDVI_Diff_Bahman_Esfand"] = (
        data["NDVI_Esfand"]
        -
        data["NDVI_Bahman"]
    )

    # --------------------------------------------------------
    # 8. NDRE Difference: Farvardin → Ordibehesht
    # --------------------------------------------------------

    data["NDRE_Diff_Farvardin_Ordibehesht"] = (
        data["NDRE_Ordibehesht"]
        -
        data["NDRE_Farvardin"]
    )

    # --------------------------------------------------------
    # 9. NDVI Difference: Aban → Azar
    # --------------------------------------------------------

    data["NDVI_Diff_Aban_Azar"] = (
        data["NDVI_Azar"]
        -
        data["NDVI_Aban"]
    )

    # --------------------------------------------------------
    # 10. VV Difference: Esfand → Farvardin
    # --------------------------------------------------------

    data["VV_Diff_Esfand_Farvardin"] = (
        data["VV_Farvardin"]
        -
        data["VV_Esfand"]
    )

    return data


# ============================================================
# 7. Apply Temporal Feature Engineering
# ============================================================

train_fe = add_temporal_features(train)

test_fe = add_temporal_features(test)


# ============================================================
# 8. Define the 10 Temporal Features
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
# 9. Create Final 50 Features
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
    "TOTAL FEATURES      :",
    len(features_50)
)


# ============================================================
# 10. Check Number of Features
# ============================================================

assert len(features_50) == 50


# ============================================================
# 11. Check Missing Features
# ============================================================

missing_train = [
    f for f in features_50
    if f not in train_fe.columns
]

missing_test = [
    f for f in features_50
    if f not in test_fe.columns
]


if missing_train:

    print("\nMissing features in TRAIN:")
    print(missing_train)


if missing_test:

    print("\nMissing features in TEST:")
    print(missing_test)


assert len(missing_train) == 0
assert len(missing_test) == 0


print("\nAll 50 features are available.")


# ============================================================
# 12. Save Processed Datasets
# ============================================================

train_fe.to_excel(
    "train_50_features.xls",
    index=False
)

test_fe.to_excel(
    "test_50_features.xls",
    index=False
)


print("\nProcessed datasets saved.")

print(
    "Train:",
    "train_50_features.xls"
)

print(
    "Test:",
    "test_50_features.xls"
)
