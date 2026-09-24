# ============================================================
# XGBoost Multi-Temporal Crop Classification
# Sentinel-1 + Sentinel-2 + Temporal Features
#
# Input:
#   - train.xls  : 70% training samples
#   - test.xls   : 30% testing samples
#   - 40-band Sentinel-1/Sentinel-2 GeoTIFF
#
# Features:
#   - 40 original multi-temporal features
#   - 10 temporal features
#   - 50 final features
#
# Classes:
#   Barley, Melon, Tomato, Wheat, Other
# ============================================================


# ============================================================
# 1. Import Libraries
# ============================================================

import os
import numpy as np
import pandas as pd
import rasterio

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# 2. Input / Output Paths
# ============================================================

# ------------------------------------------------------------
# Training and testing Excel files
# These files are already divided into 70% training
# and 30% testing samples.
# ------------------------------------------------------------

TRAIN_PATH = "/content/train.xls"
TEST_PATH = "/content/test.xls"


# ------------------------------------------------------------
# 40-band multi-temporal GeoTIFF generated in GEE
# ------------------------------------------------------------

RASTER_PATH = (
    "/content/drive/MyDrive/GEE_Stack/"
    "kaki_Stack_40Bands.tif"
)


# ------------------------------------------------------------
# Output classified raster
# ------------------------------------------------------------

OUTPUT_RASTER = (
    "/content/drive/MyDrive/GEE_Stack/"
    "kaki_XGBoost_Classification_50Features.tif"
)


# ============================================================
# 3. Read Training and Testing Data
# ============================================================

print("=" * 70)
print("Reading training and testing data...")
print("=" * 70)

train = pd.read_excel(
    TRAIN_PATH,
    engine="xlrd"
)

test = pd.read_excel(
    TEST_PATH,
    engine="xlrd"
)


print("Training samples:", len(train))
print("Testing samples :", len(test))


# ============================================================
# 4. Normalize Class Names
# ============================================================

# Correct possible typo in the original Excel files

train["Name"] = (
    train["Name"]
    .astype(str)
    .str.strip()
    .replace({"Barey": "Barley"})
)

test["Name"] = (
    test["Name"]
    .astype(str)
    .str.strip()
    .replace({"Barey": "Barley"})
)


# ============================================================
# 5. Define Months
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
# 6. Define the Original 40 Features
# ============================================================

# Sentinel-2 features
s2_features = [
    f"NDVI_{m}" for m in months
] + [
    f"NDRE_{m}" for m in months
]


# Sentinel-1 features
s1_features = [
    f"VV_{m}" for m in months
] + [
    f"VH_{m}" for m in months
] + [
    f"VVVH_{m}" for m in months
]


# All 40 original features
features_40 = s2_features + s1_features


print("\nNumber of original features:", len(features_40))


# ============================================================
# 7. Check Required Columns
# ============================================================

required_columns = ["Name"] + features_40

missing_train = [
    c for c in required_columns
    if c not in train.columns
]

missing_test = [
    c for c in required_columns
    if c not in test.columns
]


if missing_train:
    raise ValueError(
        "Missing columns in train.xls:\n"
        + "\n".join(missing_train)
    )


if missing_test:
    raise ValueError(
        "Missing columns in test.xls:\n"
        + "\n".join(missing_test)
    )


# ============================================================
# 8. Temporal Feature Engineering
# ============================================================

def create_temporal_features(df):
    """
    Create the 10 temporal features used in the study.
    """

    df = df.copy()

    # --------------------------------------------------------
    # NDRE range
    # --------------------------------------------------------

    df["NDRE_Range"] = (
        df[[f"NDRE_{m}" for m in months]].max(axis=1)
        -
        df[[f"NDRE_{m}" for m in months]].min(axis=1)
    )


    # --------------------------------------------------------
    # NDVI difference: Aban - Mehr
    # --------------------------------------------------------

    df["NDVI_Diff_Mehr_Aban"] = (
        df["NDVI_Aban"]
        -
        df["NDVI_Mehr"]
    )


    # --------------------------------------------------------
    # NDVI range
    # --------------------------------------------------------

    df["NDVI_Range"] = (
        df[[f"NDVI_{m}" for m in months]].max(axis=1)
        -
        df[[f"NDVI_{m}" for m in months]].min(axis=1)
    )


    # --------------------------------------------------------
    # NDRE difference: Esfand - Bahman
    # --------------------------------------------------------

    df["NDRE_Diff_Bahman_Esfand"] = (
        df["NDRE_Esfand"]
        -
        df["NDRE_Bahman"]
    )


    # --------------------------------------------------------
    # NDVI difference: Ordibehesht - Farvardin
    # --------------------------------------------------------

    df["NDVI_Diff_Farvardin_Ordibehesht"] = (
        df["NDVI_Ordibehesht"]
        -
        df["NDVI_Farvardin"]
    )


    # --------------------------------------------------------
    # NDRE difference: Aban - Mehr
    # --------------------------------------------------------

    df["NDRE_Diff_Mehr_Aban"] = (
        df["NDRE_Aban"]
        -
        df["NDRE_Mehr"]
    )


    # --------------------------------------------------------
    # NDVI difference: Esfand - Bahman
    # --------------------------------------------------------

    df["NDVI_Diff_Bahman_Esfand"] = (
        df["NDVI_Esfand"]
        -
        df["NDVI_Bahman"]
    )


    # --------------------------------------------------------
    # NDRE difference: Ordibehesht - Farvardin
    # --------------------------------------------------------

    df["NDRE_Diff_Farvardin_Ordibehesht"] = (
        df["NDRE_Ordibehesht"]
        -
        df["NDRE_Farvardin"]
    )


    # --------------------------------------------------------
    # NDVI difference: Azar - Aban
    # --------------------------------------------------------

    df["NDVI_Diff_Aban_Azar"] = (
        df["NDVI_Azar"]
        -
        df["NDVI_Aban"]
    )


    # --------------------------------------------------------
    # VV difference: Farvardin - Esfand
    # --------------------------------------------------------

    df["VV_Diff_Esfand_Farvardin"] = (
        df["VV_Farvardin"]
        -
        df["VV_Esfand"]
    )


    return df


# ============================================================
# 9. Create the 10 Temporal Features
# ============================================================

train_fe = create_temporal_features(train)
test_fe = create_temporal_features(test)


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


print(
    "\nNumber of temporal features:",
    len(temporal_features)
)


# ============================================================
# 10. Create the Final 50 Features
# ============================================================

features_50 = (
    features_40
    +
    temporal_features
)


print(
    "Number of final features:",
    len(features_50)
)


if len(features_50) != 50:
    raise ValueError(
        f"Expected 50 features, "
        f"but {len(features_50)} were created."
    )


# ============================================================
# 11. Check Missing Values
# ============================================================

print("\nChecking missing values...")

missing_train_values = (
    train_fe[features_50]
    .isnull()
    .sum()
    .sum()
)

missing_test_values = (
    test_fe[features_50]
    .isnull()
    .sum()
    .sum()
)


print(
    "Missing values in training:",
    missing_train_values
)

print(
    "Missing values in testing :",
    missing_test_values
)


if missing_train_values > 0:
    raise ValueError(
        "Training data contains missing values."
    )


if missing_test_values > 0:
    raise ValueError(
        "Testing data contains missing values."
    )


# ============================================================
# 12. Define Crop Classes
# ============================================================

class_order = [
    "Barley",
    "Melon",
    "Other",
    "Tomato",
    "Wheat"
]


class_mapping = {
    "Barley": 0,
    "Melon": 1,
    "Other": 2,
    "Tomato": 3,
    "Wheat": 4
}


# ============================================================
# 13. Encode Classes
# ============================================================

unknown_train = set(train_fe["Name"]) - set(class_mapping)
unknown_test = set(test_fe["Name"]) - set(class_mapping)


if unknown_train:
    raise ValueError(
        f"Unknown classes in training data: "
        f"{unknown_train}"
    )


if unknown_test:
    raise ValueError(
        f"Unknown classes in testing data: "
        f"{unknown_test}"
    )


train_fe["Class"] = (
    train_fe["Name"]
    .map(class_mapping)
)

test_fe["Class"] = (
    test_fe["Name"]
    .map(class_mapping)
)


# ============================================================
# 14. Prepare X and y
# ============================================================

X_train = train_fe[features_50].copy()
y_train = train_fe["Class"].astype(int)


X_test = test_fe[features_50].copy()
y_test = test_fe["Class"].astype(int)


print("\nFinal training shape:", X_train.shape)
print("Final testing shape :", X_test.shape)


# ============================================================
# 15. XGBoost Model Parameters
# ============================================================

xgb_params = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 5,
    "n_jobs": 2,
    "random_state": 42
}


# ============================================================
# 16. Function for XGBoost Evaluation
# ============================================================

def evaluate_feature_set(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    title
):

    model = XGBClassifier(
        **xgb_params
    )

    model.fit(
        X_train[feature_names],
        y_train
    )

    predictions = model.predict(
        X_test[feature_names]
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"{title:<30} "
        f"Accuracy = {accuracy * 100:.2f}%"
    )

    return model, predictions, accuracy


# ============================================================
# 17. Compare Feature Sets
# ============================================================

print("\n")
print("=" * 70)
print("Feature-set comparison")
print("=" * 70)


feature_sets = {

    "Sentinel-2": s2_features,

    "Sentinel-1": s1_features,

    "Sentinel-1 + Sentinel-2":
        features_40,

    "S1 + S2 + Temporal":
        features_50
}


results = []


for name, feature_list in feature_sets.items():

    model_temp, pred_temp, acc_temp = (
        evaluate_feature_set(
            X_train,
            y_train,
            X_test,
            y_test,
            feature_list,
            name
        )
    )

    results.append({
        "Feature Set": name,
        "Number of Features": len(feature_list),
        "Accuracy (%)": acc_temp * 100
    })


results_df = pd.DataFrame(results)


print("\n")
print(results_df.to_string(index=False))


# ============================================================
# 18. Final 50-Feature XGBoost Model
# ============================================================

print("\n")
print("=" * 70)
print("Training final 50-feature XGBoost model")
print("=" * 70)


final_model = XGBClassifier(
    **xgb_params
)


final_model.fit(
    X_train[features_50],
    y_train
)


# ============================================================
# 19. Final Test Prediction
# ============================================================

y_pred = final_model.predict(
    X_test[features_50]
)


final_accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\nFinal accuracy "
    f"(50 features): "
    f"{final_accuracy * 100:.2f}%"
)


# ============================================================
# 20. Classification Report
# ============================================================

print("\n")
print("=" * 70)
print("Classification Report")
print("=" * 70)


print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_order,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 21. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)


cm_df = pd.DataFrame(
    cm,
    index=class_order,
    columns=class_order
)


print("\n")
print("=" * 70)
print("Confusion Matrix")
print("=" * 70)

print(cm_df)


# ============================================================
# 22. Feature Importance
# ============================================================

importance = pd.DataFrame({
    "Feature": features_50,
    "Importance": final_model.feature_importances_
})


importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\n")
print("=" * 70)
print("Top 20 Important Features")
print("=" * 70)

print(
    importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# 23. Plot Feature Importance
# ============================================================

plt.figure(
    figsize=(10, 12)
)

plt.barh(
    importance["Feature"].iloc[::-1],
    importance["Importance"].iloc[::-1]
)

plt.xlabel(
    "XGBoost Feature Importance",
    fontsize=12
)

plt.ylabel(
    "Feature",
    fontsize=12
)

plt.title(
    "Feature Importance of 50 Features",
    fontsize=14
)

plt.tight_layout()

plt.show()


# ============================================================
# 24. Raster Classification
# ============================================================

print("\n")
print("=" * 70)
print("Starting raster classification")
print("=" * 70)


if not os.path.exists(RASTER_PATH):

    raise FileNotFoundError(
        f"Raster file not found:\n{RASTER_PATH}"
    )


with rasterio.open(RASTER_PATH) as src:

    raster = src.read()

    profile = src.profile.copy()

    transform = src.transform
    crs = src.crs

    height = src.height
    width = src.width

    nodata_value = src.nodata


print(
    "\nRaster dimensions:",
    raster.shape
)


# ============================================================
# 25. Check Raster Band Number
# ============================================================

if raster.shape[0] != 40:

    raise ValueError(
        "The input raster must contain "
        "exactly 40 bands."
    )


# ============================================================
# 26. Convert Raster to Feature Matrix
# ============================================================

# GEE band order:
#
# For every month:
#
# NDVI
# NDRE
# VV
# VH
# VVVH
#
# Therefore:
#
# 8 months × 5 features = 40 bands


raster_float = raster.astype(
    np.float32
)


# ------------------------------------------------------------
# Identify valid pixels
# ------------------------------------------------------------

valid_mask = np.all(
    np.isfinite(raster_float),
    axis=0
)


if nodata_value is not None:

    valid_mask &= np.all(
        raster != nodata_value,
        axis=0
    )


valid_pixels = (
    raster_float[:, valid_mask]
    .T
)


print(
    "Number of valid pixels:",
    valid_pixels.shape[0]
)


# ============================================================
# 27. Create DataFrame for Raster Features
# ============================================================

raster_df = pd.DataFrame(
    valid_pixels,
    columns=features_40
)


# ============================================================
# 28. Create Temporal Raster Features
# ============================================================

raster_df = create_temporal_features(
    raster_df
)


# ============================================================
# 29. Select Final 50 Raster Features
# ============================================================

X_raster = raster_df[
    features_50
]


print(
    "Raster feature matrix:",
    X_raster.shape
)


if X_raster.shape[1] != 50:

    raise ValueError(
        "Raster feature matrix does not "
        "contain 50 features."
    )


# ============================================================
# 30. Predict Crop Classes
# ============================================================

print(
    "\nPredicting crop classes..."
)


raster_predictions = final_model.predict(
    X_raster
)


raster_predictions = (
    raster_predictions.astype(
        np.uint8
    )
)


# ============================================================
# 31. Create Output Raster
# ============================================================

classified = np.full(
    (height, width),
    255,
    dtype=np.uint8
)


classified[
    valid_mask
] = raster_predictions


# ============================================================
# 32. Save Classified GeoTIFF
# ============================================================

profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress="lzw",
    nodata=255
)


with rasterio.open(
    OUTPUT_RASTER,
    "w",
    **profile
) as dst:

    dst.write(
        classified,
        1
    )


# ============================================================
# 33. Final Information
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION COMPLETED")
print("=" * 70)

print(
    f"Final accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Output raster:\n"
    f"{OUTPUT_RASTER}"
)

print(
    "\nClass codes:"
)

for class_name, class_id in class_mapping.items():

    print(
        f"{class_id} = {class_name}"
    )

print(
    "255 = NoData"
)

print("=" * 70)
