# Multi-Temporal Crop Classification Using Sentinel-1, Sentinel-2 and XGBoost

This repository contains the code used for a multi-temporal crop-classification workflow based on Sentinel-1 and Sentinel-2 data and XGBoost.

## Workflow

1. **Google Earth Engine**: Sentinel-1 and Sentinel-2 preprocessing, monthly composites, NDVI/NDRE calculation, radar variables (VV, VH, VVVH), 40-feature stack, and 5-day phenology analysis.
2. **Temporal feature engineering**: construction of 10 temporal features from the 40 monthly features.
3. **XGBoost classification**: comparison of Sentinel-2, Sentinel-1, Sentinel-1 + Sentinel-2, and Sentinel-1 + Sentinel-2 + temporal feature sets.
4. **Accuracy assessment**: overall accuracy, confusion matrix, class-wise precision/recall/F1, and feature importance.

## Study period

October 2025 to May 2026, represented in the code by the Persian-month labels:

`Mehr, Aban, Azar, Dey, Bahman, Esfand, Farvardin, Ordibehesht`

## Features

### 40 base features

- Sentinel-2: NDVI and NDRE for 8 months = 16 features.
- Sentinel-1: VV, VH and VVVH (VV − VH) for 8 months = 24 features.

### 10 temporal features

- NDRE_Range
- NDVI_Diff_Mehr_Aban
- NDVI_Range
- NDRE_Diff_Bahman_Esfand
- NDVI_Diff_Farvardin_Ordibehesht
- NDRE_Diff_Mehr_Aban
- NDVI_Diff_Bahman_Esfand
- NDRE_Diff_Farvardin_Ordibehesht
- NDVI_Diff_Aban_Azar
- VV_Diff_Esfand_Farvardin

Total: **50 input features**.

## Classes

The classification code uses the following fixed class order and IDs:

| ID | Class |
|---:|---|
| 0 | Barley |
| 1 | Melon |
| 2 | Tomato |
| 3 | Wheat |
| 4 | Other |

The code also normalizes the spelling `Barey` to `Barley`.

## Repository structure

```text
Multi-Temporal-Crop-Classification-XGBoost/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── 01_GEE/
│   └── Sentinel1_Sentinel2_MultiTemporal_Features.js
├── 02_Feature_Engineering/
│   └── Temporal_Feature_Engineering.py
├── 03_XGBoost/
│   └── XGBoost_Crop_Classification.py
└── 04_Results/
    └── README.md
```

## Input data

The XGBoost script expects two Excel files containing a `Name` column and the 40 base features listed above. The temporal features are calculated automatically.

For `.xls` input files, the script uses `xlrd`.

Example:

```bash
python 03_XGBoost/XGBoost_Crop_Classification.py --train train.xls --test test.xls --output 04_Results
```

## Google Earth Engine

The public GEE script intentionally uses placeholders for the private/user-specific assets. Replace:

```text
users/YOUR_USERNAME/YOUR_BOUNDARY_ASSET
users/YOUR_USERNAME/YOUR_SAMPLE_ASSET
```

with the appropriate assets in your own GEE account.

The script uses:

- `COPERNICUS/S2_SR_HARMONIZED`
- `COPERNICUS/S1_GRD`
- Sentinel-2 cloud filtering and QA60 masking
- 10 m processing/export
- UTM CRS `EPSG:32639` as in the source workflow

## Model parameters

The final XGBoost model uses the parameters in the original workflow:

```text
n_estimators = 300
max_depth = 5
learning_rate = 0.05
colsample_bytree = 0.8
objective = multi:softprob
eval_metric = mlogloss
num_class = 5
n_jobs = 2
random_state = 42
```

## Reproducibility note

The original Colab code contained a duplicated `NDVI_Diff_Mehr_Aban` entry in one temporal-feature list. The cleaned public version defines the intended **10 unique temporal features**, giving exactly 50 total features (40 + 10).

The original classification script also contained hard-coded values for one reported results table after calculating the values dynamically. The cleaned version removes that duplicate hard-coded table and writes the calculated results directly, reducing the risk of inconsistency between code and reported results.

## Data availability

Sentinel-1 and Sentinel-2 source data are provided by the Copernicus/Sentinel programme. Training/test reference data and derived products are not included in this repository unless explicitly released by the author.

## Citation

If you use this code, please cite the associated research article when it is published.
