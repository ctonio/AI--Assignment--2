"""
data_cleaning.py
----------------
Handles all data cleaning steps for the Titanic dataset:
  - Missing value imputation
  - Outlier detection and capping
  - Data consistency checks
  - Duplicate removal
Outputs: data/train_cleaned.csv
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────
def load_data(path: str = "data/train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. Missing value handling
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("\n[missing] Before:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Age  → median imputation (less sensitive to outliers than mean)
    age_median = df["Age"].median()
    df["Age_Missing"] = df["Age"].isnull().astype(int)   # indicator column
    df["Age"] = df["Age"].fillna(age_median)

    # Embarked → mode imputation (only 2 missing values)
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    # Cabin  → ~77 % missing; extract deck letter instead of imputing
    #          Missing cabin → deck labeled 'U' (Unknown)
    df["Deck"] = df["Cabin"].apply(
        lambda x: str(x)[0] if pd.notna(x) else "U"
    )

    # Fare   → rarely missing (test set only), impute with median
    if df["Fare"].isnull().any():
        df["Fare"].fillna(df["Fare"].median(), inplace=True)

    print("\n[missing] After:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df


# ─────────────────────────────────────────────
# 3. Outlier handling
# ─────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Fare", "Age"]:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        n_before = ((df[col] < q1) | (df[col] > q99)).sum()
        df[col] = df[col].clip(lower=q1, upper=q99)
        print(f"[outliers] {col}: capped {n_before} values to [{q1:.2f}, {q99:.2f}]")

    return df


# ─────────────────────────────────────────────
# 4. Data consistency
# ─────────────────────────────────────────────
def fix_consistency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardise Sex values to lowercase
    df["Sex"] = df["Sex"].str.lower().str.strip()

    # Standardise Embarked
    df["Embarked"] = df["Embarked"].str.upper().str.strip()

    # Remove exact duplicates (keep first)
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[consistency] Dropped {n_before - len(df)} duplicate rows")

    return df


# ─────────────────────────────────────────────
# 5. Main pipeline
# ─────────────────────────────────────────────
def clean(input_path: str = "data/train.csv",
          output_path: str = "data/train_cleaned.csv") -> pd.DataFrame:
    df = load_data(input_path)
    df = fix_consistency(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df.to_csv(output_path, index=False)
    print(f"\n[clean] Saved cleaned dataset → {output_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    clean()
