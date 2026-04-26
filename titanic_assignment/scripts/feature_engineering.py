"""
feature_engineering.py
-----------------------
Creates all derived, encoded, and transformed features
for the Titanic survival prediction task.
Input : data/train_cleaned.csv
Output: data/train_features.csv
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. Derived features
# ─────────────────────────────────────────────
def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Family size and alone flag
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # Family size category
    df["FamilyType"] = pd.cut(
        df["FamilySize"],
        bins=[0, 1, 4, 20],
        labels=["Alone", "Small", "Large"]
    )

    # Title extraction from Name
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Title"] = df["Title"].str.strip()

    # Collapse rare titles
    rare_titles = {"Lady", "Countess", "Capt", "Col", "Don",
                   "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"}
    df["Title"] = df["Title"].apply(
        lambda t: "Rare" if t in rare_titles else t
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Deck from Cabin (already extracted in cleaning; handle if not present)
    if "Deck" not in df.columns:
        df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U")

    # Age groups
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 60, 200],
        labels=["Child", "Teen", "Adult", "Senior"]
    )

    # Fare per person
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    print("[feature_eng] Derived features created.")
    return df


# ─────────────────────────────────────────────
# 2. Transformations (log, standardise)
# ─────────────────────────────────────────────
def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log-transform right-skewed features (add 1 to avoid log(0))
    df["Fare_Log"]         = np.log1p(df["Fare"])
    df["FarePerPerson_Log"]= np.log1p(df["FarePerPerson"])
    df["Age_Log"]          = np.log1p(df["Age"])

    print("[feature_eng] Log transforms applied.")
    return df


# ─────────────────────────────────────────────
# 3. Categorical encoding
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ordinal encoding for Pclass (already numeric; kept as-is)
    # One-hot encode nominal features
    ohe_cols = ["Sex", "Embarked", "Title", "Deck", "AgeGroup", "FamilyType"]

    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # Convert bool columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[feature_eng] One-hot encoding done. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 4. Interaction features (optional)
# ─────────────────────────────────────────────
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Pclass_Fare"]  = df["Pclass"] * df["Fare_Log"]
    df["Pclass_Age"]   = df["Pclass"] * df["Age"]

    print("[feature_eng] Interaction features created.")
    return df


# ─────────────────────────────────────────────
# 5. Drop raw / leaky columns
# ─────────────────────────────────────────────
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin",
             "Fare", "Age", "FarePerPerson"]


def drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[feature_eng] Dropped raw columns: {cols_to_drop}")
    return df


# ─────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────
def engineer(input_path:  str = "data/train_cleaned.csv",
             output_path: str = "data/train_features.csv") -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = create_derived_features(df)
    df = apply_transformations(df)
    df = encode_categoricals(df)
    df = create_interaction_features(df)
    df = drop_raw_columns(df)
    df.to_csv(output_path, index=False)
    print(f"\n[engineer] Saved → {output_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    engineer()
