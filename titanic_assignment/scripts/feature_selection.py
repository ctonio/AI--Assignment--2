"""
feature_selection.py
--------------------
Selects the most informative features using:
  1. Correlation analysis (remove highly correlated pairs)
  2. Random Forest feature importance ranking
  3. Recursive Feature Elimination (RFE) — bonus
Input : data/train_features.csv
Output: data/train_selected.csv  +  prints selected feature list
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 1. Correlation-based removal
# ─────────────────────────────────────────────
def remove_correlated_features(df: pd.DataFrame,
                                target: str = "Survived",
                                threshold: float = 0.90) -> pd.DataFrame:
    feat_df = df.drop(columns=[target])
    corr_matrix = feat_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"[selection] Removing {len(drop_cols)} highly correlated features: {drop_cols}")
    return df.drop(columns=drop_cols)


# ─────────────────────────────────────────────
# 2. Random Forest importance
# ─────────────────────────────────────────────
def rank_by_importance(df: pd.DataFrame,
                       target: str = "Survived",
                       top_n: int = 20) -> list:
    X = df.drop(columns=[target])
    y = df[target]

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    print("\n[selection] Top feature importances:")
    print(importance.head(top_n).to_string())

    return importance.head(top_n).index.tolist()


# ─────────────────────────────────────────────
# 3. RFE (Recursive Feature Elimination) — bonus
# ─────────────────────────────────────────────
def rfe_selection(df: pd.DataFrame,
                  target: str = "Survived",
                  n_features: int = 15) -> list:
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=n_features, step=1)
    rfe.fit(X_scaled, y)

    selected = X.columns[rfe.support_].tolist()
    print(f"\n[selection] RFE selected {len(selected)} features:")
    print(selected)
    return selected


# ─────────────────────────────────────────────
# 4. Main pipeline
# ─────────────────────────────────────────────
def select(input_path:  str = "data/train_features.csv",
           output_path: str = "data/train_selected.csv",
           target:      str = "Survived") -> pd.DataFrame:

    df = pd.read_csv(input_path)
    print(f"[selection] Input shape: {df.shape}")

    # Step 1 – remove highly correlated columns
    df = remove_correlated_features(df, target=target)

    # Step 2 – rank by RF importance and keep top 20
    top_features = rank_by_importance(df, target=target, top_n=20)

    # Step 3 – RFE for bonus refinement (keep 15 from the top 20)
    df_top = df[top_features + [target]]
    rfe_features = rfe_selection(df_top, target=target, n_features=15)

    # Final selected feature set
    final_cols = rfe_features + [target]
    df_final = df[final_cols]
    df_final.to_csv(output_path, index=False)
    print(f"\n[selection] Final dataset → {output_path}  shape={df_final.shape}")
    print(f"[selection] Selected features: {rfe_features}")
    return df_final


if __name__ == "__main__":
    select()
