# Titanic Survival Prediction — Feature Engineering Assignment

> **Assignment 2 | Artificial Intelligence**

## Overview

This project builds a predictive pipeline for Titanic survival using data cleaning, feature engineering, and feature selection. It uses the classic [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) dataset.

---

## Project Structure

```
titanic_assignment/
│
├── data/
│   ├── train.csv              # Raw training data
│   ├── train_cleaned.csv      # After Part 1 (data cleaning)
│   ├── train_features.csv     # After Part 2 (feature engineering)
│   └── train_selected.csv     # After Part 3 (feature selection)
│
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb   # Full analysis notebook
│
├── scripts/
│   ├── data_cleaning.py       # Part 1 pipeline
│   ├── feature_engineering.py # Part 2 pipeline
│   └── feature_selection.py   # Part 3 pipeline
│
├── README.md
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run scripts sequentially
```bash
cd titanic_assignment

python scripts/data_cleaning.py        # outputs data/train_cleaned.csv
python scripts/feature_engineering.py  # outputs data/train_features.csv
python scripts/feature_selection.py    # outputs data/train_selected.csv
```

### 3. Explore the notebook
```bash
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

---

## Approach

### Part 1 — Data Cleaning
| Column | Issue | Decision |
|--------|-------|----------|
| `Age` | ~20% missing | Median imputation + binary `Age_Missing` indicator |
| `Cabin` | ~77% missing | Extract deck letter; label unknown as `'U'` |
| `Embarked` | 2 missing | Mode imputation |
| `Fare`, `Age` | Outliers | Winsorise to 1st–99th percentile |
| All text cols | Inconsistent casing | `.str.lower().str.strip()` normalisation |

### Part 2 — Features Engineered
| Feature | Formula / Method | Rationale |
|---------|-----------------|-----------|
| `FamilySize` | `SibSp + Parch + 1` | Group context affects survival |
| `IsAlone` | `1 if FamilySize == 1` | Solo travellers had lower survival |
| `Title` | Regex from `Name` | Encodes gender + social class |
| `Deck` | First letter of `Cabin` | Proxy for cabin class/location |
| `AgeGroup` | Cut into Child/Teen/Adult/Senior | Non-linear age effects |
| `FarePerPerson` | `Fare / FamilySize` | Individual economic indicator |
| `Fare_Log` / `Age_Log` | `log1p(x)` | Removes right skew |
| `Pclass_Fare` | `Pclass × Fare_Log` | Class–fare interaction |
| One-hot encoding | `pd.get_dummies` on Sex, Embarked, Title, Deck, AgeGroup | Required for ML models |

### Part 3 — Feature Selection
1. **Correlation filter**: removed features with pairwise correlation > 0.90
2. **Random Forest importance**: ranked all features; kept top 20
3. **Recursive Feature Elimination (RFE)**: refined to 15 optimal features

---

## Key Findings

- **Sex / Title** are the most powerful predictors — female passengers survived at nearly 3× the rate of males.
- **Passenger class (Pclass)** strongly determines both fare and survival probability; 1st-class passengers had the highest survival rate.
- **Small families (2–4 members)** survived at higher rates than both solo travellers and large families.
- **Children** had elevated survival rates consistent with "women and children first" evacuation policy.
- **Log-transforming Fare** significantly reduces skewness and improves model compatibility.
- **Deck 'U' (unknown cabin)** is the most common category and correlates with lower-class passengers.

---

## Requirements

See `requirements.txt`. Key dependencies:
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `jupyter`
