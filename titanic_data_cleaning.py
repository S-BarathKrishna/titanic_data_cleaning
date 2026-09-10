"""
Titanic Dataset - Data Cleaning and Preprocessing
Author: [Your Name]
Purpose:
    Acquire a public dataset, explore data quality, handle missing values,
    check inconsistencies and outliers, and prepare the data for analysis.

Dataset:
    Titanic Passenger Dataset
Source:
    https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional visualization library
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# ============================================================
# 1. DATA COLLECTION
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/"
    "master/titanic.csv"
)

print("=" * 70)
print("1. DATA COLLECTION")
print("=" * 70)

df = pd.read_csv(DATA_URL)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())


# ============================================================
# 2. INITIAL DATA EXPLORATION
# ============================================================

print("\n" + "=" * 70)
print("2. INITIAL DATA EXPLORATION")
print("=" * 70)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all").T)

print("\nUnique values:")
for column in df.columns:
    print(f"{column}: {df[column].nunique(dropna=True)} unique values")


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage.round(2)
})

missing_report = missing_report.sort_values(
    by="Missing Count",
    ascending=False
)

print(missing_report)

# Save missing-value report
missing_report.to_csv("missing_value_report.csv")

# Plot missing values
missing_plot = missing_percentage[missing_percentage > 0].sort_values()

if not missing_plot.empty:
    plt.figure(figsize=(8, 5))
    missing_plot.plot(kind="barh")
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Column")
    plt.title("Missing Value Percentage")
    plt.tight_layout()
    plt.savefig("missing_values.png", dpi=200)
    plt.show()


# ============================================================
# 4. DUPLICATE RECORD CHECK
# ============================================================

print("\n" + "=" * 70)
print("4. DUPLICATE RECORD CHECK")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of exact duplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates().copy()
    print("Exact duplicate rows removed.")
else:
    print("No exact duplicate rows found.")


# ============================================================
# 5. DATA TYPE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5. DATA TYPE VALIDATION")
print("=" * 70)

numeric_columns = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Numeric columns converted/validated.")
print(df[numeric_columns].dtypes)


# ============================================================
# 6. CATEGORICAL DATA STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("6. CATEGORICAL DATA STANDARDIZATION")
print("=" * 70)

df["Sex"] = df["Sex"].astype("string").str.strip().str.lower()
df["Embarked"] = df["Embarked"].astype("string").str.strip().str.upper()

print("Sex categories:", df["Sex"].dropna().unique())
print("Embarked categories:", df["Embarked"].dropna().unique())


# ============================================================
# 7. INVALID VALUE CHECKS
# ============================================================

print("\n" + "=" * 70)
print("7. INVALID VALUE CHECKS")
print("=" * 70)

invalid_checks = {
    "Negative Age": (df["Age"] < 0).sum(),
    "Negative Fare": (df["Fare"] < 0).sum(),
    "Negative SibSp": (df["SibSp"] < 0).sum(),
    "Negative Parch": (df["Parch"] < 0).sum(),
    "Invalid Survived": (~df["Survived"].isin([0, 1])).sum(),
    "Invalid Pclass": (~df["Pclass"].isin([1, 2, 3])).sum(),
}

for check, count in invalid_checks.items():
    print(f"{check}: {count}")


# ============================================================
# 8. HANDLE MISSING AGE
# ============================================================

print("\n" + "=" * 70)
print("8. HANDLE MISSING AGE VALUES")
print("=" * 70)

age_missing_before = df["Age"].isnull().sum()
age_median = df["Age"].median()

df["Age"] = df["Age"].fillna(age_median)

age_missing_after = df["Age"].isnull().sum()

print("Missing Age before:", age_missing_before)
print("Median Age used:", age_median)
print("Missing Age after:", age_missing_after)


# ============================================================
# 9. HANDLE MISSING EMBARKED
# ============================================================

print("\n" + "=" * 70)
print("9. HANDLE MISSING EMBARKED VALUES")
print("=" * 70)

embarked_missing_before = df["Embarked"].isnull().sum()
embarked_mode = df["Embarked"].mode()[0]

df["Embarked"] = df["Embarked"].fillna(embarked_mode)

embarked_missing_after = df["Embarked"].isnull().sum()

print("Missing Embarked before:", embarked_missing_before)
print("Mode used:", embarked_mode)
print("Missing Embarked after:", embarked_missing_after)


# ============================================================
# 10. HANDLE CABIN
# ============================================================

print("\n" + "=" * 70)
print("10. CABIN MISSINGNESS ANALYSIS")
print("=" * 70)

cabin_missing = df["Cabin"].isnull().sum()
cabin_missing_percentage = (cabin_missing / len(df)) * 100

print("Missing Cabin values:", cabin_missing)
print("Missing Cabin percentage:",
      round(cabin_missing_percentage, 2), "%")

# Create a useful cabin/deck feature before removing Cabin.
# The first character represents the deck for known cabins.
df["Deck"] = df["Cabin"].str[0]

# Unknown cabin information is represented explicitly.
df["Deck"] = df["Deck"].fillna("Unknown")

# Drop original high-missingness Cabin field.
df = df.drop(columns=["Cabin"])

print("Original Cabin column removed after creating Deck.")


# ============================================================
# 11. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 70)
print("11. OUTLIER DETECTION USING IQR")
print("=" * 70)


def detect_outliers_iqr(dataframe, column):
    """
    Detect potential outliers using the 1.5 × IQR rule.
    Returns Q1, Q3, lower bound, upper bound, and count.
    """
    series = dataframe[column].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (dataframe[column] < lower_bound)
        | (dataframe[column] > upper_bound)
    )

    outlier_count = outlier_mask.sum()

    return q1, q3, lower_bound, upper_bound, outlier_count


outlier_columns = ["Age", "Fare", "SibSp", "Parch"]

outlier_results = []

for column in outlier_columns:
    q1, q3, lower, upper, count = detect_outliers_iqr(df, column)

    outlier_results.append({
        "Column": column,
        "Q1": round(q1, 2),
        "Q3": round(q3, 2),
        "IQR": round(q3 - q1, 2),
        "Lower Bound": round(lower, 2),
        "Upper Bound": round(upper, 2),
        "Potential Outliers": count
    })

outlier_report = pd.DataFrame(outlier_results)

print(outlier_report)

outlier_report.to_csv("outlier_report.csv", index=False)


# ============================================================
# 12. OUTLIER VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("12. OUTLIER VISUALIZATION")
print("=" * 70)

if SEABORN_AVAILABLE:
    for column in ["Age", "Fare"]:
        plt.figure(figsize=(8, 5))
        sns.boxplot(y=df[column])
        plt.title(f"{column} Distribution and Potential Outliers")
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(
            f"{column.lower()}_boxplot.png",
            dpi=200
        )
        plt.show()
else:
    print("Seaborn is not installed.")
    print("Install it using: pip install seaborn")


# ============================================================
# 13. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("13. FEATURE ENGINEERING")
print("=" * 70)

# FamilySize = siblings/spouses + parents/children + passenger
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Passenger traveling alone
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Age groups
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, np.inf],
    labels=[
        "Child",
        "Teen",
        "YoungAdult",
        "Adult",
        "Senior"
    ],
    right=False
)

print("New features created:")
print("- FamilySize")
print("- IsAlone")
print("- AgeGroup")


# ============================================================
# 14. SELECT ANALYSIS COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("14. SELECT ANALYSIS COLUMNS")
print("=" * 70)

analysis_columns = [
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Deck",
    "FamilySize",
    "IsAlone",
    "AgeGroup"
]

model_df = df[analysis_columns].copy()

print("Analysis dataset shape:", model_df.shape)
print(model_df.head())


# ============================================================
# 15. ENCODE CATEGORICAL VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("15. ENCODE CATEGORICAL VARIABLES")
print("=" * 70)

model_df = pd.get_dummies(
    model_df,
    columns=["Sex", "Embarked", "Deck", "AgeGroup"],
    drop_first=True,
    dtype=int
)

print("Categorical variables encoded.")
print("New shape:", model_df.shape)


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("16. FINAL VALIDATION")
print("=" * 70)

print("\nRemaining missing values:")
print(model_df.isnull().sum())

print("\nDuplicate rows:")
print(model_df.duplicated().sum())

print("\nFinal data types:")
print(model_df.dtypes)

# Assertions provide automated quality checks.
assert model_df.isnull().sum().sum() == 0
assert model_df["Survived"].isin([0, 1]).all()
assert model_df["Pclass"].isin([1, 2, 3]).all()
assert (model_df["Age"] >= 0).all()
assert (model_df["Fare"] >= 0).all()

print("\nAll final validation checks passed successfully.")


# ============================================================
# 17. SAVE CLEANED DATASET
# ============================================================

print("\n" + "=" * 70)
print("17. SAVE CLEANED DATASET")
print("=" * 70)

cleaned_file = "titanic_cleaned.csv"

model_df.to_csv(cleaned_file, index=False)

print("Cleaned dataset saved as:", cleaned_file)
print("Final dataset shape:", model_df.shape)


# ============================================================
# 18. SURVIVAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("18. SURVIVAL DISTRIBUTION")
print("=" * 70)

survival_counts = df["Survived"].value_counts().sort_index()

print("Survival counts:")
print(survival_counts)

plt.figure(figsize=(7, 5))
survival_counts.plot(kind="bar")
plt.xlabel("Survival Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.title("Titanic Survival Distribution")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("survival_distribution.png", dpi=200)
plt.show()


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("19. FINAL SUMMARY")
print("=" * 70)

print("Original dataset shape: 891 rows × 12 columns")
print("Final analytical dataset shape:", model_df.shape)
print("Age missing values handled using median imputation.")
print("Embarked missing values handled using mode imputation.")
print("Cabin converted to Deck and original Cabin field removed.")
print("Potential outliers identified using the IQR method.")
print("FamilySize, IsAlone, and AgeGroup features created.")
print("Categorical variables encoded.")
print("Final validation completed.")
print("Cleaned dataset exported to titanic_cleaned.csv")

print("\nData cleaning and preprocessing completed successfully.")
