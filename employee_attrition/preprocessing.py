# =========================================================
# EMPLOYEE ATTRITION ANALYSIS
# DATA PREPROCESSING
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# =========================================================
# BASIC DATA UNDERSTANDING
# =========================================================

print("Dataset Shape:", df.shape)

print("\nColumns:\n")
print(df.columns)

print("\nDataset Info:\n")
print(df.info())

print("\nFirst 5 Rows:\n")
print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\nMissing Values:\n")
print(df.isnull().sum())

# =========================================================
# CHECK DUPLICATES
# =========================================================

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates if present
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:", df.shape)

# =========================================================
# CHECK UNIQUE VALUES
# =========================================================

print("\nUnique Values Per Column:\n")

for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

# =========================================================
# REMOVE UNNECESSARY COLUMNS
# =========================================================
# These columns usually have same value for all rows
# and do not help the model

drop_cols = [
    'EmployeeCount',
    'Over18',
    'StandardHours'
]

df.drop(columns=drop_cols, inplace=True)

print("\nColumns After Dropping Unnecessary Features:\n")
print(df.columns)

# =========================================================
# CONVERT TARGET VARIABLE
# =========================================================
# Attrition:
# Yes -> 1
# No -> 0

df['Attrition'] = df['Attrition'].map({
    'Yes': 1,
    'No': 0
})

print("\nAttrition Value Counts:\n")
print(df['Attrition'].value_counts())

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

le = LabelEncoder()

categorical_columns = df.select_dtypes(include='object').columns

print("\nCategorical Columns:\n")
print(categorical_columns)

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

print("\nCategorical Features Encoded Successfully")

# =========================================================
# CHECK FINAL DATA TYPES
# =========================================================

print("\nFinal Data Types:\n")
print(df.dtypes)

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop('Attrition', axis=1)

y = df['Attrition']

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# =========================================================
# OPTIONAL: FEATURE SCALING
# =========================================================
# Scaling is useful for Logistic Regression
# Not mandatory for Random Forest

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed")

# =========================================================
# SAVE PREPROCESSED DATA (OPTIONAL)
# =========================================================

processed_df = pd.concat([X, y], axis=1)

processed_df.to_csv(
    "processed_employee_attrition.csv",
    index=False
)

print("\nPreprocessed Dataset Saved Successfully")

# =========================================================
# PREPROCESSING COMPLETED
# =========================================================

print("\nData Preprocessing Completed Successfully")