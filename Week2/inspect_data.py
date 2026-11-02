import pandas as pd
from pathlib import Path

# Dataset location
file_path = Path("data") / "DataCoSupplyChainDataset.csv"

# Load dataset
df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)

# 1. Dataset dimensions
print("\n--- DATASET SIZE ---")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# 2. Column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 3. First five records
print("\n--- FIRST FIVE ROWS ---")
print(df.head())

# 4. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 5. Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print("Duplicate rows:", df.duplicated().sum())

# 6. Data types
print("\n--- DATA TYPES ---")
print(df.dtypes)