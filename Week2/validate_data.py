import pandas as pd
from pathlib import Path

folder = Path(__file__).parent

# Load final dataset
df = pd.read_csv(folder / "normalized_logistics_data.csv")

print("------ FINAL DATASET VALIDATION ------")

# Dataset dimensions
print("\nDataset shape:", df.shape)

# Missing values
print("\nTotal missing values:", df.isnull().sum().sum())

# Duplicate records
print("Duplicate records:", df.duplicated().sum())

# Check normalized columns
columns = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales",
    "Order Item Quantity"
]

print("\nNormalized feature ranges:")

for column in columns:
    print(
        column,
        "Minimum:", round(df[column].min(), 3),
        "Maximum:", round(df[column].max(), 3)
    )

print("\nFinal dataset validation completed!")