import pandas as pd
from pathlib import Path

# Locate and load the original dataset
folder = Path(__file__).parent
input_file = folder / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(input_file, encoding="latin-1", low_memory=False)

print("Original dataset shape:", df.shape)

# 1. Remove personal and unnecessary information
columns_to_drop = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    "Customer Id",
    "Order Customer Id",
    "Customer City",
    "Customer State",
    "Customer Country",
    "Order Zipcode",
    "Product Description",
    "Product Image"
]

df.drop(
    columns=columns_to_drop,
    errors="ignore",
    inplace=True
)

# 2. Remove duplicate records
df.drop_duplicates(inplace=True)

# 3. Convert date columns into datetime format
date_columns = [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

# 4. Display missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 5. Display outlier counts using IQR
print("\nOutlier analysis:")

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = (
        (df[column] < lower) |
        (df[column] > upper)
    ).sum()

    print(f"{column}: {outliers}")

# 6. Save cleaned dataset
output_file = folder / "cleaned_logistics_data.csv"
df.to_csv(output_file, index=False)

print("\nOriginal rows:", 180519)
print("Cleaned dataset shape:", df.shape)
print("Cleaned dataset saved to:", output_file)