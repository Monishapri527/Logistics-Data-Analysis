import pandas as pd
from pathlib import Path

folder = Path(__file__).parent

original = pd.read_csv(
    folder / "DataCoSupplyChainDataset.csv",
    encoding="latin-1",
    low_memory=False
)

cleaned = pd.read_csv(folder / "cleaned_logistics_data.csv")
final = pd.read_csv(folder / "normalized_logistics_data.csv")

summary = f"""
WEEK 2 - LOGISTICS DATA PREPROCESSING RESULTS

1. ORIGINAL DATASET
Rows: {original.shape[0]}
Columns: {original.shape[1]}
Duplicate records: {original.duplicated().sum()}
Columns containing missing values:
{original.isnull().sum()[original.isnull().sum() > 0].to_string()}

2. CLEANING RESULTS
Rows after cleaning: {cleaned.shape[0]}
Columns after cleaning: {cleaned.shape[1]}
Missing values after cleaning: {cleaned.isnull().sum().sum()}
Duplicate records after cleaning: {cleaned.duplicated().sum()}

3. FEATURE ENGINEERING AND NORMALIZATION
Final rows: {final.shape[0]}
Final columns: {final.shape[1]}
Missing values in final dataset: {final.isnull().sum().sum()}
Duplicate records in final dataset: {final.duplicated().sum()}

4. NORMALIZATION CHECK
Selected numerical features were scaled using Min-Max normalization.
Expected range: 0 to 1.

5. OUTPUT FILES
cleaned_logistics_data.csv
normalized_logistics_data.csv
"""

output = folder / "week2_results_summary.txt"
output.write_text(summary, encoding="utf-8")

print(summary)
print("\nResults summary saved to:", output)