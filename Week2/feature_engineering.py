import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

folder = Path(__file__).parent

# Load cleaned dataset
df = pd.read_csv(folder / "cleaned_logistics_data.csv")

# 1. Extract useful date features
df["Order Year"] = pd.to_datetime(
    df["order date (DateOrders)"]
).dt.year

df["Order Month"] = pd.to_datetime(
    df["order date (DateOrders)"]
).dt.month

df["Order Day"] = pd.to_datetime(
    df["order date (DateOrders)"]
).dt.day

# 2. Select continuous numerical columns for normalization
numeric_features = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Price"
]

# 3. Apply Min-Max normalization
scaler = MinMaxScaler()

df[numeric_features] = scaler.fit_transform(
    df[numeric_features]
)

# 4. Save final dataset
output_file = folder / "normalized_logistics_data.csv"
df.to_csv(output_file, index=False)

print("Feature engineering completed!")
print("Normalized columns:", numeric_features)
print("Final dataset shape:", df.shape)
print("Saved to:", output_file)