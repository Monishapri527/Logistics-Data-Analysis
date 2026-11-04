
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. CREATE PROJECT PATHS
# --------------------------------------------------

BASE_DIR = "Week4_Predictive_Modeling_Optimization"

DATA_PATH = os.path.join(
    BASE_DIR, "data", "cleaned_logistics_data.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
VISUAL_DIR = os.path.join(BASE_DIR, "visualizations")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLOADING LOGISTICS DATASET")

data = pd.read_csv(DATA_PATH)

print("Dataset Shape:", data.shape)
print("Missing Values:", data.isnull().sum().sum())

# --------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# --------------------------------------------------

features = [
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount",
    "Product Price",
    "Benefit per order",
    "Order Item Profit Ratio"
]

target = "Days for shipping (real)"

X = data[features].copy()
y = data[target].copy()

numeric_features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount",
    "Product Price",
    "Benefit per order",
    "Order Item Profit Ratio"
]

categorical_features = ["Shipping Mode"]

print("\nFEATURE SELECTION COMPLETED")
print("Input Features:", len(features))
print("Target Variable:", target)

# --------------------------------------------------
# 4. DATA PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

# --------------------------------------------------
# 5. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDATA SPLIT COMPLETED")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# --------------------------------------------------
# 6. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

# --------------------------------------------------
# 7. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

print("\nMODEL TRAINING STARTED")

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    filename = name.lower().replace(" ", "_") + ".joblib"

    joblib.dump(
        pipeline,
        os.path.join(MODEL_DIR, filename)
    )

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

# --------------------------------------------------
# 8. SAVE MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

comparison_path = os.path.join(
    BASE_DIR, "model_comparison.csv"
)

results_df.to_csv(comparison_path, index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 9. SELECT BEST MODEL
# --------------------------------------------------

best_model_name = results_df.loc[
    results_df["MAE"].idxmin(), "Model"
]

best_model = trained_models[best_model_name]

print("\nBEST MODEL:", best_model_name)

# --------------------------------------------------
# 10. CROSS-VALIDATION
# --------------------------------------------------

print("\nPERFORMING 5-FOLD CROSS-VALIDATION")

cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

cv_mae = -cv_scores

print("Cross-validation MAE scores:", cv_mae)
print("Average CV MAE:", round(cv_mae.mean(), 4))

# Save cross-validation results
cv_df = pd.DataFrame({
    "Fold": range(1, 6),
    "MAE": cv_mae
})

cv_df.to_csv(
    os.path.join(BASE_DIR, "cross_validation_results.csv"),
    index=False
)

# --------------------------------------------------
# 11. VISUALIZATION - MAE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="MAE"
)

plt.title("Model Comparison - Mean Absolute Error")
plt.ylabel("MAE (Days)")
plt.xlabel("Machine Learning Models")
plt.tight_layout()

plt.savefig(
    os.path.join(VISUAL_DIR, "mae_comparison.png")
)

plt.close()

# --------------------------------------------------
# 12. VISUALIZATION - RMSE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE"
)

plt.title("Model Comparison - Root Mean Squared Error")
plt.ylabel("RMSE (Days)")
plt.xlabel("Machine Learning Models")
plt.tight_layout()

plt.savefig(
    os.path.join(VISUAL_DIR, "rmse_comparison.png")
)

plt.close()

# --------------------------------------------------
# 13. VISUALIZATION - R2 COMPARISON
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("Model Comparison - R2 Score")
plt.ylabel("R2 Score")
plt.xlabel("Machine Learning Models")
plt.tight_layout()

plt.savefig(
    os.path.join(VISUAL_DIR, "r2_comparison.png")
)

plt.close()

# --------------------------------------------------
# 14. ACTUAL VS PREDICTED VISUALIZATION
# --------------------------------------------------

best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.3
)

plt.xlabel("Actual Shipping Days")
plt.ylabel("Predicted Shipping Days")
plt.title("Actual vs Predicted Shipping Duration")

plt.tight_layout()

plt.savefig(
    os.path.join(VISUAL_DIR, "actual_vs_predicted.png")
)

plt.close()

# --------------------------------------------------
# 15. SAVE BEST MODEL
# --------------------------------------------------

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "best_model.joblib")
)

# --------------------------------------------------
# 16. PREDICTION EXAMPLE
# --------------------------------------------------

sample_order = X_test.iloc[[0]]

sample_prediction = best_model.predict(sample_order)[0]

print("\nSAMPLE SHIPPING TIME PREDICTION")
print("Predicted Shipping Days:", round(sample_prediction, 2))
print("Actual Shipping Days:", y_test.iloc[0])

# --------------------------------------------------
# 17. FINAL SUMMARY
# --------------------------------------------------

print("\n--------------------------------------")
print("WEEK 4 MODELING COMPLETED SUCCESSFULLY")
print("--------------------------------------")

print("Dataset Records:", len(data))
print("Models Trained:", len(models))
print("Best Model:", best_model_name)
print("Best Model MAE:", round(
    results_df.loc[
        results_df["Model"] == best_model_name, "MAE"
    ].iloc[0], 4
))
print("Average Cross-validation MAE:", round(cv_mae.mean(), 4))

print("\nGenerated Files:")
print("- Model comparison CSV")
print("- Cross-validation results CSV")
print("- Three trained model files")
print("- Best model file")
print("- Four visualization images")

print("\nAll operations completed!")
