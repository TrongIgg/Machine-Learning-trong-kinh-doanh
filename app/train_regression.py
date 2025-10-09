# %% Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "data_cars.csv"

USE_EXCEL = False
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA_FILE) if USE_EXCEL else pd.read_csv(DATA_FILE)

print("Shape:", df.shape)
print(df.head())

# %% EDA cơ bản
print("\nMissing values:")
print(df.isnull().sum())

print("\nDescribe:")
print(df.describe(include="all"))

# %% Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df["purchase_price"].dropna(), kde=True, bins=30, color="blue")
plt.title("Distribution of Purchase Price")

plt.subplot(1, 2, 2)
sns.histplot(df["selling_price"].dropna(), kde=True, bins=30, color="green")
plt.title("Distribution of Selling Price")
plt.show()

# Scatter plot
plt.figure(figsize=(6, 6))
sns.scatterplot(x="purchase_price", y="selling_price", data=df)
plt.title("Purchase vs Selling Price")
plt.show()

# Correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# %% Function train 1 target
def train_one_target(df, target):
    print(f"\n===== Training for target: {target} =====")

    # Drop NaN của target
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]

    # Identify feature types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64),
                            max_iter=300, random_state=42,
                            early_stopping=True)
    }

    # Train + Evaluate
    results = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor),
                               ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        rmse = mean_squared_error(y_val, preds, squared=False)
        r2 = r2_score(y_val, preds)

        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    results_df = pd.DataFrame(results)
    print(results_df)

    return results_df


# %% Train cho 2 target
res_purchase = train_one_target(df, "purchase_price")
res_selling = train_one_target(df, "selling_price")
