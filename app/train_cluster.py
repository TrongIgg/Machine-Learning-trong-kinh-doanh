import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from pathlib import Path
import matplotlib.pyplot as plt

# THAY ĐỔI: Train trên data demo
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "data_car_demo.csv"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(">> DATA_FILE:", DATA_FILE)
print(">> OUT_DIR:", OUT_DIR.resolve())

# Cột data
price_col = "price"
hp_col = "veenginepower"
fuel_col = "fuelconsumption"

# Load
df = pd.read_csv(DATA_FILE)
work = df[[price_col, hp_col, fuel_col]].copy()

# Làm sạch
for c in [price_col, hp_col, fuel_col]:
    work[c] = pd.to_numeric(work[c], errors="coerce")

work = work.dropna()
work = work[(work[price_col] > 0) & (work[hp_col] > 0) & (work[fuel_col] > 0)]

# Chặn outlier nhẹ
for c in [price_col, hp_col, fuel_col]:
    lo, hi = work[c].quantile([0.01, 0.99])
    work[c] = work[c].clip(lo, hi)

# Log transform
work["price_log"] = np.log1p(work[price_col])
work["hp_log"] = np.log1p(work[hp_col])

features = ["price_log", "hp_log", fuel_col]
X = work[features].to_numpy()

# Trọng số
WEIGHTS = np.array([0.8, 0.5, 2.5])
print(">> WEIGHTS:", WEIGHTS.tolist())

# Chọn k bằng silhouette
best = (-1, None, None)
scores = []
for k in range(2, 6):  # Test 2-5 clusters
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("weight", FunctionTransformer(lambda X: X * WEIGHTS)),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=42)),
    ])
    labels = pipe.fit_predict(X)
    Xw = pipe.named_steps["weight"].transform(pipe.named_steps["scaler"].transform(X))
    sil = silhouette_score(Xw, labels)
    scores.append((k, float(sil)))
    if sil > best[0]:
        best = (sil, k, pipe)

best_sil, best_k, best_pipe = best
print(">> Silhouette scores:", [(k, round(s, 3)) for k, s in scores])
print(f">> Chọn k = {best_k} (silhouette = {best_sil:.3f})")

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot([s[0] for s in scores], [s[1] for s in scores], 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Cluster quality')
plt.grid(True)
plt.savefig(OUT_DIR / 'silhouette_plot.png')
print("✓ Saved silhouette plot")

# Lưu models
scaler = best_pipe.named_steps["scaler"]
kmeans = best_pipe.named_steps["kmeans"]
dump(scaler, OUT_DIR / "cluster_scaler.joblib")
dump(kmeans, OUT_DIR / "cluster_kmeans.joblib")
pd.Series(features).to_csv(OUT_DIR / "cluster_features.csv", index=False, header=False)
pd.Series(WEIGHTS).to_csv(OUT_DIR / "cluster_weights.csv", index=False, header=False)

# GÁN CLUSTER CHO TOÀN BỘ XE (QUAN TRỌNG)
X_scaled = scaler.transform(X)
X_weighted = X_scaled * WEIGHTS
cluster_labels = kmeans.predict(X_weighted)

# Thêm cluster vào dataframe gốc
df_full = pd.read_csv(DATA_FILE)
df_full['cluster'] = -1  # Default
df_full.loc[work.index, 'cluster'] = cluster_labels

# LƯU LẠI FILE MỚI VỚI CLUSTER LABELS
output_file = ROOT / "data" / "data_car_with_clusters.csv"
df_full.to_csv(output_file, index=False)
print(f"✓ Saved data with clusters to {output_file}")

# Phân tích clusters
print("\n=== CLUSTER DISTRIBUTION ===")
print(df_full['cluster'].value_counts().sort_index())

print("\n=== CLUSTER PROFILES ===")
for cluster_id in range(best_k):
    cluster_data = df_full[df_full['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_data)} xe):")
    print(f"  Price: ${cluster_data['price'].mean() * 1000:,.0f} (±{cluster_data['price'].std() * 1000:,.0f})")
    print(f"  Power: {cluster_data['veenginepower'].mean():.0f} HP (±{cluster_data['veenginepower'].std():.0f})")
    print(f"  Fuel: {cluster_data['fuelconsumption'].mean():.1f} L/100km")

    # Top brands trong cluster
    top_brands = cluster_data['brand'].value_counts().head(3)
    print(f"  Top brands: {', '.join(top_brands.index.tolist())}")

    # Top bodytypes
    top_types = cluster_data['bodytype'].value_counts().head(3)
    print(f"  Body types: {', '.join(top_types.index.tolist())}")

# Tâm cụm
cent_scaled = kmeans.cluster_centers_
cent_unweight = cent_scaled / WEIGHTS
cent_unscaled = scaler.inverse_transform(cent_unweight)
cent_df = pd.DataFrame(cent_unscaled, columns=features)
cent_df["price"] = np.expm1(cent_df["price_log"])
cent_df["horsepower"] = np.expm1(cent_df["hp_log"])
cent_df = cent_df.drop(columns=["price_log", "hp_log"])
cent_df.to_csv(OUT_DIR / "cluster_centers.csv", index=False)

print("\n>> Đã lưu model vào:", OUT_DIR.resolve())
