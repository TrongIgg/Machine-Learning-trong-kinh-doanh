import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

# Đường dẫn dự án
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "data_cars.csv"
USE_EXCEL = False
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(">> DATA_FILE:", DATA_FILE)
print(">> OUT_DIR:", OUT_DIR.resolve())
#Cột data
price_col = "price"
hp_col    = "veenginepower"
fuel_col  = "fuelconsumption"

# ===== Load =====
df = pd.read_excel(DATA_FILE) if USE_EXCEL else pd.read_csv(DATA_FILE)
work = df[[price_col, hp_col, fuel_col]].copy()

# ===== Làm sạch =====
for c in [price_col, hp_col, fuel_col]:
    work[c] = pd.to_numeric(work[c], errors="coerce")

work = work.dropna()
work = work[(work[price_col] > 0) & (work[hp_col] > 0) & (work[fuel_col] > 0)]

# Chặn outlier nhẹ
for c in [price_col, hp_col, fuel_col]:
    lo, hi = work[c].quantile([0.01, 0.99])
    work[c] = work[c].clip(lo, hi)

# Log cho 2 cột lệch
work["price_log"] = np.log1p(work[price_col])
work["hp_log"]    = np.log1p(work[hp_col])

features = ["price_log", "hp_log", fuel_col]
X = work[features].to_numpy()

#lấy mẫu cho nhanh nếu data rất lớn
if len(X) > 20000:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=20000, replace=False)
    X = X[idx]

# Ranh giới
WEIGHTS = np.array([0.8, 0.5, 2.5])
print(">> WEIGHTS:", WEIGHTS.tolist())

# Chọn k bằng silhouette
best = (-1, None, None)
scores = []
for k in range(2, 6):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("weight", FunctionTransformer(lambda X: X * WEIGHTS)),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=42)),
    ])
    labels = pipe.fit_predict(X)
    # silhouette tính trên dữ liệu đã scale + weight
    Xw = pipe.named_steps["weight"].transform(pipe.named_steps["scaler"].transform(X))
    sil = silhouette_score(Xw, labels)
    scores.append((k, float(sil)))
    if sil > best[0]:
        best = (sil, k, pipe)

best_sil, best_k, best_pipe = best
print(">> Silhouette:", [(k, round(s, 3)) for k, s in scores])
print(f">> Chọn k = {best_k} (silhouette = {best_sil:.3f})")

# ===== Lưu artefacts =====
scaler = best_pipe.named_steps["scaler"]
kmeans = best_pipe.named_steps["kmeans"]
dump(scaler, OUT_DIR / "scaler.joblib")
dump(kmeans, OUT_DIR / "kmeans.joblib")
pd.Series(features).to_csv(OUT_DIR / "feature_order.csv", index=False, header=False)
pd.Series(WEIGHTS).to_csv(OUT_DIR / "weights.csv", index=False, header=False)  # NEW

# ===== Tâm cụm
cent_scaled   = kmeans.cluster_centers_
# bỏ weight trước khi inverse_transform để trả về thang scale và rồi thang gốc
cent_unweight = cent_scaled / WEIGHTS
cent_unscaled = scaler.inverse_transform(cent_unweight)
cent_df = pd.DataFrame(cent_unscaled, columns=features)
cent_df["price"]      = np.expm1(cent_df["price_log"])
cent_df["horsepower"] = np.expm1(cent_df["hp_log"])
cent_df = cent_df.drop(columns=["price_log", "hp_log"])
cent_df.to_csv(OUT_DIR / "cluster_centers.csv", index=False)

print(">> Đã lưu model vào:", OUT_DIR.resolve())
print(cent_df.sort_values("price"))
