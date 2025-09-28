import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ====== CẤU HÌNH ======
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "data_cars.csv"

USE_EXCEL = False
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cột dữ liệu
price_col = "price"
hp_col = "veenginepower"
fuel_col = "fuelconsumption"

# ====== ĐỌC DỮ LIỆU ======
df = pd.read_excel(DATA_FILE) if USE_EXCEL else pd.read_csv(DATA_FILE)
work = df[[price_col, hp_col, fuel_col]].copy()

# ====== LÀM SẠCH ======
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
work["hp_log"] = np.log1p(work[hp_col])

features = ["price_log", "hp_log", fuel_col]
X = work[features].to_numpy()

# Trọng số ưu tiên
WEIGHTS = np.array([0.8, 0.5, 2.5])
print(">> WEIGHTS:", WEIGHTS.tolist())

# Pipeline chuẩn hóa + trọng số
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("weight", FunctionTransformer(lambda X: X * WEIGHTS)),
])

X_scaled = pipe.fit_transform(X)

# Lưu artefact
dump(pipe.named_steps["scaler"], OUT_DIR / "scaler.joblib")
pd.Series(features).to_csv(OUT_DIR / "feature_order.csv", index=False, header=False)
pd.Series(WEIGHTS).to_csv(OUT_DIR / "weights.csv", index=False, header=False)

# ====== HUẤN LUYỆN KNN ======
n_neighbors = 5  # số xe gợi ý gần nhất
nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
nn_model.fit(X_scaled)
dump(nn_model, OUT_DIR / "knn_model.joblib")

print(">> Đã huấn luyện KNN và lưu model vào:", OUT_DIR.resolve())


# ====== HÀM GỢI Ý ======
def suggest_cars(price, hp, fuel, n_neighbors=5):
    """
    price, hp, fuel: thông số nhu cầu của user
    """
    vec = np.array([[np.log1p(price), np.log1p(hp), fuel]])
    vec_scaled = pipe.transform(vec)
    dists, indices = nn_model.kneighbors(vec_scaled, n_neighbors=n_neighbors)
    results = work.iloc[indices[0]].copy()
    results["distance"] = dists[0]
    return results, vec_scaled


# ====== TEST GỢI Ý ======
user_price = 50000
user_hp = 150
user_fuel = 7
results, vec_scaled = suggest_cars(user_price, user_hp, user_fuel)
print("Gợi ý 5 xe gần nhất cho nhu cầu của bạn:")
print(results)

# ====== VẼ BIỂU ĐỒ ======

# 1. Histogram phân bố các tiêu chí
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(work["price"], bins=30)
axes[0].set_title("Giá xe")

axes[1].hist(work["veenginepower"], bins=30)
axes[1].set_title("Công suất (HP)")

axes[2].hist(work["fuelconsumption"], bins=30)
axes[2].set_title("Tiêu thụ nhiên liệu")
plt.suptitle("Phân bố các tiêu chí xe")
plt.tight_layout()
plt.show()

# 2. Scatter 2D PCA (toàn bộ xe + nhu cầu user)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
vec_2d = pca.transform(vec_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=10, alpha=0.5, label='Cars')
plt.scatter(vec_2d[:, 0], vec_2d[:, 1], c='red', s=100, label='User need')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Phân bố xe trong không gian 2D (PCA)')
plt.legend()
plt.show()

# 3. Heatmap khoảng cách KNN của 10 xe đầu tiên
distances, indices = nn_model.kneighbors(X_scaled[:10], n_neighbors=10)
sns.heatmap(distances, cmap="YlGnBu")
plt.title("Khoảng cách KNN của 10 xe đầu tiên")
plt.xlabel("Neighbor")
plt.ylabel("Xe")
plt.show()
