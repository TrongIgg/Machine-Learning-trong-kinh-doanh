from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

app = Flask(__name__)

# ==== Đường dẫn models ====
ROOT = Path(__file__).resolve().parents[1]   # .../FinalProject
MODELS_DIR = ROOT / "models"

# ==== Nạp artefacts ====
scaler = load(MODELS_DIR / "scaler.joblib")
kmeans = load(MODELS_DIR / "kmeans.joblib")
feature_order = pd.read_csv(MODELS_DIR / "feature_order.csv", header=None)[0].tolist()
# NEW: nạp weights (được lưu từ train_cluster.py)
WEIGHTS = pd.read_csv(MODELS_DIR / "weights.csv", header=None)[0].to_numpy()

print("FEATURE ORDER:", feature_order)
print("WEIGHTS:", WEIGHTS.tolist())

# ==== Đặt tên cụm (khớp centers của bạn) ====
cluster_names = {
    0: "Hiệu năng/Sang",
    1: "Tiết kiệm/Phổ thông",
}

# ---- Tiện ích: chuyển chuỗi -> số, chấp nhận '5,5'
def to_float(x: object) -> float:
    return float(str(x).strip().replace(",", "."))

# ===================== API =====================

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/centers")
def centers():
    """Trả về tâm cụm ở thang GỐC (đã được export khi train) để giải thích mô hình."""
    df = pd.read_csv(MODELS_DIR / "cluster_centers.csv")
    return df.to_json(orient="records")

@app.post("/debug_cluster")
def debug_cluster():
    """Trả về khoảng cách tới từng tâm sau khi scale + weight (để thấy vì sao chọn cụm)."""
    data = request.get_json(force=True)
    try:
        price = to_float(data["price"])
        hp    = to_float(data["veenginepower"])
        fuel  = to_float(data["fuelconsumption"])
    except Exception:
        return jsonify({"error": "Thiếu/sai kiểu: price, veenginepower, fuelconsumption"}), 400

    x  = np.array([[np.log1p(price), np.log1p(hp), fuel]], dtype=float)
    xs = scaler.transform(x)
    xs = xs * WEIGHTS  # VERY IMPORTANT: áp trọng số đúng như lúc train

    # khoảng cách tới center trong không gian weighted
    dists = np.linalg.norm(xs - kmeans.cluster_centers_, axis=1)
    picked = int(np.argmin(dists))
    return jsonify({"dists": dists.tolist(), "picked": picked, "names": cluster_names})

@app.post("/cluster")
def cluster():
    """
    Body JSON mẫu:
    {
      "price": 25000,
      "veenginepower": 150,
      "fuelconsumption": 6.5
    }
    """
    data = request.get_json(force=True)

    # --- Parse + validate ---
    try:
        price = to_float(data["price"])
        hp    = to_float(data["veenginepower"])
        fuel  = to_float(data["fuelconsumption"])
    except Exception:
        return jsonify({"error": "Thiếu/sai kiểu: price, veenginepower, fuelconsumption"}), 400

    for name, v in [("price", price), ("veenginepower", hp), ("fuelconsumption", fuel)]:
        if not np.isfinite(v) or v <= 0:
            return jsonify({"error": f"{name} phải là số > 0 (dấu thập phân dùng chấm hoặc phẩy)."}), 400

    # --- Preprocess như lúc train ---
    x  = np.array([[np.log1p(price), np.log1p(hp), fuel]], dtype=float)
    xs = scaler.transform(x)
    xs = xs * WEIGHTS  # VERY IMPORTANT

    # --- Dự đoán cụm trong không gian weighted ---
    label = int(kmeans.predict(xs)[0])

    center = kmeans.cluster_centers_[label]  # center trong không gian weighted
    dist   = float(np.linalg.norm(xs - center, axis=1)[0])

    return jsonify({
        "cluster_id": label,
        "cluster_name": cluster_names.get(label, f"Cluster {label}"),
        "distance_to_center": dist,
        "used_features": feature_order
    })

# ===================== Trang demo =====================

@app.get("/")
def demo_page():
    # Trang HTML đơn giản gọi /cluster bằng fetch
    return """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Car Cluster Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { color-scheme: light dark; }
    body { font-family: Arial, sans-serif; max-width: 720px; margin: 40px auto; line-height:1.5; }
    label { display:block; margin-top:12px; font-weight:600; }
    input { width: 100%; padding: 10px; margin-top: 6px; border-radius: 8px; border: 1px solid #ccc; }
    button { margin-top: 16px; padding: 10px 16px; border-radius: 10px; border: 0; background:#1f6feb; color:#fff; cursor: pointer; }
    button:hover { opacity: .9; }
    pre { background:#111; color:#0f0; padding:14px; border-radius:12px; overflow:auto; }
  </style>
</head>
<body>
  <h1>Phân cụm xe – KMeans</h1>
  <p>Nhập thông số và bấm <b>Phân cụm</b> để xem cụm dự đoán.</p>

  <label>Giá (USD)
    <input id="price" type="number" inputmode="decimal" min="1" step="100" value="22000" required>
  </label>

  <label>Công suất (hp) – veenginepower
    <input id="hp" type="number" inputmode="decimal" min="1" step="5" value="120" required>
  </label>

  <label>Mức tiêu hao (L/100km) – fuelconsumption
    <input id="fuel" type="number" inputmode="decimal" min="0.1" step="0.1" value="6.2" required>
  </label>

  <button onclick="run()">Phân cụm</button>

  <h3>Kết quả</h3>
  <pre id="out">...</pre>

<script>
function toNum(v){ return Number(String(v).trim().replace(',', '.')); }

async function run(){
  const payload = {
    price:           toNum(document.getElementById('price').value),
    veenginepower:   toNum(document.getElementById('hp').value),
    fuelconsumption: toNum(document.getElementById('fuel').value)
  };

  if (!payload.price || !payload.veenginepower || !payload.fuelconsumption){
    document.getElementById('out').textContent = "Vui lòng nhập đủ 3 ô và > 0.";
    return;
  }

  const res  = await fetch('/cluster', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  const text = await res.text();
  try {
    document.getElementById('out').textContent = JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    document.getElementById('out').textContent = text;
  }
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
