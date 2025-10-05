# FinalProject – Car Market Clustering

## 1) Cài môi trường
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## 2) Train mô hình
python src/train_cluster.py

## 3) Chạy API
python api/app.py
# POST http://127.0.0.1:5000/cluster
# body:
# {
#   "price": 25000,
#   "veenginepower": 150,
#   "fuelconsumption": 6.5
# }
