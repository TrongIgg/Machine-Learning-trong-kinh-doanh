import pandas as pd
import numpy as np

# GIẢ SỬ DataFrame của bạn đã được tải
df = pd.read_csv("cars_cleaned_final.csv")

# --- 1. Thống kê mô tả cho các biến Định lượng ---

# Danh sách các biến định lượng quan trọng cần kiểm tra
numerical_cols = ['weighttotal', 'emissionsco2_num', 'modeldate', 'seatingcapacity', 'fuelcapacity', 'speed_num',
                  'payload', 'veenginepower']  # Thêm/bớt cột tùy theo dữ liệu thực tế

print("--- THỐNG KÊ MÔ TẢ CHO CÁC BIẾN ĐỊNH LƯỢNG (GIÁ VÀ ĐẶC TRƯNG CƠ BẢN) ---")
# Sử dụng .describe() để tính toán Count, Mean, Std, Min, Max, Quartiles
# Thêm các chỉ số Skewness và Kurtosis thủ công
stats_df = df[numerical_cols].describe().T

# Tính toán Skewness và Kurtosis
stats_df['skewness'] = df[numerical_cols].skew()
stats_df['kurtosis'] = df[numerical_cols].kurt()

print(stats_df)
print("\n" + "=" * 80 + "\n")

# --- 2. Thống kê mô tả (Tần suất) cho các biến Phân loại ---

# Danh sách các biến phân loại quan trọng
categorical_cols = ['brand', 'bodytype', 'fueltype', 'seatingcapacity']

print("--- TẦN SUẤT CHO CÁC BIẾN PHÂN LOẠI QUAN TRỌNG ---")
for col in categorical_cols:
    print(f"\n--- Phân bố của cột '{col}' ---")

    # Tính toán tần suất và tỷ lệ
    value_counts = df[col].value_counts(dropna=False)
    value_percentages = df[col].value_counts(normalize=True, dropna=False) * 100

    # Kết hợp và in ra
    freq_df = pd.DataFrame({
        'Count': value_counts,
        'Percentage (%)': value_percentages.round(2)
    })
    print(freq_df)
    print("-" * (len(col) + 20))
