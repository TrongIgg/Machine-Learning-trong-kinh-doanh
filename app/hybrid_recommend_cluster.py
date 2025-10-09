import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Load cluster models
try:
    cluster_scaler = load(MODELS_DIR / "cluster_scaler.joblib")
    cluster_kmeans = load(MODELS_DIR / "cluster_kmeans.joblib")
    cluster_features = pd.read_csv(MODELS_DIR / "cluster_features.csv", header=None)[0].tolist()
    cluster_weights = pd.read_csv(MODELS_DIR / "cluster_weights.csv", header=None)[0].tolist()
    CLUSTER_ENABLED = True
    print("✓ Cluster models loaded successfully")
except Exception as e:
    print(f"⚠️ Cluster models not found: {e}")
    CLUSTER_ENABLED = False


def parse_survey_data(form_data):
    """Parse dữ liệu từ form survey"""
    survey = {
        'budget_min': float(form_data.get('budget_min', 0) or 0) / 1000,
        'budget_max': float(form_data.get('budget_max', 100000) or 100000) / 1000,
        'body_types': form_data.getlist('body_types'),
        'preferred_brands': [b.strip() for b in form_data.get('preferred_brands', '').split(',') if b.strip()],
        'fuel_type': form_data.get('fuel_type', ''),
        'seating_capacity': int(form_data.get('seating_capacity', 5) or 5),
        'engine_power_hp': int(form_data.get('engine_power_hp', 150) or 150),
        'usage': form_data.get('usage', ''),
        'is_electric': form_data.get('is_electric', '') == 'Yes',
        'number_of_doors': int(form_data.get('number_of_doors', 4) or 4),
        'top_speed_kmh': int(form_data.get('top_speed_kmh', 180) or 180),
        'co2_emissions_g_km': int(form_data.get('co2_emissions_g_km', 150) or 150),
    }
    return survey


def assign_cluster_to_car(car_row):
    """Gán cluster cho 1 xe dựa vào trained model"""
    if not CLUSTER_ENABLED:
        return -1

    try:
        # Tạo features giống như khi train
        price_log = np.log1p(car_row['price']) if car_row['price'] > 0 else 0
        hp_log = np.log1p(car_row['veenginepower']) if car_row['veenginepower'] > 0 else 0
        fuel = car_row['fuelconsumption'] if car_row['fuelconsumption'] > 0 else 7.0

        X = np.array([[price_log, hp_log, fuel]])
        X_scaled = cluster_scaler.transform(X)
        X_weighted = X_scaled * cluster_weights

        cluster_id = cluster_kmeans.predict(X_weighted)[0]
        return int(cluster_id)
    except:
        return -1


def get_user_preferred_cluster(survey_data):
    """Xác định cluster phù hợp với user dựa vào khảo sát"""
    if not CLUSTER_ENABLED:
        return -1

    try:
        # Tính toán từ survey
        avg_budget = (survey_data['budget_min'] + survey_data['budget_max']) / 2
        price_log = np.log1p(avg_budget)
        hp_log = np.log1p(survey_data['engine_power_hp'])

        # Ước tính fuel consumption dựa vào engine power
        # Xe mạnh hơn thường tốn nhiều xăng hơn
        estimated_fuel = 5.0 + (survey_data['engine_power_hp'] - 100) * 0.02
        estimated_fuel = max(4.0, min(estimated_fuel, 15.0))

        X = np.array([[price_log, hp_log, estimated_fuel]])
        X_scaled = cluster_scaler.transform(X)
        X_weighted = X_scaled * cluster_weights

        cluster_id = cluster_kmeans.predict(X_weighted)[0]
        return int(cluster_id)
    except:
        return -1


def calculate_rule_score(car, survey):
    """Tính điểm dựa trên các rule cứng"""
    score = 0.0
    weights = []

    # 1. Budget (weight: 3.0)
    if survey['budget_min'] <= car['price'] <= survey['budget_max']:
        score += 3.0
        weights.append(3.0)
    elif car['price'] < survey['budget_min']:
        ratio = car['price'] / survey['budget_min']
        score += 3.0 * max(0, ratio)
        weights.append(3.0)
    else:
        ratio = survey['budget_max'] / car['price']
        score += 3.0 * max(0, ratio * 0.5)
        weights.append(3.0)

    # 2. Body type (weight: 2.5)
    if survey['body_types']:
        body_match = any(
            str(car.get('bodytype', '')).lower() == bt.lower()
            for bt in survey['body_types']
        )
        if body_match:
            score += 2.5
        weights.append(2.5)

    # 3. Brand (weight: 1.5)
    if survey['preferred_brands']:
        brand_match = any(
            str(car.get('brand', '')).lower() == pb.lower()
            for pb in survey['preferred_brands']
        )
        if brand_match:
            score += 1.5
        weights.append(1.5)

    # 4. Fuel type (weight: 2.0)
    if survey['fuel_type']:
        if str(car.get('fueltype', '')).lower() == survey['fuel_type'].lower():
            score += 2.0
        weights.append(2.0)

    # 5. Seating capacity (weight: 1.5)
    if car.get('seatingcapacity'):
        seat_diff = abs(car['seatingcapacity'] - survey['seating_capacity'])
        if seat_diff == 0:
            score += 1.5
        elif seat_diff <= 1:
            score += 1.0
        elif seat_diff <= 2:
            score += 0.5
        weights.append(1.5)

    # 6. Engine power (weight: 1.5)
    if car.get('veenginepower') and car['veenginepower'] > 0:
        power_ratio = car['veenginepower'] / survey['engine_power_hp']
        if 0.8 <= power_ratio <= 1.2:
            score += 1.5
        elif 0.6 <= power_ratio <= 1.4:
            score += 1.0
        elif 0.4 <= power_ratio <= 1.6:
            score += 0.5
        weights.append(1.5)

    # Normalize
    max_score = sum(weights) if weights else 1.0
    return score / max_score


def calculate_knn_score_with_cluster(car, survey, car_df, user_cluster):
    """Tính KNN score với ưu tiên cluster"""
    try:
        # Prepare features
        numeric_features = ['price', 'veenginepower', 'fuelconsumption', 'seatingcapacity']

        # User vector
        user_vector = np.array([
            (survey['budget_min'] + survey['budget_max']) / 2,
            survey['engine_power_hp'],
            7.0,  # default fuel
            survey['seating_capacity']
        ]).reshape(1, -1)

        # Car vectors - lọc xe hợp lệ
        valid_cars = car_df[
            (car_df['price'] > 0) &
            (car_df['veenginepower'] > 0) &
            (car_df['fuelconsumption'] > 0) &
            (car_df['seatingcapacity'] > 0)
            ].copy()

        if len(valid_cars) < 5:
            return 0.5

        # Gán cluster cho các xe (nếu chưa có)
        if 'cluster' not in valid_cars.columns or valid_cars['cluster'].isna().any():
            valid_cars['cluster'] = valid_cars.apply(assign_cluster_to_car, axis=1)

        car_vectors = valid_cars[numeric_features].values

        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        car_vectors_scaled = scaler.fit_transform(car_vectors)
        user_vector_scaled = scaler.transform(user_vector)

        # KNN
        k = min(20, len(valid_cars))
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(car_vectors_scaled)

        distances, indices = knn.kneighbors(user_vector_scaled)

        # Kiểm tra xe hiện tại có trong top-k không
        car_idx = car.name if hasattr(car, 'name') else None
        if car_idx is None:
            return 0.5

        try:
            car_position = valid_cars.index.get_loc(car_idx)
        except:
            return 0.3

        if car_position in indices[0]:
            rank = np.where(indices[0] == car_position)[0][0]
            base_score = 1.0 - (rank / k)

            # BONUS: Nếu xe cùng cluster với user -> tăng điểm
            if CLUSTER_ENABLED and user_cluster >= 0:
                car_cluster = valid_cars.loc[car_idx, 'cluster']
                if car_cluster == user_cluster:
                    base_score = min(1.0, base_score * 1.3)  # Tăng 30%

            return base_score

        # Nếu không trong top-k, tính khoảng cách trực tiếp
        car_vector_scaled = scaler.transform(
            valid_cars.loc[car_idx, numeric_features].values.reshape(1, -1)
        )
        dist = np.linalg.norm(user_vector_scaled - car_vector_scaled)
        max_dist = distances[0][-1]

        score = max(0, 1 - (dist / (max_dist + 1e-6))) * 0.5

        # BONUS cluster
        if CLUSTER_ENABLED and user_cluster >= 0:
            car_cluster = valid_cars.loc[car_idx, 'cluster']
            if car_cluster == user_cluster:
                score = min(1.0, score * 1.3)

        return score

    except Exception as e:
        print(f"⚠️ KNN error: {e}")
        return 0.3


def recommend_cars(survey_data, car_df, top_n=5):
    """
    Hàm chính: Recommendation với Cluster
    """
    if car_df.empty:
        return pd.DataFrame()

    # Xác định cluster của user
    user_cluster = get_user_preferred_cluster(survey_data)
    print(f"👤 User preferred cluster: {user_cluster}")

    # Gán cluster cho tất cả xe (nếu chưa có)
    if 'cluster' not in car_df.columns:
        print("🔄 Assigning clusters to cars...")
        car_df['cluster'] = car_df.apply(assign_cluster_to_car, axis=1)
        print(f"✓ Clusters assigned: {car_df['cluster'].value_counts().to_dict()}")

    # Tính điểm cho từng xe
    scores = []
    for idx, car in car_df.iterrows():
        try:
            rule_score = calculate_rule_score(car, survey_data)
            knn_score = calculate_knn_score_with_cluster(car, survey_data, car_df, user_cluster)

            # Kết hợp điểm: 60% rule + 40% KNN
            final_score = 0.6 * rule_score + 0.4 * knn_score

            # BONUS CLUSTER: Nếu xe cùng cluster với user -> tăng 15% tổng điểm
            if CLUSTER_ENABLED and user_cluster >= 0 and car.get('cluster') == user_cluster:
                final_score = min(1.0, final_score * 1.15)

            scores.append({
                'index': idx,
                'final_score': final_score,
                'rule_score': rule_score,
                'knn_score': knn_score,
                'cluster': car.get('cluster', -1)
            })
        except Exception as e:
            print(f"⚠️ Error processing car {idx}: {e}")
            continue

    if not scores:
        return pd.DataFrame()

    # Sắp xếp và lấy top N
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values('final_score', ascending=False)
    top_indices = scores_df.head(top_n)['index'].tolist()

    # Kết quả
    result = car_df.loc[top_indices].copy()
    result['final_score'] = scores_df.set_index('index').loc[top_indices, 'final_score'].values
    result['rule_score'] = scores_df.set_index('index').loc[top_indices, 'rule_score'].values
    result['knn_score'] = scores_df.set_index('index').loc[top_indices, 'knn_score'].values
    result['car_index'] = top_indices

    print(f"\n🎯 Top {top_n} recommendations:")
    for i, (idx, row) in enumerate(result.iterrows(), 1):
        print(f"{i}. {row['brand']} {row['model']} | "
              f"Score: {row['final_score']:.2f} | Cluster: {row.get('cluster', -1)}")

    return result
