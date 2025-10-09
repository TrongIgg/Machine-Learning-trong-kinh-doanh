import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# ===== LOAD CLUSTER MODELS =====
try:
    cluster_scaler = load(MODELS_DIR / "scaler.joblib")
    cluster_kmeans = load(MODELS_DIR / "kmeans.joblib")

    # Load feature order và weights
    feature_order_df = pd.read_csv(MODELS_DIR / "feature_order.csv")
    cluster_features = feature_order_df['feature'].tolist()

    weights_df = pd.read_csv(MODELS_DIR / "weights.csv")
    cluster_weights = weights_df['weight'].values

    CLUSTER_ENABLED = True
    print("✓ Cluster models loaded successfully")
    print(f"  - Features: {cluster_features}")
    print(f"  - Weights: {cluster_weights.tolist()}")
except Exception as e:
    print(f"⚠️ Cluster models not found: {e}")
    CLUSTER_ENABLED = False
    cluster_features = []
    cluster_weights = np.array([1.0, 1.0, 1.0])

# ===== LOAD KNN MODEL =====
try:
    knn_model = load(MODELS_DIR / "knn_model.joblib")
    KNN_ENABLED = True
    print("✓ KNN model loaded successfully")
except Exception as e:
    print(f"⚠️ KNN model not found: {e}")
    KNN_ENABLED = False


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


def prepare_cluster_features(car_row):
    """Chuẩn bị features cho cluster (giống train_cluster.py)"""
    try:
        price = car_row.get('price', 0)
        hp = car_row.get('veenginepower', 0)
        fuel = car_row.get('fuelconsumption', 0)

        if price <= 0 or hp <= 0 or fuel <= 0:
            return None

        # Log transform giống train
        price_log = np.log1p(price)
        hp_log = np.log1p(hp)

        return np.array([[price_log, hp_log, fuel]])
    except:
        return None


def assign_cluster_to_car(car_row):
    """Gán cluster cho xe"""
    if not CLUSTER_ENABLED:
        return -1

    try:
        X = prepare_cluster_features(car_row)
        if X is None:
            return -1

        # Scale và weight
        X_scaled = cluster_scaler.transform(X)
        X_weighted = X_scaled * cluster_weights

        # Predict cluster
        cluster_id = cluster_kmeans.predict(X_weighted)[0]
        return int(cluster_id)
    except Exception as e:
        print(f"⚠️ Error assigning cluster: {e}")
        return -1


def get_user_preferred_cluster(survey_data):
    """Xác định cluster phù hợp với user"""
    if not CLUSTER_ENABLED:
        return -1

    try:
        # Tính từ survey
        avg_budget = (survey_data['budget_min'] + survey_data['budget_max']) / 2
        price_log = np.log1p(avg_budget)
        hp_log = np.log1p(survey_data['engine_power_hp'])

        # Estimate fuel consumption
        estimated_fuel = 5.0 + (survey_data['engine_power_hp'] - 100) * 0.02
        estimated_fuel = max(4.0, min(estimated_fuel, 15.0))

        X = np.array([[price_log, hp_log, estimated_fuel]])
        X_scaled = cluster_scaler.transform(X)
        X_weighted = X_scaled * cluster_weights

        cluster_id = cluster_kmeans.predict(X_weighted)[0]
        return int(cluster_id)
    except Exception as e:
        print(f"⚠️ Error getting user cluster: {e}")
        return -1


def calculate_rule_score(car, survey):
    """Tính điểm dựa trên rules"""
    score = 0.0
    weights = []

    # 1. Budget (3.0)
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

    # 2. Body type (2.5)
    if survey['body_types']:
        body_match = any(
            str(car.get('bodytype', '')).lower() == bt.lower()
            for bt in survey['body_types']
        )
        if body_match:
            score += 2.5
        weights.append(2.5)

    # 3. Brand (1.5)
    if survey['preferred_brands']:
        brand_match = any(
            str(car.get('brand', '')).lower() == pb.lower()
            for pb in survey['preferred_brands']
        )
        if brand_match:
            score += 1.5
        weights.append(1.5)

    # 4. Fuel type (2.0)
    if survey['fuel_type']:
        if str(car.get('fueltype', '')).lower() == survey['fuel_type'].lower():
            score += 2.0
        weights.append(2.0)

    # 5. Seating (1.5)
    if car.get('seatingcapacity'):
        seat_diff = abs(car['seatingcapacity'] - survey['seating_capacity'])
        if seat_diff == 0:
            score += 1.5
        elif seat_diff <= 1:
            score += 1.0
        elif seat_diff <= 2:
            score += 0.5
        weights.append(1.5)

    # 6. Engine power (1.5)
    if car.get('veenginepower') and car['veenginepower'] > 0:
        power_ratio = car['veenginepower'] / survey['engine_power_hp']
        if 0.8 <= power_ratio <= 1.2:
            score += 1.5
        elif 0.6 <= power_ratio <= 1.4:
            score += 1.0
        elif 0.4 <= power_ratio <= 1.6:
            score += 0.5
        weights.append(1.5)

    max_score = sum(weights) if weights else 1.0
    return score / max_score


def calculate_knn_score(car, survey, car_df, user_cluster):
    """Tính KNN score với cluster bonus"""
    try:
        # Features
        numeric_features = ['price', 'veenginepower', 'fuelconsumption', 'seatingcapacity']

        # User vector
        user_vector = np.array([
            (survey['budget_min'] + survey['budget_max']) / 2,
            survey['engine_power_hp'],
            7.0,
            survey['seating_capacity']
        ]).reshape(1, -1)

        # Valid cars
        valid_cars = car_df[
            (car_df['price'] > 0) &
            (car_df['veenginepower'] > 0) &
            (car_df['fuelconsumption'] > 0) &
            (car_df['seatingcapacity'] > 0)
            ].copy()

        if len(valid_cars) < 5:
            return 0.5

        # Gán cluster nếu chưa có
        if 'cluster' not in valid_cars.columns or valid_cars['cluster'].isna().any():
            valid_cars['cluster'] = valid_cars.apply(assign_cluster_to_car, axis=1)

        car_vectors = valid_cars[numeric_features].values

        # Scale
        scaler = StandardScaler()
        car_vectors_scaled = scaler.fit_transform(car_vectors)
        user_vector_scaled = scaler.transform(user_vector)

        # KNN
        k = min(20, len(valid_cars))
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(car_vectors_scaled)

        distances, indices = knn.kneighbors(user_vector_scaled)

        # Check xe hiện tại
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

            # CLUSTER BONUS
            if CLUSTER_ENABLED and user_cluster >= 0:
                car_cluster = valid_cars.loc[car_idx, 'cluster']
                if car_cluster == user_cluster:
                    base_score = min(1.0, base_score * 1.25)  # +25%

            return base_score

        # Fallback
        car_vector_scaled = scaler.transform(
            valid_cars.loc[car_idx, numeric_features].values.reshape(1, -1)
        )
        dist = np.linalg.norm(user_vector_scaled - car_vector_scaled)
        max_dist = distances[0][-1]

        score = max(0, 1 - (dist / (max_dist + 1e-6))) * 0.5

        # CLUSTER BONUS
        if CLUSTER_ENABLED and user_cluster >= 0:
            car_cluster = valid_cars.loc[car_idx, 'cluster']
            if car_cluster == user_cluster:
                score = min(1.0, score * 1.25)

        return score

    except Exception as e:
        print(f"⚠️ KNN error: {e}")
        return 0.3


def recommend_cars(survey_data, car_df, top_n=6):
    """
    Main recommendation function
    top_n: Số lượng xe gợi ý (default=6 để grid đẹp)
    """
    if car_df.empty:
        return pd.DataFrame()

    # User cluster
    user_cluster = get_user_preferred_cluster(survey_data)
    print(f"👤 User preferred cluster: {user_cluster}")

    # Gán cluster cho xe
    if 'cluster' not in car_df.columns:
        print("🔄 Assigning clusters...")
        car_df['cluster'] = car_df.apply(assign_cluster_to_car, axis=1)
        print(f"✓ Clusters: {car_df['cluster'].value_counts().to_dict()}")

    # Calculate scores
    scores = []
    for idx, car in car_df.iterrows():
        try:
            rule_score = calculate_rule_score(car, survey_data)
            knn_score = calculate_knn_score(car, survey_data, car_df, user_cluster)

            # Combine: 60% rule + 40% KNN
            final_score = 0.6 * rule_score + 0.4 * knn_score

            # CLUSTER BOOST: +15% nếu cùng cluster
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

    # Sort và lấy top N
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values('final_score', ascending=False)
    top_indices = scores_df.head(top_n)['index'].tolist()

    # Result
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
