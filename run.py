import os
import csv
import json
import sqlite3
import pandas as pd
import sys
from pathlib import Path
import hashlib

# Import hybrid recommendation module
sys.path.append(str(Path(__file__).parent / 'app'))
try:
    from hybrid_recommend import recommend_cars, parse_survey_data
except ImportError:
    print("⚠️ Warning: hybrid_recommend not found, using fallback")
    recommend_cars = None
    parse_survey_data = None

from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)
app.secret_key = 'your_secret_key_here_change_in_production'

# File paths
ROOT_DIR = Path(__file__).parent
DATA_FILE = ROOT_DIR / "data" / "data_car_demo.csv"
USERS_FILE = ROOT_DIR / "data" / "users.csv"
SURVEY_FILE = ROOT_DIR / "data" / "survey_data.csv"
DB_FILE = ROOT_DIR / "data" / "app.db"

# Đảm bảo thư mục data tồn tại
(ROOT_DIR / "data").mkdir(exist_ok=True)


# ===== DATABASE SETUP =====
def init_db():
    """Khởi tạo database và tables"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Bảng recommendations
    c.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            email TEXT PRIMARY KEY,
            survey_data TEXT NOT NULL,
            car_indices TEXT NOT NULL,
            scores TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print("✓ Database initialized")


init_db()


# ===== CUSTOM FILTER =====
@app.template_filter('format_number')
def format_number(value):
    """Format number với dấu phẩy ngăn cách hàng nghìn"""
    try:
        return "{:,.0f}".format(float(value))
    except (ValueError, TypeError):
        return value


# ===== USER MANAGEMENT =====
def hash_password(password):
    """Hash mật khẩu với SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def save_user(full_name, email, password):
    """Lưu user vào CSV"""
    file_exists = USERS_FILE.exists()
    with open(USERS_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['full_name', 'email', 'password_hash', 'created_at'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'full_name': full_name,
            'email': email,
            'password_hash': hash_password(password),
            'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })


def get_user(email):
    """Lấy thông tin user từ CSV"""
    if not USERS_FILE.exists():
        return None
    try:
        users_df = pd.read_csv(USERS_FILE)
        user = users_df[users_df['email'] == email]
        if user.empty:
            return None
        return user.iloc[0].to_dict()
    except Exception as e:
        print(f"⚠️ Error reading users: {e}")
        return None


def verify_password(email, password):
    """Kiểm tra password"""
    user = get_user(email)
    if not user:
        return False
    return user['password_hash'] == hash_password(password)


# ===== CAR DATA =====
def load_cars_with_images():
    """Load dữ liệu xe với xử lý ảnh"""
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"✓ Loaded {len(df)} cars from {DATA_FILE}")

        if 'img' in df.columns:
            df['image_path'] = df['img']
        else:
            df['image_path'] = 'img/default.jpg'

        df['display_name'] = df['brand'].astype(str) + ' ' + df['model'].astype(str)

        def format_price(price):
            try:
                if pd.notna(price) and price > 0:
                    return f"${float(price) * 1000:,.0f}"
                return "Liên hệ"
            except:
                return "Liên hệ"

        df['price_display'] = df['price'].apply(format_price)
        return df
    except Exception as e:
        print(f"✗ Error loading cars: {e}")
        return pd.DataFrame()


car_df = load_cars_with_images()


# ===== SURVEY & RECOMMENDATIONS =====
def save_survey(email, survey_data):
    """Lưu survey data vào CSV"""
    import time
    survey_id = f"S{int(time.time())}"
    file_exists = SURVEY_FILE.exists()

    with open(SURVEY_FILE, mode='a', newline='', encoding='utf-8') as f:
        fieldnames = ['survey_id', 'email', 'submitted_at'] + list(survey_data.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {
            'survey_id': survey_id,
            'email': email,
            'submitted_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        row.update(survey_data)
        writer.writerow(row)


def save_recommendations_to_db(email, survey_data, recommended_df):
    """Lưu recommendations vào database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        now = pd.Timestamp.now().isoformat()

        # Lưu indices và scores
        car_indices = recommended_df.index.tolist()
        scores = {
            'final': recommended_df['final_score'].tolist(),
            'rule': recommended_df['rule_score'].tolist(),
            'knn': recommended_df['knn_score'].tolist()
        }

        c.execute('''
            INSERT OR REPLACE INTO recommendations 
            (email, survey_data, car_indices, scores, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            email,
            json.dumps(survey_data),
            json.dumps(car_indices),
            json.dumps(scores),
            now,
            now
        ))

        conn.commit()
        conn.close()
        print(f"✓ Saved recommendations to DB for {email}")
        return True
    except Exception as e:
        print(f"✗ DB save error: {e}")
        return False


def load_recommendations_from_db(email):
    """Load recommendations từ database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            SELECT survey_data, car_indices, scores 
            FROM recommendations 
            WHERE email = ?
        ''', (email,))

        result = c.fetchone()
        conn.close()

        if not result:
            return None, None

        survey_data = json.loads(result[0])
        car_indices = json.loads(result[1])
        scores = json.loads(result[2])

        # Merge với car_df
        recommended_cars = car_df.iloc[car_indices].copy()
        recommended_cars['final_score'] = scores['final']
        recommended_cars['rule_score'] = scores['rule']
        recommended_cars['knn_score'] = scores['knn']
        recommended_cars['car_index'] = car_indices

        return survey_data, recommended_cars.to_dict('records')

    except Exception as e:
        print(f"✗ DB load error: {e}")
        return None, None


# ===== ROUTES =====
@app.route('/')
def index():
    """Trang chủ với pagination và recommendations"""
    page = request.args.get('page', 1, type=int)
    per_page = 12

    user_logged_in = session.get('logged_in', False)
    recommended_cars = session.get('recommended_cars', [])
    survey_completed = session.get('survey_completed', False)

    start = (page - 1) * per_page
    end = start + per_page
    total_cars = len(car_df)
    total_pages = max((total_cars + per_page - 1) // per_page, 1)

    if recommended_cars and page == 1:
        recommended_ids = [car.get('car_index') for car in recommended_cars]
        remaining_df = car_df[~car_df.index.isin(recommended_ids)]
        cars = remaining_df.iloc[start:end].to_dict('records')
    else:
        cars = car_df.iloc[start:end].to_dict('records')

    return render_template('index.html',
                           cars=cars,
                           recommended_cars=recommended_cars if page == 1 else [],
                           survey_completed=survey_completed,
                           current_page=page,
                           total_pages=total_pages,
                           total_cars=total_cars,
                           user_logged_in=user_logged_in,
                           user_name=session.get('user_name', ''),
                           user_email=session.get('user_email', ''))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if verify_password(email, password):
            user = get_user(email)
            session['logged_in'] = True
            session['user_email'] = email
            session['user_name'] = user['full_name']

            # Load recommendations từ DB
            survey_data, recommended_cars = load_recommendations_from_db(email)

            if survey_data and recommended_cars:
                session['survey_data'] = survey_data
                session['recommended_cars'] = recommended_cars
                session['survey_completed'] = True
                print(f"✓ Loaded {len(recommended_cars)} recommendations for {email}")

            print(f"✓ Login successful: {email}")
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Email hoặc mật khẩu không đúng!")

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout và clear session"""
    session.clear()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not full_name or not email or not password:
            return render_template('register.html', error="Vui lòng điền đầy đủ thông tin!")

        if get_user(email):
            return render_template('register.html', error="Email đã được đăng ký!")

        try:
            save_user(full_name, email, password)
            print(f"✓ Registration successful: {full_name} - {email}")

            session['logged_in'] = True
            session['user_email'] = email
            session['user_name'] = full_name
            return redirect(url_for('survey'))

        except Exception as e:
            print(f"✗ Registration error: {e}")
            return render_template('register.html', error="Đã xảy ra lỗi, vui lòng thử lại!")

    return render_template('register.html')



@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """Survey page với recommendation generation"""
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            if parse_survey_data:
                survey_data = parse_survey_data(request.form)
            else:
                survey_data = {
                    'budget_min': int(request.form.get('budget_min', 0) or 0) / 1000,
                    'budget_max': int(request.form.get('budget_max', 100000) or 100000) / 1000,
                    'body_types': request.form.getlist('body_types'),
                    'fuel_type': request.form.get('fuel_type', ''),
                    'seating_capacity': int(request.form.get('seating_capacity', 5) or 5),
                    'engine_power_hp': int(request.form.get('engine_power_hp', 150) or 150),
                }

            email = session.get('user_email', 'guest')
            save_survey(email, survey_data)
            print(f"✓ Survey saved for {email}")

            if recommend_cars and not car_df.empty:
                # THAY ĐỔI TẠI ĐÂY: top_n=5 → top_n=6
                recommended = recommend_cars(survey_data, car_df, top_n=6)  # ← SỬA Ở ĐÂY

                # Lưu vào DB
                save_recommendations_to_db(email, survey_data, recommended)

                # Lưu vào session
                session['survey_completed'] = True
                session['recommended_cars'] = recommended.to_dict('records')
                session['survey_data'] = survey_data

                print(f"✓ Generated {len(recommended)} recommendations")
            else:
                print("⚠️ Recommendation unavailable")
                session['recommended_cars'] = []

        except Exception as e:
            print(f"✗ Survey processing error: {e}")
            import traceback
            traceback.print_exc()
            session['recommended_cars'] = []

        return redirect(url_for('index'))

    return render_template('survey.html')


@app.errorhandler(404)
def not_found(e):
    return "404 - Page not found", 404


@app.errorhandler(500)
def server_error(e):
    return "500 - Server error", 500


if __name__ == '__main__':
    print("=" * 50)
    print("🚗 Car Recommendation System")
    print(f"📊 Loaded {len(car_df)} cars")
    print(f"📁 Data directory: {ROOT_DIR / 'data'}")
    print(f"💾 Database: {DB_FILE}")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
