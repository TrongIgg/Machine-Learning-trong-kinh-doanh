# app.py - Hiển thị xe với ảnh thật
import os
import csv
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)
app.secret_key = 'your_secret_key_here'

# Load dữ liệu xe
DATA_FILE = Path(__file__).parent / "data" / "data_car_demo.csv"


def load_cars_with_images():
    """Load dữ liệu xe - sử dụng cột img có sẵn"""
    df = pd.read_csv(DATA_FILE)

    # Sử dụng cột 'img' có sẵn từ CSV
    if 'img' in df.columns:
        df['image_path'] = df['img']
    else:
        df['image_path'] = 'img/default.jpg'

    # Tạo tên hiển thị
    df['display_name'] = df['brand'].astype(str) + ' ' + df['model'].astype(str)

    # Format giá
    df['price_display'] = df['price'].apply(
        lambda x: f"${x * 1000:,.0f}" if pd.notna(x) and x > 0 else "Liên hệ"
    )

    print(f"✓ Loaded {len(df)} cars with images")
    return df


# Load data khi khởi động
car_df = load_cars_with_images()


@app.route('/')
def index():
    """Trang chủ với pagination"""
    page = request.args.get('page', 1, type=int)
    per_page = 12

    start = (page - 1) * per_page
    end = start + per_page

    total_cars = len(car_df)
    total_pages = (total_cars + per_page - 1) // per_page

    cars = car_df.iloc[start:end].to_dict('records')

    # Lấy thông tin user từ session
    user_logged_in = session.get('logged_in', False)
    user_name = session.get('user_name', '')
    user_email = session.get('user_email', '')

    return render_template('index.html',
                           cars=cars,
                           current_page=page,
                           total_pages=total_pages,
                           total_cars=total_cars,
                           user_logged_in=user_logged_in,
                           user_name=user_name,
                           user_email=user_email)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        print(f"Login: {email}")

        # TODO: Lấy tên từ database thực tế
        # Tạm thời dùng email làm tên
        user_name = email.split('@')[0]  # Lấy phần trước @ làm tên

        session['logged_in'] = True
        session['user_email'] = email
        session['user_name'] = user_name
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        password = request.form.get('password')
        print(f"Register: {full_name} - {email}")

        session['logged_in'] = True
        session['user_email'] = email
        session['user_name'] = full_name  # Lưu tên
        return redirect(url_for('survey'))
    return render_template('register.html')


@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        data = request.form.to_dict()
        data['email'] = session.get('user_email', 'guest')
        print("Survey submitted:", data)

        filename = "survey_data.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

        return redirect(url_for('index'))

    if 'logged_in' not in session:
        return redirect(url_for('login'))

    return render_template('survey.html')

if __name__ == '__main__':
    app.run(debug=True)
