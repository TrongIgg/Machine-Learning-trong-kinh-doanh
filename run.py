from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import json
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Paths
ROOT = Path(__file__).resolve().parent

# Sample car data (thay thế cho việc load từ CSV)
SAMPLE_CARS = [
    {
        'id': '1',
        'brand': 'Mercedes-Benz',
        'model': 'S-Class 2024',
        'type': 'sedan',
        'price': 3200000000,
        'horsepower': 429,
        'fuel_consumption': 8.5,
        'specs': 'V6 3.0L • Hybrid • AWD • 429 HP'
    },
    {
        'id': '2',
        'brand': 'BMW',
        'model': 'X7 2024',
        'type': 'suv',
        'price': 4800000000,
        'horsepower': 523,
        'fuel_consumption': 11.2,
        'specs': 'V8 4.4L • Twin Turbo • AWD • 523 HP'
    },
    {
        'id': '3',
        'brand': 'Audi',
        'model': 'Q8 2024',
        'type': 'suv',
        'price': 3900000000,
        'horsepower': 340,
        'fuel_consumption': 9.8,
        'specs': 'V6 3.0L • TFSI • Quattro • 340 HP'
    },
    {
        'id': '4',
        'brand': 'Lexus',
        'model': 'LX 600 2024',
        'type': 'suv',
        'price': 7200000000,
        'horsepower': 409,
        'fuel_consumption': 13.1,
        'specs': 'V6 3.5L • Twin Turbo • AWD • 409 HP'
    },
    {
        'id': '5',
        'brand': 'Porsche',
        'model': '911 Carrera',
        'type': 'coupe',
        'price': 8500000000,
        'horsepower': 379,
        'fuel_consumption': 8.9,
        'specs': 'H6 3.0L • Twin Turbo • RWD • 379 HP'
    },
    {
        'id': '6',
        'brand': 'Bentley',
        'model': 'Continental GT',
        'type': 'coupe',
        'price': 15800000000,
        'horsepower': 626,
        'fuel_consumption': 14.7,
        'specs': 'W12 6.0L • Twin Turbo • AWD • 626 HP'
    }
]


# Database setup
def init_db():
    with sqlite3.connect('users.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            preferences TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            car_id TEXT,
            FOREIGN KEY (username) REFERENCES users (username)
        )''')
        conn.commit()


init_db()


def format_price(price):
    """Format price to VND currency"""
    return f"{price:,.0f} VND"


def get_user_favorites(username):
    """Get user's favorite cars"""
    if not username:
        return []
    with sqlite3.connect('users.db') as conn:
        c = conn.cursor()
        c.execute("SELECT car_id FROM favorites WHERE username = ?", (username,))
        return [row[0] for row in c.fetchall()]


def get_filtered_cars(min_price=None, max_price=None, brand=None, car_type=None):
    """Filter cars based on criteria"""
    filtered = SAMPLE_CARS.copy()

    if min_price:
        filtered = [car for car in filtered if car['price'] >= min_price]
    if max_price:
        filtered = [car for car in filtered if car['price'] <= max_price]
    if brand:
        filtered = [car for car in filtered if car['brand'].lower() == brand.lower()]
    if car_type:
        filtered = [car for car in filtered if car['type'].lower() == car_type.lower()]

    return filtered


@app.route('/')
def index():
    current_user = session.get('username')
    cars = SAMPLE_CARS
    user_favorites = get_user_favorites(current_user)

    # Add favorite status to each car
    for car in cars:
        car['is_favorite'] = car['id'] in user_favorites

    return render_template('index.html',
                           cars=cars,
                           current_user=current_user,
                           favorite_count=len(user_favorites),
                           format_price=format_price)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            return render_template('login.html', error="Vui lòng nhập đầy đủ thông tin")

        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
            user = c.fetchone()

            if user:
                session['username'] = username
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error="Sai tên đăng nhập hoặc mật khẩu")

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        if not username or not password:
            return render_template('register.html', error="Vui lòng nhập đầy đủ thông tin")

        if password != confirm_password:
            return render_template('register.html', error="Mật khẩu không khớp")

        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ?", (username,))

            if c.fetchone():
                return render_template('register.html', error="Tên đăng nhập đã tồn tại")

            c.execute("INSERT INTO users (username, password, preferences) VALUES (?, ?, ?)",
                      (username, password, None))
            conn.commit()

            session['username'] = username
            return redirect(url_for('survey'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        try:
            budget = float(request.form['budget'])
            horsepower = float(request.form['horsepower'])
            fuel = float(request.form['fuel'])

            preferences = {
                'budget': budget,
                'horsepower': horsepower,
                'fuel_consumption': fuel
            }

            if 'username' in session:
                with sqlite3.connect('users.db') as conn:
                    c = conn.cursor()
                    c.execute("UPDATE users SET preferences = ? WHERE username = ?",
                              (json.dumps(preferences), session['username']))
                    conn.commit()
            else:
                session['preferences'] = preferences

            return redirect(url_for('index'))

        except (ValueError, KeyError):
            return render_template('survey.html', error="Vui lòng nhập số hợp lệ")

    return render_template('survey.html')


@app.route('/filter', methods=['POST'])
def filter_cars():
    try:
        min_price = request.form.get('min_price')
        max_price = request.form.get('max_price')
        brand = request.form.get('brand')
        car_type = request.form.get('type')

        min_price = float(min_price) if min_price else None
        max_price = float(max_price) if max_price else None

        filtered_cars = get_filtered_cars(min_price, max_price, brand, car_type)

        current_user = session.get('username')
        user_favorites = get_user_favorites(current_user)

        # Add favorite status
        for car in filtered_cars:
            car['is_favorite'] = car['id'] in user_favorites

        return render_template('index.html',
                               cars=filtered_cars,
                               current_user=current_user,
                               favorite_count=len(user_favorites),
                               format_price=format_price)

    except (ValueError, TypeError):
        return redirect(url_for('index'))


@app.route('/search', methods=['POST'])
def search():
    search_term = request.form.get('search', '').lower()

    if not search_term:
        return redirect(url_for('index'))

    filtered_cars = [
        car for car in SAMPLE_CARS
        if search_term in car['brand'].lower()
           or search_term in car['model'].lower()
           or search_term in car['type'].lower()
    ]

    current_user = session.get('username')
    user_favorites = get_user_favorites(current_user)

    for car in filtered_cars:
        car['is_favorite'] = car['id'] in user_favorites

    return render_template('index.html',
                           cars=filtered_cars,
                           current_user=current_user,
                           favorite_count=len(user_favorites),
                           format_price=format_price,
                           search_term=search_term)


@app.route('/favorite', methods=['POST'])
def toggle_favorite():
    data = request.get_json()
    car_id = data.get('car_id')
    username = session.get('username')

    if not username:
        return jsonify({'success': False, 'message': 'Vui lòng đăng nhập'})

    if not car_id:
        return jsonify({'success': False, 'message': 'ID xe không hợp lệ'})

    with sqlite3.connect('users.db') as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM favorites WHERE username = ? AND car_id = ?", (username, car_id))

        if c.fetchone():
            # Remove from favorites
            c.execute("DELETE FROM favorites WHERE username = ? AND car_id = ?", (username, car_id))
            action = 'removed'
        else:
            # Add to favorites
            c.execute("INSERT INTO favorites (username, car_id) VALUES (?, ?)", (username, car_id))
            action = 'added'

        conn.commit()

        # Get updated favorite count
        c.execute("SELECT COUNT(*) FROM favorites WHERE username = ?", (username,))
        favorite_count = c.fetchone()[0]

        return jsonify({
            'success': True,
            'action': action,
            'favorite_count': favorite_count
        })


@app.route('/favorites')
def favorites():
    username = session.get('username')

    if not username:
        return redirect(url_for('login'))

    user_favorites = get_user_favorites(username)
    favorite_cars = [car for car in SAMPLE_CARS if car['id'] in user_favorites]

    for car in favorite_cars:
        car['is_favorite'] = True

    return render_template('index.html',
                           cars=favorite_cars,
                           current_user=username,
                           favorite_count=len(user_favorites),
                           format_price=format_price,
                           page_title="Xe yêu thích")


# Template filters
@app.template_filter('format_price')
def format_price_filter(price):
    return format_price(price)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
