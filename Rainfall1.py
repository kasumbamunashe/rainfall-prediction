from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import psycopg2
import bcrypt
import re
import joblib
import pandas as pd
import numpy as np
import requests
import json
from psycopg2.extras import DictCursor
from datetime import datetime
from pydantic import BaseModel, confloat, validator
from typing import Optional, Dict, List, Any
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change this to a secure secret key

# Configuration
WEATHERAPI_API_KEY = "12649ebeb7cf4fb3ae4213124251403"  # Replace with your actual API key


# Database setup
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        port="8080",
        dbname="Rainfall-Prediction",
        user="postgres",
        password="Munashe056"
    )
    conn.cursor_factory = DictCursor
    return conn


# Initialize database
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # Create users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            fullname TEXT NOT NULL,
            address TEXT NOT NULL,
            phone_number TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            city TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Create rainfall_predictions table with category
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rainfall_predictions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            humidity FLOAT NOT NULL,
            temperature FLOAT NOT NULL,
            wind_speed FLOAT NOT NULL,
            predicted_rainfall FLOAT NOT NULL,
            category TEXT NOT NULL,
            prediction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


init_db()

# Load model assets
MODEL = joblib.load('model.joblib')
FEATURES = joblib.load('feature_names.joblib')
with open('rainfall_categories.json') as f:
    CATEGORIES = json.load(f)


# Pydantic models for validation
class RainfallRequest(BaseModel):
    humidity: confloat(ge=0, le=100)  # 0-100%
    temperature: confloat(ge=-40, le=60)  # -40°C to 60°C
    wind_speed: confloat(ge=0, le=300)  # 0-300 km/h

    @validator('*')
    def round_values(cls, v):
        return round(v, 2) if isinstance(v, float) else v


class UserRegistration(BaseModel):
    fullname: str
    address: str
    phone_number: str
    email: str
    username: str
    city: str
    password: str


# Helper functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(hashed_password: str, password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed_password.encode())


def validate_email(email: str) -> bool:
    regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(regex, email) is not None


def validate_phone(phone: str) -> bool:
    regex = r"^\+263\d{9}$"
    return re.match(regex, phone) is not None


def validate_password(password: str) -> bool:
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True


# Prediction function with agricultural insights
def predict_with_insights(humidity: float, temperature: float, wind_speed: float) -> Dict[str, Any]:
    try:
        input_data = np.array([[humidity, temperature, wind_speed]])
        prediction = MODEL.predict(input_data)[0]

        # Determine category
        category = next(
            (cat for cat, vals in CATEGORIES.items()
             if vals['min'] <= prediction < vals['max']),
            "Unknown"
        )

        # Generate recommendations
        recommendations = {
            "Low": [
                "Plant drought-resistant crops (e.g., sorghum, millet)",
                "Implement drip irrigation systems",
                "Apply mulching to conserve soil moisture"
            ],
            "Moderate": [
                "Suitable for wheat, maize, and legumes",
                "Consider supplemental irrigation during dry spells",
                "Practice crop rotation for soil health"
            ],
            "High": [
                "Ideal for rice, sugarcane, and banana cultivation",
                "Excellent for tea and coffee plantations",
                "Install proper drainage to prevent waterlogging"
            ],
            "Excessive": [
                "Plant flood-resistant varieties (e.g., deepwater rice)",
                "Construct raised beds for crops",
                "Implement terracing to control erosion"
            ]
        }.get(category, [])

        return {
            "prediction_mm": round(float(prediction), 2),
            "category": category,
            "description": CATEGORIES.get(category, {}).get("description", ""),
            "recommendations": recommendations
        }

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")


# Routes
@app.route("/")
def landing():
    return render_template("LandingPage.html")


@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        user_data = UserRegistration(**data)

        if not validate_email(user_data.email):
            return jsonify({"error": "Invalid email address"}), 400

        if not validate_phone(user_data.phone_number):
            return jsonify({"error": "Invalid phone number. Use +263 format."}), 400

        if not validate_password(user_data.password):
            return jsonify({
                "error": "Password must be at least 8 characters with uppercase, lowercase, number, and special character"
            }), 400

        hashed_password = hash_password(user_data.password)

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO users (fullname, address, phone_number, email, username, city, password)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                user_data.fullname, user_data.address, user_data.phone_number,
                user_data.email, user_data.username, user_data.city, hashed_password
            ))
            conn.commit()
            return jsonify({"message": "Registration successful!"}), 201
        except psycopg2.IntegrityError as e:
            if "users_email_key" in str(e):
                return jsonify({"error": "Email already exists"}), 400
            elif "users_username_key" in str(e):
                return jsonify({"error": "Username already exists"}), 400
            return jsonify({"error": "Registration failed"}), 400
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()

        if user and check_password(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return jsonify({
                "message": "Login successful",
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"]
                }
            }), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401
    finally:
        cur.close()
        conn.close()


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"}), 200


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("landing"))

    weather_data = get_weather_news()
    return render_template("index.html", weather_data=weather_data)


@app.route("/predict-rainfall", methods=["POST"])
def predict_rainfall():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json()
        request_data = RainfallRequest(**data)

        # Make prediction
        result = predict_with_insights(
            humidity=request_data.humidity,
            temperature=request_data.temperature,
            wind_speed=request_data.wind_speed
        )

        # Save to database
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO rainfall_predictions 
                (user_id, humidity, temperature, wind_speed, predicted_rainfall, category)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                session["user_id"],
                request_data.humidity,
                request_data.temperature,
                request_data.wind_speed,
                result["prediction_mm"],
                result["category"]
            ))
            conn.commit()
        finally:
            cur.close()
            conn.close()

        return jsonify({
            "status": "success",
            "prediction": result["prediction_mm"],
            "category": result["category"],
            "description": result["description"],
            "recommendations": result["recommendations"]
        }), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/historical-data", methods=["GET"])
def get_historical_data():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    limit = request.args.get('limit', default=10, type=int)
    if limit < 1 or limit > 100:
        return jsonify({"error": "Limit must be between 1 and 100"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, humidity, temperature, wind_speed, predicted_rainfall, category, 
                   prediction_date AT TIME ZONE 'UTC' AT TIME ZONE 'Africa/Harare' as prediction_date
            FROM rainfall_predictions
            WHERE user_id = %s
            ORDER BY prediction_date DESC
            LIMIT %s
        """, (session["user_id"], limit))

        data = cur.fetchall()
        return jsonify([dict(row) for row in data]), 200
    finally:
        cur.close()
        conn.close()


@app.route('/get_weather_news', methods=['GET'])
def get_weather_news():
    if "user_id" not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT city FROM users WHERE id = %s", (session["user_id"],))
        user_data = cur.fetchone()
        if not user_data:
            return jsonify({'error': 'User not found'}), 404

        location = user_data["city"]
        url = f"https://wttr.in/{location}?format=%t+%h+%w+%C"
        headers = {'User-Agent': 'Mozilla/5.0'}

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return jsonify({'error': 'Weather service unavailable'}), 503

        parts = response.text.split(' ', 3)
        now = datetime.now()

        return jsonify({
            'temperature': parts[0],
            'humidity': parts[1],
            'wind': parts[2],
            'condition': parts[3] if len(parts) > 3 else 'N/A',
            'time': now.strftime("%H:%M:%S"),
            'date': now.strftime("%B %d, %Y"),
            'day': now.strftime("%A"),
            'location': location
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": "1.0"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)