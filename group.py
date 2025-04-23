from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import psycopg2
import bcrypt
import re
import joblib
import pandas as pd
import json
import numpy as np
import requests  # Add this for API requests
from psycopg2.extras import DictCursor
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management

# OpenWeatherMap API Key
WEATHERAPI_API_KEY = "12649ebeb7cf4fb3ae4213124251403"  # Replace with your actual API key


# Database setup (unchanged)
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",  # Hostname of the PostgreSQL server
        port="8080",  # Port on which PostgreSQL is running
        dbname="Rainfall-Prediction",  # Name of the database
        user="postgres",  # Username for the database
        password="Munashe056"
    )
    conn.cursor_factory = DictCursor
    return conn


# Initialize database (unchanged)
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

    # Create rainfall_data table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rainfall_predictions (
            id SERIAL PRIMARY KEY,
            humidity FLOAT NOT NULL,
            temperature FLOAT NOT NULL,
            wind_speed FLOAT NOT NULL,
            predicted_rainfall FLOAT NOT NULL,
            prediction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


init_db()

# Load the trained models (unchanged)
MODEL = joblib.load('rainfall_model_20250409_0441.joblib')
FEATURES = joblib.load('feature_names.joblib')
with open('rainfall_categories.json') as f:
    CATEGORIES = json.load(f)


# Validation functions (unchanged)
def validate_email(email):
    """Validate email format using regex."""
    regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(regex, email) is not None


def validate_phone(phone):
    """Validate Zimbabwe phone number format (+263 followed by 9 digits)."""
    regex = r"^\+263\d{9}$"
    return re.match(regex, phone) is not None


def validate_password(password):
    """
    Validate password complexity:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    """
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):  # At least one uppercase letter
        return False
    if not re.search(r"[a-z]", password):  # At least one lowercase letter
        return False
    if not re.search(r"\d", password):  # At least one digit
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):  # At least one special character
        return False
    return True


# Password hashing (unchanged)
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def check_password(hashed_password, password):
    return bcrypt.checkpw(password.encode(), hashed_password)


def validate_temperature(temperature):
    """Validate that the temperature is within realistic bounds."""
    if not isinstance(temperature, (int, float)):
        return False
    if temperature < -89.2 or temperature > 56.7:
        return False
    return True


def validate_year(year):
    """
    Validate the year input.
    - Must be an integer or a string that can be converted to an integer.
    - Must be between 1900 and the current year + 10.
    """
    try:
        year = int(year)  # Convert to integer if it's a string
    except (ValueError, TypeError):
        return False

    current_year = datetime.now().year
    if year < 1900 or year > current_year + 10:
        return False

    return True


# Routes (unchanged except for new /weather-news route)
@app.route("/")
def landing():
    """Render the landing page by default."""
    return render_template("LandingPage.html")


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    fullname = data.get("fullname")
    address = data.get("address")
    phone_number = data.get("phone_number")
    email = data.get("email")
    username = data.get("username")
    city = data.get("city")
    password = data.get("password")

    # Validate input
    if not all([fullname, address, phone_number, email, username, city, password]):
        return jsonify({"error": "All fields are required"}), 400

    if not validate_email(email):
        return jsonify({"error": "Invalid email address"}), 400

    if not validate_phone(phone_number):
        return jsonify({"error": "Invalid phone number. Use +263 format."}), 400

    if not validate_password(password):
        return jsonify({
            "error": "Password must be at least 8 characters long and include: "
                     "1 uppercase letter, 1 lowercase letter, 1 digit, and 1 special character."
        }), 400

    # Hash password
    hashed_password = hash_password(password)

    # Save user to database
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO users (fullname, address, phone_number, email, username,city,password)
            VALUES (%s, %s, %s, %s, %s, %s,%s)
        """, (fullname, address, phone_number, email, username, city, hashed_password.decode()))
        conn.commit()
        return jsonify({"message": "Registration successful! Please login."}), 201
    except psycopg2.IntegrityError as e:
        if "duplicate key value violates unique constraint" in str(e):
            if "users_email_key" in str(e):
                return jsonify({"error": "Email already exists"}), 400
            elif "users_username_key" in str(e):
                return jsonify({"error": "Username already exists"}), 400
        return jsonify({"error": "Registration failed"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        # Render the login page
        return render_template("Login.html")
    elif request.method == "POST":
        # Handle login logic
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        # Validate input
        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400

        # Fetch user from database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password(user["password"].encode(), password):
            # Set session for the logged-in user
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return jsonify({"message": "Login successful!", "redirect": url_for("dashboard")}), 200
        else:
            return jsonify({"error": "Invalid username or password"}), 401


@app.route("/dashboard")
def dashboard():
    """Render the dashboard page for logged-in users."""
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Fetch weather news for Harare
    weather_data = get_weather_news()  # No argument passed
    return render_template("index.html", weather_data=weather_data)


@app.route("/logout", methods=["POST"])
def logout():
    """Log out the user by clearing the session."""
    session.clear()  # Clear the session data
    return jsonify({"message": "Logged out successfully!"}), 200


@app.route("/change-password", methods=["POST"])
def change_password():
    """Change the user's password."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    current_password = data.get("current_password")
    new_password = data.get("new_password")
    confirm_new_password = data.get("confirm_new_password")

    # Validate input
    if not all([current_password, new_password, confirm_new_password]):
        return jsonify({"error": "All fields are required"}), 400

    if new_password != confirm_new_password:
        return jsonify({"error": "New passwords do not match"}), 400

    if not validate_password(new_password):
        return jsonify({
            "error": "Password must be at least 8 characters long and include: "
                     "1 uppercase letter, 1 lowercase letter, 1 digit, and 1 special character."
        }), 400

    # Fetch user from database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (session["user_id"],))
    user = cur.fetchone()

    if not user or not check_password(user["password"].encode(), current_password):
        cur.close()
        conn.close()
        return jsonify({"error": "Current password is incorrect"}), 400

    # Hash new password
    hashed_password = hash_password(new_password)

    # Update password in database
    cur.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_password.decode(), session["user_id"]))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Password changed successfully!"}), 200


@app.route("/edit-profile", methods=["POST"])
def edit_profile():
    """Edit the user's profile."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    fullname = data.get("fullname")
    email = data.get("email")
    phone_number = data.get("phone_number")
    city = data.get("city")

    # Validate input
    if not all([fullname, email, phone_number, city]):
        return jsonify({"error": "All fields are required"}), 400

    if not validate_email(email):
        return jsonify({"error": "Invalid email address"}), 400

    if not validate_phone(phone_number):
        return jsonify({"error": "Invalid phone number. Use +263 format."}), 400

    # Update profile in database
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE users
            SET fullname = %s, email = %s, phone_number = %s
            WHERE id = %s
        """, (fullname, email, phone_number, city, session["user_id"]))
        conn.commit()
        return jsonify({"message": "Profile updated successfully!"}), 200
    except psycopg2.IntegrityError as e:
        # Handle unique constraint violation (e.g., duplicate email)
        if "duplicate key value violates unique constraint" in str(e):
            return jsonify({"error": "Email already exists"}), 400
        else:
            return jsonify({"error": "Profile update failed"}), 400
    except Exception as e:
        # Handle other exceptions
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()


# Fetch Rainfall Analytics Data (unchanged)
@app.route("/api/rainfall-analytics")
def rainfall_analytics():
    """Fetch historical rainfall data from the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Fetch all rainfall data
        cur.execute("SELECT year, rainfall FROM rainfall_data ORDER BY year ASC")
        data = cur.fetchall()
        # Convert to a list of dictionaries
        rainfall_data = [{"year": row["year"], "rainfall": row["rainfall"]} for row in data]
        return jsonify(rainfall_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route("/predict-rainfall", methods=["POST"])
def predict_rainfall():
    # Get input data from the frontend
    data = request.get_json()

    # Extract and validate all required parameters
    humidity = data.get("humidity")
    temperature = data.get("temperature")
    wind_speed = data.get("wind_speed")

    # Validate all inputs
    if None in (humidity, temperature, wind_speed):
        return jsonify({
            "error": "All parameters are required",
            "required_parameters": ["humidity", "temperature", "wind_speed"]
        }), 400

    validation_errors = []

    # Validate humidity (0-100%)
    if not (0 <= float(humidity) <= 100):
        validation_errors.append("Humidity must be between 0% and 100%")

    # Validate temperature (realistic Earth temperatures)
    if not (-89.2 <= float(temperature) <= 56.7):
        validation_errors.append("Temperature must be between -89.2°C and 56.7°C")

    # Validate wind speed (0-400 km/h)
    if not (0 <= float(wind_speed) <= 400):
        validation_errors.append("Wind speed must be between 0 and 400 km/h")

    if validation_errors:
        return jsonify({
            "error": "Invalid parameter values",
            "details": validation_errors
        }), 400

    # Prepare input for the models
    try:
        input_data = pd.DataFrame({
            'Humidity (%)': [float(humidity)],
            'Annual Mean Temp': [float(temperature)],
            'Wind Speed (km/h)': [float(wind_speed)]
        })

        # Make prediction
        predicted_rainfall = model.predict(input_data)[0]
        predicted_rainfall = round(float(predicted_rainfall), 2)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

    # Save the prediction to database
    try:
        conn = get_db_connection()  # Get PostgreSQL connection
        cur = conn.cursor()

        # Insert the prediction into the database
        cur.execute("""
            INSERT INTO rainfall_predictions 
            (humidity, temperature, wind_speed, predicted_rainfall, prediction_date)
            VALUES (%s, %s, %s, %s, NOW())
        """, (humidity, temperature, wind_speed, predicted_rainfall))

        conn.commit()  # Commit the transaction to save the data

    except Exception as e:
        # Log the error but don't fail the request
        app.logger.error(f"Database save failed: {str(e)}")
    finally:
        if 'conn' in locals():
            cur.close()  # Close the cursor
            conn.close()  # Close the database connection

    # Return successful prediction
    return jsonify({
        "prediction": predicted_rainfall,
        "units": "mm",
        "parameters": {
            "humidity": f"{humidity}%",
            "temperature": f"{temperature}°C",
            "wind_speed": f"{wind_speed} km/h"
        }
    }), 200


@app.route("/api/historical-data", methods=["GET"])
def get_historical_data():
    """Fetch historical rainfall prediction data from the database."""
    try:
        # Get optional query parameters for filtering
        limit = request.args.get('limit', default=100, type=int)
        sort = request.args.get('sort', default='desc')  # 'asc' or 'desc'

        # Validate parameters
        if limit < 1 or limit > 1000:
            return jsonify({"error": "Limit must be between 1 and 1000"}), 400

        if sort not in ['asc', 'desc']:
            return jsonify({"error": "Sort must be either 'asc' or 'desc'"}), 400

        # Connect to database
        conn = get_db_connection()
        cur = conn.cursor()

        # Build query with sorting
        query = """
            SELECT id, humidity, temperature, wind_speed, predicted_rainfall, 
                   prediction_date AT TIME ZONE 'UTC' AT TIME ZONE 'Africa/Harare' as prediction_date
            FROM rainfall_predictions
            ORDER BY prediction_date {}
            LIMIT %s
        """.format('ASC' if sort == 'asc' else 'DESC')

        cur.execute(query, (limit,))
        data = cur.fetchall()

        # Convert to list of dictionaries
        historical_data = []
        for row in data:
            historical_data.append({
                "id": row["id"],
                "humidity": row["humidity"],
                "temperature": row["temperature"],
                "wind_speed": row["wind_speed"],
                "predicted_rainfall": row["predicted_rainfall"],
                "prediction_date": row["prediction_date"].isoformat() if row["prediction_date"] else None
            })

        return jsonify({
            "success": True,
            "data": historical_data,
            "count": len(historical_data)
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


# New Route: Fetch Weather News
@app.route('/get_weather_news', methods=['GET'])  # Changed to GET since we're getting user from session
def get_weather_news():
    try:
        # Get user ID from session (you'll need to implement session management)
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'User not logged in'}), 401

        # Get user's city from database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT city FROM users WHERE id = %s", (user_id,))
        user_data = cur.fetchone()
        cur.close()
        conn.close()

        if not user_data:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        location = user_data[0]  # Get city from query result

        # Using wttr.in to get weather data
        url = f"https://wttr.in/{location}?format=%t+%h+%w+%C"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return jsonify({'success': False, 'error': 'Failed to fetch weather data'}), 500

        parts = response.text.split(' ', 3)  # Split into 4 parts

        # Get current date and time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%B %d, %Y")
        day_of_week = now.strftime("%A")

        return jsonify({
            'success': True,
            'temperature': parts[0],
            'humidity': parts[1],
            'wind': parts[2],
            'condition': parts[3] if len(parts) > 3 else 'N/A',
            'time': current_time,
            'date': current_date,
            'day': day_of_week,
            'location': location
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Run the app
if __name__ == "__main__":
    app.run(debug=True)