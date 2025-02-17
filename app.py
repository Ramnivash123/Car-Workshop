from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from PIL import Image  # Import the Image module



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Use SQLite (change for MySQL/PostgreSQL)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'ajbscajlcbaljcnalndc'

db = SQLAlchemy(app)

# User model
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  

class Receiptss(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    car_name = db.Column(db.String(100), nullable=False)
    engine_oil_cost = db.Column(db.Integer, nullable=False)
    oil_filter_cost = db.Column(db.Integer, nullable=False)
    air_filter_cost = db.Column(db.Integer, nullable=False)
    update_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Create database tables before running
with app.app_context():
    db.create_all()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]  # Hash passwords in real applications
        
        # Check if user already exists
        existing_user = Users.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered!", "danger")
            return redirect(url_for("register"))

        # Insert into database
        new_user = Users(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful! You can now log in.", "success")
        return redirect(url_for("register2"))

    return render_template("register.html")

@app.route("/register2", methods=["GET", "POST"])
def register2():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = Users.query.filter_by(email=email, password=password).first()
        if user:
            session["user_id"] = user.id
            session["username"] = user.username
            flash("Login successful!", "success")
            return redirect(url_for("service"))

        flash("Invalid email or password.", "danger")
        return redirect(url_for("register2"))

    return render_template("register2.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

@app.route("/service")
def service():
    if "user_id" not in session:
        flash("Please log in to continue.", "warning")
        return redirect(url_for("register2"))

    return render_template("service.html", username=session["username"])



@app.route('/upload_receipt', methods=['GET', 'POST'])
def upload_receipt():
    if request.method == 'POST':
        car_name = request.form["car_name"]
        engine_oil_cost = request.form['engine_oil_cost']
        oil_filter_cost = request.form['oil_filter_cost']
        air_filter_cost = request.form['air_filter_cost']
        update_date = datetime.utcnow()

        # Assuming you have the 'car_name' and 'username' available in your context
        new_receipt = Receiptss(
            username=session["username"],  # Replace with actual username
            car_name=car_name,   # Replace with actual car name
            engine_oil_cost=engine_oil_cost,
            oil_filter_cost=oil_filter_cost,
            air_filter_cost=air_filter_cost,
            update_date=update_date
        )
        
        db.session.add(new_receipt)
        db.session.commit()

        return redirect(url_for('view_receipts'))
    
    return render_template('upload_receipt.html')


@app.route('/view_receipts', methods=['GET'])
def view_receipts():
    receipts = Receiptss.query.all()  # Fetch all receipts from the database
    return render_template('view_receipts.html', receipts=receipts)


from flask import render_template, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import calendar

def prepare_model_data(receipts):
    """Prepare data for the model from receipts."""
    data = []
    for receipt in receipts:
        month = receipt.update_date.strftime("%b")
        data.append({
            "Date": month,
            "Total_Cost": receipt.engine_oil_cost + receipt.oil_filter_cost + receipt.air_filter_cost,
            "Engine_Oil_Cost": receipt.engine_oil_cost,
            "Oil_Filter_Cost": receipt.oil_filter_cost,
            "Air_Filter_Cost": receipt.air_filter_cost
        })
    return pd.DataFrame(data)

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'GET':
        # Get unique car names for the dropdown
        cars = db.session.query(Receiptss.car_name).distinct().all()
        cars = [car[0] for car in cars]
        return render_template('analysis.html', cars=cars)
    
    # Handle POST request
    car_name = request.form.get('car_name')
    
    # Get receipts for the selected car
    receipts = Receiptss.query.filter_by(car_name=car_name).order_by(Receiptss.update_date).all()
    
    if not receipts:
        return render_template('analysis.html', error="No data available for selected car")
    
    # Prepare data for the model
    df = prepare_model_data(receipts)
    
    # Month mapping and data preparation (same as in the model code)
    month_mapping = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5,
        "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
        "Nov": 11, "Dec": 12
    }
    df["Month_Num"] = df["Date"].map(month_mapping)
    
    # Add cyclical features
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month_Num']/12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month_Num']/12)
    
    # Prepare features
    X = df[["Month_Num", "Month_Sin", "Month_Cos", "Engine_Oil_Cost", "Oil_Filter_Cost", "Air_Filter_Cost"]]
    y = df["Total_Cost"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model with best parameters from grid search
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Generate predictions for all months
    future_months = np.array(range(1, 13))
    future_sin = np.sin(2 * np.pi * future_months/12)
    future_cos = np.cos(2 * np.pi * future_months/12)
    
    # Create future dataset using averages
    future_data = pd.DataFrame({
        "Month_Num": future_months,
        "Month_Sin": future_sin,
        "Month_Cos": future_cos,
        "Engine_Oil_Cost": [df["Engine_Oil_Cost"].mean()] * 12,
        "Oil_Filter_Cost": [df["Oil_Filter_Cost"].mean()] * 12,
        "Air_Filter_Cost": [df["Air_Filter_Cost"].mean()] * 12
    })
    
    # Scale and predict
    future_data_scaled = scaler.transform(future_data)
    predicted_costs = model.predict(future_data_scaled)
    
    # Calculate metrics
    y_pred = model.predict(X_scaled)
    accuracy = round((1 - abs(np.mean((y - y_pred) / y))) * 100, 2)
    
    # Prepare analysis data for template
    analysis_data = {
        'metrics': {
            'avg_cost': int(df['Total_Cost'].mean()),
            'next_month': int(predicted_costs[datetime.now().month % 12]),
            'total_spent': int(df['Total_Cost'].sum()),
            'accuracy': accuracy
        },
        'importance': {
            'labels': ['Engine Oil', 'Oil Filter', 'Air Filter'],
            'values': model.feature_importances_[-3:].tolist()  # Last 3 features are costs
        },
        'trends': {
            'months': list(calendar.month_abbr)[1:],
            'actual': df['Total_Cost'].tolist(),
            'predicted': predicted_costs.tolist()
        }
    }
    
    # Get car list for dropdown
    cars = db.session.query(Receiptss.car_name).distinct().all()
    cars = [car[0] for car in cars]
    
    return render_template('analysis.html', 
                         cars=cars,
                         analysis_data=analysis_data,
                         selected_car=car_name)

if __name__ == "__main__":
    app.run(debug=True)
