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

class Receipt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    car_name = db.Column(db.String(100), nullable=False)
    engine_oil_cost = db.Column(db.Integer, nullable=False)
    oil_filter_cost = db.Column(db.Integer, nullable=False)
    air_filter_cost = db.Column(db.Integer, nullable=False)
    update_date = db.Column(db.Date, nullable=False)

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
        update_date_str = request.form['update_date']  # This is a string from the form

        # Convert string date to Python date object
        update_date = datetime.strptime(update_date_str, '%Y-%m-%d').date()

        # Assuming you have the 'car_name' and 'username' available in your context
        new_receipt = Receipt(
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
    receipts = Receipt.query.all()  # Fetch all receipts from the database
    return render_template('view_receipts.html', receipts=receipts)


import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import io
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from flask import render_template, request
from flask import Flask, jsonify



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import calendar

def prepare_model_data(car_name):
    """Prepare data for the model from receipts of a particular car."""
    # Query data from the Receipts table
    receipts = Receipt.query.filter_by(car_name=car_name).order_by(Receipt.update_date).all()
    
    # Prepare data
    data = []
    for receipt in receipts:
        month_name = calendar.month_abbr[receipt.update_date.month]  # Convert to "Jan", "Feb", etc.
        data.append({
            "Month": month_name,
            "Month_Num": receipt.update_date.month,  # Numeric month for sorting
            "Total_Cost": receipt.engine_oil_cost + receipt.oil_filter_cost + receipt.air_filter_cost
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Check if DataFrame is empty
    if df.empty:
        print(f"No data found for car: {car_name}")
        return df
    
    # Ensure we have all months (Jan-Dec) in the DataFrame
    all_months = list(calendar.month_abbr[1:])  # Skip the empty string in month_abbr[0]
    existing_months = df["Month"].unique()
    missing_months = [month for month in all_months if month not in existing_months]

    # Create a DataFrame for missing months
    missing_data = pd.DataFrame({
        "Month": missing_months,
        "Month_Num": [list(calendar.month_abbr).index(month) for month in missing_months],  # Correct month index
        "Total_Cost": [0] * len(missing_months)  # Assuming total cost 0 for missing months
    })

    # Concatenate the missing months with the original DataFrame
    df = pd.concat([df, missing_data], ignore_index=True)
    
    # Sort data by month number and aggregate the costs
    df = df.groupby(["Month", "Month_Num"]).sum().reset_index().sort_values("Month_Num")
    
    # Predict missing months using linear regression
    if len(missing_months) > 0:
        # Prepare data for regression
        X = df[df["Total_Cost"] > 0]["Month_Num"].values.reshape(-1, 1)
        y = df[df["Total_Cost"] > 0]["Total_Cost"].values
        
        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict for missing months
        missing_months_num = df[df["Total_Cost"] == 0]["Month_Num"].values.reshape(-1, 1)
        predicted_costs = model.predict(missing_months_num)
        
        # Update the DataFrame with predicted costs
        df.loc[df["Total_Cost"] == 0, "Total_Cost"] = predicted_costs
    
    return df

def generate_cost_chart(df):
    """Generate a line chart with actual vs. predicted costs."""
    # Check if DataFrame is empty
    if df.empty:
        print("DataFrame is empty. Cannot generate chart.")
        return None
    
    # Encode month names to numbers
    le = LabelEncoder()
    df["Month_Encoded"] = le.fit_transform(df["Month"])

    # Train RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[["Month_Encoded"]], df["Total_Cost"])
    
    # Predict future values
    df["Predicted_Cost"] = model.predict(df[["Month_Encoded"]])

    # Plot actual vs. predicted costs
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df["Month"], y=df["Predicted_Cost"], marker='s', linestyle='--', label="Predicted Cost")
    
    plt.xlabel("Month")
    plt.ylabel("Total Cost")
    plt.title("Car Maintenance Cost Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    
    # Convert plot to base64 image
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'GET':
        cars = db.session.query(Receipt.car_name).distinct().all()
        cars = [car[0] for car in cars]
        return render_template('analysis.html', cars=cars)

    car_name = request.form.get('car_name')
    df = prepare_model_data(car_name)
    
    # Check if DataFrame is empty
    if df.empty:
        return render_template('analysis.html', cars=cars, selected_car=car_name, error="No data found for the selected car.")
    
    # Generate chart
    chart_img = generate_cost_chart(df)
    
    # Get car list for dropdown
    cars = db.session.query(Receipt.car_name).distinct().all()
    cars = [car[0] for car in cars]

    cost = df["Predicted_Cost"].to_list()

    for i in range(1,13):
        if cost[i] >cost[i-1]:
            max=i-1
            break

    return render_template('analysis.html', cars=cars, selected_car=car_name, chart_img=chart_img, max=max)
    

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'ramnivash19022003@gmail.com'
EMAIL_PASSWORD = 'mcig idjb abah djlz'  # Use an app-specific password for security

@app.route('/send-alert', methods=['POST'])
def send_alert():
    try:
        # Month mapping
        month_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        # Get the month with the maximum cost rise
        max_month = month_map.get(max, "Invalid month")

        # Create the email
        subject = "Cost Analysis Alert"
        body = f"The month with the maximum cost rise is: {max_month}"
        sender_email = EMAIL_ADDRESS
        receiver_email = "mathan32097@gmail.com"

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(sender_email, receiver_email, message.as_string())

        return jsonify({"message": "Alert sent successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)