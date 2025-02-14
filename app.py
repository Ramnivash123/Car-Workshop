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

class Receipts(db.Model):
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


@app.route('/analysis')
def analysis():
    if "user_id" not in session:
        flash("Please log in to continue.", "warning")
        return redirect(url_for("register2"))

    return render_template("analysis.html", username=session["username"])


@app.route('/upload_receipt', methods=['GET', 'POST'])
def upload_receipt():
    if request.method == 'POST':
        car_name = request.form["car_name"]
        engine_oil_cost = request.form['engine_oil_cost']
        oil_filter_cost = request.form['oil_filter_cost']
        air_filter_cost = request.form['air_filter_cost']
        update_date = datetime.utcnow()

        # Assuming you have the 'car_name' and 'username' available in your context
        new_receipt = Receipts(
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
    receipts = Receipts.query.all()  # Fetch all receipts from the database
    return render_template('view_receipts.html', receipts=receipts)

if __name__ == "__main__":
    app.run(debug=True)
