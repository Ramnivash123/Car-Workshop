import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Sample dataset
data = {
    "Date": ["2025-02-12", "2025-03-10", "2025-01-25", "2025-02-20", "2025-04-05"],
    "Engine_Oil_Cost": [1500, 2000, 1200, 1800, 2500],
    "Oil_Filter_Cost": [500, 600, 400, 550, 800],
    "Air_Filter_Cost": [300, 400, 200, 350, 700]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert date to numerical format (days since a reference date)
reference_date = datetime(2025, 1, 1)
df["Days"] = df["Date"].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d") - reference_date).days)

# Features (X) and Target (y)
X = df[["Days"]]  # Using only 'Days' as input
y = df.drop(columns=["Date", "Days"])  # Predicting cost values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regression Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Prediction for a new date
input_date = "2025-02-12"  # Example input date
input_days = (datetime.strptime(input_date, "%Y-%m-%d") - reference_date).days
input_scaled = scaler.transform(np.array([[input_days]]))
predicted_costs = model.predict(input_scaled)[0]

# Display results
print("Predicted Costs on", input_date)
print("Engine Oil Cost: ₹", round(predicted_costs[0]))
print("Oil Filter Cost: ₹", round(predicted_costs[1]))
print("Air Filter Cost: ₹", round(predicted_costs[2]))
