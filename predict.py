import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample dataset
data = {
    "Date": ["Jan", "Mar", "Apr", "Jun", "Jul"],
    "Total_Cost": [2300, 3000, 1800, 2700, 4000],
    "Engine_Oil_Cost": [1500, 2000, 1200, 1800, 2500],
    "Oil_Filter_Cost": [500, 600, 400, 550, 800],
    "Air_Filter_Cost": [300, 400, 200, 350, 700]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Mapping months to numerical values (1-12)
month_mapping = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5,
    "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
    "Nov": 11, "Dec": 12
}
df["Month_Num"] = df["Date"].map(month_mapping)

# Features (X) and Target (y) – Removed 'Total_Cost' from X
X = df[["Month_Num", "Engine_Oil_Cost", "Oil_Filter_Cost", "Air_Filter_Cost"]]
y = df["Total_Cost"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regression Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred) if not np.isnan(r2_score(y_test, y_pred)) else 0.0

# Generate predictions for all months (Jan to Dec)
future_months = np.array(range(1, 13)).reshape(-1, 1)  # 1 to 12

# Create DataFrame with Month_Num and dummy values for other columns
future_data = pd.DataFrame({
    "Month_Num": future_months.flatten(),
    "Engine_Oil_Cost": [df["Engine_Oil_Cost"].mean()] * 12,  # Use mean values
    "Oil_Filter_Cost": [df["Oil_Filter_Cost"].mean()] * 12,
    "Air_Filter_Cost": [df["Air_Filter_Cost"].mean()] * 12
})

# Scale using the same StandardScaler
future_data_scaled = scaler.transform(future_data)

# Predict using the trained model
predicted_costs = model.predict(future_data_scaled)

# Convert to DataFrame
predicted_df = pd.DataFrame({
    "Month": list(month_mapping.keys()),  # Assign month names
    "Total_Cost": predicted_costs
})

# Convert actual dates to months for plotting
df["Month"] = df["Date"]  # Keep month names for plotting

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_df["Month"], predicted_df["Total_Cost"], label="Predicted Total Cost", color="blue", linestyle="dashed")

# Plot actual data points
plt.scatter(df["Month"], df["Total_Cost"], color="red", label="Actual Total Cost", marker="o")

# Display accuracy metrics on the graph
accuracy_text = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR² Score: {r2:.4f}"
plt.text(0.05, 0.7, accuracy_text, transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Formatting
plt.xlabel("Month")
plt.ylabel("Cost (₹)")
plt.title("Predicted Price Variation for the Year")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
