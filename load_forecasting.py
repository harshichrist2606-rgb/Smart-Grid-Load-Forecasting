# Smart Grid Load Forecasting
# Simple Linear Regression Model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("==========================================")
print("       SMART GRID LOAD FORECASTING")
print("==========================================")

# Historical hourly load data in kW
load_data = np.array([
    320, 300, 285, 275, 270, 280,
    310, 350, 400, 450, 470, 490,
    510, 500, 480, 465, 450, 470,
    520, 570, 610, 590, 540, 450
])

# Create hour values
hours = np.arange(1, len(load_data) + 1)

# Prepare data for machine learning
X = hours.reshape(-1, 1)
y = load_data

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict historical load
predicted_load = model.predict(X)

# Predict next hour
next_hour = np.array([[25]])
next_load = model.predict(next_hour)[0]

# Calculate load statistics
average_load = np.mean(load_data)
maximum_load = np.max(load_data)
minimum_load = np.min(load_data)

# Calculate Mean Absolute Error
mae = np.mean(np.abs(load_data - predicted_load))

# Calculate approximate accuracy
accuracy = max(0, 100 - (mae / average_load * 100))

# Find peak load hour
peak_hour = np.argmax(load_data) + 1

# Display results
print("\n----------- LOAD ANALYSIS -----------")
print(f"Average Load       : {average_load:.2f} kW")
print(f"Maximum Load       : {maximum_load:.2f} kW")
print(f"Minimum Load       : {minimum_load:.2f} kW")

print("\n----------- FORECAST -----------")
print("Next Hour          : Hour 25")
print(f"Predicted Load     : {next_load:.2f} kW")

print("\n----------- MODEL PERFORMANCE -----------")
print(f"Mean Absolute Error: {mae:.2f} kW")
print(f"Approx. Accuracy   : {accuracy:.2f}%")

print("\n----------- PEAK LOAD -----------")
print(f"Peak Load          : {maximum_load:.2f} kW")
print(f"Peak Occurs At     : Hour {peak_hour}")

print("\n==========================================")

# Plot actual and predicted load
plt.figure(figsize=(10, 5))

plt.plot(
    hours,
    load_data,
    marker="o",
    label="Actual Load"
)

plt.plot(
    hours,
    predicted_load,
    linestyle="--",
    label="Predicted Load"
)

plt.scatter(
    25,
    next_load,
    marker="*",
    s=150,
    label="Next Hour Forecast"
)

plt.title("Smart Grid Load Forecasting")
plt.xlabel("Hour")
plt.ylabel("Load (kW)")
plt.grid(True)
plt.legend()

plt.show()
