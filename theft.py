import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Generate synthetic power distribution data
np.random.seed(42)
num_samples = 1000

# Simulated power consumption in kWh
power_consumption = np.random.randint(100, 1000, size=num_samples)

# Simulated voltage in volts
voltage = np.random.uniform(220, 240, size=num_samples)

# Simulated current in amps
current = np.random.uniform(5, 20, size=num_samples)

# Create DataFrame
power_data = pd.DataFrame({
    'power_consumption': power_consumption,
    'voltage': voltage,
    'current': current
})

# Display a sample of the synthetic data
print("Sample of synthetic power distribution data:")
print(power_data.head())

# Write synthetic data to a CSV file
power_data.to_csv('synthetic_power_distribution_data.csv', index=False)

# Load data from CSV
data = pd.read_csv('synthetic_power_distribution_data.csv')

# Preprocessing
X = data.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model initialization
model = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination based on your data

# Model fitting
model.fit(X_scaled)

# Predictions
predictions = model.predict(X_scaled)

# Add predictions to original DataFrame
data['anomaly'] = predictions

# Filter out the anomalies for further investigation
power_theft_cases = data[data['anomaly'] == -1]

# Check if there are any potential power theft cases
if len(power_theft_cases) > 0:
    print("Suspicious activity detected. Potential power theft cases found.")
    print("Power theft cases:")
    print(power_theft_cases)
else:
    print("No suspicious activity detected. Power distribution is normal.")
