import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Sample data
data = {
    'generation_biomass': [447.0, 449.0, 448.0, 438.0, 428.0],
    'generation_fossil_brown_coal_lignite': [2000.0, 2100.0, 2050.0, 2150.0, 2200.0],
    'generation_fossil_gas': [1000.0, 1050.0, 1025.0, 1075.0, 1100.0],
    'total_load_actual': [5000.0, 5100.0, 5050.0, 5150.0, 5200.0],
    'price_day_ahead': [50.10, 48.10, 47.33, 42.27, 38.41],
    'price_actual': [65.41, 64.92, 64.48, 59.32, 56.04]
}

# Create DataFrame
df = pd.DataFrame(data)

# For demonstration, let's create a mock 'outage' column based on some condition
df['outage'] = np.where(df['total_load_actual'] > df['total_load_actual'].quantile(0.95), 1, 0)

# Drop unnecessary columns and handle missing values
df = df.dropna()
X = df.drop('outage', axis=1)
y = df['outage']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print evaluation metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

# Function to predict outages and trigger preventive measures
def predict_and_prevent_outages(new_data):
    # Standardize the new data
    new_data = scaler.transform(new_data)

    # Predict outages
    predictions = model.predict(new_data)

    # Trigger preventive measures for predicted outages
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            print(f"Potential outage detected in instance {i}. Triggering preventive measures...")

# Example new data
# Replace the placeholder [...] with actual new data you want to predict on
new_data = np.array([
    [450.0, 2100.0, 1050.0, 5100.0, 49.0, 65.0],  # Example row
    [430.0, 2200.0, 1100.0, 5200.0, 39.0, 55.0]   # Example row
])

# Predict and prevent outages
predict_and_prevent_outages(new_data)
