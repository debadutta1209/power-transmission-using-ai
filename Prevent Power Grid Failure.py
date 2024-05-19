import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('C:/Users/Debabrata/Desktop/energy-consumption-generation-prices-and-weather.csv')

# Inspect the first few rows of the dataset and the column names
print(data.head())
print(data.columns)

# Check for missing values
print(data.isnull().sum())

# Assuming 'outage' is the target column that indicates power failures
# Create a mock 'outage' column based on 'total load actual'
data['outage'] = np.where(data['total load actual'] > data['total load actual'].quantile(0.95), 1, 0)

# Handle missing values by filling them with the mean or another strategy
data.fillna(data.mean(), inplace=True)

# Drop the 'time' column or any other irrelevant columns
data.drop(columns=['time'], inplace=True)

# Define features and target variable
X = data.drop('outage', axis=1)
y = data['outage']

# Verify that X and y are not empty
print(X.shape, y.shape)

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
print(classification_report(y_test, y_pred))

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
# Replace [...] with actual new data you want to predict on
# Ensure that the new data has the same number of features as the training data
new_data = np.array([[...]])  # Example placeholder

# Predict and prevent outages
predict_and_prevent_outages(new_data)
