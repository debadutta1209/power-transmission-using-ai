import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy import signal


# Function to apply Continuous Wavelet Transformation (CWT) to data
def apply_cwt(data):
    widths = np.arange(1, 31)  # Define the range of widths for the wavelet transform
    cwt_matrix = signal.cwt(data, signal.ricker, widths)  # Apply the Ricker wavelet
    cwt_flattened = cwt_matrix.flatten()  # Flatten the matrix into a single feature vector
    return cwt_flattened


# Function to extract features from the dataset
def extract_features(data):
    features = []
    for recording in data:
        coeffs = apply_cwt(recording)
        features.append(coeffs)
    return np.array(features)


# Function to train SVM model
def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_classifier = SVC(kernel='rbf')  # Radial Basis Function (RBF) kernel is commonly used
    svm_classifier.fit(X_train, y_train)
    return svm_classifier, X_test, y_test


# Function to evaluate the SVM model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=1))


# Main function for deployment
def main():
    # Sample data for demonstration purposes
    data = {
        'generation_biomass': [447.0, 449.0, 448.0, 438.0, 428.0],
        'generation_fossil_brown_coal_lignite': [2000.0, 2100.0, 2050.0, 2150.0, 2200.0],
        'generation_fossil_gas': [1000.0, 1050.0, 1025.0, 1075.0, 1100.0],
        'total_load_actual': [5000.0, 5100.0, 5050.0, 5150.0, 5200.0],
        'price_day_ahead': [50.10, 48.10, 47.33, 42.27, 38.41],
        'price_actual': [65.41, 64.92, 64.48, 59.32, 56.04]
    }

    df = pd.DataFrame(data)

    # Mock 'outage' column based on some condition for demonstration
    df['outage'] = np.where(df['total_load_actual'] > df['total_load_actual'].quantile(0.95), 1, 0)

    # Drop unnecessary columns and handle missing values
    df = df.dropna()
    X = df.drop('outage', axis=1).values
    y = df['outage'].values

    # Extract features from the data using CWT
    X_features = extract_features(X)

    # Train SVM model
    svm_model, X_test, y_test = train_svm(X_features, y)

    # Evaluate the trained model
    print("Evaluation Report:")
    evaluate_model(svm_model, X_test, y_test)

    # Deployment phase with new data
    # Assuming new_data contains new voltage recordings (replace this with actual new data)
    new_data = np.array([
        [450.0, 2100.0, 1050.0, 5100.0, 49.0, 65.0],  # Example row
        [430.0, 2200.0, 1100.0, 5200.0, 39.0, 55.0]  # Example row
    ])
    new_data_features = extract_features(new_data)
    predicted_faults = svm_model.predict(new_data_features)
    print("Predicted faults for new data:", predicted_faults)


if __name__ == "__main__":
    main()
