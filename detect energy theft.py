import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet


# Generate synthetic energy demand data
def generate_synthetic_energy_demand():
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='H')
    np.random.seed(42)
    energy_demand = 5000 + np.random.normal(0, 200, len(dates))
    data = pd.DataFrame({'ds': dates, 'y': energy_demand})
    data.to_csv('synthetic_energy_demand.csv', index=False)
    print("Synthetic energy demand data saved as 'synthetic_energy_demand.csv'")
    return data


# Generate synthetic renewable energy output data
def generate_synthetic_renewable_energy_output():
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='H')
    np.random.seed(42)
    wind_speed = np.random.uniform(0, 25, len(dates))
    temperature = np.random.uniform(-10, 35, len(dates))
    sunlight = np.random.uniform(0, 1, len(dates))
    humidity = np.random.uniform(0, 100, len(dates))
    energy_output = 1000 + wind_speed * 30 + temperature * 2 + sunlight * 500 + np.random.normal(0, 50, len(dates))
    data = pd.DataFrame({
        'date': dates,
        'wind_speed': wind_speed,
        'temperature': temperature,
        'sunlight': sunlight,
        'humidity': humidity,
        'energy_output': energy_output
    })
    data.to_csv('synthetic_renewable_energy_output.csv', index=False)
    print("Synthetic renewable energy output data saved as 'synthetic_renewable_energy_output.csv'")
    return data


# Forecasting energy demand with Facebook Prophet
def forecast_energy_demand(data):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=365)  # Forecasting for the next year
    forecast = model.predict(future)
    fig = model.plot(forecast)
    plt.title('Energy Demand Forecast')
    plt.xlabel('Date')
    plt.ylabel('Energy Demand')
    plt.show()


# Prepare the data for LSTM
def create_sequences(data, target, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i + sequence_length]
        y = target[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Forecasting renewable energy output with LSTM
def forecast_renewable_energy_output(data):
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    features = data[['wind_speed', 'temperature', 'sunlight', 'humidity']]
    target = data['energy_output']

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    sequence_length = 30  # 30 days sequence
    X, y = create_sequences(scaled_features, target.values, sequence_length)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    predictions = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Energy Output')
    plt.plot(predictions, label='Predicted Energy Output')
    plt.title('Renewable Energy Output Forecast')
    plt.xlabel('Time')
    plt.ylabel('Energy Output')
    plt.legend()
    plt.show()


# Generate synthetic energy usage data
def generate_synthetic_data(num_samples=10000):
    np.random.seed(42)
    customer_ids = np.arange(1, num_samples + 1)

    # Generate normal usage patterns
    normal_usage = np.random.normal(loc=100, scale=20, size=num_samples)

    # Introduce some fraudulent patterns
    fraud_percentage = 0.1
    num_fraud = int(fraud_percentage * num_samples)
    fraud_indices = np.random.choice(num_samples, num_fraud, replace=False)
    fraud_usage = normal_usage[fraud_indices] * np.random.uniform(1.5, 3.0, num_fraud)  # 50% to 200% more usage

    # Combine normal and fraudulent usage
    usage = normal_usage.copy()
    usage[fraud_indices] = fraud_usage

    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': customer_ids,
        'energy_usage': usage,
        'is_fraud': [1 if i in fraud_indices else 0 for i in range(num_samples)]
    })

    return data


# Load and prepare the data
def load_and_prepare_data():
    # Load the synthetic data
    data = generate_synthetic_data()

    # Generate additional features
    data['usage_deviation'] = np.abs(data['energy_usage'] - data['energy_usage'].mean())
    data['high_usage'] = (data['energy_usage'] > data['energy_usage'].quantile(0.95)).astype(int)

    return data


# Train the model to detect energy theft
def train_model(data):
    # Prepare the feature matrix X and target vector y
    X = data[['energy_usage', 'usage_deviation', 'high_usage']]
    y = data['is_fraud']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model


# Function to detect fraud in new data
def detect_fraud(model, new_data):
    predictions = model.predict(new_data)
    return predictions


def main():
    # Generate and forecast energy demand
    print("Generating and forecasting energy demand...")
    energy_demand_data = generate_synthetic_energy_demand()
    forecast_energy_demand(energy_demand_data)

    # Generate and forecast renewable energy output
    print("Generating and forecasting renewable energy output...")
    renewable_energy_data = generate_synthetic_renewable_energy_output()
    forecast_renewable_energy_output(renewable_energy_data)

    # Detect energy theft
    print("Detecting energy theft...")
    data = load_and_prepare_data()
    model = train_model(data)

    # Example new data (Replace with actual new data for real use case)
    new_data = pd.DataFrame({
        'energy_usage': [150, 250, 90, 300],  # Example values
        'usage_deviation': [10, 20, 5, 30],  # Example values
        'high_usage': [1, 1, 0, 1]  # Example values
    })

    # Detect fraud in new data
    fraud_predictions = detect_fraud(model, new_data)
    print(f"Fraud Predictions: {fraud_predictions}")


if __name__ == "__main__":
    main()
