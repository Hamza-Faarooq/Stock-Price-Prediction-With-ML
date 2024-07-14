# Stock-Price-Prediction-With-ML

# Stock Price Prediction Using Machine Learning

## Overview
Can you predict the future? In this project, you'll explore using machine learning models in Python on Google Colab to forecast stock prices based on historical data.

## What You Will Be Learning
- Understand the limitations and complexities of stock price prediction.
- Implement machine learning models like LSTMs (Long Short-Term Memory) in Python for time series forecasting.
- Evaluate the performance of your models and interpret the predictions with caution.

## Tools and Libraries
- **Programming Language**: Python
- **Platform**: Google Colab (free Jupyter notebook environment)
- **Libraries**:
  - TensorFlow or PyTorch: Deep learning frameworks
  - NumPy: For numerical operations
  - Pandas: For data manipulation
  - scikit-learn: For machine learning tools and evaluation

## Dataset
- Dataset: [Google Stock Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction/data)

## Expected Results
The expected results include a model that forecasts future stock prices based on historical data. The model's performance will be evaluated using appropriate metrics, and predictions will be visualized.

## How to Run the Project
1. **Setup Environment**: Install necessary libraries.
2. **Upload and Load Dataset**: Handle the dataset file from Kaggle.
3. **Preprocess Data**: Prepare the data for training the machine learning model.
4. **Build and Train Model**: Implement and train the LSTM model using TensorFlow or PyTorch.
5. **Evaluate Model**: Assess the model's performance using appropriate metrics.
6. **Make Predictions**: Use the trained model to predict future stock prices.
7. **Visualize Results**: Plot the predictions against the actual stock prices for visualization.

## Code
```python
# Setup Environment
!pip install numpy pandas scikit-learn tensorflow matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset from Kaggle
from google.colab import files
uploaded = files.upload()

# Load the CSV file
df = pd.read_csv('google_stock_data.csv')

# Display the first few rows of the dataset
print(df.head())

# Preprocess data
# Assuming 'Close' is the target variable
data = df['Close'].values
data = data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create datasets for LSTM model
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='Actual Prices')
plt.plot(df['Date'][time_step:len(train_predict) + time_step], train_predict, label='Train Predict')
plt.plot(df['Date'][len(train_predict) + (time_step * 2) + 1:len(df) - 1], test_predict, label='Test Predict')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
