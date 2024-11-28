# Study Long Short Term Memory (LSTM) for Time Series Prediction

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate sine wave data
x = np.linspace(0, 100, 1000)
y = np.sin(x)

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(y, seq_length)

X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X)
plt.plot(y, label='True Data')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
