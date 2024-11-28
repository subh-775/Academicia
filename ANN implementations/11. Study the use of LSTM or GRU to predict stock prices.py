# Study the use of LSTM/GRU to predict stock prices

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load stock price data (example with random values)
data = np.sin(np.linspace(0, 100, 1000))  # Replace with real stock price data
seq_length = 50

# Prepare sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X)
plt.plot(data, label='True Data')
plt.plot(np.arange(seq_length, len(data)), predictions, label='Predictions')
plt.legend()
plt.show()

