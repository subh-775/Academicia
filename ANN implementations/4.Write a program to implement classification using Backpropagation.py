# Write a program to implement classification using Backpropagation

import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(1)
weights = np.random.rand(2, 1)
bias = np.random.rand(1)
learning_rate = 0.1

# Training
for epoch in range(10000):
    # Forward pass
    layer_output = sigmoid(np.dot(X, weights) + bias)
    
    # Backpropagation
    error = y - layer_output
    adjustments = error * sigmoid_derivative(layer_output)
    
    weights += np.dot(X.T, adjustments) * learning_rate
    bias += np.sum(adjustments) * learning_rate

print("Trained Weights:\n", weights)
print("Trained Bias:\n", bias)
