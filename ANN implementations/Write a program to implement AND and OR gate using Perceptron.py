#Write a program to implement AND and OR gate using Perceptron

import numpy as np

# Define AND gate inputs and outputs
def AND_gate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return X, y

# Define OR gate inputs and outputs
def OR_gate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    return X, y

def perceptron_gate(X, y, epochs=10, lr=0.1):
    weights = np.zeros(X.shape[1])
    bias = 0
    for _ in range(epochs):
        for i, x in enumerate(X):
            prediction = np.dot(x, weights) + bias
            if (y[i] == 1 and prediction <= 0) or (y[i] == 0 and prediction > 0):
                weights += lr * (y[i] - (prediction > 0)) * x
                bias += lr * (y[i] - (prediction > 0))
    return weights, bias

# AND gate
X, y = AND_gate()
weights, bias = perceptron_gate(X, y)
print("AND Gate Weights:", weights, "Bias:", bias)

# OR gate
X, y = OR_gate()
weights, bias = perceptron_gate(X, y)
print("OR Gate Weights:", weights, "Bias:", bias)
