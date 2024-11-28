#Write a program to implement Perceptron

import numpy as np

# Perceptron function
def perceptron(X, y, learning_rate=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0

    for _ in range(epochs):
        for i, x in enumerate(X):
            activation = np.dot(x, weights) + bias
            if activation * y[i] <= 0:
                weights += learning_rate * y[i] * x
                bias += learning_rate * y[i]
    return weights, bias

# Example data
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y = np.array([1, -1, -1, -1])

weights, bias = perceptron(X, y)
print("Weights:", weights)
print("Bias:", bias)
