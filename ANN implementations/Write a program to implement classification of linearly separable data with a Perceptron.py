# Write a program to implement classification of linearly separable data with a Perceptron

import numpy as np
import matplotlib.pyplot as plt

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

# Example data (linearly separable)
X = np.array([[2, 7], [8, 1], [7, 5], [6, 2], [4, 8], [1, 6]])
y = np.array([1, -1, -1, -1, 1, 1])  # Labels (1 or -1)

weights, bias = perceptron(X, y)

# Plotting
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], color='red' if y[i] == 1 else 'blue')

# Decision boundary
x_values = np.linspace(0, 10, 100)
y_values = -(weights[0] * x_values + bias) / weights[1]
plt.plot(x_values, y_values, label='Decision Boundary', color='black')
plt.legend()
plt.show()
