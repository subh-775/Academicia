# Write a program to implement functions: purelin(n), binary threshold(hardlim(n)), haradlims(n), tansig(n), logsig(n)

import numpy as np

def purelin(n):
    return n

def hardlim(n):
    return np.where(n >= 0, 1, 0)

def hardlims(n):
    return np.where(n >= 0, 1, -1)

def tansig(n):
    return np.tanh(n)

def logsig(n):
    return 1 / (1 + np.exp(-n))

# Example inputs
n = np.array([-2, -1, 0, 1, 2])

print("Purelin:", purelin(n))
print("Hardlim:", hardlim(n))
print("Hardlims:", hardlims(n))
print("Tansig:", tansig(n))
print("Logsig:", logsig(n))
