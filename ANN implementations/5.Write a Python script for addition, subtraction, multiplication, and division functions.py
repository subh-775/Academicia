#Write a Python script for addition, subtraction, multiplication, and division functions

def add_matrices(A, B):
    return A + B

def subtract_matrices(A, B):
    return A - B

def multiply_matrices(A, B):
    return A * B

def divide_matrices(A, B):
    return A / B

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Addition:\n", add_matrices(A, B))
print("Subtraction:\n", subtract_matrices(A, B))
print("Multiplication:\n", multiply_matrices(A, B))
print("Division:\n", divide_matrices(A, B))
