import numpy as np

A = np.array([[4,9,9],[9,1,6],[9,2,3]])
print("Matrix A :",A)

B = np.array(([2,2],[5,7],[4,4]))
print("Matrix B:",B)

print(np.matmul(A,B))

print(A @ B)

try:
    np.matmul(B, A)
except ValueError as err:
    print(err)

try:
    B @ A
except ValueError as err:
    print(err)

x = np.array([1, -2, -5])
y = np.array([4, 3, -1])

print("Shape of vector x:", x.shape)
print("Number of dimensions of vector x:", x.ndim)
print("Shape of vector x, reshaped to a matrix:", x.reshape((3, 1)).shape)
print("Number of dimensions of vector x, reshaped to a matrix:", x.reshape((3, 1)).ndim)

np.matmul(x,y)

try:
    np.matmul(x.reshape((3, 1)), y.reshape((3, 1)))
except ValueError as err:
    print(err)

print(np.dot(A, B))

print(A - 2)