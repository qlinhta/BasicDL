# Linear Algebra application for Machine Learning

import numpy as np

# Create a 2D array
A = np.array([[1, 2], [3, 4]])
print(A)

# Create a 3D array
B = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(B)

# Create a 3D array of zeros
C = np.zeros((2, 3, 4))
print(C)

# Create a 3D array of ones
D = np.ones((2, 3, 4))

# Create a 3D array of random numbers
E = np.random.random((2, 3, 4))

# Create a 3D array of random integers
F = np.random.randint(10, size=(3, 4, 5))

# Create a 3D array of random integers
G = np.random.randint(10, size=(3, 4, 5))

# Create a function calculate the dot product of two vectors
def dot_product(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])

# Write the test for the dot product function
def test_dot_product():
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert dot_product(a, b) == 32
    print("Test passed!")

# Create a function calculate the sum of two vectors
def vector_sum(a, b):
    return [a[i] + b[i] for i in range(len(a))]

# Write the test for the vector sum function
def test_vector_sum():
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert vector_sum(a, b) == [5, 7, 9]
    print("Test passed!")

# Create a function calculate the sum of multiple vectors
def vector_sum(*args):
    return [sum(arg[i] for arg in args) for i in range(len(args[0]))]

# Write the test for the vector sum function
def test_vector_sum():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    assert vector_sum(a, b, c) == [12, 15, 18]
    print("Test passed!")

# Create a function calculate the subtraction of two vectors
def vector_subtract(a, b):
    return [a[i] - b[i] for i in range(len(a))]

# Write the test for the vector subtraction function
def test_vector_subtract():
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert vector_subtract(a, b) == [-3, -3, -3]
    print("Test passed!")

# Create a function calculate the subtraction of two matrices
def matrix_subtract(a, b):
    return [vector_subtract(a[i], b[i]) for i in range(len(a))]

# Write the test for the matrix subtraction function
def test_matrix_subtract():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8, 9], [10, 11, 12]]
    assert matrix_subtract(a, b) == [[-6, -6, -6], [-6, -6, -6]]
    print("Test passed!")

# Create a function calculate the sum of two matrices
def matrix_sum(a, b):
    return [vector_sum(a[i], b[i]) for i in range(len(a))]

# Write the test for the matrix sum function
def test_matrix_sum():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8, 9], [10, 11, 12]]
    assert matrix_sum(a, b) == [[8, 10, 12], [14, 16, 18]]
    print("Test passed!")


# Create a function calculate determinant of a matrix
def matrix_determinant(a):
    if len(a) == 1:
        return a[0][0]
    else:
        return sum([(-1) ** i * a[0][i] * matrix_determinant([row[:i] + row[i+1:] for row in a[1:]]) for i in range(len(a))])

# Write the test for the matrix determinant function
def test_matrix_determinant():
    a = [[1, 2], [3, 4]]
    assert matrix_determinant(a) == -2
    print("Test passed!")

# Create a function calculate the transpose of a matrix
def matrix_transpose(a):
    return [[row[i] for row in a] for i in range(len(a[0]))]

# Write the test for the matrix transpose function
def test_matrix_transpose():
    a = [[1, 2], [3, 4]]
    assert matrix_transpose(a) == [[1, 3], [2, 4]]
    print("Test passed!")

# Create a function calculate eigenvalues of a matrix
def matrix_eigenvalues(a):
    return np.linalg.eigvals(a)

# Create a function calculate eigenvectors of a matrix
def matrix_eigenvectors(a):
    return np.linalg.eig(a)[1]

# Create a function calculate the inverse of a matrix
def matrix_inverse(a):
    return np.linalg.inv(a)

# Create a function find the rank of a matrix
def matrix_rank(a):
    return np.linalg.matrix_rank(a)

# Write the test for the matrix rank function
def test_matrix_rank():
    a = [[1, 2], [3, 4]]
    assert matrix_rank(a) == 2
    print("Test passed!")

# Create a function calculate the trace of a matrix
def matrix_trace(a):
    return sum([a[i][i] for i in range(len(a))])

# Write the test for the matrix trace function
def test_matrix_trace():
    a = [[1, 2], [3, 4]]
    assert matrix_trace(a) == 5
    print("Test passed!")

# Create a function compute Singular Value Decomposition of a matrix
def matrix_svd(a):
    return np.linalg.svd(a)

# Write the test for the matrix SVD function
def test_matrix_svd():
    a = [[1, 2], [3, 4]]
    assert matrix_svd(a) == (array([[-0.40455358, -0.9145143 ], [-0.9145143 ,  0.40455358]]), array([5.4649857 , 0.36596619]), array([[-0.57604844, -0.81741556], [-0.81741556,  0.57604844]]))
    print("Test passed!")

# Create a function compute the Moore-Penrose pseudoinverse of a matrix
def matrix_pseudoinverse(a):
    return np.linalg.pinv(a)

# Write the test for the matrix pseudoinverse function
def test_matrix_pseudoinverse():
    a = [[1, 2], [3, 4]]
    assert matrix_pseudoinverse(a) == [[-2. ,  1. ], [ 1.5, -0.5]]
    print("Test passed!")

# Create a function compute the Cholesky decomposition of a matrix
def matrix_cholesky(a):
    return np.linalg.cholesky(a)

# Write the test for the matrix Cholesky decomposition function
def test_matrix_cholesky():
    a = [[1, 2], [3, 4]]
    assert matrix_cholesky(a) == [[1., 0.], [3., 0.81649658]]
    print("Test passed!")

