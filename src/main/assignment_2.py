import numpy as np
import sympy as sp
from scipy.linalg import solve

### Question 1: Neville's Interpolation ###
def neville_interpolation(x_values, y_values, x_target):
    """Computes Neville's Interpolation at a given x_target."""
    n = len(x_values)
    Q = np.zeros((n, n))

    # Initialize the diagonal with f(x) values
    for i in range(n):
        Q[i][0] = y_values[i]

    # Compute Neville's table
    for j in range(1, n):  # Column index
        for i in range(n - j):  # Row index
            Q[i][j] = ((x_target - x_values[i + j]) * Q[i][j - 1] -
                       (x_target - x_values[i]) * Q[i + 1][j - 1]) / (x_values[i] - x_values[i + j])

    return Q[0][n-1]  # The top-right value is our result

# Given data for Question 1
x_vals_q1 = [3.6, 3.8, 3.9]
y_vals_q1 = [1.675, 1.436, 1.318]
x_target_q1 = 3.7

# Compute interpolation for Question 1
result_q1 = neville_interpolation(x_vals_q1, y_vals_q1, x_target_q1)
print(f"Question 1: Neville's Method - Interpolated value at x = {x_target_q1}: {result_q1}\n")


### Question 2 & 3: Newton's Forward Interpolation ###
def forward_difference_table(x_values, y_values):
    """Computes the forward difference table correctly."""
    n = len(y_values)
    diff_table = np.zeros((n, n))

    # First column is f(x)
    for i in range(n):
        diff_table[i][0] = y_values[i]

    # Compute forward differences with step size correction
    for j in range(1, n):  # Column index
        for i in range(n - j):  # Row index
            diff_table[i][j] = (diff_table[i+1][j-1] - diff_table[i][j-1]) / (x_values[i + j] - x_values[i])

    return diff_table

def evaluate_newton_forward(x_values, diff_table, x_target):
    """Evaluates Newton's Forward Interpolation at a given x_target."""
    x0 = x_values[0]
    h = x_values[1] - x_values[0]
    s = (x_target - x0) / h

    result = diff_table[0][0]
    term = 1

    for i in range(1, len(x_values)):
        term *= (s - (i - 1)) / i
        result += term * diff_table[0][i]

    return result

# Given data for Newton's Method
x_vals_q2 = np.array([7.2, 7.4, 7.5, 7.6])
y_vals_q2 = np.array([23.5492, 25.3913, 26.8224, 27.4589])

diff_table_q2 = forward_difference_table(x_vals_q2, y_vals_q2)

print("\nQuestion 2: Forward Differences")
for i in range(3):  # Print only first 3 forward differences
    print(diff_table_q2[0][i+1])

x_target_q3 = 7.3
approximation_q3 = evaluate_newton_forward(x_vals_q2, diff_table_q2, x_target_q3)
print(f"\nQuestion 3: Approximation for f({x_target_q3}): {approximation_q3}")
### Question 4: Hermite Interpolation ###



def hermite_divided_difference_table(x_values, y_values, dy_values):
    """Constructs the divided difference table for Hermite interpolation."""
    n = len(x_values)
    size = 2 * n  # Because each x-value appears twice
    table = np.zeros((size, size))

    z_values = np.zeros(size)
    f_values = np.zeros(size)

    # Fill z_values and f_values with duplicate x-values and function values
    for i in range(n):
        z_values[2*i] = x_values[i]
        z_values[2*i+1] = x_values[i]
        f_values[2*i] = y_values[i]
        f_values[2*i+1] = y_values[i]

    # First column: x-values
    table[:, 0] = z_values
    # Second column: f(x) values
    table[:, 1] = f_values

    # First-order divided differences
    for i in range(n):
        table[2*i+1, 2] = dy_values[i]  # f'(x) goes directly into first-order differences
        if i != 0:
            table[2*i, 2] = (table[2*i, 1] - table[2*i-1, 1]) / (z_values[2*i] - z_values[2*i-1])

    # Higher-order divided differences
    for j in range(3, size):  # Starting from the third column
        for i in range(size - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (z_values[i+j-1] - z_values[i])

    return table

# Given data for Hermite Interpolation
x_vals_q4 = np.array([3.6, 3.8, 3.9])
y_vals_q4 = np.array([1.675, 1.436, 1.318])
dy_vals_q4 = np.array([-1.195, -1.188, -1.182])

# Compute the Hermite divided difference table
hermite_table = hermite_divided_difference_table(x_vals_q4, y_vals_q4, dy_vals_q4)

# Print the corrected Hermite interpolation table
print("\nQuestion 4: Corrected Hermite Interpolation Table")
print(hermite_table)


### Question 5: Cubic Spline Interpolation ###
def cubic_spline_matrices(x_values, y_values):
    """Computes matrix A, vector b, and vector x for cubic spline interpolation."""
    n = len(x_values) - 1
    h = np.diff(x_values)

    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = (3/h[i]) * (y_values[i+1] - y_values[i]) - (3/h[i-1]) * (y_values[i] - y_values[i-1])

    c = solve(A, b)

    return A, b, c

x_vals_q5 = np.array([2, 5, 8, 10])
y_vals_q5 = np.array([3, 5, 7, 9])

A_q5, b_q5, c_q5 = cubic_spline_matrices(x_vals_q5, y_vals_q5)

print("\nQuestion 5: Cubic Spline Matrices")
print("Matrix A:")
print(A_q5)
print("\nVector b:")
print(b_q5)
print("\nVector x (c coefficients):")
print(c_q5)