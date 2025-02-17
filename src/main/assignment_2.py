import numpy as np
import sympy as sp
from scipy.linalg import solve

### Question 1: Neville's Interpolation ###
def neville_interpolation(x_values, y_values, x_target):
    """Uses Neville's iterative method to approximate f(x) at a given target point."""
    n = len(x_values)
    neville_table = np.zeros((n, n))

    # Initialize the diagonal with f(x) values
    for i in range(n):
        neville_table[i, 0] = y_values[i]

    # Compute Neville's table
    for j in range(1, n):
        for i in range(n - j):
            neville_table[i, j] = ((x_target - x_values[i + j]) * neville_table[i, j - 1] -
                                    (x_target - x_values[i]) * neville_table[i + 1, j - 1]) / (x_values[i] - x_values[i + j])

    return neville_table[0][n-1]

# Given data for Question 1
x_vals_q1 = [3.6, 3.8, 3.9]
y_vals_q1 = [1.675, 1.436, 1.318]
x_target_q1 = 3.7

# Compute interpolation for Question 1
result_q1 = neville_interpolation(x_vals_q1, y_vals_q1, x_target_q1)
print("-" * 40)
print(f"Question 1: Neville's Method - Interpolated value at x = {x_target_q1}: {result_q1}")
print("-" * 40)

### Question 2 & 3: Newton’s Forward Difference Interpolation ###
def compute_newton_differences():
    """Constructs the forward difference table and approximates f(7.3) using Newton's method."""
    x_values = np.array([7.2, 7.4, 7.5, 7.6], dtype=float)
    y_values = np.array([23.5492, 25.3913, 26.8224, 27.4589], dtype=float)
    num_points = len(x_values)

    difference_table = np.zeros((num_points, num_points))
    difference_table[:, 0] = y_values  

    for col in range(1, num_points):
        for row in range(col, num_points):
            difference_table[row, col] = (difference_table[row, col - 1] - difference_table[row - 1, col - 1]) / (x_values[row] - x_values[row - col])

    print("\n" + "-" * 40)
    print("Question 2: Newton's Forward Differences")
    for i in range(1, num_points):
        print(difference_table[i, i])
    print("-" * 40)

    # Question 3: Approximate f(7.3)
    x_query = 7.3
    interpolation = difference_table[0, 0]
    increment = 1  

    for i in range(1, num_points):
        increment *= (x_query - x_values[i - 1])
        interpolation += increment * difference_table[i, i]

    print(f"Question 3: Interpolated value for f({x_query}): {interpolation}")
    print("-" * 40)

### Question 4: Hermite Interpolation ###
def construct_hermite_table():
    """Constructs and prints the Hermite divided difference table."""
    x_points = [3.6, 3.6, 3.8, 3.8, 3.9, 3.9]
    function_values = [1.675, 1.675, 1.436, 1.436, 1.318, 1.318]
    derivative_values = [-1.195, -1.195, -1.188, -1.188, -1.182, -1.182]

    total_points = len(x_points)
    hermite_table = np.zeros((total_points, total_points))

    hermite_table[:, 0] = x_points
    hermite_table[:, 1] = function_values

    for i in range(1, total_points):
        if x_points[i] == x_points[i - 1]:
            hermite_table[i, 2] = derivative_values[i]
        else:
            hermite_table[i, 2] = (hermite_table[i, 1] - hermite_table[i - 1, 1]) / (x_points[i] - x_points[i - 1])

    for col in range(3, total_points):
        for row in range(col - 1, total_points):
            hermite_table[row, col] = (hermite_table[row, col - 1] - hermite_table[row - 1, col - 1]) / (x_points[row] - x_points[row - col + 1])

    print("\nQuestion 4: Hermite Interpolation Table")
    for row in range(total_points):
        print("[", end="")
        for col in range(total_points - 1):
            print(f"{hermite_table[row][col]: 12.8e} ", end="")
        print("]")
    print("-" * 40)

### Question 5: Cubic Spline Interpolation ###
def generate_cubic_spline_system():
    """Constructs and displays the cubic spline system (Matrix A, Vector b, Vector x)."""
    x_data = np.array([2, 5, 8, 10], dtype=float)
    y_data = np.array([3, 5, 7, 9], dtype=float)
    total_nodes = len(x_data)

    h_intervals = np.zeros(total_nodes - 1, dtype=float)
    for i in range(total_nodes - 1):
        h_intervals[i] = x_data[i + 1] - x_data[i]

    A_matrix = np.zeros((total_nodes, total_nodes), dtype=float)
    A_matrix[0, 0] = 1.0
    A_matrix[total_nodes - 1, total_nodes - 1] = 1.0

    for i in range(1, total_nodes - 1):
        A_matrix[i, i - 1] = h_intervals[i - 1]
        A_matrix[i, i] = 2.0 * (h_intervals[i - 1] + h_intervals[i])
        A_matrix[i, i + 1] = h_intervals[i]

    b_vector = np.zeros(total_nodes, dtype=float)
    for i in range(1, total_nodes - 1):
        term1 = (3.0 / h_intervals[i]) * (y_data[i + 1] - y_data[i])
        term2 = (3.0 / h_intervals[i - 1]) * (y_data[i] - y_data[i - 1])
        b_vector[i] = term1 - term2

    c_coeffs = np.linalg.solve(A_matrix, b_vector)

    print("\nQuestion 5: Cubic Spline System")
    print("Matrix A:")
    for row in A_matrix:
        print("[", end="")
        for val in row:
            print(f"{val:4.1f} ", end="")
        print("]")

    print("\nVector b:")
    print("[", end="")
    for val in b_vector:
        print(f"{val:5.2f} ", end="")
    print("]")

    print("\nVector x (c coefficients):")
    print("[ 0.", end="")
    for val in c_coeffs[1:-1]:  
        print(f" {val:.8f}", end="")
    print(" 0.]\n")

### **Main Function to Run All Questions**
def run_interpolation_procedures():
    """Executes all interpolation methods for Questions 1-5."""
    compute_newton_differences()
    construct_hermite_table()
    generate_cubic_spline_system()

if __name__ == "__main__":
    run_interpolation_procedures()
