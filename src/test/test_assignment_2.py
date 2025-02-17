import unittest
import numpy as np
import io
import re
from contextlib import redirect_stdout
from main.assignment_2 import (
    neville_interpolation,
    compute_newton_differences,
    construct_hermite_table,
    generate_cubic_spline_system
)

class TestAssignment2(unittest.TestCase):

    def test_neville_interpolation(self):
        """Test Neville’s method."""
        x_values = [3.6, 3.8, 3.9]
        y_values = [1.675, 1.436, 1.318]
        x_target = 3.7
        expected_result = 1.5549999999999995
        result = neville_interpolation(x_values, y_values, x_target)
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_newton_forward_differences(self):
        """Test forward difference table."""
        expected_differences = [9.2105, 17.00166666666675, -141.82916666666722]
        output = io.StringIO()
        with redirect_stdout(output):
            compute_newton_differences()
        
        output_lines = output.getvalue().split("\n")
        computed_values = [float(line.strip()) for line in output_lines if line.strip().replace('.', '', 1).replace('-', '', 1).isdigit()]
        
        self.assertEqual(len(computed_values), len(expected_differences))
        for computed, expected in zip(computed_values, expected_differences):
            self.assertAlmostEqual(computed, expected, places=6)

    def test_newton_interpolation_value(self):
        """Test Newton's interpolation for f(7.3)."""
        expected_value = 24.016574999999992
        output = io.StringIO()
        with redirect_stdout(output):
            compute_newton_differences()

        output_lines = output.getvalue().split("\n")

        for line in output_lines:
            match = re.search(r"Interpolated value for f\(7.3\):\s*([-+]?\d*\.\d+)", line)
            if match:
                computed_value = float(match.group(1))
                self.assertAlmostEqual(computed_value, expected_value, places=6)
                return

        self.fail("Newton's interpolation output not found.")

    def test_hermite_interpolation(self):
        """Test Hermite interpolation table."""
        expected_output_part = "[ 3.60000000e+00  1.67500000e+00  0.00000000e+00"
        output = io.StringIO()
        with redirect_stdout(output):
            construct_hermite_table()
        
        self.assertIn(expected_output_part, output.getvalue())

    def test_cubic_spline_system(self):
        """Test cubic spline system matrices."""
        expected_matrix_part = "[ 1.0  0.0  0.0  0.0 ]"
        expected_vector_part = "[ 0.00  0.00  1.00  0.00 ]"
        expected_coefficients_part = "[ 0. -0.02702703 0.10810811 0.]"
        
        output = io.StringIO()
        with redirect_stdout(output):
            generate_cubic_spline_system()
        
        output_text = output.getvalue()
        self.assertIn(expected_matrix_part, output_text)
        self.assertIn(expected_vector_part, output_text)
        self.assertIn(expected_coefficients_part, output_text)

if __name__ == "__main__":
    unittest.main()
