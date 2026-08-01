from __future__ import annotations

import math
import unittest

from engine import Value


class TestAutogradEngine(unittest.TestCase):

    def test_basic_addition_and_multiplication(self):
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        d = a * b + c
        d.backward()

        self.assertEqual(d.value, 4.0)
        self.assertEqual(a.gradient, -3.0)
        self.assertEqual(b.gradient, 2.0)

    def test_variable_reuse_gradient_accumulation(self):
        # f(x) = x * x + x -> f'(x) = 2x + 1; for x = 3, f'(3) = 7
        x = Value(3.0)
        y = x * x + x
        y.backward()

        self.assertEqual(y.value, 12.0)
        self.assertEqual(x.gradient, 7.0)

    def test_tanh_activation_backward(self):
        x = Value(0.8814)
        y = x.tanh()
        y.backward()

        expected_tanh = math.tanh(0.8814)
        expected_grad = 1.0 - expected_tanh**2
        self.assertAlmostEqual(y.value, expected_tanh, places=4)
        self.assertAlmostEqual(x.gradient, expected_grad, places=4)

    def test_relu_activation_backward(self):
        x1 = Value(2.0)
        y1 = x1.relu()
        y1.backward()
        self.assertEqual(x1.gradient, 1.0)

        x2 = Value(-1.5)
        y2 = x2.relu()
        y2.backward()
        self.assertEqual(x2.gradient, 0.0)

    def test_power_operation(self):
        x = Value(3.0)
        y = x**3
        y.backward()
        self.assertEqual(y.value, 27.0)
        self.assertEqual(x.gradient, 27.0)  # 3 * x^2 = 27

    def test_power_edge_cases(self):
        # x = 0, exponent = 0 (should be 1, derivative 0)
        x = Value(0.0)
        y = x**0
        y.backward()
        self.assertEqual(y.value, 1.0)
        self.assertEqual(x.gradient, 0.0)

        # x = 0, exponent = 2 (derivative 0)
        x = Value(0.0)
        y = x**2
        y.backward()
        self.assertEqual(y.value, 0.0)
        self.assertEqual(x.gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
