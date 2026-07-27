import unittest
import numpy as np

from tensor_engine import Tensor


class TestTensorEngine(unittest.TestCase):
    """
    Unit tests for the lightweight Tensor autograd engine.
    """

    def test_tensor_initialization(self):
        """Tensor should correctly initialize data and gradients."""

        tensor = Tensor([[1, 2], [3, 4]])

        np.testing.assert_allclose(tensor.data, np.array([[1.0, 2.0], [3.0, 4.0]]))

        np.testing.assert_allclose(tensor.grad, np.zeros((2, 2)))

    def test_addition_gradient(self):
        """Gradients should flow correctly through addition."""

        a = Tensor([[1.0, 2.0]])
        b = Tensor([[5.0, 6.0]])

        c = a + b

        loss = Tensor(c.data.sum(), (c,))

        def backward():
            c.grad += np.ones_like(c.data)

        loss._backward = backward

        loss.backward()

        expected = np.ones((1, 2))

        np.testing.assert_allclose(a.grad, expected)
        np.testing.assert_allclose(b.grad, expected)

    def test_broadcast_addition(self):
        """Broadcasted addition should correctly reduce gradients."""

        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        b = Tensor([[10.0, 20.0, 30.0]])

        c = a + b

        loss = Tensor(c.data.sum(), (c,))

        def backward():
            c.grad += np.ones_like(c.data)

        loss._backward = backward

        loss.backward()

        np.testing.assert_allclose(a.grad, np.ones_like(a.data))

        np.testing.assert_allclose(b.grad, np.array([[2.0, 2.0, 2.0]]))

    def test_matrix_multiplication_gradients(self):
        """Verify analytical gradients for matrix multiplication."""

        x = Tensor([[1.0, 2.0]])

        w = Tensor([[3.0], [4.0]])

        y = x @ w

        y.backward()

        expected_x = np.array([[3.0, 4.0]])
        expected_w = np.array([[1.0], [2.0]])

        np.testing.assert_allclose(x.grad, expected_x)
        np.testing.assert_allclose(w.grad, expected_w)

    def test_invalid_matrix_shapes(self):
        """Invalid matrix multiplication should raise ValueError."""

        a = Tensor([[1.0, 2.0]])
        b = Tensor([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            _ = a @ b

    def test_relu_gradient(self):
        """Negative values should have zero gradient."""

        x = Tensor([[-2.0, 3.0, -1.0, 5.0]])

        y = x.relu()

        loss = Tensor(y.data.sum(), (y,))

        def backward():
            y.grad += np.ones_like(y.data)

        loss._backward = backward

        loss.backward()

        expected = np.array([[0.0, 1.0, 0.0, 1.0]])

        np.testing.assert_allclose(x.grad, expected)

    def test_relu_zero_boundary(self):
        """Gradient at exactly zero should be zero."""

        x = Tensor([[0.0]])

        y = x.relu()

        y.backward()

        np.testing.assert_allclose(x.grad, np.array([[0.0]]))

    def test_zero_grad(self):
        """zero_grad should reset gradients."""

        x = Tensor([[1.0, 2.0]])
        w = Tensor([[3.0], [4.0]])

        y = x @ w

        y.backward()

        x.zero_grad()
        w.zero_grad()

        np.testing.assert_allclose(x.grad, np.zeros_like(x.data))

        np.testing.assert_allclose(w.grad, np.zeros_like(w.data))

    def test_gradient_accumulation(self):
        """Using the same tensor twice should accumulate gradients."""

        x = Tensor([[5.0]])

        y = x + x

        y.backward()

        np.testing.assert_allclose(x.grad, np.array([[2.0]]))

    def test_backward_requires_scalar(self):
        """Backward should reject non-scalar outputs."""

        x = Tensor([[1.0, 2.0]])

        y = x.relu()

        with self.assertRaises(RuntimeError):
            y.backward()

    def test_multi_layer_graph(self):
        """Verify gradients propagate through a multi-layer graph."""

        x = Tensor([[1.0, 2.0]])

        w1 = Tensor([[1.0, -1.0], [2.0, 3.0]])

        w2 = Tensor([[1.0], [2.0]])

        hidden = x @ w1
        activated = hidden.relu()
        output = activated @ w2

        output.backward()

        self.assertEqual(x.grad.shape, x.data.shape)
        self.assertEqual(w1.grad.shape, w1.data.shape)
        self.assertEqual(w2.grad.shape, w2.data.shape)

        self.assertTrue(np.any(x.grad != 0))
        self.assertTrue(np.any(w1.grad != 0))
        self.assertTrue(np.any(w2.grad != 0))

    def test_repr(self):
        """__repr__ should include Tensor information."""

        tensor = Tensor([[1.0]])

        representation = repr(tensor)

        self.assertIn("Tensor", representation)
        self.assertIn("shape", representation)
        self.assertIn("grad", representation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
