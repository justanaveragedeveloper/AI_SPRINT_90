import unittest
import numpy as np

# Adjust imports to match your project layout
from tensor_engine import Tensor
from nn.module import Module, Parameter
from nn.layers import Linear
from nn.losses import MSELoss, BCELoss
from nn.optimizers import SGD


class TestNNPrimitives(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    # ------------------------------------------------------------
    # Original tests (unchanged)
    # ------------------------------------------------------------

    def test_parameter_initialization(self):
        p = Parameter(np.ones((3, 3)))
        self.assertTrue(p.requires_grad)
        self.assertTrue(isinstance(p, Tensor))

    def test_module_parameter_tracking(self):
        class SimpleNet(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(4, 8)
                self.fc2 = Linear(8, 1)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        net = SimpleNet()
        params = list(net.parameters())
        # fc1: W(4,8), B(1,8); fc2: W(8,1), B(1,1) -> Total 4 parameter Tensors
        self.assertEqual(len(params), 4)

    def test_linear_layer_forward_shape(self):
        layer = Linear(in_features=10, out_features=5)
        x = Tensor(np.random.randn(32, 10))
        out = layer(x)
        self.assertEqual(out.shape, (32, 5))

    def test_mse_loss_forward_and_backward(self):
        pred = Tensor(np.array([[2.0], [3.0]]), requires_grad=True)
        target = Tensor(np.array([[1.0], [1.0]]))
        criterion = MSELoss()
        loss = criterion(pred, target)

        # Analytical loss: ((2-1)^2 + (3-1)^2) / 2 = (1 + 4)/2 = 2.5
        self.assertAlmostEqual(loss.data.item(), 2.5, places=5)

        loss.backward()
        # dL/dPred = 2 * (pred - target) / N = [1.0, 2.0]
        expected_grad = np.array([[1.0], [2.0]])
        np.testing.assert_allclose(pred.grad, expected_grad, rtol=1e-5)

    def test_sgd_optimization_step(self):
        # Optimize simple scalar function: L = W^2 -> dL/dW = 2W
        w = Parameter(np.array([[3.0]]))
        optimizer = SGD([w], lr=0.1)

        loss = w * w
        loss.backward()

        optimizer.step()
        # Updated W = 3.0 - 0.1 * (2 * 3.0) = 3.0 - 0.6 = 2.4
        self.assertAlmostEqual(w.data.item(), 2.4, places=5)

        optimizer.zero_grad()
        self.assertAlmostEqual(w.grad.item(), 0.0, places=5)

    def test_full_training_loop(self):
        # Learn linear mapping y = 2x + 1
        X_data = np.array([[1.0], [2.0], [3.0], [4.0]])
        Y_data = np.array([[3.0], [5.0], [7.0], [9.0]])

        model = Linear(1, 1)
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = MSELoss()

        initial_loss = criterion(model(Tensor(X_data)), Tensor(Y_data)).data.item()

        for _ in range(100):
            optimizer.zero_grad()
            predictions = model(Tensor(X_data))
            loss = criterion(predictions, Tensor(Y_data))
            loss.backward()
            optimizer.step()

        final_loss = criterion(model(Tensor(X_data)), Tensor(Y_data)).data.item()
        self.assertLess(final_loss, initial_loss)

    # ------------------------------------------------------------
    # New tests (added based on code review)
    # ------------------------------------------------------------

    def test_bce_loss_forward_and_backward(self):
        """Test Binary Cross-Entropy loss forward and backward passes."""
        pred = Tensor(np.array([[0.9], [0.1]]), requires_grad=True)
        target = Tensor(np.array([[1.0], [0.0]]))
        criterion = BCELoss()

        loss = criterion(pred, target)
        self.assertTrue(np.isfinite(loss.data.item()))

        loss.backward()
        # BCELoss gradient should be finite and non-zero
        self.assertTrue(np.all(np.isfinite(pred.grad)))
        # For pred=0.9, target=1.0: gradient should be negative (pushing pred higher)
        self.assertLess(pred.grad[0, 0], 0)
        # For pred=0.1, target=0.0: gradient should be positive (pushing pred lower)
        self.assertGreater(pred.grad[1, 0], 0)

    def test_bce_loss_numerical_stability(self):
        """Test BCE loss with extreme probabilities (0 and 1) to verify clipping."""
        pred = Tensor(np.array([[0.0], [1.0]]), requires_grad=True)
        target = Tensor(np.array([[0.0], [1.0]]))
        criterion = BCELoss()

        loss = criterion(pred, target)
        # Loss should be finite (not -inf or nan)
        self.assertTrue(np.isfinite(loss.data.item()))

        loss.backward()
        # Gradients should also be finite
        self.assertTrue(np.all(np.isfinite(pred.grad)))

    def test_bce_loss_shape_mismatch(self):
        """Test BCE loss raises ValueError for shape mismatch."""
        pred = Tensor(np.random.randn(4, 3))
        target = Tensor(np.random.randn(5, 3))
        criterion = BCELoss()

        with self.assertRaises(ValueError):
            criterion(pred, target)

    def test_bce_loss_invalid_input(self):
        """Test BCE loss raises ValueError for predictions outside [0, 1]."""
        pred = Tensor(np.array([[2.0], [-1.0]]), requires_grad=True)
        target = Tensor(np.array([[1.0], [0.0]]))
        criterion = BCELoss()

        with self.assertRaises(ValueError):
            criterion(pred, target)

    def test_linear_validation_negative_features(self):
        """Test Linear layer raises ValueError for invalid feature dimensions."""
        with self.assertRaises(ValueError):
            Linear(in_features=-5, out_features=10)
        with self.assertRaises(ValueError):
            Linear(in_features=5, out_features=-10)
        with self.assertRaises(ValueError):
            Linear(in_features=0, out_features=10)

    def test_linear_forward_validation(self):
        """Test Linear layer input validation in forward pass."""
        layer = Linear(in_features=10, out_features=5)

        # Wrong input type
        with self.assertRaises(TypeError):
            layer(np.random.randn(32, 10))

        # Wrong input dimension
        with self.assertRaises(ValueError):
            x = Tensor(np.random.randn(32, 8))  # Should be 10
            layer(x)

        # Wrong input shape (1D instead of 2D)
        with self.assertRaises(ValueError):
            x = Tensor(np.random.randn(10))
            layer(x)

    def test_bias_broadcasting(self):
        """
        Test that bias gradients are correctly reduced via broadcasting.

        When bias has shape (1, out_features) and batch size > 1,
        the gradient should be summed across the batch dimension.
        """
        layer = Linear(in_features=3, out_features=2)
        x = Tensor(np.random.randn(64, 3))
        y = layer(x)

        # Create a dummy gradient for the output
        y.grad = np.ones_like(y.data)
        y._backward()

        # Bias gradient shape should be (1, 2) after reduction
        self.assertEqual(layer.bias.grad.shape, (1, 2))

        # All values in bias gradient should be 64 (sum over batch)
        expected = np.full((1, 2), 64.0)
        np.testing.assert_allclose(layer.bias.grad, expected, rtol=1e-5)

    def test_xavier_initialization(self):
        """Test that Xavier uniform initialization produces weights within bounds."""
        in_features, out_features = 10, 20
        layer = Linear(in_features, out_features)

        a = np.sqrt(6.0 / (in_features + out_features))

        # Check weights are within [-a, a]
        self.assertTrue(np.all(layer.weight.data >= -a))
        self.assertTrue(np.all(layer.weight.data <= a))

        # Check bias is initialized to zeros (if enabled)
        self.assertTrue(np.all(layer.bias.data == 0))

    def test_requires_grad_propagation(self):
        """Test that requires_grad propagates through operations."""
        a = Tensor([1.0], requires_grad=True)
        b = Tensor([2.0])

        # a + b should require gradients because a does
        c = a + b
        self.assertTrue(c.requires_grad)

        # a * b should require gradients because a does
        d = a * b
        self.assertTrue(d.requires_grad)

        # b + b should NOT require gradients because neither does
        e = b + b
        self.assertFalse(e.requires_grad)

    def test_parameter_replacement(self):
        """Test that replacing a parameter correctly removes the old one."""

        class Net(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter([1.0])
                self.weight = Parameter([2.0])

        net = Net()
        params = list(net.parameters())
        # Should only have one parameter (the second one)
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].data.item(), 2.0)

    def test_parameter_set_to_none(self):
        """Test that setting a parameter to None removes it from tracking."""

        class Net(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter([1.0])
                self.bias = Parameter([2.0])
                self.bias = None

        net = Net()
        params = list(net.parameters())
        # Should only have 'weight', since 'bias' was set to None
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].data.item(), 1.0)

    def test_optimizer_validation(self):
        """Test SGD optimizer validation."""
        # Invalid learning rate
        with self.assertRaises(ValueError):
            SGD([Parameter([1.0])], lr=0.0)
        with self.assertRaises(ValueError):
            SGD([Parameter([1.0])], lr=-0.1)

        # Empty parameter list
        with self.assertRaises(ValueError):
            SGD([])

        # Invalid parameter type
        with self.assertRaises(TypeError):
            SGD([1, 2, 3])

    def test_mse_loss_shape_mismatch(self):
        """Test MSE loss raises ValueError for shape mismatch."""
        pred = Tensor(np.random.randn(4, 3))
        target = Tensor(np.random.randn(5, 3))
        criterion = MSELoss()

        with self.assertRaises(ValueError):
            criterion(pred, target)

    def test_mse_loss_input_types(self):
        """Test MSE loss raises TypeError for non-Tensor inputs."""
        criterion = MSELoss()

        with self.assertRaises(TypeError):
            criterion(np.array([1.0]), Tensor([1.0]))

        with self.assertRaises(TypeError):
            criterion(Tensor([1.0]), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
