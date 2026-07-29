import numpy as np

from typing import Optional

from .module import Module, Parameter
from tensor_engine import Tensor


class Linear(Module):
    """
    A fully-connected (dense) linear layer.

    Performs the transformation:
        Y = X @ W + B

    where:
        X : (batch_size, in_features)
        W : (in_features, out_features)
        B : (1, out_features)

    Weight initialisation uses Xavier / Glorot uniform:
        W ~ U(-a, a) where a = sqrt(6 / (in_features + out_features))

    Bias is initialised to zeros.

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    bias : bool, default=True
        If True, the layer learns an additive bias term.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Initialise the Linear layer.

        Parameters
        ----------
        in_features : int
            Dimensionality of the input (d_in).

        out_features : int
            Dimensionality of the output (d_out).

        bias : bool, default=True
            If True, include a learnable bias.

        Raises
        ------
        ValueError
            If in_features or out_features are not positive integers.
        """
        # Validate dimensions
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        super().__init__()

        # ---- Xavier / Glorot uniform initialisation ----
        a = np.sqrt(6.0 / (in_features + out_features))
        w_data = np.random.uniform(
            low=-a,
            high=a,
            size=(in_features, out_features)
        )

        # ---- Register weight as a Parameter ----
        self.weight = Parameter(w_data)

        # ---- Register bias as a Parameter (if enabled) ----
        self.bias: Optional[Parameter] = None
        if bias:
            b_data = np.zeros((1, out_features))
            self.bias = Parameter(b_data)

        # Store meta-data for reference
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, out_features).

        Raises
        ------
        TypeError
            If x is not a Tensor.
        ValueError
            If x is not 2‑dimensional or its second dimension does not match in_features.
        """
        # Validate input type
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor, got {type(x).__name__}")

        # Validate input shape
        if x.data.ndim != 2:
            raise ValueError(
                f"Input tensor must be 2‑dimensional (batch, features), "
                f"got shape {x.data.shape}"
            )
        if x.data.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input features {self.in_features}, " f"got {x.data.shape[1]}"
            )

        # Compute X @ W
        out = x @ self.weight

        # Add bias (broadcasted along the batch dimension)
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        """
        Return a readable representation of the Linear layer.
        """
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )
