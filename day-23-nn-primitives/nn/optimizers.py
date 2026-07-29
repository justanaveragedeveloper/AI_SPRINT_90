from typing import Any, Iterable

from .module import Parameter


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Performs a parameter update:
        W = W - lr * grad_W

    where grad_W is the accumulated gradient of the loss with respect to W.

    Parameters
    ----------
    params : Iterable[Parameter]
        An iterable of Parameter objects to be optimised.

    lr : float
        Learning rate (step size). Must be positive.

    Attributes
    ----------
    params : list[Parameter]
        List of parameters to update.
    lr : float
        Learning rate.

    Raises
    ------
    ValueError
        If lr is not positive, or if no parameters are provided.
    TypeError
        If any element in params is not a Parameter.
    """

    def __init__(self, params: Iterable[Parameter], lr: float = 0.01) -> None:
        """
        Initialize the SGD optimizer.

        Parameters
        ----------
        params : Iterable[Parameter]
            Parameters to be updated.
        lr : float, default=0.01
            Learning rate. Must be > 0.

        Raises
        ------
        ValueError
            If lr <= 0 or if params is empty.
        TypeError
            If any element in params is not a Parameter.
        """
        # Validate learning rate
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")

        # Convert to list to allow multiple iterations
        self.params = list(params)

        # Validate that parameters were provided
        if not self.params:
            raise ValueError(
                "Optimizer received no parameters. "
                "Make sure model.parameters() is not empty."
            )

        # Validate each parameter is a Parameter instance
        for param in self.params:
            if not isinstance(param, Parameter):
                raise TypeError(
                    f"Expected Parameter, got {type(param).__name__}. "
                    "Only Parameter objects should be passed to the optimizer."
                )

        self.lr = lr

    def step(self) -> None:
        """
        Perform a single optimisation step.

        Updates each parameter in-place using:
            data -= lr * grad

        The update is performed directly on the NumPy array, so no gradient
        tracking is involved (equivalent to torch.no_grad()).

        Only parameters with requires_grad=True are updated, which allows
        freezing specific parameters by setting requires_grad=False.
        """
        for param in self.params:
            # Only update if the parameter requires gradients
            if param.requires_grad:
                # In-place update using NumPy's array arithmetic
                param.data -= self.lr * param.grad

    def zero_grad(self) -> None:
        """
        Reset gradients of all parameters to zero.

        This should be called before each backward pass to prevent
        gradient accumulation across batches.
        """
        for param in self.params:
            param.zero_grad()

    def __repr__(self) -> str:
        """
        Return a readable representation of the SGD optimizer.
        """
        return f"SGD(lr={self.lr}, " f"param_count={len(self.params)})"
