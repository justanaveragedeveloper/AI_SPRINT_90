from typing import Any, Iterator

from tensor_engine import Tensor


class Parameter(Tensor):
    """
    A Tensor subclass that marks a tensor as a learnable parameter.

    By default, requires_grad is set to True, making it track gradients
    during the backward pass.
    """

    def __init__(self, data) -> None:
        """
        Create a Parameter.

        Parameters
        ----------
        data
            Numeric scalar, list, or NumPy array containing the initial values.
        """
        super().__init__(data, requires_grad=True)


class Module:
    """
    Base class for all neural network modules.

    Your models should subclass this class. Modules can contain nested
    modules and Parameter instances. The class automatically tracks them
    recursively using Python's __setattr__ hook.

    Provides:
        - parameters()  : iterates over all learnable weights/biases.
        - zero_grad()   : resets gradients of all parameters to zero.
        - forward()     : must be overridden by subclasses.
        - __call__()    : calls forward().
    """

    def __init__(self) -> None:
        """
        Initialize the module with empty registries for submodules and parameters.
        """
        # Use object.__setattr__ to bypass our own __setattr__ during initialization
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_parameters", {})

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override setattr to automatically register nested modules and parameters.

        When you assign a Module or Parameter as an attribute (e.g., self.fc = Linear(...)),
        it is automatically added to the internal registries for recursive traversal.

        If you assign a different module/parameter with the same name, the old one is
        automatically unregistered. Assigning None removes it from the registries.
        """
        # Check if we are assigning None – remove from registries if it exists
        if value is None:
            # Pop from both registries (safe to pop even if not present)
            self._parameters.pop(name, None)
            self._modules.pop(name, None)
        else:
            # If assigning a Module or Parameter, unregister any previous entry with the same name
            # to prevent stale references.
            self._parameters.pop(name, None)
            self._modules.pop(name, None)

            # Now register the new object
            if isinstance(value, Module):
                self._modules[name] = value
            elif isinstance(value, Parameter):
                self._parameters[name] = value

        # Always set the attribute normally
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Parameter]:
        """
        Recursively yield all Parameter instances in this module and its children.

        Yields
        ------
        Parameter
            Every learnable tensor in the module hierarchy.
        """
        # Yield parameters directly owned by this module
        for param in self._parameters.values():
            yield param

        # Recurse into all child modules
        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self) -> None:
        """
        Reset gradients of all parameters in the module and its children to zero.

        This should be called before each backward pass during training to avoid
        accumulating gradients from multiple iterations.
        """
        for param in self.parameters():
            param.zero_grad()

    def forward(self, *args: Any, **kwargs: Any):
        """
        Define the forward pass of the module.

        Must be overridden by every concrete subclass.

        Raises
        ------
        NotImplementedError
            If a subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()."
        )

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Allow the module to be called as a function.

        This simply delegates to forward().
        """
        return self.forward(*args, **kwargs)
