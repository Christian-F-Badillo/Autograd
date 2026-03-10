from autograd.nodes import Node, Scalar
import math


class Sin(Node):
    def __init__(self, x: int | float | Node, label: str = "") -> None:
        if isinstance(x, (int, float)):
            x = Scalar(x, label="Scalar")

        self.x = x
        forward = math.sin(x.value)
        requires_grad = getattr(x, "requires_grad", False)

        super().__init__(
            value=forward,
            label=label,
            _parents={x},
            _op="sin",
            requires_grad=requires_grad,
        )

    def _backward(self):
        if self.requires_grad and getattr(self.x, "requires_grad", False):
            self.x._grad += self.grad * math.cos(self.x.value)

    @property
    def _node_type(self) -> str:
        return "Variable" if self.requires_grad else "Scalar"


class Cos(Node):
    def __init__(self, x: int | float | Node, label: str = "") -> None:
        if isinstance(x, (int, float)):
            x = Scalar(x, label="Scalar")

        self.x = x
        forward = math.cos(x.value)
        requires_grad = getattr(x, "requires_grad", False)

        super().__init__(
            value=forward,
            label=label,
            _parents={x},
            _op="cos",
            requires_grad=requires_grad,
        )

    def _backward(self):
        if self.requires_grad and getattr(self.x, "requires_grad", False):
            self.x._grad += self.grad * -math.sin(self.x.value)

    @property
    def _node_type(self) -> str:
        return "Variable" if self.requires_grad else "Scalar"


class Exp(Node):
    def __init__(self, x: int | float | Node, label: str = "") -> None:
        if isinstance(x, (int, float)):
            x = Scalar(x, label="Scalar")

        self.x = x
        forward = math.exp(x.value)
        requires_grad = getattr(x, "requires_grad", False)

        super().__init__(
            value=forward,
            label=label,
            _parents={x},
            _op="exp",
            requires_grad=requires_grad,
        )

    def _backward(self):
        if self.requires_grad and getattr(self.x, "requires_grad", False):
            self.x._grad += self.grad * self.value

    @property
    def _node_type(self) -> str:
        return "Variable" if self.requires_grad else "Scalar"


class Log(Node):
    def __init__(self, x: int | float | Node, label: str = "") -> None:
        if isinstance(x, (int, float)):
            x = Scalar(x, label="Scalar")

        self.x = x
        forward = math.log(x.value)
        requires_grad = getattr(x, "requires_grad", False)

        super().__init__(
            value=forward,
            label=label,
            _parents={x},
            _op="log",
            requires_grad=requires_grad,
        )

    def _backward(self):
        if self.requires_grad and getattr(self.x, "requires_grad", False):
            self.x._grad += self.grad * (1 / self.x.value)

    @property
    def _node_type(self) -> str:
        return "Variable" if self.requires_grad else "Scalar"


class ReLU(Node):
    def __init__(self, x: int | float | Node, label: str = "") -> None:
        if isinstance(x, (int, float)):
            x = Scalar(x, label="Scalar")

        self.x = x
        forward = x.value if x.value > 0 else 0.0
        requires_grad = getattr(x, "requires_grad", False)

        super().__init__(
            value=forward,
            label=label,
            _parents={x},
            _op="relu",
            requires_grad=requires_grad,
        )

    def _backward(self):
        if self.requires_grad and getattr(self.x, "requires_grad", False):
            local_grad = 1.0 if self.x.value > 0 else 0.0
            self.x._grad += self.grad * local_grad

    @property
    def _node_type(self) -> str:
        return "Variable" if self.requires_grad else "Scalar"


class Sigmoid(Node):
    def __init__(self, x: int | float | Node, label: str = "") -> None:
        if isinstance(x, (int, float)):
            x = Scalar(x, label="Scalar")

        self.x = x
        if x.value < -100:
            forward = 0.0
        else:
            forward = 1.0 / (1.0 + math.exp(-x.value))

        requires_grad = getattr(x, "requires_grad", False)

        super().__init__(
            value=forward,
            label=label,
            _parents={x},
            _op="σ",
            requires_grad=requires_grad,
        )

    def _backward(self):
        if self.requires_grad and getattr(self.x, "requires_grad", False):
            local_grad = self.value * (1.0 - self.value)
            self.x._grad += self.grad * local_grad

    @property
    def _node_type(self) -> str:
        return "Variable" if self.requires_grad else "Scalar"


class Tanh(Node):
    def __init__(self, x: int | float | Node, label: str = "") -> None:
        if isinstance(x, (int, float)):
            x = Scalar(x, label="Scalar")

        self.x = x
        forward = math.tanh(x.value)
        requires_grad = getattr(x, "requires_grad", False)

        super().__init__(
            value=forward,
            label=label,
            _parents={x},
            _op="tanh",
            requires_grad=requires_grad,
        )

    def _backward(self):
        if self.requires_grad and getattr(self.x, "requires_grad", False):
            local_grad = 1.0 - (self.value**2)
            self.x._grad += self.grad * local_grad

    @property
    def _node_type(self) -> str:
        return "Variable" if self.requires_grad else "Scalar"


# ----------------------------------------------------------------------------------------------
# Wrappers for Functionals Classes
# ----------------------------------------------------------------------------------------------


def sin(x: int | float | Node, label: str = "") -> Sin:
    return Sin(x, label=label)


def cos(x: int | float | Node, label: str = "") -> Cos:
    return Cos(x, label=label)


def exp(x: int | float | Node, label: str = "") -> Exp:
    return Exp(x, label=label)


def log(x: int | float | Node, label: str = "") -> Log:
    return Log(x, label=label)


def relu(x: int | float | Node, label: str = "") -> ReLU:
    return ReLU(x, label=label)


def sigmoid(x: int | float | Node, label: str = "") -> Sigmoid:
    return Sigmoid(x, label=label)


def tanh(x: int | float | Node, label: str = "") -> Tanh:
    return Tanh(x, label=label)
