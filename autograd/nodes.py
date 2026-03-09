from __future__ import annotations
from typing import Set
import math

from matplotlib import rcdefaults


class Node:
    def __init__(
        self,
        value: int | float = 0,
        label: str = "",
        requires_grad: bool = True,
        _parents: Set[Node] = set(),
        _op: str = "",
    ) -> None:
        self._label = label
        self._data = value
        self._parents = set(_parents) if _parents else set()
        self._op = _op if _op else ""
        self._grad = 0.0
        self._backward = lambda: None
        self._requires_grad = requires_grad

    @property
    def label(self):
        return self._label

    @property
    def grad(self) -> float:
        return self._grad

    @property
    def value(self) -> float | int:
        return self._data

    @property
    def parents(self) -> Set[Node]:
        return self._parents

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def zero_grad(self) -> None:
        topo_order = self._build_topologycal_sort()

        for node in topo_order:
            node._to_zero_grad()

    def backward(self):
        topo_order = self._build_topologycal_sort()

        self._grad = 1.0
        for node in reversed(topo_order):
            node._backward()

    def __add__(self, other):
        if not isinstance(other, (float, int, Node)):
            raise ArithmeticError(f"Can't operate Variable and {type(other)}")

        if isinstance(other, (float, int)):
            other = Scalar(other, label="Scalar")

        result = Variable(self.value + other.value, _parents={self, other}, _op="+")

        def _backward():
            if self.requires_grad:
                self._grad += result.grad * 1

            if getattr(other, "requires_grad", False):
                other._grad += result.grad * 1

        result._backward = _backward

        return result

    def __sub__(self, other):
        if not isinstance(other, (float, int, Node)):
            raise ArithmeticError(f"Can't operate Variable and {type(other)}")

        if isinstance(other, (float, int)):
            other = Scalar(other, label="Scalar")

        result = Variable(self.value - other.value, _parents={self, other}, _op="-")

        def _backward():
            if self.requires_grad:
                self._grad += result.grad * 1

            if getattr(other, "requires_grad", False):
                other._grad += result.grad * -1

        result._backward = _backward

        return result

    def __mul__(self, other):
        if not isinstance(other, (float, int, Node)):
            raise ArithmeticError(f"Can't operate Variable and {type(other)}")

        if isinstance(other, (float, int)):
            other = Scalar(other, label="Scalar")

        result = Variable(self.value * other.value, _parents={self, other}, _op="*")

        def _backward():
            if self.requires_grad:
                self._grad += result.grad * other.value

            if getattr(other, "requires_grad", False):
                other._grad += result.grad * self.value

        result._backward = _backward

        return result

    def __truediv__(self, other):
        if not isinstance(other, (float, int, Node)):
            raise ArithmeticError(f"Can't operate Variable and {type(other)}")

        if isinstance(other, (float, int)):
            other = Scalar(other, label="Scalar")

        if other.value == 0:
            raise ZeroDivisionError("Found zero division")

        result = Variable(self.value / other.value, _parents={self, other}, _op="÷")

        def _backward():
            if self.requires_grad:
                self._grad += result.grad * (1 / other.value)

            if getattr(other, "requires_grad", False):
                other._grad += result.grad * (-self.value / (other.value**2))

        result._backward = _backward

        return result

    def __pow__(self, other):
        if not isinstance(other, (float, int, Node)):
            raise ArithmeticError(f"Can't operate Variable and {type(other)}")

        if isinstance(other, (float, int)):
            other = Scalar(other, label="Scalar")

        result = Variable(self.value**other.value, _parents={self, other}, _op="pow")

        def _backward():
            if self.requires_grad:
                self._grad += result.grad * (
                    other.value * (self.value) ** (other.value - 1)
                )
            if getattr(other, "requires_grad", False):
                other._grad += result.grad * (
                    (self.value**other.value) * math.log(self.value)
                )

        result._backward = _backward

        return result

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        if not isinstance(other, (float, int, Node)):
            raise ArithmeticError(f"Can't operate Variable and {type(other)}")

        if isinstance(other, (float, int)):
            other = Scalar(other, label="Scalar")

        result = Variable(other.value - self.value, _parents={self, other}, _op="-")

        def _backward():
            if self.requires_grad:
                self._grad += result.grad * -1

            if getattr(other, "requires_grad", False):
                other._grad += result.grad * 1

        result._backward = _backward

        return result

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        if not isinstance(other, (float, int, Node)):
            raise ArithmeticError(f"Can't operate Variable and {type(other)}")

        if isinstance(other, (float, int)):
            other = Scalar(other, label="Scalar")

        if self.value == 0:
            raise ZeroDivisionError("Found zero division")

        result = Variable(other.value / self.value, _parents={self, other}, _op="÷")

        def _backward():
            if getattr(other, "requires_grad", False):
                other._grad += result.grad * (1 / self.value)

            if self.requires_grad:
                self._grad += result.grad * (-other.value / (self.value**2))

        result._backward = _backward

        return result

    def _to_zero_grad(self) -> None:
        self._grad = 0.0

    def _build_topologycal_sort(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build(parent)
                topo.append(v)

        build(self)
        return topo

    @property
    def _node_type(self) -> str:
        return "Node"


class Variable(Node):
    def __init__(
        self,
        value: int | float = 0.0,
        label: str = "",
        _parents: Set[Node] = set(),
        _op: str = "",
    ):
        super().__init__(value, label, requires_grad=True, _parents=_parents, _op=_op)

    def __repr__(self) -> str:
        return f"Variable({self.value})"

    def __str__(self) -> str:
        return f"Variable({self.value})"

    @property
    def _node_type(self) -> str:
        return "Variable"


class Scalar(Node):
    def __init__(
        self,
        value: int | float,
        label: str = "",
        _parents: Set[Node] = set(),
        _op: str = "",
    ) -> None:
        """
        Class to compute all the scalar operations.
        """

        super().__init__(value, label, requires_grad=False, _parents=_parents, _op=_op)

    def _backward(self):
        pass

    def __repr__(self) -> str:
        return f"Scalar( {self._data} )"

    def __str__(self) -> str:
        return f"Scalar( {self._data} )"

    @property
    def _node_type(self) -> str:
        return "Scalar"
