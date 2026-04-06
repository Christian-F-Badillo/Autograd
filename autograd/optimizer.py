from abc import ABC, abstractmethod
from autograd.nodes import Node
from typing import Dict, List, Optional, Callable
import math
import numpy as np


class Optimizer(ABC):
    @property
    @abstractmethod
    def lr(self) -> float:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def _setup(self, vars: List[Node]) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        self._lr = lr

    @property
    def lr(self) -> float:
        return self._lr

    def _setup(self, vars: List[Node]):
        self._vars = vars

    def step(self):
        for var in self._vars:
            var.value -= self.lr * var.grad


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = True,
    ) -> None:
        self._lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad

    @property
    def lr(self) -> float:
        return self._lr

    def _setup(self, vars: List[Node]):
        self._vars = vars
        self.m = [0.0 for _ in vars]
        self.v = [0.0 for _ in vars]

        if self.amsgrad:
            self.v_max = [0.0 for _ in vars]

        self.t = 0

    def step(self):
        self.t += 1

        bias_correction1 = 1 - self.beta1**self.t
        bias_correction2 = 1 - self.beta2**self.t

        for i, var in enumerate(self._vars):
            grad = var.grad

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            if self.amsgrad:
                self.v_max[i] = max(self.v_max[i], self.v[i])

                v_hat = self.v_max[i] / bias_correction2
            else:
                v_hat = self.v[i] / bias_correction2

            m_hat = self.m[i] / bias_correction1

            var.value -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


class BFGS(Optimizer):
    def __init__(self, lr: float = 1.0, epsilon: float = 1e-8) -> None:
        """
        Optimizador Cuasi-Newton BFGS.
        Nota: El 'lr' en BFGS estándar asume un line-search (Búsqueda Lineal).
        Para funciones de juguete, lr=1.0 o un valor pequeño constante puede bastar.
        """
        self._lr = lr
        self.epsilon = epsilon

    @property
    def lr(self) -> float:
        return self._lr

    def _setup(self, vars: List[Node]):
        self._vars = vars
        self.n = len(vars)

        self.H = np.eye(self.n)

        self.x_old = None
        self.grad_old = None

    def step(self):
        x_curr = np.array([var.value for var in self._vars], dtype=float)
        grad_curr = np.array([var.grad for var in self._vars], dtype=float)

        if self.x_old is not None:
            s_k = x_curr - self.x_old
            y_k = grad_curr - self.grad_old

            dot_ys = np.dot(y_k, s_k)
            if dot_ys > self.epsilon:
                rho = 1.0 / dot_ys
                I = np.eye(self.n)

                A = I - rho * np.outer(s_k, y_k)
                B = I - rho * np.outer(y_k, s_k)

                self.H = A @ self.H @ B + rho * np.outer(s_k, s_k)

        p_k = -self.H @ grad_curr

        self.x_old = x_curr.copy()
        self.grad_old = grad_curr.copy()

        for i, var in enumerate(self._vars):
            var.value += self.lr * p_k[i]


def minimize(
    fn: Callable[[], Node],
    targets: List[Node],
    optimizer: Optimizer,
    max_iter: int = 1000,
    tol: float = 1e-6,
    data: Optional[List[List[int | float]]] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    if batch_size and not data:
        raise ValueError("Can not use batch size without data.")

    initial_graph = fn()
    vars_in_scope = initial_graph._build_topologycal_sort()
    vars_set = set(targets)

    if not vars_set.issubset(vars_in_scope):
        raise ValueError(
            f"Targets {vars_set.difference(vars_in_scope)} are out of scope"
        )

    optimizer._setup(targets)

    targets_labels = [
        node.label if node.label else f"Target {i}" for i, node in enumerate(targets)
    ]

    if not data:  # Fn is a normal function to minimize
        for _ in range(max_iter):
            loss_node = fn()

            loss_node.zero_grad()

            loss_node.backward()

            optimizer.step()

            if all(abs(node.grad) < tol for node in targets):
                print(
                    "=" * 80,
                    "\nAll variables converged successfully\n",
                    "=" * 80,
                    sep="",
                )
                return {
                    label: node.value for label, node in zip(targets_labels, targets)
                }

        print(
            "=" * 80,
            "\nMax iteration reached. Not all variables converged.\n",
            "=" * 80,
            sep="",
        )

        return {label: node.value for label, node in zip(targets_labels, targets)}
