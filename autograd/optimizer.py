from abc import ABC, abstractmethod
from autograd.nodes import Node
from typing import Dict, List, Optional, Callable


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


def minimize(
    fn: Callable[[], Node],
    targets: List[Node],
    optimizer: Optimizer,
    max_iter: int = 1000,
    tol: float = 1e-6,
    data: Optional[List[List[int | float]]] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, float | int]:
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

    if not data:  # Fn is a normal function to minimize
        prev_grad = [0.0 for _ in targets]

        for _ in range(max_iter):
            loss_node = fn()

            loss_node.zero_grad()

            loss_node.backward()

            optimizer.step()

            grad_diffs = [
                abs(prev_grad[i] - node.grad) for i, node in enumerate(targets)
            ]
            prev_grad = [node.grad for node in targets]

            if all(diff < tol for diff in grad_diffs):
                print(
                    "=" * 80,
                    "\nAll variables converged successfully\n",
                    "=" * 80,
                    sep="",
                )
                return {node.label: node.value for node in targets}

        print(
            "=" * 80,
            "\nMax iteration reached. Not all variables converged.\n",
            "=" * 80,
            sep="",
        )

        return {node.label: node.value for node in targets}
