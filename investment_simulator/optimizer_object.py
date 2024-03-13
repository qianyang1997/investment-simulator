"""Optimizer class to solve convex optimization problems.

Public methods:
    `add_variable`:
    `add_objective`:
    `add_constraint`:
    `add_metric`:
    `optimize`:
    `save_result`:
"""

import cvxpy as cp
import numpy as np
from typing import Union, Optional, Dict, List
from investment_simulator.config import set_logger


logger = set_logger("OPTIMIZER")


class Metric:

    def __init__(
        self, name: str, desc: str, expr: Union[np.number, np.ndarray, cp.Expression]
    ):
        self.name = name
        self.type = "metric"
        self.desc = desc
        self.expr = expr

    def __call__(self):
        return self.get_value()

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"Type: {self.type}\n"
            f"Description: {self.desc}\n"
            f"Value: {self.get_value()}"
        )

    def get_value(self):
        if isinstance(self.expr, cp.Expression):
            return self.expr.value
        else:
            return self.expr


class Variable:

    def __init__(self, name, desc, dim, **kwargs):
        self.name = name
        self.type = "variable"
        self.desc = desc
        self.var = cp.Variable(name=name, shape=dim, **kwargs)

    def __call__(self):
        return self.var.value

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"Type: {self.type}\n"
            f"Description: {self.desc}\n"
            f"Value: {self.var.value}"
        )


class Objective:

    def __init__(self, name: str, desc: str, obj_func: cp.Expression):
        self.name = name
        self.type = "objective"
        self.desc = desc
        self.obj_func = obj_func

    def __call__(self):
        return self.obj_func.value

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"Type: {self.type}\n"
            f"Description: {self.desc}\n"
            f"Value: {self.obj_func.value}"
        )


class Constraint:
    def __init__(self, name, desc, cons_expr: cp.Expression):
        self.name = name
        self.type = "constraint"
        self.desc = desc
        self.cons_expr = cons_expr
        self.dual_value = None

    def __call__(self):
        return self.dual_value

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"Type: {self.type}\n"
            f"Description: {self.desc}\n"
            f"Dual value: {self.dual_value}"
        )


class Optimizer:

    _optimizer_objectives = {}
    _optimizer_constraints = {}
    _optimizer_variables = {}
    _optimizer_metrics = {}

    def __init__(self):
        self.problem = None

    def clear(self):
        Optimizer._optimizer_objectives.clear()
        Optimizer._optimizer_constraints.clear()
        Optimizer._optimizer_variables.clear()
        Optimizer._optimizer_metrics.clear()

    @classmethod
    def update_objectives(cls, value: Union[Objective, Dict[str, Objective]]):
        if isinstance(value, Objective):
            value = {value.name: value}
        Optimizer._optimizer_objectives.update(value)

    @classmethod
    def update_constraints(cls, value: Union[Constraint, Dict[str, Constraint]]):
        if isinstance(value, Constraint):
            value = {value.name: value}
        Optimizer._optimizer_constraints.update(value)

    @classmethod
    def update_variables(cls, value: Union[Variable, Dict[str, Variable]]):
        """https://www.cvxpy.org/api_reference/cvxpy.expressions.html#variable"""
        if isinstance(value, Variable):
            value = {value.name: value}
        Optimizer._optimizer_variables.update(value)

    @classmethod
    def update_metrics(cls, value: Union[Metric, Dict[str, Metric]]):
        if isinstance(value, Metric):
            value = {value.name: value}
        Optimizer._optimizer_metrics.update(value)

    @classmethod
    def update_dual_values(cls, lis_constraints: List[cp.Expression]):
        # assign dual values to constraints
        counter = 0
        for c in Optimizer._optimizer_constraints:
            Optimizer._optimizer_constraints[c].dual_value = lis_constraints[
                counter
            ].dual_value
            counter += 1

    def optimize(self):
        """Run the optimizer instance.

        Raises:
            Exception: Having multiple objectives is not allowed.
        """
        # if multi-objective, raise Exception
        if len(self._optimizer_objectives) > 1:
            raise Exception("Having multiple objectives is not allowed.")
        # input objective
        objective = [v.obj_func for _, v in self._optimizer_objectives.items()][0]
        constraints = [v.cons_expr for _, v in self._optimizer_constraints.items()]
        # define and solve the problem
        self.problem = cp.Problem(objective, constraints)
        self.problem.solve()
        self.update_dual_values(constraints)
