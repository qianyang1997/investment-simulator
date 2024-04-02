import cvxpy as cp
from typing import Optional, List, Dict, Tuple
from investment_simulator.optimizer_object import (
    Constraint,
    Metric,
    Variable,
    Optimizer,
)


def set_weights(tickers: List[str]):
    weights = Optimizer._optimizer_metrics.get("weights")
    if not weights:
        # add weights optimizer
        Optimizer.update_variables(
            Variable(
                name="weights",
                desc="Portfolio allocation % for each ticker",
                dim=len(tickers),
            )
        )
        # ensure portfolio shares sum to 100%
        Optimizer.update_constraints(
            Constraint(
                name="sum(shares)==1",
                desc="ensure portfolio shares sum to 100%",
                cons_expr=cp.sum(Optimizer._optimizer_variables["weights"].var) == 1,
            )
        )


def set_ema(return_metric: Metric, window: int):
    ema = Optimizer._optimizer_metrics.get("ema")
    if not ema:
        # add ema variables to optimizer
        Optimizer.update_variables(
            Variable(
                name="ema",
                desc="Exponential moving average of portfolio return",
                dim=return_metric.expr.size,
            )
        )
        # define constraints for ema variables
        alpha = 2 / (window + 1)
        base_case = (
            Optimizer._optimizer_variables["ema"].var[0] == return_metric.expr[0]
        )
        Optimizer.update_constraints(
            Constraint(
                name="Recursively assign ema: base case",
                desc="Assign first value of ema to be the first value of the time series.",
                cons_expr=base_case,
            )
        )
        recursive_case = (
            Optimizer._optimizer_variables["ema"].var[1:]
            == alpha * return_metric.expr[1:]
            + (1 - alpha) * Optimizer._optimizer_variables["ema"].var[:-1]
        )
        Optimizer.update_constraints(
            Constraint(
                name="recursively assign ema: recursive case",
                desc="Assign each subsequent value of ema to be a weighted average of the current value and the previous ema.",
                cons_expr=recursive_case,
            )
        )
