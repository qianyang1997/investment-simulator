import cvxpy as cp
from investment_simulator.optimizer_object import Metric, Optimizer
from investment_simulator import utils


def get_simple_return(data, tickers) -> Metric:
    metric = Metric(
        name="simple return",
        desc="% nominal growth or loss against initial input price.",
        expr=utils.get_percent_change_against_initial_state(data[tickers])
        @ Optimizer._optimizer_variables["weights"].var,
    )
    Optimizer.update_metrics(metric)
    return metric


def get_dod_return(data, tickers) -> Metric:
    metric = Metric(
        name="DoD return",
        desc="% DoD change in portfolio return.",
        expr=utils.get_percent_change_over_time(data[tickers])
        @ Optimizer._optimizer_variables["weights"].var,
    )
    Optimizer.update_metrics(metric)
    return metric


def get_return_against_benchmark(data, tickers, benchmark) -> Metric:
    nominal_return = utils.get_percent_change_against_initial_state(data[tickers])
    benchmark_return = utils.get_percent_change_against_initial_state(data[benchmark])
    metric = Metric(
        name=f"return against {benchmark}",
        desc=f"Nominal return minus benchmark return ({benchmark}).",
        expr=nominal_return @ Optimizer._optimizer_variables["weights"].var
        - benchmark_return,
    )
    Optimizer.update_metrics(metric)
    return metric


def get_drawdown(data, tickers) -> Metric:
    nominal_portfolio_return = (
        utils.get_percent_change_against_initial_state(data[tickers])
        @ Optimizer._optimizer_variables["weights"].var
    )
    metric = Metric(
        name="drawdown",
        desc="% drop of portfolio value since the maximum-to-date.",
        expr=cp.cummax(nominal_portfolio_return) - nominal_portfolio_return,
    )
    Optimizer.update_metrics(metric)
    return metric


def get_covariance_matrix(data, tickers) -> Metric:
    metric = Metric(
        name="covariance matrix",
        desc="Covariance matrix of historical ticker prices.",
        expr=utils.get_covariance_matrix(data[tickers]),
    )
    Optimizer.update_metrics(metric)
    return metric
