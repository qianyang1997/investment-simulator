import cvxpy as cp
import pandas as pd
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


## GHOST


class TimeSeriesMetrics:

    def delta_return(
        self, price_data: pd.DataFrame, weights: cp.Variable, freq="D"
    ) -> cp.Expression:
        data = price_data.copy()
        data.index = pd.to_datetime(data.index)
        if freq != "D":
            data = data.resample(freq).mean()
        returns = self._get_percent_change_in_metric(data)
        return returns @ weights

    def time_series_portfolio_inflation_adjusted_return(
        self, price_data: pd.DataFrame, cpi_data: pd.DataFrame
    ) -> pd.DataFrame:
        nominal_return = self.time_series_portfolio_nominal_return(price_data)
        inflation_rate = self.inflation_rate(cpi_data)
        inflation_adjusted_return = (1 + nominal_return) / (1 - inflation_rate) - 1
        return inflation_adjusted_return

    def input_portfolio_price(
        self, price_data: pd.DataFrame, weights: cp.Variable
    ) -> cp.Expression:
        initial_value = self._get_initial_array(price_data)
        return initial_value @ weights

    def output_portfolio_price(
        self, price_data: pd.DataFrame, weights: cp.Variable
    ) -> cp.Expression:
        final_value = self._get_final_array(price_data)
        return final_value @ weights

    def time_series_portfolio_price(
        self, price_data: pd.DataFrame, weights: cp.Variable
    ) -> cp.Expression:
        return price_data @ weights

    def inflation_adjusted_return(
        self, price_data: pd.DataFrame, cpi_data, weights: cp.Variable
    ) -> cp.Expression:
        return (1 + self.nominal_return(price_data, weights)) / (
            1 + self.inflation_rate(cpi_data)
        ) - 1
