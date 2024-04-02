import cvxpy as cp
from investment_simulator.set_variables import set_ema
from investment_simulator.optimizer_object import Metric, Optimizer


def get_historical_min(ts_metric: Metric) -> Metric:
    metric = Metric(
        name=f"{ts_metric.name} (historical min)",
        desc=f"Historical minimum of {ts_metric.name}.",
        expr=cp.min(ts_metric.expr),
    )
    Optimizer.update_metrics(metric)
    return metric


def get_historical_max(ts_metric: Metric) -> Metric:
    metric = Metric(
        name=f"{ts_metric.name} (historical max)",
        desc=f"Historical maximum of {ts_metric.name}.",
        expr=cp.max(ts_metric.expr),
    )
    Optimizer.update_metrics(metric)
    return metric


def get_historical_avg(ts_metric: Metric) -> Metric:
    metric = Metric(
        name=f"{ts_metric.name} (historical average)",
        desc=f"Historical average of {ts_metric.name}.",
        expr=cp.mean(ts_metric.expr),
    )
    Optimizer.update_metrics(metric)
    return metric


def get_ema_weighted_avg(ts_metric: Metric, smoothing_window: int) -> Metric:
    set_ema(ts_metric, window=smoothing_window)
    metric = Metric(
        name=f"{ts_metric.name} (exponential weighted average)",
        desc=f"Exponential moving average of {ts_metric.name} on return date.",
        expr=Optimizer._optimizer_variables["ema"].var[-1],
    )
    Optimizer.update_metrics(metric)
    return metric


def get_final_point_value(ts_metric: Metric) -> Metric:
    metric = Metric(
        name=f"{ts_metric.name} (value on return date)",
        desc=f"Value of {ts_metric.name} on return date.",
        expr=ts_metric.expr[-1],
    )
    Optimizer.update_metrics(metric)
    return metric


def get_percent_outperforming_days(ts_metric: Metric, threshold: float = 0) -> Metric:
    metric = Metric(
        name=f"{ts_metric.name} (% outperforming days)",
        desc=f"% days where portfolio outperforms benchmark.",
        expr=cp.sum(ts_metric.expr > threshold) / ts_metric.expr.size,
    )
    Optimizer.update_metrics(metric)
    return metric


def get_percent_underperforming_days(ts_metric: Metric, threshold: float = 0) -> Metric:
    metric = Metric(
        name=f"{ts_metric.name} (% underperforming days)",
        desc=f"% days where portfolio underperforms.",
        expr=cp.sum(ts_metric.expr < threshold) / ts_metric.expr.size,
    )
    Optimizer.update_metrics(metric)
    return metric


def get_ema_deviation(ts_metric: Metric, smoothing_window: int) -> Metric:
    set_ema(ts_metric, window=smoothing_window)
    metric = Metric(
        name=f"{ts_metric.name} (deviation from historical weighted average)",
        desc=f"Standard deviation of {ts_metric.name} against the ema smoothing function.",
        expr=cp.std(ts_metric - Optimizer._optimizer_variables["ema"].var),
    )
    Optimizer.update_metrics(metric)
    return metric


def get_portfolio_variance(cov_matrix: Metric) -> cp.Expression:
    metric = Metric(
        name=f"portfolio variance (classical method)",
        desc=f"Quadratic form of portfolio covariance matrix.",
        expr=cp.quad_form(
            Optimizer._optimizer_variables["weights"].var, cov_matrix.expr
        ),
    )
    Optimizer.update_metrics(metric)
    return metric
