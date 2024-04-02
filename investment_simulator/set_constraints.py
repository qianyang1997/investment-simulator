import cvxpy as cp
from investment_simulator import get_ts_metrics as ts
from investment_simulator import get_agg_metrics as agg
from investment_simulator.optimizer_object import Constraint, Optimizer


def keep_long_positions_only() -> Constraint:
    constraint = Constraint(
        name=f"keep long positions only",
        desc=f"Keep long positions only.",
        cons_expr=Optimizer._optimizer_variables["weights"].var >= 0,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_return_above_threshold(data, tickers, threshold: float) -> Constraint:
    nominal_return = ts.get_simple_return(data, tickers)
    final_return = agg.get_final_point_value(nominal_return)
    constraint = Constraint(
        name=f"keep return above {threshold:.2%}",
        desc=f"Ensure nominal return exceeds {threshold:.2%} on output date.",
        cons_expr=final_return.expr >= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_avg_return_above_threshold(data, tickers, threshold: float) -> Constraint:
    nominal_return = ts.get_simple_return(data, tickers)
    avg_return = agg.get_historical_avg(nominal_return)
    constraint = Constraint(
        name=f"keep average return above {threshold:.2%}",
        desc=f"Ensure average historical return (arithmetic mean) exceeds {threshold:.2%}.",
        cons_expr=avg_return.expr >= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_ema_return_above_threshold(
    data, tickers, smoothing_window: int, threshold: float
) -> Constraint:
    nominal_return = ts.get_simple_return(data, tickers)
    ema_return = agg.get_ema_weighted_avg(nominal_return, smoothing_window)
    constraint = Constraint(
        name=f"keep ema return above {threshold:.2%}",
        desc=f"Ensure the exponential moving average of nominal return exceeds {threshold:.2%}.",
        cons_expr=ema_return.expr >= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_avg_dod_return_above_threshold(data, tickers, threshold: float) -> Constraint:
    dod_return = ts.get_dod_return(data, tickers)
    avg_return = agg.get_historical_avg(dod_return)
    constraint = Constraint(
        name=f"keep dod return above {threshold:.2%}",
        desc=f"Ensure average day-over-day return exceeds {threshold:.2%}",
        cons_expr=avg_return.expr >= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_ema_dod_return_above_threshold(
    data, tickers, smoothing_window: int, threshold: float
) -> Constraint:
    dod_return = ts.get_dod_return(data, tickers)
    ema_return = agg.get_ema_weighted_avg(dod_return, smoothing_window)
    constraint = Constraint(
        name=f"keep ema dod return above {threshold:.2%}",
        desc=f"Ensure the exponential moving average of day-over-day return exceeds {threshold:.2%}.",
        cons_expr=ema_return.expr >= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_return_against_benchmark_above_threshold(
    data, tickers, benchmark, threshold: float
) -> Constraint:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    final_return = agg.get_final_point_value(delta)
    constraint = Constraint(
        name=f"keep return against {benchmark} above {threshold:.2%}",
        desc=f"Ensure the nominal return on output date is {threshold:.2%} greater than benchmark return ({benchmark})",
        cons_expr=final_return.expr >= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_avg_return_against_benchmark_above_threshold(
    data, tickers, benchmark, threshold: float
) -> Constraint:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    avg_return = agg.get_historical_avg(delta)
    constraint = Constraint(
        name=f"keep average return against {benchmark} above {threshold:.2%}",
        desc=f"Ensure the nominal return (arithmetic mean) is {threshold:.2%} greater than benchmark return ({benchmark}) on average.",
        cons_expr=avg_return.expr >= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_ema_return_against_benchmark_above_threshold(
    data, tickers, smoothing_window: int, benchmark, threshold: float
) -> Constraint:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    final_return = agg.get_ema_weighted_avg(delta, smoothing_window)
    constraint = Constraint(
        name=f"keep ema return against {benchmark} above {threshold:.2%}",
        desc=f"Ensure the exponential moving average of return against benchmark (nominal return minus benchmark return) exceeds {threshold:.2%}.",
        cons_expr=final_return.expr >= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_percent_outperforming_days_above_threshold(
    data, tickers, benchmark, threshold: float
) -> Constraint:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    percent_days = agg.get_percent_outperforming_days(delta)
    constraint = Constraint(
        name=f"keep % outperforming days above {threshold:.2%} (return against {benchmark})",
        desc=f"Ensure % days where nominal return outperforms benchmark return ({benchmark}) exceeds {threshold:.2%}.",
        cons_expr=percent_days.expr >= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def cap_maximal_loss_at_threshold(data, tickers, threshold: float) -> Constraint:
    nominal_return = ts.get_simple_return(data, tickers)
    min_return = agg.get_historical_min(nominal_return)
    constraint = Constraint(
        name=f"cap maximal loss at {threshold:.2%}",
        desc=f"Ensure maximal loss does not exceed {threshold:.2%}.",
        cons_expr=min_return.expr >= -threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def cap_maximal_dod_loss_at_threshold(data, tickers, threshold: float) -> Constraint:
    dod_return = ts.get_dod_return(data, tickers)
    min_dod_return = agg.get_historical_min(dod_return)
    constraint = Constraint(
        name=f"cap maximal single-day loss at {threshold:.2%}",
        desc=f"Ensure maximal single-day loss does not exceed {threshold:.2%}.",
        cons_expr=min_dod_return.expr >= -threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def cap_maximal_drawdown_at_threshold(data, tickers, threshold: float) -> Constraint:
    drawdown = ts.get_drawdown(data, tickers)
    max_drawdown = agg.get_historical_max(drawdown)
    constraint = Constraint(
        name=f"cap maximal drawdown at {threshold:.2%}",
        desc=f"Ensure maximal drawdown does not exceed {threshold:.2%}.",
        cons_expr=max_drawdown.expr <= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def cap_avg_drawdown_at_threshold(data, tickers, threshold: float) -> Constraint:
    drawdown = ts.get_drawdown(data, tickers)
    avg_drawdown = agg.get_historical_avg(drawdown)
    constraint = Constraint(
        name=f"cap average drawdown at {threshold:.2%}",
        desc=f"Ensure average drawdown does not exceed {threshold:.2%}.",
        cons_expr=avg_drawdown.expr <= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def cap_ema_drawdown_at_threshold(
    data, tickers, smoothing_window: int, threshold: float
) -> Constraint:
    drawdown = ts.get_drawdown(data, tickers)
    ema_drawdown = agg.get_ema_weighted_avg(drawdown, smoothing_window)
    constraint = Constraint(
        name=f"cap ema drawdown at {threshold:.2%}",
        desc=f"Ensure the exponential moving average of drawdown does not exceed {threshold:.2%}.",
        cons_expr=ema_drawdown.expr <= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def cap_percent_underperforming_days_at_threshold(
    data, tickers, benchmark, threshold: float
) -> Constraint:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    percent_days = agg.get_percent_underperforming_days(delta)
    constraint = Constraint(
        name=f"cap % underperforming days (return against {benchmark})",
        desc=f"Ensure the % days where nominal return underperforms against benchmark return does not exceed {threshold:.2%}.",
        cons_expr=percent_days.expr <= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_ema_deviation_below_threshold(
    data, tickers, smoothing_window: int, threshold: float
) -> Constraint:
    nominal_return = ts.get_simple_return(data, tickers)
    ema_deviation = agg.get_ema_deviation(nominal_return, smoothing_window)
    constraint = Constraint(
        name=f"keep ema deviation below {threshold:.2%}",
        desc=f"Ensure the degree to which nominal return deviates from exponential moving average does not exceed {threshold:.2%}.",
        cons_expr=ema_deviation.expr <= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint


def keep_portfolio_variance_below_threshold(
    data, tickers, threshold: float
) -> Constraint:
    covariance_matrix = ts.get_covariance_matrix(data, tickers)
    volatility = agg.get_portfolio_variance(covariance_matrix)
    constraint = Constraint(
        name=f"keep portfolio variance below {threshold:.2%}",
        desc=f"Ensure total portfolio variance (quadratic form of covariance) does not exceed {threshold}.",
        cons_expr=volatility.expr <= threshold,
    )
    Optimizer.update_constraints(constraint)
    return constraint
