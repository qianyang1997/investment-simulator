import cvxpy as cp
from investment_simulator import get_ts_metrics as ts
from investment_simulator import get_agg_metrics as agg
from investment_simulator.optimizer_object import Objective, Optimizer


def maximize_return(data, tickers) -> Objective:
    nominal_return = ts.get_simple_return(data, tickers)
    final_return = agg.get_final_point_value(nominal_return)
    objective = Objective(
        name="maximize return",
        desc="Maximize nominal return on output date.",
        obj_func=cp.Maximize(final_return.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def maximize_avg_return(data, tickers) -> Objective:
    nominal_return = ts.get_simple_return(data, tickers)
    avg_return = agg.get_historical_avg(nominal_return)
    objective = Objective(
        name="maximize average return",
        desc="Maximize average historical return (arithmetic mean).",
        obj_func=cp.Maximize(avg_return.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def maximize_ema_return(data, tickers, smoothing_window: int) -> Objective:
    nominal_return = ts.get_simple_return(data, tickers)
    ema_return = agg.get_ema_weighted_avg(nominal_return, smoothing_window)
    objective = Objective(
        name="maximize ema return",
        desc="Maximize the exponential moving average of the return on final output date.",
        obj_func=cp.Maximize(ema_return.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def maximize_avg_dod_return(data, tickers) -> Objective:
    dod_return = ts.get_dod_return(data, tickers)
    avg_return = agg.get_historical_avg(dod_return)
    objective = Objective(
        name="maximize average DoD return",
        desc="Maximize average day-over-day return (arithmetic mean).",
        obj_func=cp.Maximize(avg_return.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def maximize_ema_dod_return(data, tickers, smoothing_window: int) -> Objective:
    dod_return = ts.get_dod_return(data, tickers)
    ema_return = agg.get_ema_weighted_avg(dod_return, smoothing_window)
    objective = Objective(
        name="maximize ema DoD return",
        desc="Maximize the exponential moving average of the day-over-day return.",
        obj_func=cp.Maximize(ema_return.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def maximize_return_against_benchmark(data, tickers, benchmark) -> Objective:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    final_return = agg.get_final_point_value(delta)
    objective = Objective(
        name=f"maximize return against benchmark ({benchmark})",
        desc=f"Maximize nominal return minus benchmark return ({benchmark}) on final output date.",
        obj_func=cp.Maximize(final_return.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def maximize_avg_return_against_benchmark(data, tickers, benchmark) -> Objective:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    avg_return = agg.get_historical_avg(delta)
    objective = Objective(
        name=f"maximize average return against benchmark ({benchmark})",
        desc=f"Maximize the historical average (arithmetic mean) of return against benchmark return (nominal return minus benchmark return).",
        obj_func=cp.Maximize(avg_return.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def maximize_ema_return_against_benchmark(
    data, tickers, benchmark, smoothing_window
) -> Objective:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    final_return = agg.get_ema_weighted_avg(delta, smoothing_window)
    objective = Objective(
        name=f"maximize ema return against benchmark ({benchmark})",
        desc=f"Maximize the exponential weighted average of return against benchmark (nominal return minus benchmark return)",
        obj_func=cp.Maximize(final_return.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def maximize_percent_outperforming_days(data, tickers, benchmark) -> Objective:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    percent_days = agg.get_percent_outperforming_days(delta)
    objective = Objective(
        name=f"maximize % outperforming days (return against {benchmark})",
        desc=f"Maximize % days where nominal return outperforms benchmark return ({benchmark}).",
        obj_func=cp.Maximize(percent_days.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def minimize_maximal_loss(data, tickers) -> Objective:
    nominal_return = ts.get_simple_return(data, tickers)
    min_return = agg.get_historical_min(nominal_return)
    objective = Objective(
        name="minimize maximal loss",
        desc="Minimize the greatest amount of loss a portfolio could incur.",
        obj_func=cp.Maximize(min_return.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def minimize_days_with_loss(data, tickers, threshold: float) -> Objective:
    dod_return = ts.get_dod_return(data, tickers)
    days_with_loss = agg.get_percent_underperforming_days(dod_return, -threshold)
    objective = Objective(
        name="minimize % days with loss",
        desc=f"Minimize % of days with single-day loss exceeding {threshold: .2%}.",
        obj_func=cp.Minimize(days_with_loss.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def minimize_maximal_drawdown(data, tickers) -> Objective:
    drawdown = ts.get_drawdown(data, tickers)
    max_drawdown = agg.get_historical_max(drawdown)
    objective = Objective(
        name="minimize maximal drawdown",
        desc="Minimize maximal drop in position.",
        obj_func=cp.Minimize(max_drawdown.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def minimize_avg_drawdown(data, tickers) -> Objective:
    drawdown = ts.get_drawdown(data, tickers)
    avg_drawdown = agg.get_historical_avg(drawdown)
    objective = Objective(
        name="minimize average drawdown",
        desc="Minimize average drawdown (arithmetic mean).",
        obj_func=cp.Minimize(avg_drawdown.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def minimize_ema_drawdown(data, tickers, smoothing_window: int) -> Objective:
    drawdown = ts.get_drawdown(data, tickers)
    ema_drawdown = agg.get_ema_weighted_avg(drawdown, smoothing_window)
    objective = Objective(
        name="minimize ema drawdown",
        desc="Minimize the exponential weighted average of drawdown",
        obj_func=cp.Minimize(ema_drawdown.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def minimize_underperforming_days(data, tickers, benchmark) -> Objective:
    delta = ts.get_return_against_benchmark(data, tickers, benchmark)
    percent_days = agg.get_percent_underperforming_days(delta)
    objective = Objective(
        name=f"minimize % underperforming days (return against {benchmark})",
        desc=f"Minimize % days where nominal return underperforms benchmark return ({benchmark}).",
        obj_func=cp.Minimize(percent_days.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def minimize_ema_deviation(data, tickers, smoothing_window: int) -> Objective:
    nominal_return = ts.get_simple_return(data, tickers)
    ema_deviation = agg.get_ema_deviation(nominal_return, smoothing_window)
    objective = Objective(
        name="minimize ema deviation",
        desc="Minimize the degree to which nominal return deviates from the exponential moving average.",
        obj_func=cp.Minimize(ema_deviation.expr),
    )
    Optimizer.update_objectives(objective)
    return objective


def minimize_classical_volatility(data, tickers) -> Objective:
    covariance_matrix = ts.get_covariance_matrix(data, tickers)
    volatility = agg.get_portfolio_variance(covariance_matrix)
    objective = Objective(
        name="minimize classical volatility",
        desc="Minimize total portfolio variance (by calculating quadratic form of covariance).",
        obj_func=cp.Minimize(volatility.expr),
    )
    Optimizer.update_objectives(objective)
    return objective
