import numpy as np
import pandas as pd


def get_initial_array(sorted_df: pd.DataFrame) -> np.ndarray:
    return sorted_df.iloc[0, :].to_numpy()


def get_final_array(sorted_df: pd.DataFrame) -> np.ndarray:
    return sorted_df.iloc[-1, :].to_numpy()


def get_percent_change_against_initial_state(sorted_df: pd.DataFrame) -> np.ndarray:
    initial_value = get_initial_array(sorted_df)
    returns = (sorted_df.to_numpy() - initial_value) / initial_value
    return returns


def get_percent_change_over_time(sorted_df: pd.DataFrame) -> np.ndarray:
    return np.diff(sorted_df.to_numpy(), axis=0) / sorted_df.to_numpy()[:-1, :]


def get_covariance_matrix(sorted_df: pd.DataFrame) -> np.ndarray:
    dod_returns = get_percent_change_over_time(sorted_df).T
    return np.cov(dod_returns)


def inflation_rate(cpi_data: pd.DataFrame) -> np.ndarray:
    initial_cpi = get_initial_array(cpi_data)
    inflation = (cpi_data - initial_cpi) / initial_cpi
    return inflation
