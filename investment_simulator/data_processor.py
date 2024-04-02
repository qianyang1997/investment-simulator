from functools import reduce
import json
import pandas as pd
from typing import List

from investment_simulator.config import DATA_DIR, set_logger
from investment_simulator.data_retriever import DataRetriever


logger = set_logger("DATA PROCESSOR")


class DataProcessor:

    def __init__(self, min_date, max_date):
        self.data_elements = {}
        self.data = pd.DataFrame()
        self.min_date = min_date
        self.max_date = max_date
        self.data_retriever = DataRetriever()

    def _resample(self, data: pd.DataFrame, frequency: str = "D"):
        """Resample the data to desired frequency."""
        resampled_data = data.copy()
        resampled_data.index = pd.to_datetime(resampled_data.index)
        resampled_data = resampled_data.sort_index()
        resampled_data = resampled_data.resample(frequency).ffill()
        resampled_data.index = resampled_data.index.strftime("%Y-%m-%d")

        return resampled_data

    def retrieve_price(self, tickers: List[str]):
        prices = []
        for ticker in tickers:
            # retrieve raw json data
            try:
                with open(f"{DATA_DIR}/{ticker}.json", "r") as f:
                    price = json.loads(f.read())["Time Series (Daily)"]
            except FileNotFoundError:
                price = self.data_retriever.get_price_data(ticker)
            # reformat into dataframe
            price = pd.DataFrame.from_dict(price, orient="index")
            price = price[["4. close"]].rename(columns={"4. close": ticker})
            prices.append(price)
        # join all tickers data
        prices = reduce(
            lambda x, y: x.merge(y, how="inner", left_index=True, right_index=True),
            prices,
        )
        # append to data element
        self.data_elements["price"] = prices

    def retrieve_cpi(self):
        # retrieve raw cpi data
        try:
            with open(f"{DATA_DIR}/CPI.json", "r") as f:
                cpi = json.loads(f.read())["data"]
        except FileNotFoundError:
            cpi = self.data_retriever.get_cpi_data()
        # reformat into dataframe
        cpi = pd.DataFrame.from_records(cpi).set_index("date")
        cpi = cpi.rename(columns={"value": "CPI"})
        # resample from monthly to daily
        cpi = self._resample(cpi, frequency="D")
        # append to data element
        self.data_elements["CPI"] = cpi

    def refresh_dataframe(self):
        self.data = reduce(
            lambda x, y: x.merge(y, how="inner", left_index=True, right_index=True),
            list(self.data_elements.values()),
        )
        if self.min_date:
            self.data = self.data[self.data.index >= self.min_date]
        if self.max_date:
            self.data = self.data[self.data.index <= self.max_date]
        self.data = self.data.sort_index().astype(float)
        print(
            f"Historical data range: {self.data.iloc[0].name} - {self.data.iloc[-1].name}"
        )  # TODO
