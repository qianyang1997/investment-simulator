import json
import requests
from investment_simulator.config import CREDS, DATA_DIR


class DataRetriever:

    def __init__(self):
        self.api_key = CREDS["API_KEY"]

    def _get_url(self, **kwargs):
        query = []
        for k, v in kwargs.items():
            if v:
                query.append(f"{k}={v}")
        query = "&".join(query)
        url = f"https://www.alphavantage.co/query?{query}"
        return url

    def _get_data(self, **kwargs):
        url = self._get_url(**kwargs)
        r = requests.get(url)
        data = r.json()
        return json.dumps(data)

    def _write_data(self, data, symbol):
        with open(f"{DATA_DIR}/{symbol}.json", "w") as f:
            f.write(data)

    def get_price_data(self, ticker):
        data = self._get_data(
            function="TIME_SERIES_DAILY",
            symbol=ticker,
            apikey=self.api_key,
            outputsize="full",
            datatype="json",
        )
        self._write_data(data, ticker)

    def get_cpi_data(self):
        data = self._get_data(
            function="CPI", apikey=self.api_key, outputsize="full", datatype="json"
        )
        self._write_data(data, "CPI")

    def get_inflation_data(self):
        data = self._get_data(
            function="INFLATION",
            apikey=self.api_key,
            outputsize="full",
            datatype="json",
        )
        self._write_data(data, "INFLATION")
        return data
