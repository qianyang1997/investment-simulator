import re
from datetime import datetime
import json
import pandas as pd
from typing import List
from investment_simulator.config import DATA_DIR
from investment_simulator.data_processor import DataProcessor
from investment_simulator.optimizer_object import Optimizer
from investment_simulator.set_variables import set_weights


class InvestmentSimulator(Optimizer):

    def __init__(
        self,
        tickers: List[str],
        input_date: str = None,
        output_date: str = None,
    ):
        super().__init__()
        set_weights(tickers)
        self.tickers = tickers
        self.input_date = input_date
        self.output_date = output_date
        self.data = self._retrieve_data()
        self.report = self._initialize_report()

    def _initialize_report(self):
        return {
            "status": None,
            "tickers": self.tickers,
            "value": None,
            "objective": None,
            "allocation": {},
            "metrics": {},
            "constraints": {},
        }

    def _retrieve_data(self) -> pd.DataFrame:
        dp = DataProcessor(self.input_date, self.output_date)
        dp.retrieve_price(self.tickers)
        dp.retrieve_cpi()
        dp.refresh_dataframe()
        return dp.data

    def generate_allocation(self):
        allocation = self._optimizer_variables["weights"].var.value
        allocation_dic = {}
        for ticker, allo in zip(self.tickers, allocation):
            if allo > 1e-6:
                allocation_dic[ticker] = allo
        return allocation_dic

    def generate_report(self):
        self.report["status"] = self.problem.status
        self.report["tickers"] = self.tickers
        self.report["value"] = self.problem.value
        self.report["allocation"] = self.generate_allocation()
        for obj in self._optimizer_objectives:
            self.report["objective"] = str(self._optimizer_objectives[obj])
        for met in self._optimizer_metrics:
            if re.match(r".*\(.*\)", self._optimizer_metrics[met].name):
                self.report["metrics"][met] = str(self._optimizer_metrics[met])
        for con in self._optimizer_constraints:
            self.report["constraints"][con] = str(self._optimizer_constraints[con])

    def save_report(self):
        """Save optimized results."""
        with open(f"{DATA_DIR}/output/{datetime.now()}.json", "w") as f:
            f.write(json.dumps(self.report, indent=4))
