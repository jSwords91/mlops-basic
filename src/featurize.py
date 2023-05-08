import pandas as pd
from dataclasses import dataclass
from utils import read_config
from typing import List

import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class Featurizer:
    dataframe: pd.DataFrame

    def _add_date_features(self) -> pd.DataFrame:
        self.dataframe['year'] = self.dataframe['date'].dt.year
        self.dataframe['month'] = self.dataframe['date'].dt.month
        self.dataframe['quarter'] = self.dataframe['date'].dt.quarter
        return self.dataframe

    def _lagger(self, lag: int) -> pd.DataFrame:
        self.dataframe[f'lag_{lag}'] = self.dataframe["y"].shift(lag)
        return self.dataframe

    def _add_lag_features(self, lag_range: List[int]) -> pd.DataFrame:
        for lag in lag_range:
            self._lagger(lag)
        return self.dataframe

    def _difference(self, diff: int) -> pd.DataFrame:
        self.dataframe[f'diff_{diff}'] = self.dataframe["y"].diff(diff)
        return self.dataframe

    def _add_diff_features(self, diff_range: List[int]) -> pd.DataFrame:
        for diff in diff_range:
            self._difference(diff)
        return self.dataframe

    def _add_rolling_mean(self, periods: int) -> pd.DataFrame:
        self.dataframe[f"rolling_mean_n{periods}"] = self.dataframe["y"].rolling(periods, center=False).mean()
        return self.dataframe

    def _get_classification_label(self) -> pd.DataFrame:
        self.dataframe['target'] = (self.dataframe['y'].diff(periods=-1) < 0).astype(int)
        return self.dataframe

    def run(self) -> pd.DataFrame:
        self._add_date_features()
        self._get_classification_label()
        self.dataframe = self._add_lag_features(lag_range=[1, 2, 3, 4, 5, 5, 7])
        self.dataframe = self._add_diff_features(diff_range=[1, 2, 3, 4, 5, 5, 7])
        self.dataframe = self._add_rolling_mean(periods=7)
        return self.dataframe.fillna(0.0)
