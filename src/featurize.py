import pandas as pd
from dataclasses import dataclass
from utils import read_config


class Featurizer:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def _add_date_features(self) -> pd.DataFrame:
        self.dataframe['year'] = self.dataframe['date'].dt.year
        self.dataframe['month'] = self.dataframe['date'].dt.month
        self.dataframe['quarter'] = self.dataframe['date'].dt.quarter

        return self.dataframe

    def _get_classification_label(self) -> pd.DataFrame:
        self.dataframe['target'] = (self.dataframe['y'].diff(periods=-1) < 0).astype(int)

    def _drop_original_unused(self) -> pd.DataFrame:
        return self.dataframe.drop(columns=["y"])

    def run(self) -> pd.DataFrame:
        self._add_date_features()
        self._get_classification_label()
        #self.dataframe = self._drop_original_unused()
        return self.dataframe