import pandas as pd
from dataclasses import dataclass
from ingestion import DataIngestion
from process import DataProcess
from utils import read_config

@dataclass
class Featurizer:
    dataframe: pd.DataFrame

    def add_date_features(self) -> pd.DataFrame:
        self.dataframe['year'] = self.dataframe['date'].dt.year
        self.dataframe['month'] = self.dataframe['date'].dt.month
        self.dataframe['quarter'] = self.dataframe['date'].dt.quarter

        return self.dataframe

    def get_classification_label(self) -> pd.DataFrame:
        self.dataframe['target'] = (self.dataframe['y'].diff(periods=-1) < 0).astype(int)

    def drop_original_y(self) -> pd.DataFrame:
        return self.dataframe.drop(columns=["y"])

    def run(self) -> pd.DataFrame:
        self.add_date_features()
        self.get_classification_label()
        #self.drop_original_y()
        return self.dataframe
