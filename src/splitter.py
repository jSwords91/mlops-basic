import pandas as pd
from dataclasses import dataclass

from utils import read_config



@dataclass
class TimeSeriesSplitConfig:
    train_ratio: float
    dev_ratio: float
    test_ratio: float

class TimeSeriesSplit:
    def __init__(self, dataframe: pd.DataFrame, config: TimeSeriesSplitConfig):
        self.dataframe = dataframe
        self.cfg = config

    def split(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        self.dataframe = self.dataframe.sort_values(by="date").drop(columns=["date"])
        train_split_index = int(self.cfg.train_ratio * len(self.dataframe))
        dev_split_index = train_split_index + int(self.cfg.dev_ratio * len(self.dataframe))

        train_data = self.dataframe.iloc[:train_split_index]
        dev_data = self.dataframe.iloc[train_split_index:dev_split_index]
        test_data = self.dataframe.iloc[dev_split_index:]
        print(len(self.dataframe))
        print(len(train_data) + len(dev_data) + len(dev_data))

        assert len(self.dataframe) == len(train_data) + len(dev_data) + len(test_data), "Dataframes mismatch"

        return train_data, dev_data, test_data
