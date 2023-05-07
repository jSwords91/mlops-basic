from typing import List, Dict
import pandas as pd
from dataclasses import dataclass
from utils import read_config
from ingestion import DataIngestion

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

@dataclass
class DataCleanConfig:
    date_columns: List[str]
    int_columns: List[str]
    column_name_mapping: Dict[str, str]

class DataClean:
    def __init__(self, dataframe: pd.DataFrame, config: DataCleanConfig):
        self.dataframe = dataframe
        self.cfg = config

    def clean_date_columns(self) -> None:
        for col in self.cfg.date_columns:
            self.dataframe[col] = pd.to_datetime(self.dataframe[col], errors='coerce')

    def clean_int_columns(self) -> None:
        for col in self.cfg.int_columns:
            self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce', downcast='integer')

    def clean_column_names(self) -> None:
        self.dataframe.rename(columns=self.cfg.column_name_mapping, inplace=True)

    def run(self) -> pd.DataFrame:
        self.clean_date_columns()
        self.clean_int_columns()
        self.clean_column_names()
        return self.dataframe
