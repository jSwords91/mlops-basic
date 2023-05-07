from typing import Optional, Dict, Any
import pandas as pd
from dataclasses import dataclass
from utils import read_config

@dataclass
class DataIngestion:
    file_path: str

    def load_data(self) -> Optional[pd.DataFrame]:
        try:
            data = pd.read_csv(self.file_path, sep="\t")
            return data
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None

    def run(self) -> Optional[pd.DataFrame]:
        return self.load_data()