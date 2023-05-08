import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessorConfig:
    dataframe: pd.DataFrame
    target_column: str

class DataPreprocessor:

    def __init__(self, config: PreprocessorConfig):
        self.cfg = config
        self.numeric_columns = self._get_numeric_columns()

    def _get_numeric_columns(self):
        numeric_columns = self.cfg.dataframe.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns.remove(self.cfg.target_column)
        return numeric_columns

    def build_pipeline(self) -> Any:
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_columns)
            ])

        return preprocessor