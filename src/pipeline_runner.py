import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from utils import read_config
from ingestion import DataIngestion
from clean import DataClean, DataCleanConfig
from featurize import Featurizer
from splitter import TimeSeriesSplit, TimeSeriesSplitConfig

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


@dataclass
class PreprocessorConfig:
    dataframe: pd.DataFrame
    target_column: str

class DataPreprocessor:

    def __init__(self, config: PreprocessorConfig):
        self.cfg = config
        self.numeric_columns = self._get_numeric_columns()

    def _get_numeric_columns(self):
        numeric_columns = self.cfg.dataframe.select_dtypes(include=[pd.np.number]).columns.tolist()
        numeric_columns.remove(self.cfg.target_column)
        return numeric_columns

    def build_pipeline(self) -> Any:
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_columns)
            ])

        return preprocessor

@dataclass
class ModelFitConfig:
    train_data: pd.DataFrame
    dev_data: pd.DataFrame
    target_column: str
    preprocessor: Any

class ModelFit:

    def __init__(self, config: ModelFitConfig):
        self.cfg = config
        self.model = LogisticRegression()
        self.train_data = self.cfg.preprocessor.fit_transform(self.cfg.train_data.drop(columns=[self.cfg.target_column]))
        self.dev_data = self.cfg.preprocessor.transform(self.cfg.dev_data.drop(columns=[self.cfg.target_column]))

    def fit(self) -> None:
        y_train = self.cfg.train_data[self.cfg.target_column]
        self.model.fit(self.train_data, y_train)

@dataclass
class ModelEvaluateConfig:
    model: Any
    train_data: pd.DataFrame
    dev_data: pd.DataFrame
    target_column: str
    preprocessor: Any

class ModelEvaluate:

    def __init__(self, config: ModelEvaluateConfig):
        self.cfg = config
        self.train_data = self.cfg.preprocessor.transform(self.cfg.train_data.drop(columns=[self.cfg.target_column]))
        self.dev_data = self.cfg.preprocessor.transform(self.cfg.dev_data.drop(columns=[self.cfg.target_column]))

    def evaluate(self) -> (float, float):
        train_accuracy = self._evaluate(self.train_data, self.cfg.train_data[self.cfg.target_column])
        dev_accuracy = self._evaluate(self.dev_data, self.cfg.dev_data[self.cfg.target_column])
        return train_accuracy, dev_accuracy

    def _evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        y_pred = self.cfg.model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

if __name__ == "__main__":
    config_file = "config/config.yaml"
    config = read_config(config_file)

    data = DataIngestion(**config["data_ingestion"]).run()

    date_columns = config["data_processing"]["date_columns"]
    int_columns = config["data_processing"]["int_columns"]
    column_name_mapping = config["data_processing"]["column_name_mapping"]


    clean_config = DataCleanConfig(date_columns, int_columns, column_name_mapping)
    cleaned_data = DataClean(data, clean_config).run()

    featurized_data = Featurizer(cleaned_data).run()
    print(featurized_data)
    timeseries_split_config = TimeSeriesSplitConfig(**config["train_test_split"])
    train_data, dev_data, test_data = TimeSeriesSplit(featurized_data, timeseries_split_config).split()

    print(train_data)

    preprocessor_config = PreprocessorConfig(dataframe=train_data, target_column=config["target"]["variable_name"])
    data_preprocessor = DataPreprocessor(config=preprocessor_config)
    preprocessor = data_preprocessor.build_pipeline()

    fit_config = ModelFitConfig(train_data=train_data, dev_data=dev_data, target_column=config["target"]["variable_name"], preprocessor=preprocessor)
    model_fit = ModelFit(config=fit_config)

    model_fit.fit()

    evaluate_config = ModelEvaluateConfig(model=model_fit.model, train_data=train_data, dev_data=dev_data, target_column=config["target"]["variable_name"], preprocessor=preprocessor)
    model_evaluate = ModelEvaluate(config=evaluate_config)
    train_accuracy, dev_accuracy = model_evaluate.evaluate()

    print(f"Train accuracy: {train_accuracy}")
    print(f"Dev accuracy: {dev_accuracy}")