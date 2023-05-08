import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@dataclass
class ModelFitConfig:
    train_data: pd.DataFrame
    dev_data: pd.DataFrame
    target_column: str
    preprocessor: Any
    random_forest_params: dict

class ModelFit:
    def __init__(self, config: ModelFitConfig):
        self.cfg = config
        self.model = RandomForestClassifier(**self.cfg.random_forest_params)
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

    def _predict(self, X: pd.DataFrame, index: pd.Index) -> pd.Series:
        predictions = self.cfg.model.predict(X)
        return pd.Series(predictions, index=index)

    def predict_and_evaluate(self, data: pd.DataFrame) -> (pd.DataFrame, float):
        X = self.cfg.preprocessor.transform(data.drop(columns=[self.cfg.target_column]))
        y_true = data[self.cfg.target_column]

        y_pred = self._predict(X, index=data.index)
        data_with_predictions = data.assign(prediction=y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        return data_with_predictions, accuracy

    def evaluate(self) -> (float, float):
        _, train_accuracy = self.predict_and_evaluate(self.cfg.train_data)
        _, dev_accuracy = self.predict_and_evaluate(self.cfg.dev_data)
        return train_accuracy, dev_accuracy



class xModelEvaluate:

    def __init__(self, config: ModelEvaluateConfig):
        self.cfg = config
        self.train_data = self.cfg.preprocessor.transform(self.cfg.train_data.drop(columns=[self.cfg.target_column]))
        self.dev_data = self.cfg.preprocessor.transform(self.cfg.dev_data.drop(columns=[self.cfg.target_column]))
        print(self.train_data)

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        return self.cfg.model.predict(X)

    def evaluate(self) -> (float, float):
        train_accuracy = self._evaluate(self.train_data, self.cfg.train_data[self.cfg.target_column])
        dev_accuracy = self._evaluate(self.dev_data, self.cfg.dev_data[self.cfg.target_column])
        return train_accuracy, dev_accuracy

    def _evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        y_pred = self._predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
