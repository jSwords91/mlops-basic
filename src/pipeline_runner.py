import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional

from utils import read_config
from ingestion import DataIngestion
from process import DataProcess
from featurize import Featurizer
from splitter import TimeSeriesSplit

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


@dataclass
class Model:
    preprocessor: Any
    logreg_model: Any

    def build_pipeline(self) -> Any:
        return Pipeline(steps=[('preprocessor', self.preprocessor),
                               ('logreg', self.logreg_model)])

@dataclass
class Preprocessor:
    numeric_columns: Any
    categorical_columns: Any

    def build_pipeline(self) -> Any:
        # Define the preprocessing steps for numeric and categorical columns
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

        # Combine the preprocessing steps for numeric and categorical columns using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_columns),
                ('cat', categorical_transformer, self.categorical_columns)
            ])

        return preprocessor

@dataclass
class LogisticRegressionModel:
    solver: str
    C: float

    def build_model(self) -> Any:
        return LogisticRegression(solver=self.solver, C=self.C)

@dataclass
class ModelEvaluator:
    pipeline: Any
    train_data: Any
    dev_data: Any

    def evaluate_model(self) -> Any:
        # Train the pipeline on the training data
        self.pipeline.fit(self.train_data.drop(columns=['target']), self.train_data['target'])

        # Evaluate the pipeline on the training data
        train_predictions = self.pipeline.predict(self.train_data.drop(columns=['target']))
        train_accuracy = accuracy_score(self.train_data['target'], train_predictions)

        # Evaluate the pipeline on the dev data
        dev_predictions = self.pipeline.predict(self.dev_data.drop(columns=['target']))
        dev_accuracy = accuracy_score(self.dev_data['target'], dev_predictions)

        return train_accuracy, dev_accuracy



if __name__ == "__main__":
    config_file = "config/config.yaml"
    config = read_config(config_file)

    data = DataIngestion(**config["data_ingestion"]).run()

    date_columns = config["data_processing"]["date_columns"]
    int_columns = config["data_processing"]["int_columns"]
    column_name_mapping = config["data_processing"]["column_name_mapping"]

    processed_data = DataProcess(data, date_columns, int_columns, column_name_mapping).run()

    featurized_data = Featurizer(processed_data).run()

    train_data, dev_data, test_data = TimeSeriesSplit(featurized_data, **config["train_test_split"]).split()
    print(train_data)

    preprocessor_pipeline = Preprocessor(**config["preprocessing"]).build_pipeline()

    logreg = LogisticRegressionModel(**config["logistic_regression"]).build_model()

    pipeline = Model(preprocessor_pipeline, logreg).build_pipeline()

    evaluator = ModelEvaluator(pipeline, train_data, dev_data)

    print(evaluator.evaluate_model())