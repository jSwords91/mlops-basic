from utils import read_config
from ingestion import DataIngestion
from clean import DataClean, DataCleanConfig
from featurize import Featurizer
from splitter import TimeSeriesSplit, TimeSeriesSplitConfig
from preprocess import PreprocessorConfig, DataPreprocessor
from modelling import ModelFitConfig, ModelFit, ModelEvaluateConfig, ModelEvaluate

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
    timeseries_split_config = TimeSeriesSplitConfig(**config["train_test_split"])
    train_data, dev_data, test_data = TimeSeriesSplit(featurized_data, timeseries_split_config).split()

    preprocessor_config = PreprocessorConfig(dataframe=train_data, target_column=config["target"]["variable_name"])
    data_preprocessor = DataPreprocessor(config=preprocessor_config)
    preprocessor = data_preprocessor.build_pipeline()

    model_fit_config = ModelFitConfig(
        train_data=train_data,
        dev_data=dev_data,
        target_column=config["target"]["variable_name"],
        preprocessor=preprocessor,
        random_forest_params=config["random_forest_parameters"]
    )

    model_fit = ModelFit(model_fit_config)
    model_fit.fit()

    evaluate_config = ModelEvaluateConfig(
        model=model_fit.model,
        train_data=train_data,
        dev_data=dev_data,
        target_column=config["target"]["variable_name"],
        preprocessor=preprocessor
    )

    model_evaluate = ModelEvaluate(config=evaluate_config)
    train_accuracy, dev_accuracy = model_evaluate.evaluate()


    print(f"Train accuracy: {train_accuracy}")
    print(f"Dev accuracy: {dev_accuracy}")

    test_data_with_predictions, test_accuracy = model_evaluate.predict_and_evaluate(test_data)
    print(test_data_with_predictions)
    print(f"Test accuracy: {test_accuracy}")

