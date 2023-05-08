# mlops-basic
A basic example of MLOps implementation using the air passengers dataset.

This project demonstrates how to build a time series classification solution using machine learning, while following software engineering and MLOps best practices. The goal is to provide a high-level guide on how to structure a time series classification project and ensure that the code is maintainable, scalable, and easy to understand. In this project, we aim to classify whether the number of passengers in the next month will increase or not.

This is not an exhaustive attempt at producing the best model, rather a resource to help you get started structuring ML projects.

Steps you'd also want to incorporate in a more full solution: Orchestration, Docker deployment, Monitoring etc.

## Overview
The project consists of several steps:

* Data ingestion
* Data cleaning
* Feature engineering
* Splitting the data into train, dev, and test sets
* Preprocessing the data
* Model training
* Model evaluation

Each of these steps has its own dedicated module, ensuring separation of concerns and making it easier to maintain and extend the code.

## Prerequisites
To run this project, you need to have Python 3.6 or later installed, along with the following libraries:

```python
pandas
numpy
scikit-learn
```

## Usage
Clone the repository and navigate to the project directory.

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Configure the config/config.yaml file with the desired parameters, such as the file path for data ingestion, column names for data processing, train-test split ratios, and random forest hyperparameters.

Run the main script:

```bash
python main.py
```

The script will perform data ingestion, cleaning, feature engineering, and preprocessing before training a Random Forest classifier. It will then evaluate the model's accuracy on the train and dev sets and print the results.
