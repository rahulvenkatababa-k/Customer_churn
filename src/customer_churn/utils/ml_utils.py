import os
import mlflow
import truststore

truststore.inject_into_ssl()
from src.customer_churn.configuration.configuration import config

def configure_mlflow() -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

def log_parameters(parameters:dict) ->None:
    if parameters:
        mlflow.log_params(params=parameters)

def log_metrics(metrics:dict) ->None :
    numeric_metrics = {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)

def log_pipeline_duration(duration_seconds:float) -> None:
    mlflow.log_metric("Pipeline duration" , duration_seconds)