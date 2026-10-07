import os
import mlflow
import truststore
from mlflow.models import infer_signature

truststore.inject_into_ssl()
from src.customer_churn.configuration.configuration import config
from src.customer_churn.logging.logger import logger

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

def register_model(
        model,
        model_name: str,
        X_sample,
) -> str:
    """Log a trained model to MLFLOW and register it in the model registry
        Args:
            model: The already-trained, selected model.
            model_name: Stable name for the registered model.
            X_sample: Representative input rows for the model signature.

        Returns:
            The MLflow model URI.
        """

    X_sample = X_sample.copy()
    integer_columns = X_sample.select_dtypes(include=['int64']).columns
    X_sample[integer_columns] = X_sample[integer_columns].astype("float64")
    signature = infer_signature(X_sample, model.predict(X_sample))
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name= f"{model_name}_artifact",
        registered_model_name= model_name,
        signature= signature,
        input_example=X_sample
    )

    logger.info(
        "Registered model '%s' with URI '%s'",
        model_name,
        model_info.model_uri,
    )

    return model_info.model_uri
    

def load_model(model_uri: str):
    configure_mlflow()
    return mlflow.sklearn.load_model(model_uri)