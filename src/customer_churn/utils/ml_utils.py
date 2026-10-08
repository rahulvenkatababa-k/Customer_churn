import os
import mlflow
import truststore
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

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
        input_example=X_sample,
        skops_trusted_types = [
            "catboost.core.CatBoostClassifier",
            "imblearn.pipeline.Pipeline",
        ]
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


def promote_model_to_alias(
    model_uri: str,
    model_name: str,
    alias: str,
) -> str:
    """
    Promote an already-registered model version to an MLflow alias.

    Args:
        model_uri: Exact model URI returned by register_model().
        model_name: Registered MLflow model name.
        alias: Alias to assign.

    Returns:
        The promoted model version.
    """

    client = MlflowClient()

    model_versions = client.search_model_versions(
        filter_string=f"name='{model_name}'"
    )

    matching_versions = [
        version
        for version in model_versions
        if version.source == model_uri
    ]

    if not matching_versions:
        raise ValueError(
            f"No registered model version found for "
            f"model '{model_name}' with URI '{model_uri}'."
        )

    if len(matching_versions) > 1:
        raise ValueError(
            f"Multiple registered model versions found for "
            f"model '{model_name}' with URI '{model_uri}'."
        )

    model_version = matching_versions[0].version

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version,
    )

    logger.info(
        "Promoted model '%s' version '%s' to alias '%s'",
        model_name,
        model_version,
        alias,
    )

    return str(model_version)