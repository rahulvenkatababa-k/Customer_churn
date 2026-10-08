import sys

from prefect import flow, task

from src.customer_churn.pipeline.training_pipeline import TrainingPipeline
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging.logger import logger


@task(
    name="Data Ingestion",
    retries=3,
    retry_delay_seconds=30,
    timeout_seconds=120
)
def data_ingestion(pipeline: TrainingPipeline):
    return pipeline.start_data_ingestion()


@task(
    name="Data Validation",
    timeout_seconds=60
)
def data_validation(
    pipeline: TrainingPipeline,
    data_ingestion_artifact
):
    return pipeline.start_data_validation(
        data_ingestion_artifact=data_ingestion_artifact
    )


@task(
    name="Data Transformation",
    timeout_seconds=60
)
def data_transformation(
    pipeline: TrainingPipeline,
    data_validation_artifact
):
    return pipeline.start_data_transformation(
        data_validation_artifact=data_validation_artifact
    )


@task(
    name="Feature Engineering",
    timeout_seconds=60
)
def feature_engineering(
    pipeline: TrainingPipeline,
    data_transformation_artifact
):
    return pipeline.start_feature_engineering(
        data_transformation_artifact=data_transformation_artifact
    )


@task(
    name="Model Training",
    timeout_seconds=900
)
def model_training(
    pipeline: TrainingPipeline,
    feature_engineering_artifact
):
    return pipeline.start_model_trainer(
        feature_engineering_artifact=feature_engineering_artifact
    )


@task(
    name="Model Evaluation",
    timeout_seconds=180
)
def model_evaluation(
    pipeline: TrainingPipeline,
    feature_engineering_artifact,
    model_trainer_artifact
):
    return pipeline.start_model_evaluation(
        feature_engineering_artifact=feature_engineering_artifact,
        model_trainer_artifact=model_trainer_artifact
    )


@task(
    name="Model Pusher",
    retries=3,
    retry_delay_seconds=30,
    timeout_seconds=60
)
def model_pusher(
    pipeline: TrainingPipeline,
    model_evaluation_artifact
):
    return pipeline.start_model_pusher(
        model_evaluation_artifact=model_evaluation_artifact
    )


@flow(name="customer-churn-training-pipeline")
def customer_churn_training_flow():

    pipeline = TrainingPipeline()

    try:

        # 1. Data Ingestion
        data_ingestion_artifact = data_ingestion(
            pipeline
        )

        # 2. Data Validation
        data_validation_artifact = data_validation(
            pipeline,
            data_ingestion_artifact
        )

        # 3. Data Transformation
        data_transformation_artifact = data_transformation(
            pipeline,
            data_validation_artifact
        )

        # 4. Feature Engineering
        feature_engineering_artifact = feature_engineering(
            pipeline,
            data_transformation_artifact
        )

        # 5. Model Training
        model_trainer_artifact = model_training(
            pipeline,
            feature_engineering_artifact
        )

        # 6. Model Evaluation
        model_evaluation_artifact = model_evaluation(
            pipeline,
            feature_engineering_artifact,
            model_trainer_artifact
        )

        # 7. Model Pusher
        model_pusher(
            pipeline,
            model_evaluation_artifact
        )

    except Exception as e:
        logger.error(
            f"Pipeline Execution failed: "
            f"{type(e).__name__}: {e}"
        )
        raise CustomerChurnException(
            e,
            sys,
            "Prefect Training Pipeline"
        )


if __name__ == "__main__":
    customer_churn_training_flow()