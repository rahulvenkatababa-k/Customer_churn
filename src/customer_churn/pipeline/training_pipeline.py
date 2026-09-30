import sys
import time

from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging.logger import logger

from src.customer_churn.entity.config_entity import (TrainingPipelineConfig, 
                                                     DataIngestionConfig, DataValidationConfig)
from src.customer_churn.entity.artifact_entity import (DataIngestionArtifact,DataValidationArtifact)
from src.customer_churn.components.data_ingestion import DataIngestion
from src.customer_churn.components.data_validation import DataValidation


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            return data_ingestion_artifact

        except Exception as e:
            logger.info("%s", e)
            raise CustomerChurnException(e,sys, "Training Pipeline", "start data ingestion") from e

    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact) ->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config=data_validation_config,
                                             data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Training Pipeline", "start data validation") from e

    def run_pipeline(self):
        try:
            pipeline_start_time = time.perf_counter()

            logger.info("--- Training Pipeline Started ---")

            stage_start_time = time.perf_counter()

            logger.info(" Starting Data Ingestion ")

            data_ingestion_artifact = self.start_data_ingestion()

            stage_duration = time.perf_counter() - stage_start_time


            logger.info(
                f" Data Ingestion Completed | "
                f"Time: {stage_duration:.2f} seconds | "
                f"Artifact: {data_ingestion_artifact}"
                )

            stage_start_time = time.perf_counter()
            
            logger.info(" Starting Data Validation ")

            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

            stage_duration = time.perf_counter() - stage_start_time

            logger.info(
                            f" Data Validation Completed | "
                            f"Time: {stage_duration:.2f} seconds | "
                            f"Artifact: {data_validation_artifact}"
                            )

            total_duration = time.perf_counter() - pipeline_start_time

            logger.info(
                "========== TRAINING PIPELINE COMPLETED =========="
            )

            logger.info(
                f"Total pipeline execution time: "
                f"{total_duration:.2f} seconds"
            )
        except Exception as e:
        
            raise CustomerChurnException(e,sys, "Training Pipeline", "run pipeline") from e