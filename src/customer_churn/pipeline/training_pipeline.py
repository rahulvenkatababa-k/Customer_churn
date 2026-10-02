import sys
import time

from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging.logger import logger

from src.customer_churn.entity.config_entity import (TrainingPipelineConfig, 
                                                     DataIngestionConfig, DataValidationConfig,
                                                     DataTransformationConfig, FeatureEngineeringConfig,
                                                     ModelTrainerConfig, ModelEvaluationConfig
                                                     )
from src.customer_churn.entity.artifact_entity import (DataIngestionArtifact,DataValidationArtifact,
                                                       DataTransformationArtifact,FeatureEngineeringArtifact,
                                                       ModelTrainerArtifact, ModelEvaluationArtifact)
from src.customer_churn.components.data_ingestion import DataIngestion
from src.customer_churn.components.data_validation import DataValidation
from src.customer_churn.components.data_transformation import DataTransformation
from src.customer_churn.components.feature_engineering import FeatureEngineering
from src.customer_churn.components.model_trainer import ModelTrainer
from src.customer_churn.components.model_evaluation import ModelEvaluation


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

    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Training Pipeline", "start data transformation") from e


    def start_feature_engineering(self, 
                                  data_transformation_artifact: DataTransformationArtifact) -> FeatureEngineeringArtifact:
        try:
            feature_engineering_config = FeatureEngineeringConfig(training_pipeline_config=self.training_pipeline_config)

            feature_engineering = FeatureEngineering(data_transformation_artifact=data_transformation_artifact, 
                                                     feature_engineering_config=feature_engineering_config,)

            feature_engineering_artifact = (feature_engineering.initiate_feature_engineering())

            return feature_engineering_artifact

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys,"Training Pipeline","start feature engineering",) from e

    def start_model_trainer(self, 
                            feature_engineering_artifact: FeatureEngineeringArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(feature_engineering_artifact=feature_engineering_artifact,
                                         model_trainer_config=model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Training Pipeline", " start model trainer") from e

    def start_model_evaluation(self,
                               model_trainer_artifact:ModelTrainerArtifact,
                               feature_engineering_artifact:FeatureEngineeringArtifact) ->ModelEvaluationArtifact:
        try:
            model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
            model_evaluation = ModelEvaluation(model_evaluation_config=model_evaluation_config,
                                               model_trainer_artifact=model_trainer_artifact,
                                               feature_engineering_artifact=feature_engineering_artifact)

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            return model_evaluation_artifact
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Training Pipeline", "start model evaluation") from e

    def run_pipeline(self):
        try:
            pipeline_start_time = time.perf_counter()

            logger.info("--- Training Pipeline Started ---")

            logger.info("-----------------------------------------------------------------------------")

            stage_start_time = time.perf_counter()

            logger.info(" Starting Data Ingestion ")

            data_ingestion_artifact = self.start_data_ingestion()

            stage_duration = time.perf_counter() - stage_start_time


            logger.info(
                f" Data Ingestion Completed | "
                f"Time: {stage_duration:.2f} seconds | "
                f"Artifact: {data_ingestion_artifact}"
                )

            logger.info("-----------------------------------------------------------------------------")

            stage_start_time = time.perf_counter()
            
            logger.info(" Starting Data Validation ")

            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

            stage_duration = time.perf_counter() - stage_start_time

            logger.info(
                            f" Data Validation Completed | "
                            f"Time: {stage_duration:.2f} seconds | "
                            f"Artifact: {data_validation_artifact}"
                            )
            logger.info("-----------------------------------------------------------------------------")

            stage_start_time = time.perf_counter()
            
            logger.info(" Starting Data Transformation ")

            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)

            stage_duration = time.perf_counter() - stage_start_time


            logger.info(
                f" Data Transformation Completed | "
                f"Time: {stage_duration:.2f} seconds | "
                f"Artifact: {data_transformation_artifact}"
                )
            logger.info("-----------------------------------------------------------------------------")

            stage_start_time = time.perf_counter()
                        
            logger.info(" Starting Feature Engineering ")
            
            feature_engineering_artifact = self.start_feature_engineering(data_transformation_artifact=data_transformation_artifact)

            stage_duration = time.perf_counter() - stage_start_time
            
            
            logger.info(
                f" Feature Engineering Completed | "
                f"Time: {stage_duration:.2f} seconds | "
                f"Artifact: {feature_engineering_artifact}"
                )
            logger.info("-----------------------------------------------------------------------------")

            stage_start_time = time.perf_counter()
                                    
            logger.info(" Starting Model Training ")
            
            model_trainer_artifact = self.start_model_trainer(feature_engineering_artifact=feature_engineering_artifact)

            stage_duration = time.perf_counter() - stage_start_time
            
            
            logger.info(
                f" Model Training Completed | "
                f"Time: {stage_duration:.2f} seconds | "
                f"Artifact: {model_trainer_artifact}"
                )
            logger.info("-----------------------------------------------------------------------------")

            stage_start_time = time.perf_counter()
                                                
            logger.info(" Starting Model Evaluation ")
            
            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,
                                                                 feature_engineering_artifact=feature_engineering_artifact)

            stage_duration = time.perf_counter() - stage_start_time
            
            
            logger.info(
                f" Model Evaluation Completed | "
                f"Time: {stage_duration:.2f} seconds | "
                f"Artifact: {model_evaluation_artifact}"
                )
            logger.info("-----------------------------------------------------------------------------")
            












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