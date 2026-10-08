import os
from src.customer_churn.constants import training_pipeline
from datetime import datetime

class TrainingPipelineConfig:
    def __init__(self):
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        self.artifact_dir = training_pipeline.ARTIFACTS_DIR_NAME
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = os.path.join(self.artifact_dir, timestamp)

        self.timestamp = timestamp
        self.random_state = training_pipeline.RANDOM_STATE

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_name, 
                                               training_pipeline.DATA_INGESTION_DIR_NAME)

        self.ingested_data_dir = os.path.join(self.data_ingestion_dir, 
                                              training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME)
        self.versioned_data_dir = training_pipeline.DATA_VERSIONED_DIR
        self.versioned_data_path = os.path.join(
            self.versioned_data_dir,
            training_pipeline.DATA_VERSIONED_DATASET_FILE_NAME
        )

class DataValidationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_name,
                                                     training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.validated_data_dir : str = os.path.join(self.data_validation_dir,
                                                     training_pipeline.DATA_VALIDATION_VALID_DIR_NAME)
        self.validation_report : str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_REPORT_FILE_NAME)
        self.validated_data_path : str = os.path.join(self.validated_data_dir, training_pipeline.DATA_VALIDATION_VALIDATED_DATAFRAME_NAME)
        
class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_name,
                                                    training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_data_dir = os.path.join(self.data_transformation_dir,
                                                 training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME)
        self.transformed_data_path  = os.path.join( self.transformed_data_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATAFRAME_NAME)

class FeatureEngineeringConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.feature_engineering_dir = os.path.join(training_pipeline_config.artifact_name,
                                                    training_pipeline.FEATURE_ENGINEERING_DIR_NAME)
        self.engineered_data_dir = os.path.join(self.feature_engineering_dir,
                                                training_pipeline.FEATURE_ENGINEERING_ENGINEERED_DIR_NAME)
        self.X_train_path = os.path.join(self.engineered_data_dir,
                                              training_pipeline.FEATURE_ENGINEERING_X_TRAIN_FILE_NAME)
        self.y_train_path = os.path.join(self.engineered_data_dir,
                                                      training_pipeline.FEATURE_ENGINEERING_Y_TRAIN_FILE_NAME)
        self.X_test_path = os.path.join(self.engineered_data_dir,
                                                      training_pipeline.FEATURE_ENGINEERING_X_TEST_FILE_NAME)
        self.y_test_path = os.path.join(self.engineered_data_dir,
                                                      training_pipeline.FEATURE_ENGINEERING_Y_TEST_FILE_NAME)
        self.X_val_path = os.path.join(self.engineered_data_dir,
                                       training_pipeline.FEATURE_ENGINEERING_X_VAL_FILE_NAME)
        self.y_val_path = os.path.join(self.engineered_data_dir,
                                       training_pipeline.FEATURE_ENGINEERING_Y_VAL_FILE_NAME)
        self.preprocessor_file_path = os.path.join(self.engineered_data_dir,
                                                   training_pipeline.FEATURE_ENGINEERING_PREPROCESSOR_FILE_NAME)
        self.random_state = training_pipeline.RANDOM_STATE

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_name,
                                              training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.trained_model_dir = os.path.join(self.model_trainer_dir,
                                              training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR_NAME)
        self.model_path = os.path.join(self.trained_model_dir,
                                       training_pipeline.MODEL_TRAINER_FILE_NAME)
        self.random_state = training_pipeline_config.random_state

class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.model_evaluation_dir = os.path.join(training_pipeline_config.artifact_name,
                                                 training_pipeline.MODEL_EVALUATION_DIR_NAME)
        self.model_evaluation_report_path = os.path.join(self.model_evaluation_dir,
                                                         training_pipeline.MODEL_EVALUATION_REPORT_FILE_NAME)
        self.model_evaluation_confusion_matrix_dir = os.path.join(self.model_evaluation_dir,
                                                                   training_pipeline.MODEL_EVALUATION_CONFUSION_MATRIX_DIR_NAME)
        
class ModelPusherConfig:
    def __init__(self):
        self.model_name = training_pipeline.MODEL_PUSHER_MODEL_NAME
        self.model_alias = training_pipeline.MODEL_PUSHER_MODEL_ALIAS