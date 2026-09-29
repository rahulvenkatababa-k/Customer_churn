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

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_name, 
                                               training_pipeline.DATA_INGESTION_DIR_NAME)

        self.ingested_data_dir = os.path.join(self.data_ingestion_dir, 
                                              training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME)