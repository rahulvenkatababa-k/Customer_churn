import os
import sys
import json
import pandas as pd
import pandera.pandas as pa

from src.customer_churn.logging.logger import logger
from src.customer_churn.exception.exception import CustomerChurnException

from src.customer_churn.entity.config_entity import DataValidationConfig
from src.customer_churn.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.customer_churn.entity.schema import CustomerChurnSchema

class DataValidation:
    def __init__(self,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config

    def _schema_validation(self, dataframe:pd.DataFrame):
        try:
            schema = CustomerChurnSchema.get_schema()
            schema.validate(dataframe,lazy=True)

            return True, None
        except pa.errors.SchemaErrors as e:
            logger.error("%s", e)
            return False, e.failure_cases
        except Exception as e:
            raise CustomerChurnException(e,sys,"Data Validation","Checking schema_validation") from e

    def _load_data(self) ->pd.DataFrame:
        try:
            data = pd.read_csv(self.data_ingestion_artifact.ingested_data_path)
            logger.info("Ingested Dataframe loaded sucessfully")
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys,"Data Validation","load data") from e

        return data
    def _save_validated_dataframe(self, dataframe:pd.DataFrame, file_name: str) -> pd.DataFrame:
        try:
            os.makedirs(self.data_validation_config.validated_data_dir, exist_ok=True)
            file_path = os.path.join(self.data_validation_config.validated_data_dir, file_name)
            dataframe.to_csv(file_path, index=False)

            logger.info(f"validated dataframe save sucessfully at {file_path}")
            return file_path
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys,"Data Validation", "save validated dataraframe")from e

    def _generate_validation_report(self,
                                    validation_status:bool,
                                    failure_cases = None):
        try:
            report = {
                "validation_status": validation_status,
                "failure_cases" : failure_cases.to_dict(orient = 'records') if failure_cases is not None else []
            }
            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            report_file_path = self.data_validation_config.validation_report

            with open(report_file_path, 'w') as file:
                json.dump(report, file, indent=4, default=str)

                logger.info(f"validation report saved at : {report_file_path}")

            return report_file_path

            
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Data Validation", "generate validation report") from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            dataframe = self._load_data()
            validation_status, failure_cases = self._schema_validation(dataframe=dataframe)
            if validation_status:
                valid_dataframe_file_path = self._save_validated_dataframe(dataframe=dataframe,
                                                                       file_name="validated_customer_churn.csv")
            else:
                valid_dataframe_file_path = ""
            validation_report_file_path = self._generate_validation_report(
                validation_status=validation_status,
                failure_cases=failure_cases
            )
            data_validation_artifact = DataValidationArtifact(validated_status=validation_status,
                                                              validated_data_path=valid_dataframe_file_path,
                                                              validation_report_path=validation_report_file_path)
            return data_validation_artifact

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Data validation", "initiate data validation")from e
        
        