import os
import sys
import pandas as pd

from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging.logger import logger

from src.customer_churn.entity.config_entity import DataTransformationConfig
from src.customer_churn.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact

class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig,
                        data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            logger.error("%s",e)
            raise CustomerChurnException(e,sys,"Data transformation", "init fucntion") from e

    def _load_data(self) ->pd.DataFrame:
        try:
            if self.data_validation_artifact.validated_data_path is None:
                raise ValueError("validated file path is None. Data Valdiation Failed")
            data = pd.read_csv(self.data_validation_artifact.validated_data_path)


            return data
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys,"Data transformation", "load data") from e

    def _transform_data(self, dataframe:pd.DataFrame) ->pd.DataFrame:
        try:
            df = dataframe.copy()

            df.TotalCharges = pd.to_numeric(df.TotalCharges, errors= 'coerce')
            null_count = df.isnull().sum()
            logger.info(f"Before handling misisng values count is : \n {null_count} ")

            df.TotalCharges = df.TotalCharges.fillna(0)
            null_count = df.isnull().sum()
            logger.info(f"After handling Misisng Values count is : \n {null_count}")

            return df
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys,"Data transformation", "transform data") from e

    def _save_transformed_data(self, dataframe:pd.DataFrame) -> str:
        try:
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            file_path = self.data_transformation_config.transformed_data_path

            dataframe.to_csv(file_path, index=False)

            logger.info(f"Transformed Data file is saved successfully at : {file_path}")

            return file_path
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys,"Data transformation", "save transformed data") from e
        


    def initiate_data_transformation(self):
        try:
            data = self._load_data()
            df = self._transform_data(dataframe=data)
            transformed_data = self._save_transformed_data(dataframe=df)
            data_transformation_artifact = DataTransformationArtifact(transformed_data_path=transformed_data)

            return data_transformation_artifact
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys,"Data transformation", "initiate data transformation") from e