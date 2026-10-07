import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging.logger import logger

from src.customer_churn.entity.artifact_entity import (DataTransformationArtifact,FeatureEngineeringArtifact)
from src.customer_churn.entity.config_entity import FeatureEngineeringConfig


class FeatureEngineering:

    NUMERICAL_COLUMNS = ["tenure","MonthlyCharges","TotalCharges"]

    BINARY_COLUMNS = ["SeniorCitizen"]

    TARGET_COLUMN = "Churn"
    ID_COLUMN = "customerID"

    def __init__(self,data_transformation_artifact: DataTransformationArtifact,
                 feature_engineering_config: FeatureEngineeringConfig,):
        self.data_transformation_artifact = data_transformation_artifact
        self.feature_engineering_config = feature_engineering_config
        self.preprocessor = None

    def _load_data(self) -> pd.DataFrame:
        try:
            data_path = (self.data_transformation_artifact.transformed_data_path)

            if data_path is None:
                raise ValueError("Transformed data file path is None.")

            df = pd.read_csv(data_path)

            logger.info("Transformed dataset loaded successfully. Shape: %s",df.shape,)

            return df

        except Exception as e:
            logger.exception("Failed to load transformed data.")
            raise CustomerChurnException(e, sys, "Feature Engineering", "load data") from e

    def _prepare_features_and_target(self,dataframe: pd.DataFrame,) -> tuple[pd.DataFrame, pd.Series]:
        try:
            df = dataframe.copy()

            required_columns = {
                self.TARGET_COLUMN,
                self.ID_COLUMN,
                *self.NUMERICAL_COLUMNS,
                *self.BINARY_COLUMNS,
            }

            missing_columns = required_columns - set(df.columns)

            if missing_columns:
                raise ValueError(f"Required columns missing: {missing_columns}")

            if df[self.TARGET_COLUMN].isna().any():
                raise ValueError("Target column contains missing values.")

            y = df[self.TARGET_COLUMN].map({"No": 0,"Yes": 1})

            if y.isna().any():
                raise ValueError("Churn contains missing or unexpected values.")

            y = y.astype("int64")

            X = df.drop(columns=[self.TARGET_COLUMN, self.ID_COLUMN])

            logger.info("Features and target separated. X: %s, y: %s",X.shape,y.shape,)

            return X, y

        except Exception as e:
            logger.exception("Failed to prepare features and target.")
            raise CustomerChurnException(e, sys, "Feature Engineering","prepare features and target") from e

    def _split_train_test(self,X: pd.DataFrame,
                          y: pd.Series,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.30,random_state=self.feature_engineering_config.random_state,stratify=y,)

            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=self.feature_engineering_config.random_state, stratify=y_temp)


            logger.info("Train/test split completed. ""X_train=%s,X_val=%s , X_test=%s",X_train.shape,X_val.shape,X_test.shape,)

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.exception("Failed to split train/test data.")
            raise CustomerChurnException(e, sys, "Feature Engineering", "split train test") from e



    def _save_feature_datasets(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        try:
            config = self.feature_engineering_config

            # Use paths defined by FeatureEngineeringConfig.
            output_paths = {
                "X_train": config.X_train_path,
                "X_test": config.X_test_path,
                "y_train": config.y_train_path,
                "y_test": config.y_test_path,
                "X_val": config.X_val_path,
                "y_val": config.y_val_path,
            }

            # Create parent directories for configured paths.
            for path in output_paths.values():
                os.makedirs(
                    os.path.dirname(os.fspath(path)) or ".",
                    exist_ok=True,
                )

            X_train.to_csv(
                output_paths["X_train"], index=False
            )
            X_test.to_csv(
                output_paths["X_test"], index=False
            )

            # Preserve the target as Yes/No for now.
            y_train.to_frame(name=self.TARGET_COLUMN).to_csv(
                output_paths["y_train"], index=False
            )
            y_test.to_frame(name=self.TARGET_COLUMN).to_csv(
                output_paths["y_test"], index=False
            )
            X_val.to_csv(output_paths["X_val"], index=False)

            y_val.to_frame(name=self.TARGET_COLUMN).to_csv(
                output_paths["y_val"], index=False
            )

            logger.info(
                "X_train saved to: %s",
                output_paths["X_train"],
            )
            logger.info(
                "X_test saved to: %s",
                output_paths["X_test"],
            )
            logger.info(
                "y_train saved to: %s",
                output_paths["y_train"],
            )
            logger.info(
                "y_test saved to: %s",
                output_paths["y_test"],
            )
            logger.info(
                "X_val saved to: %s",
                output_paths["X_val"],
            )
            logger.info(
                "y_val saved to: %s",
                output_paths["y_val"],
            )

        except Exception as e:
            logger.exception("Failed to save feature datasets.")
            raise CustomerChurnException(
                e, sys, "Feature Engineering",
                "save feature datasets"
            ) from e

    def initiate_feature_engineering(
        self,
    ) -> FeatureEngineeringArtifact:
        try:
            logger.info("Feature engineering started.")

            # 1. Load cleaned data.
            df = self._load_data()

            # 2. Separate X and y; remove customerID.
            X, y = self._prepare_features_and_target(df)

            
            # 3. Split into train, validation, and test sets.
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            ) = self._split_train_test(X, y)

            # 4. Save the raw feature splits.
            self._save_feature_datasets(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
            )

            config = self.feature_engineering_config

            # 5. Return the updated artifact.
            artifact = FeatureEngineeringArtifact(
                X_train_path=config.X_train_path,
                y_train_path=config.y_train_path,
                X_val_path=config.X_val_path,
                y_val_path=config.y_val_path,
                X_test_path=config.X_test_path,
                y_test_path=config.y_test_path,
            )

            logger.info(
                "Feature engineering completed successfully."
            )

            return artifact

        except Exception as e:
            logger.exception("Feature engineering failed.")
            raise CustomerChurnException(
                e, sys, "Feature Engineering",
                "initiate feature engineering"
            ) from e