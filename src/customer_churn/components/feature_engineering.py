import os
import sys
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
                          y: pd.Series,) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y,)

            logger.info("Train/test split completed. ""X_train=%s, X_test=%s",X_train.shape,X_test.shape,)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.exception("Failed to split train/test data.")
            raise CustomerChurnException(e, sys, "Feature Engineering", "split train test") from e

    def _create_preprocessor(self,X_train: pd.DataFrame,) -> ColumnTransformer:
        try:
            numerical_columns = self.NUMERICAL_COLUMNS.copy()
            binary_columns = self.BINARY_COLUMNS.copy()

            categorical_columns = [
                column for column in X_train.columns 
                if column not in numerical_columns + binary_columns]

            for column in numerical_columns:
                if not pd.api.types.is_numeric_dtype(X_train[column]):
                    raise TypeError(f"{column} must be numeric before scaling.")

            if not pd.api.types.is_numeric_dtype(X_train["SeniorCitizen"]):
                raise TypeError("SeniorCitizen must be numeric for passthrough.")

            logger.info("Numerical columns: %s", numerical_columns)
            logger.info("Binary columns: %s", binary_columns)
            logger.info("Categorical columns: %s", categorical_columns)

            transformers = [
                (
                    "numerical",
                    StandardScaler(),
                    numerical_columns,
                ),
                (
                    "binary",
                    "passthrough",
                    binary_columns,
                ),
            ]

            if categorical_columns:
                transformers.append(
                    (
                        "categorical",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                        categorical_columns,
                    )
                )

            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                verbose_feature_names_out=False,
            )

            logger.info("Preprocessor created successfully.")

            return self.preprocessor

        except Exception as e:
            logger.exception("Failed to create preprocessor.")
            raise CustomerChurnException(
                e, sys, "Feature Engineering", "create preprocessor"
            ) from e

    def _fit_and_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            if self.preprocessor is None:
                raise ValueError(
                    "Preprocessor has not been created."
                )

            X_train_transformed = self.preprocessor.fit_transform(
                X_train
            )

            X_test_transformed = self.preprocessor.transform(
                X_test
            )

            feature_names = (
                self.preprocessor.get_feature_names_out()
            )

            X_train_df = pd.DataFrame(
                X_train_transformed,
                columns=feature_names,
                index=X_train.index,
            )

            X_test_df = pd.DataFrame(
                X_test_transformed,
                columns=feature_names,
                index=X_test.index,
            )

            logger.info(
                "Transformation completed. "
                "X_train=%s, X_test=%s",
                X_train_df.shape,
                X_test_df.shape,
            )

            return X_train_df, X_test_df

        except Exception as e:
            logger.exception(
                "Failed to fit and transform features."
            )
            raise CustomerChurnException(
                e, sys, "Feature Engineering", "fit and transform"
            ) from e

    def _save_feature_datasets(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        try:
            config = self.feature_engineering_config

            # Use paths defined by FeatureEngineeringConfig.
            output_paths = {
                "X_train": config.X_train_path,
                "X_test": config.X_test_path,
                "y_train": config.y_train_path,
                "y_test": config.y_test_path,
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

        except Exception as e:
            logger.exception("Failed to save feature datasets.")
            raise CustomerChurnException(
                e, sys, "Feature Engineering",
                "save feature datasets"
            ) from e

    def _save_preprocessor(self) -> None:
        try:
            if self.preprocessor is None:
                raise ValueError(
                    "Cannot save an unfitted preprocessor."
                )

            preprocessor_path = (
                self.feature_engineering_config.preprocessor_file_path
            )

            os.makedirs(
                os.path.dirname(
                    os.fspath(preprocessor_path)
                ) or ".",
                exist_ok=True,
            )

            joblib.dump(
                self.preprocessor,
                preprocessor_path,
            )

            logger.info(
                "Preprocessor saved to: %s",
                preprocessor_path,
            )

        except Exception as e:
            logger.exception("Failed to save preprocessor.")
            raise CustomerChurnException(
                e, sys, "Feature Engineering", "save preprocessor"
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

            # 3. Split before fitting the preprocessor.
            X_train, X_test, y_train, y_test = (
                self._split_train_test(X, y)
            )

            # 4. Create explicit preprocessing groups.
            self._create_preprocessor(X_train)

            # 5. Fit on train; transform train and test.
            X_train, X_test = self._fit_and_transform(
                X_train, X_test
            )

            # 6. Save all four datasets to configured paths.
            self._save_feature_datasets(
                X_train,
                X_test,
                y_train,
                y_test,
            )

            # 7. Save the fitted preprocessor.
            self._save_preprocessor()

            config = self.feature_engineering_config

            # 8. Return the artifact using the agreed contract.
            artifact = FeatureEngineeringArtifact(
                X_train_path=config.X_train_path,
                y_train_path=config.y_train_path,
                X_test_path=config.X_test_path,
                y_test_path=config.y_test_path,
                preprocessor_file_path=config.preprocessor_file_path,
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