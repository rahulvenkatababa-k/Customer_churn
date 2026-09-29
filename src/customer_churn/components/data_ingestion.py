import os
import re
import sys

import pandas as pd
from sqlalchemy import create_engine, text

from src.customer_churn.logging.logger import logger
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.entity.artifact_entity import DataIngestionArtifact
from src.customer_churn.entity.config_entity import DataIngestionConfig
from src.customer_churn.configuration.configuration import config


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CustomerChurnException(
                e,
                sys,
                "DataIngestion",
                "Initializing data ingestion"
            ) from e

    def _get_engine(self):
        try:
            return create_engine(config.postgres_sql_uri)

        except Exception as e:
            raise CustomerChurnException(
                e,
                sys,
                "DataIngestion",
                "Creating PostgreSQL engine"
            ) from e

    def _get_latest_table(self, engine):
        try:
            with engine.connect() as connection:
                result = connection.execute(
                    text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_type = 'BASE TABLE'
                          AND table_name ~ '^customer_churn_v[0-9]+$'
                        """
                    )
                )

                table_names = result.scalars().all()

            if not table_names:
                raise ValueError(
                    "No versioned customer churn tables found "
                    "in the public schema."
                )

            versioned_tables = []

            for table_name in table_names:
                match = re.fullmatch(
                    r"customer_churn_v([0-9]+)",
                    table_name
                )

                if match:
                    versioned_tables.append(
                        (int(match.group(1)), table_name)
                    )

            if not versioned_tables:
                raise ValueError(
                    "No valid customer_churn_vN tables were discovered."
                )

            dataset_version, latest_table = max(
                versioned_tables,
                key=lambda item: item[0]
            )

            logger.info(
                "Latest dataset table discovered: %s (version %s)",
                latest_table,
                dataset_version
            )

            return latest_table, dataset_version

        except Exception as e:
            raise CustomerChurnException(
                e,
                sys,
                "DataIngestion",
                "Discovering latest dataset table"
            ) from e


    def _read_data(self, engine, table_name):
        try:
            quoted_table = '"' + table_name.replace('"', '""') + '"'

            query = text(
                f"SELECT * FROM public.{quoted_table}"
            )

            df = pd.read_sql(query, engine)

            return df

        except Exception as e:
            raise CustomerChurnException(
                e,
                sys,
                "DataIngestion",
                f"Reading data from {table_name}"
            ) from e

    def _validate_dataframe(self, df, table_name):
        try:
            if df.empty:
                raise ValueError(
                    f"Table '{table_name}' contains no records."
                )

            if len(df.columns) == 0:
                raise ValueError(
                    f"Table '{table_name}' contains no columns."
                )

            if not df.columns.is_unique:
                raise ValueError(
                    f"Table '{table_name}' has duplicate column names."
                )

            logger.info(
                "Successfully read %s rows and %s columns from %s",
                len(df),
                len(df.columns),
                table_name
            )

        except Exception as e:
            raise CustomerChurnException(
                e,
                sys,
                "DataIngestion",
                f"Validating dataframe from {table_name}"
            ) from e

    def load_data(self) -> tuple[pd.DataFrame, int]:
        engine = None

        try:
            engine = self._get_engine()

            latest_table, dataset_version = self._get_latest_table(
                engine
            )

            df = self._read_data(
                engine,
                latest_table
            )

            self._validate_dataframe(
                df,
                latest_table
            )

            return df, dataset_version

        except CustomerChurnException:
            raise

        except Exception as e:
            logger.exception(
                "Failed to load data from PostgreSQL"
            )

            raise CustomerChurnException(
                e,
                sys,
                "DataIngestion",
                "Loading data from PostgreSQL"
            ) from e

        finally:
            if engine is not None:
                engine.dispose()

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            df, dataset_version = self.load_data()

            os.makedirs(
                self.data_ingestion_config.ingested_data_dir,
                exist_ok=True
            )

            ingested_data_path = os.path.join(
                self.data_ingestion_config.ingested_data_dir,
                f"customer_churn_v{dataset_version}.csv"
            )

            df.to_csv(
                ingested_data_path,
                index=False
            )

            if not os.path.isfile(ingested_data_path):
                raise FileNotFoundError(
                    f"Ingested file was not created: "
                    f"{ingested_data_path}"
                )

            if os.path.getsize(ingested_data_path) == 0:
                raise ValueError(
                    f"Ingested file is empty: "
                    f"{ingested_data_path}"
                )

            logger.info(
                "Ingested dataset saved to %s",
                ingested_data_path
            )

            artifact = DataIngestionArtifact(
                ingested_data_path=ingested_data_path,
                dataset_version=dataset_version
            )

            logger.info(
                "Data ingestion completed successfully"
            )

            return artifact

        except CustomerChurnException:
            raise

        except Exception as e:
            logger.exception(
                "Data ingestion pipeline failed"
            )

            raise CustomerChurnException(
                e,
                sys,
                "DataIngestion",
                "Initiating data ingestion"
            ) from e