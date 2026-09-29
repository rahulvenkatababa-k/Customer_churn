import os

from dotenv import load_dotenv

from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging.logger import logger


load_dotenv()


class Configuration:
    def __init__(self):
        self.postgres_sql_uri = self._get_required_value(
            "POSTGRES_SQL_URI"
        )
        self.model_name = self._get_required_value("MODEL_NAME")

    @staticmethod
    def _get_required_value(variable_name):
        value = os.getenv(variable_name)

        if value is None or not value.strip():
            error = ValueError(
                f"Required configuration variable "
                f"'{variable_name}' is missing"
            )

            logger.error(str(error))
            raise error

        return value


config = Configuration()