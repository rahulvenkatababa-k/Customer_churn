import os, sys
from src.customer_churn.logging.logger import logger

class CustomerChurnException(Exception):
    def __init__(self, original_error, error_details:sys, pipeline=None, operation=None):
        super().__init__(str(original_error))

        self.original_error = original_error

        _, _, exc_tb = error_details.exc_info()

        self.pipeline = pipeline
        self.operation = operation

        if exc_tb is not None:
            self.filename = exc_tb.tb_frame.f_code.co_filename
            self.lineno = exc_tb.tb_lineno
        else:
            self.filename = None
            self.lineno = None

    def __str__(self):
        return (
            f"Pipeline/Component : [{self.pipeline}] | "
            f" Operation : [{self.operation}] | "
            f" Error occured in python script name [{self.filename}] "
            f" at line number [{self.lineno}] "
            f" with error message - [{self.original_error}]"
        )














if __name__ == "__main__":
    from src.customer_churn.logging.logger import logger

    try:
        try:
            raise ConnectionError("PostgreSQL connection refused")

        except Exception as e:
            raise CustomerChurnException(
                original_error=e,
                error_details=sys,
                pipeline="CustomerChurn",
                operation="Connect to PostgreSQL",
            ) from e

    except CustomerChurnException as e:
        logger.exception("%s", e)

        print("\nCustom exception information:")
        print(e)

        print(
            "Original exception type:",
            type(e.original_error).__name__,
        )
        print(
            "Original exception message:",
            str(e.original_error),
        )
        print(
            "Original exception preserved as cause:",
            e.__cause__ is e.original_error,
        )