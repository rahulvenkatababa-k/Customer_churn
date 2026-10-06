import sys
from src.customer_churn.logging.logger import logger
from src.customer_churn.exception.exception import CustomerChurnException

from src.customer_churn.pipeline.training_pipeline import TrainingPipeline

if __name__ == '__main__':
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        logger.error("%s", e)
        raise CustomerChurnException(e,sys, "main", "main")