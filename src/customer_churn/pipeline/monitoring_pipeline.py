
from prefect import flow, task

from src.customer_churn.components.production_drift_monitor import (
    run_production_drift_check,
)
from src.customer_churn.logging.logger import logger


@task(
    name="Production Drift Check",
    retries=2,
    retry_delay_seconds=60,
    timeout_seconds=300,
)
def production_drift_check() -> dict:
    return run_production_drift_check()


@flow(name="customer-churn-production-monitoring")
def customer_churn_monitoring_flow() -> dict:
    result = production_drift_check()
    logger.info("Production monitoring result: %s", result)
    return result


if __name__ == "__main__":
    customer_churn_monitoring_flow()
