import os
import logging
from datetime import datetime

log_file = f"{datetime.now().strftime('%d_%m_%Y %H_%M_%S')}.log"
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

log = os.path.join(log_dir, log_file)

log_format = "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"

formatter = logging.Formatter(log_format)

file_handler = logging.FileHandler(log)

console_handler = logging.StreamHandler()

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)



# Temporary logging test
if __name__ == "__main__":
    logger.info("INFO: Logging configuration is working!")
    logger.warning("WARNING: This is a test warning.")
    logger.error("ERROR: This is a test error.")

    print(f"\nLog file created at: {log}")