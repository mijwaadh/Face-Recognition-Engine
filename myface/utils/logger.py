import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Sets up a logger with output to both console and a rotating log file.
    
    Args:
        name: Name of the logger.
        log_dir: Directory where log files are stored.
        log_level: Severity level threshold for logging.
        
    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is re-initialized
    if logger.handlers:
        return logger

    # Map string log levels
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logger.setLevel(numeric_level)

    # Formatter definition
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "app.log")
        
        # Rotating file: Max 5MB per file, max 3 backup files
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"WARNING: Could not initialize log file handler: {e}")

    return logger
