"""
Centralized logging system for the application.
"""

import logging
import logging.handlers
from pathlib import Path
from src.reliability_analysis.utils.config import LOG_DIR, LOG_LEVEL, LOG_FORMAT, APP_NAME


def setup_logging(name: str = APP_NAME) -> logging.Logger:
    """
    Configures and returns a logger with both console and file handlers.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    log_file = LOG_DIR / f"{name.lower().replace(' ', '_')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()
