"""
Sistema de logging centralizado para la aplicación.
"""

import logging
import logging.handlers
from pathlib import Path
from src.reliability_analysis.utils.config import LOG_DIR, LOG_LEVEL, LOG_FORMAT, APP_NAME


def setup_logging(name: str = APP_NAME) -> logging.Logger:
    """
    Configura y retorna un logger con manejo a archivo y consola.
    
    Args:
        name: Nombre del logger
        
    Returns:
        logging.Logger: Logger configurado
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
