import logging
import os
from pathlib import Path
from typing import Dict

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

def setup_logging(log_level: str = 'info'):
    """Setup logging configuration"""
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=LOG_LEVELS.get(log_level.lower(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'voice_agent.log'),
            logging.StreamHandler()
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

# Setup basic logging
setup_logging()
