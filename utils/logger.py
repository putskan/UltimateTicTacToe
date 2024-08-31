import logging
import os
from pathlib import Path
from typing import Union


def get_logger(log_name: str, log_dir_name: Union[Path, str] = None, log_to_console: bool = False) -> logging.Logger:
    """
    Create and configure a logger
    :param log_name: The log name
    :param log_dir_name: The log dest directory
    :param log_to_console: Whether to log to console
    :return Configured logger
    """
    assert log_to_console or log_dir_name, 'Must provide at least one of log_to_console, log_dir_name'
    # Create a custom logger
    logger = logging.getLogger(f'{log_name}_logger')
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if the logger is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # Create file handler
        if log_dir_name is not None:
            os.makedirs(log_dir_name, exist_ok=True)
            log_file_name = os.path.join(log_dir_name, f'{log_name}.log')
            file_handler = logging.FileHandler(log_file_name, mode='w')
            file_handler.setLevel(logging.INFO)
            # Create formatter and add it to the handlers
            file_handler.setFormatter(formatter)
            # Add file handler to the logger
            logger.addHandler(file_handler)

        if log_to_console:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            # Add console handler to the logger
            logger.addHandler(console_handler)

    return logger
