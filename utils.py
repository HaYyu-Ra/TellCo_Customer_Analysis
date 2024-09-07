# utils.py
import os
import logging

def check_file_exists(file_path):
    """Checks if a given file exists; raises an error if it doesn't."""
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
