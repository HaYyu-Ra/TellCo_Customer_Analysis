# extract.py
import pandas as pd
import logging
from utils import check_file_exists
from src.config import FILE_PATHS

def extract_data():
    """Extracts data from specified files and loads them into pandas DataFrames."""
    # Validate file existence
    for path in FILE_PATHS.values():
        check_file_exists(path)

    # Load data into DataFrames
    field_descriptions_df = pd.read_excel(FILE_PATHS["field_descriptions"])
    week1_data_df_1 = pd.read_excel(FILE_PATHS["week1_challenge_data_1"])
    week1_data_df_2 = pd.read_csv(FILE_PATHS["week1_challenge_data_2"])

    logging.info("Data extracted successfully.")
    return field_descriptions_df, week1_data_df_1, week1_data_df_2
