# etl_process.py
import logging
from logging_config import setup_logging
from extract import extract_data
from transform import transform_data
from load import load_data_to_db

def etl_process():
    """Executes the full ETL process: extract, transform, and load."""
    try:
        # Extraction
        field_descriptions_df, week1_data_df_1, week1_data_df_2 = extract_data()
        
        # Transformation
        combined_data_df = transform_data(field_descriptions_df, week1_data_df_1, week1_data_df_2)
        
        # Loading
        load_data_to_db(combined_data_df)
        
    except Exception as e:
        logging.error(f"An error occurred during the ETL process: {e}")

if __name__ == "__main__":
    setup_logging()
    etl_process()
