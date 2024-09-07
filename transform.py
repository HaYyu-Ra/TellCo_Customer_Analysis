# transform.py
import pandas as pd
import logging
from clean import clean_data

def transform_data(field_descriptions_df, week1_data_df_1, week1_data_df_2):
    """Performs transformations on the extracted data, such as combining the datasets."""
    # Combine two DataFrames vertically
    combined_data_df = pd.concat([week1_data_df_1, week1_data_df_2], axis=0, ignore_index=True)
    
    # Clean the combined DataFrame
    combined_data_df = clean_data(combined_data_df)
    
    logging.info("Data transformed successfully.")
    return combined_data_df
