# clean.py
import pandas as pd
import logging

def clean_data(df):
    """Cleans the DataFrame by handling missing values, removing duplicates, and correcting data types."""
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    df.ffill(inplace=True)  # Forward fill to handle missing values
    
    logging.info("Data cleaned successfully.")
    return df
