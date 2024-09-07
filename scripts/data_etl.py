# src/data_processing.py

import pandas as pd
from sqlalchemy import create_engine
from src.config import DB_URL, FILE_PATHS

def extract_data():
    """Extract data from the Excel and CSV files."""
    data_1 = pd.read_excel(FILE_PATHS['week1_challenge_data_1'])
    data_2 = pd.read_csv(FILE_PATHS['week1_challenge_data_2'])
    return data_1, data_2

def clean_data(df):
    """Clean and prepare the dataset."""
    # Example: Handling missing values
    df.fillna(df.mean(), inplace=True)
    return df

def load_to_db(df, table_name):
    """Load cleaned data into the PostgreSQL database."""
    engine = create_engine(DB_URL)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
# scripts/data_etl.py

from src.data_processing import extract_data, clean_data, load_to_db

# Extract data
data_1, data_2 = extract_data()

# Clean data
cleaned_data_1 = clean_data(data_1)
cleaned_data_2 = clean_data(data_2)

# Load data to PostgreSQL
load_to_db(cleaned_data_1, 'user_data_xdr')
load_to_db(cleaned_data_2, 'user_data_sessions')

print("Data successfully loaded into the PostgreSQL database!")
