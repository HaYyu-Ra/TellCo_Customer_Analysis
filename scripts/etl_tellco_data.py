import os
import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection parameters
DATABASE_URL = "postgresql://postgres:admin@localhost:5432/tellco_analysis"
engine = create_engine(DATABASE_URL)

# Define the path for the data file
data_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\TellCo_Customer_Analysis\data\Week1_challenge_data_source(CSV).csv'

# List of tables to load data into
tables = [
    'customer_data',
    'field_descriptions',
    'telecom_data',
    'tellco_cleaned_data',
    'tellco_combined_data',
    'user_behavior',
    'week1_challenge_data_1',
    'week1_challenge_data_2',
    'week1_challenge_data_source',
    'week1_data',
    'week1_data_csv',
    'week1_data_excel',
    'week1_data_source',
    'week1_data_source_csv',
    'xdr_data',
    'xdr_records',
    'xdr_sessions'
]

def load_data(file_path, table_name):
    """
    Load data into the specified PostgreSQL table.
    """
    if os.path.exists(file_path):
        print(f"Loading data for table '{table_name}' from {file_path}")
        
        # Load the CSV file as a pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Clean data
        df = clean_data(df)

        # Load the data into the PostgreSQL table
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"Data successfully loaded into table '{table_name}'")
    else:
        print(f"Data file for table '{table_name}' not found or not specified.")

def clean_data(df):
    """
    Clean data by handling missing values and fixing data types.
    """
    for column in df.columns:
        if df[column].dtype == 'float64':
            df[column] = df[column].fillna(0)  # Fill NaN with 0 for float columns
        else:
            df[column] = df[column].fillna('')  # Fill NaN with empty string for other columns
    return df

# Load the data into each specified table
for table_name in tables:
    load_data(data_file_path, table_name)
