import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from src.config import db_url, csv_file_user_behavior

# Load the data from CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Handle missing values
def clean_data(df):
    df.fillna(df.mean(), inplace=True)
    return df

# Create database connection
def create_db_connection():
    engine = create_engine(db_url)
    return engine

# Save cleaned data to the database
def save_data_to_db(df, table_name, engine):
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)

# Feature engineering for xDR sessions and total volume
def feature_engineering(df):
    df['total_volume'] = df['total_DL'] + df['total_UL']
    return df

if __name__ == '__main__':
    data = load_data(csv_file_user_behavior)
    cleaned_data = clean_data(data)
    featured_data = feature_engineering(cleaned_data)
    
    # Save to DB
    engine = create_db_connection()
    save_data_to_db(featured_data, 'user_behavior', engine)
