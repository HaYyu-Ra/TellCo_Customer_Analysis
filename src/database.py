import pandas as pd
from sqlalchemy import create_engine

DB_URL = 'postgresql://postgres:admin@localhost:5432/tellco_analysis'

def create_db_connection(db_url=DB_URL):
    """
    Creates a connection to the PostgreSQL database.
    
    Args:
        db_url (str): The database connection URL.
    
    Returns:
        sqlalchemy.engine.base.Connection: A connection to the database.
    """
    engine = create_engine(db_url)
    connection = engine.connect()
    return connection

def store_data_to_db(df, table_name, if_exists='replace'):
    """
    Stores a DataFrame into the SQL database.
    
    Args:
        df (pd.DataFrame): Data to be stored.
        table_name (str): Name of the table in the database.
        if_exists (str): Behavior when the table already exists. Options are 'fail', 'replace', 'append'.
    """
    connection = create_db_connection()
    df.to_sql(table_name, con=connection, if_exists=if_exists, index=False)
    connection.close()

def load_data_from_db(table_name):
    """
    Loads data from the SQL database.
    
    Args:
        table_name (str): The name of the table to load data from.
    
    Returns:
        pd.DataFrame: Data loaded from the SQL database.
    """
    connection = create_db_connection()
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=connection)
    connection.close()
    return df
