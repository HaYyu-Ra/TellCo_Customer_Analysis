# load.py
from sqlalchemy import create_engine, text
import logging
from src.config import DB_URL

def load_data_to_db(combined_data_df):
    """Loads the transformed data into a PostgreSQL database."""
    engine = create_engine(DB_URL)

    try:
        with engine.connect() as conn:
            # Create an insert statement
            insert_stmt = text("""
                INSERT INTO xdr_data ("Bearer Id", "Start ms", "End ms", "Dur. (ms)", "IMSI", "MSISDN/Number", "Last Location Name", "Total DL (Bytes)")
                VALUES (:bearer_id, :start_ms, :end_ms, :dur_ms, :imsi, :msisdn_number, :last_location_name, :total_dl_bytes)
            """)

            # Insert data into the table
            for index, row in combined_data_df.iterrows():
                try:
                    conn.execute(insert_stmt, {
                        'bearer_id': row.get("Bearer Id"),
                        'start_ms': row.get("Start ms"),
                        'end_ms': row.get("End ms"),
                        'dur_ms': row.get("Dur. (ms)"),
                        'imsi': row.get("IMSI"),
                        'msisdn_number': row.get("MSISDN/Number"),
                        'last_location_name': row.get("Last Location Name"),
                        'total_dl_bytes': row.get("Total DL (Bytes)")
                    })
                except Exception as e:
                    logging.error(f"Error inserting row {index}: {e}")
            
            logging.info("Data loaded into the database successfully.")

    except Exception as e:
        logging.error(f"An error occurred while loading data into the database: {e}")

    finally:
        engine.dispose()
