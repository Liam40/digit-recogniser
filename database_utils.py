import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_connection(dbname='digit_recogniser'):
    """Connect to the PostgreSQL database."""
    return psycopg2.connect(
        dbname=dbname,
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )

def initialise_database():
    """Ensure the database and table exist."""
    try:
        # Attempt to create the 'digit-recogniser' database if it doesn't exist
        conn = get_db_connection("postgres")
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'digit_recogniser'")
            if not cursor.fetchone():
                print("hello")
                cursor.execute("CREATE DATABASE digit_recogniser")

        # Ensure the table exists in the 'digit-recogniser' database
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS history (
                    timestamp TIMESTAMP PRIMARY KEY,
                    pred VARCHAR(4),
                    label VARCHAR(4)
                )
            """))
        conn.commit()

    except psycopg2.Error as e:
        raise Exception(f"Error initialising database: {e}")

def save_to_database(timestamp, pred, label):
    """Save a record to the `history` table."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("INSERT INTO history (timestamp, pred, label) VALUES (%s, %s, %s)"),
                    (timestamp, pred, label),
                )
            conn.commit()
    except psycopg2.Error as e:
        raise Exception(f"Error saving to database: {e}")

def fetch_history():
    """Fetch all records from the `history` table."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
                return cursor.fetchall()
    except psycopg2.Error as e:
        raise Exception(f"Error fetching history: {e}")