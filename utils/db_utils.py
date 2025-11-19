import psycopg2
import random
import os
from psycopg2.extras import RealDictCursor  # for dict-like cursor
from dotenv import load_dotenv
load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 5432))  # default port 5432
    )

def get_ad_for_cluster(cluster_id):
    """Get one random ad for the given cluster"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)  # dict-like rows
    cursor.execute("SELECT * FROM ads WHERE cluster = %s", (cluster_id,))
    ads = cursor.fetchall()
    cursor.close()
    conn.close()

    if ads:
        return random.choice(ads)  # pick one ad randomly
    return None

def log_campaign(customer_id, ad_id, email):
    """Log the sent ad to the logs table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO logs (customer_id, ad_id, email) VALUES (%s, %s, %s)",
        (customer_id, ad_id, email)
    )
    conn.commit()
    cursor.close()
    conn.close()
