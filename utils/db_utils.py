import psycopg2
import random
from psycopg2.extras import RealDictCursor  # for dict-like cursor

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        user="postgres",         # your Postgres user
        password="Lenevo5ryzen7",
        dbname="customer_segmentation"
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
