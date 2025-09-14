import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="customer_segmentation"
    )

def get_ad_for_cluster(cluster_id):
    """Get one ad for the given cluster"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM ads WHERE cluster = %s LIMIT 1", (cluster_id,))
    ad = cursor.fetchone()
    cursor.close()
    conn.close()
    return ad

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
