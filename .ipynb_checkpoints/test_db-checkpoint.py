import psycopg2
from psycopg2 import sql


def get_db_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            user="postgres",  # your Postgres user
            password="Lenevo5ryzen7",
            dbname="customer_segmentation"
        )
        print("✅ Connected to PostgreSQL successfully!")
        return conn
    except psycopg2.Error as err:
        print(f"❌ Error: {err}")


# Test DB
conn = get_db_connection()
if conn:
    cursor = conn.cursor()

    # Insert a test ad
    cursor.execute("""
        INSERT INTO ads (cluster, ad_text)
        VALUES (%s, %s)
        RETURNING id
    """, (2, 'This is a test ad'))
    inserted_id = cursor.fetchone()[0]
    conn.commit()
    print(f"Inserted ad with id: {inserted_id}")

    # Fetch all ads
    cursor.execute("SELECT * FROM ads")
    for row in cursor.fetchall():
        print(row)

    cursor.close()
    conn.close()
