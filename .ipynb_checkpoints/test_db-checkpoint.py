import mysql.connector

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # XAMPP default
            database="customer_segmentation"
        )
        print("✅ Connected to database successfully!")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Error: {err}")

# Test DB
conn = get_db_connection()
if conn:
    cursor = conn.cursor()
    
    # Insert a test ad
    cursor.execute("""
        INSERT INTO ads (id, cluster, ad_text)
        VALUES (0, 2, 'This is a test ad')
    """)
    conn.commit()
    
    # Fetch all ads
    cursor.execute("SELECT * FROM ads")
    for row in cursor.fetchall():
        print(row)
    
    cursor.close()
    conn.close()
