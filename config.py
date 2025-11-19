import os
from dotenv import load_dotenv, find_dotenv

# Load .env file
load_dotenv(find_dotenv())

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 5432))   # default to 5432 if missing
}
