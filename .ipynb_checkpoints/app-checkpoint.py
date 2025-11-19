import os
import pandas as pd
import joblib
import smtplib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import mysql.connector

# Initialize Flask app
app = Flask(__name__)

# ✅ Load centroids & scaler (already saved from Jupyter)
centroids = joblib.load("centroids.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- Helper: Assign cluster ----------------
def assign_cluster(new_data, scaler, centroids):
    """Assigns cluster index based on nearest centroid"""
    new_data_scaled = scaler.transform([new_data])
    distances = np.linalg.norm(new_data_scaled - centroids, axis=1)
    return int(np.argmin(distances))

def assign_clusters_bulk(features, scaler, centroids):
    """Assign clusters for a whole DataFrame"""
    scaled_features = scaler.transform(features)
    clusters = []
    for row in scaled_features:
        distances = np.linalg.norm(row - centroids, axis=1)
        clusters.append(np.argmin(distances))
    return clusters

# ---------------- DB Connection ----------------
def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",      # default XAMPP user
        password="",      # default is empty in XAMPP
        database="customer_segmentation"
    )
    return conn

def get_ad_for_cluster(cluster_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM ads WHERE cluster_id = %s LIMIT 1"
    cursor.execute(query, (cluster_id,))
    ad = cursor.fetchone()
    cursor.close()
    conn.close()
    return ad

def log_campaign(ad_id, email, cluster_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "INSERT INTO campaigns (ad_id, customer_email, cluster_id) VALUES (%s, %s, %s)"
    cursor.execute(query, (ad_id, email, cluster_id))
    conn.commit()
    cursor.close()
    conn.close()

# ---------------- Email Function ----------------
def send_email(receiver_email, subject, body):
    sender_email = "your_email@gmail.com"
    password = "your_password"   # ⚠️ use Gmail app password

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"✅ Email sent to {receiver_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

# ---------------- Flask Routes ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check which form was submitted
        if "single_submit" in request.form:  # Single user form
            email = request.form.get("email")
            try:
                age = float(request.form.get("age"))
                income = float(request.form.get("income"))
                spending = float(request.form.get("spending_score"))
            except Exception:
                return "Please enter valid numeric values."

            features = [age, income, spending]

            # Predict cluster
            cluster_id = assign_cluster(features, scaler, centroids)
            ad = get_ad_for_cluster(cluster_id)

            if ad:
                subject = ad["title"]
                body = f"<h2>{ad['title']}</h2><p>{ad['content']}</p>"
                if ad["image_url"]:
                    body += f"<br><img src='{ad['image_url']}' width='300'>"

                # Send email + log
                send_email(email, subject, body)
                log_campaign(ad["id"], email, cluster_id)

                return f"✅ Email sent to {email} for Cluster {cluster_id}"

        elif "bulk_submit" in request.form:  # Bulk CSV form
            file = request.files["file"]
            if not file:
                return "No file uploaded"

            df = pd.read_csv(file)

            if "Email" not in df.columns:
                return "CSV must contain an 'Email' column"

            emails = df["Email"]
            features = df.drop(columns=["Email"])
            clusters = assign_clusters_bulk(features, scaler, centroids)
            df["Cluster"] = clusters

            for idx, row in df.iterrows():
                cluster_id = row["Cluster"]
                email = row["Email"]

                ad = get_ad_for_cluster(cluster_id)
                if ad:
                    subject = ad["title"]
                    body = f"<h2>{ad['title']}</h2><p>{ad['content']}</p>"
                    if ad["image_url"]:
                        body += f"<br><img src='{ad['image_url']}' width='300'>"

                    send_email(email, subject, body)
                    log_campaign(ad["id"], email, cluster_id)

            return render_template("result.html", tables=[df.to_html(classes="data")], titles=df.columns.values)

    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
