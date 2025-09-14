import os
import mysql.connector  # fixed import
from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, url_for, session, flash
from utils.db_utils import get_ad_for_cluster, log_campaign
from utils.email_utils import send_email

# ---------------- DB CONFIG ----------------
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "customer_segmentation"
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# ---------------- UPLOAD CONFIG ----------------
UPLOAD_FOLDER = "static/images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "my_super_secret_key_123"

# Load model
centroids = joblib.load("Model/centroids.pkl")
scaler = joblib.load("Model/scaler.pkl")

centroids_scaled = scaler.transform(centroids)

def assign_cluster(new_data):
    new_data_scaled = scaler.transform([new_data])
    distances = np.linalg.norm(new_data_scaled - centroids_scaled, axis=1)
    return int(np.argmin(distances))


# ---------------- DASHBOARD ----------------
@app.route("/")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])

# ---------------- Single ----------------
@app.route("/single", methods=["GET", "POST"])
def single_input():
    errors = {}  # Dictionary to store errors
    values = {}  # Dictionary to keep entered values

    if request.method == "POST":
        email = request.form.get("email")
        values["email"] = email

        try:
            balance = float(request.form.get("balance"))
            purchases = float(request.form.get("purchases"))
            cash_advance = float(request.form.get("cash_advance"))
            credit_limit = float(request.form.get("credit_limit"))
            payments = float(request.form.get("payments"))
            full_payment = float(request.form.get("full_payment"))
            purchases_freq = float(request.form.get("purchases_freq"))
            cash_adv_freq = float(request.form.get("cash_adv_freq"))
        except:
            errors["general"] = "Please enter valid numeric values for all fields."
            return render_template("single.html", errors=errors, values=request.form)

        # --- Validation ---
        if not (0 <= balance <= 100000):
            errors["balance"] = "Balance out of range (0-100000)"
        if not (0 <= purchases <= 100000):
            errors["purchases"] = "Purchases out of range (0-100000)"
        if not (0 <= cash_advance <= 100000):
            errors["cash_advance"] = "Cash Advance out of range (0-100000)"
        if not (0 <= credit_limit <= 200000):
            errors["credit_limit"] = "Credit Limit out of range (0-200000)"
        if not (0 <= payments <= 200000):
            errors["payments"] = "Payments out of range (0-200000)"
        if full_payment not in [0, 1]:
            errors["full_payment"] = "Full Payment must be 0 or 1"
        if not (0 <= purchases_freq <= 1):
            errors["purchases_freq"] = "Purchases Frequency out of range (0-1)"
        if not (0 <= cash_adv_freq <= 1):
            errors["cash_adv_freq"] = "Cash Advance Frequency out of range (0-1)"

        if errors:
            return render_template("single.html", errors=errors, values=request.form)

        # Predict cluster
        features = [balance, purchases, cash_advance, credit_limit, payments, full_payment, purchases_freq, cash_adv_freq]
        cluster_id = assign_cluster(features)

        # Save customer
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO customers 
            (email, balance, purchases, cash_advance, credit_limit, payments, full_payment, purchases_freq, cash_adv_freq, cluster)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (email, *features, cluster_id))
        conn.commit()
        customer_id = cursor.lastrowid
        cursor.close()
        conn.close()

        # Fetch ad
        ad = get_ad_for_cluster(cluster_id)
        if not ad:
            return render_template("single_result.html", results=[{"email": email, "status": "❌ No ad found", "cluster": cluster_id}])

        # Send email
        image_path = ad.get("image_url").lstrip("/") if ad.get("image_url") else None
        send_email(email, f"Ad for Cluster {cluster_id}", f"<h2>Special Offer for You!</h2><p>{ad['ad_text']}</p>", image_path=image_path)

        # Log campaign
        log_campaign(customer_id, ad["id"], email)

        results = [{"email": email, "status": "✅ Email sent", "cluster": cluster_id}]
        return render_template("single_result.html", results=results)

    return render_template("single.html", errors=errors, values=values)


# ---------------- BULK UPLOAD ----------------
@app.route("/bulk", methods=["GET", "POST"])
def bulk_input():
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if not uploaded_file:
            flash("❌ No file uploaded", "danger")
            return redirect(request.url)

        filename = uploaded_file.filename
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif filename.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        except Exception as e:
            flash(f"❌ Error reading file: {e}", "danger")
            return redirect(request.url)

        required_columns = [
            "email", "balance", "purchases", "cash_advance",
            "credit_limit", "payments", "full_payment",
            "purchases_freq", "cash_adv_freq"
        ]
        for col in required_columns:
            if col not in df.columns:
                flash(f"❌ Missing column: {col}", "danger")
                return redirect(request.url)

        results = []

        for _, row in df.iterrows():
            email = row.get("email", "Unknown")
            try:
                features = [
                    float(row["balance"]), float(row["purchases"]), float(row["cash_advance"]),
                    float(row["credit_limit"]), float(row["payments"]), float(row["full_payment"]),
                    float(row["purchases_freq"]), float(row["cash_adv_freq"])
                ]
                if any(f < 0 for f in features):
                    raise ValueError("Negative value detected")
            except Exception as e:
                results.append({"email": email, "status": "❌ Invalid data", "error": str(e)})
                continue

            cluster_id = assign_cluster(features)

            # Save customer
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO customers
                    (email, balance, purchases, cash_advance, credit_limit, payments, full_payment, purchases_freq, cash_adv_freq, cluster)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (email, *features, cluster_id))
                conn.commit()
                customer_id = cursor.lastrowid
                cursor.close()
                conn.close()
            except Exception as e:
                results.append({"email": email, "status": "❌ DB error", "error": str(e)})
                continue

            ad = get_ad_for_cluster(cluster_id)
            if not ad:
                results.append({"email": email, "status": "❌ No ad for cluster"})
                continue

            try:
                image_path = ad.get("image_url").lstrip("/") if ad.get("image_url") else None
                send_email(email, f"Ad for Cluster {cluster_id}", f"<h2>Special Offer for You!</h2><p>{ad['ad_text']}</p>", image_path=image_path)
                log_campaign(customer_id, ad["id"], email)
                results.append({"email": email, "status": "✅ Email sent", "cluster": cluster_id})
            except Exception as e:
                results.append({"email": email, "status": "❌ Email failed", "error": str(e)})

        return render_template("bulk_result.html", results=results)

    return render_template("upload.html")

# ---------------- ADS MANAGEMENT ----------------
@app.route("/ads", methods=["GET", "POST"])
def ads_management():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    if request.method == "POST":
        cluster_id = request.form.get("cluster_id")
        ad_text = request.form.get("ad_text")

        # Handle image file upload
        image_file = request.files.get("image_file")
        image_url = None
        if image_file and image_file.filename != "" and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image_file.save(filepath)
            image_url = f"/static/images/{filename}"  # store relative URL

        # Insert into database
        cursor.execute(
            "INSERT INTO ads (cluster, ad_text, image_url) VALUES (%s, %s, %s)",
            (cluster_id, ad_text, image_url)
        )
        conn.commit()

    # Fetch all ads
    cursor.execute("SELECT * FROM ads")
    ads = cursor.fetchall()

    cursor.close()
    conn.close()
    return render_template("ads.html", ads=ads)

# ---------------- DELETE AD ----------------
@app.route("/ads/delete/<int:ad_id>", methods=["POST"])
def delete_ad(ad_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Optionally: delete image from static folder
    cursor.execute("SELECT image_url FROM ads WHERE id=%s", (ad_id,))
    result = cursor.fetchone()
    if result and result[0]:
        image_path = result[0].lstrip("/")  # remove leading '/'
        if os.path.exists(image_path):
            os.remove(image_path)

    # Delete ad from database
    cursor.execute("DELETE FROM ads WHERE id=%s", (ad_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect(url_for("ads_management"))

# ---------------- EDIT AD ----------------
@app.route("/ads/edit/<int:ad_id>", methods=["GET", "POST"])
def edit_ad(ad_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    if request.method == "POST":
        cluster_id = request.form.get("cluster_id")
        ad_text = request.form.get("ad_text")

        # Handle new image upload
        image_file = request.files.get("image_file")
        image_url = None
        if image_file and image_file.filename != "" and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image_file.save(filepath)
            image_url = f"/static/images/{filename}"
            cursor.execute(
                "UPDATE ads SET cluster=%s, ad_text=%s, image_url=%s WHERE id=%s",
                (cluster_id, ad_text, image_url, ad_id)
            )
        else:
            # Update without changing image
            cursor.execute(
                "UPDATE ads SET cluster=%s, ad_text=%s WHERE id=%s",
                (cluster_id, ad_text, ad_id)
            )
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for("ads_management"))

    # GET request: fetch ad data to prefill form
    cursor.execute("SELECT * FROM ads WHERE id=%s", (ad_id,))
    ad = cursor.fetchone()
    cursor.close()
    conn.close()
    return render_template("edit_ad.html", ad=ad)

# ---------- Register ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_pw = generate_password_hash(password)

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, hashed_pw)
            )
            conn.commit()
            flash("✅ Registered successfully! Please login.", "success")
            return redirect(url_for("login"))
        except:
            flash("⚠️ Username or Email already exists", "danger")
        finally:
            cur.close()
            conn.close()
    return render_template("register.html")

# ---------- Login ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        else:
            flash("❌ Invalid email or password", "danger")
    return render_template("login.html")

# ---------- Logout ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    print(">>> Starting Flask App <<<")
    app.run(debug=True)
