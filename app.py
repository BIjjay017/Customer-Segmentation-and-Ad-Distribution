import os
import io
import csv
from flask import Response
from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, url_for, session, flash
from utils.db_utils import get_ad_for_cluster, log_campaign
from utils.email_utils import send_email
import psycopg2
from psycopg2.extras import RealDictCursor


# ---------------- DB CONFIG ----------------
db_config = {
    "host": "localhost",
    "user": "postgres",
    "password": "Lenevo5ryzen7",
    "dbname": "customer_segmentation"
}

def get_db_connection():
    return psycopg2.connect(**db_config)


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

# Get exact training feature names (in the order scaler expects)
if hasattr(scaler, "feature_names_in_"):
    FEATURE_NAMES = list(scaler.feature_names_in_)   # e.g. ['BALANCE','PURCHASES',...]
else:
    # fallback if scaler doesn't carry names (adjust if your training order differs)
    FEATURE_NAMES = [
        "BALANCE","PURCHASES","CASH_ADVANCE","CREDIT_LIMIT",
        "PAYMENTS","PRC_FULL_PAYMENT","PURCHASES_FREQUENCY","CASH_ADVANCE_FREQUENCY"
    ]

# Normalized lowercase names that match form/CSV headers you use in the app
FEATURE_NAMES_LOWER = [n.strip().lower().replace(" ", "_") for n in FEATURE_NAMES]

# Small synonyms mapping (expand if your CSV/form uses different names)
SYNONYMS = {
    "prc_full_payment": "full_payment",
    "purchases_frequency": "purchases_freq",
    "cash_advance_frequency": "cash_adv_freq"
}

def assign_cluster(new_data):
    """
    new_data: 1D list/array of raw feature values in the normalized lower-order:
      FEATURE_NAMES_LOWER order, e.g. ['balance','purchases',...]
    It constructs a DataFrame with exact column names scaler expects, transforms, and finds nearest centroid.
    Returns: int cluster index
    """
    # Ensure input is length-matched
    if len(new_data) != len(FEATURE_NAMES_LOWER):
        raise ValueError(f"Expected {len(FEATURE_NAMES_LOWER)} features in order {FEATURE_NAMES_LOWER}")

    # Build a DataFrame with normalized columns then rename to scaler's original column names
    df = pd.DataFrame([new_data], columns=FEATURE_NAMES_LOWER)
    df.columns = FEATURE_NAMES  # now matches scaler.feature_names_in_
    scaled = scaler.transform(df)   # scaler.transform accepts DataFrame with those column names
    distances = np.linalg.norm(scaled - centroids, axis=1)  # centroids already scaled
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
        except Exception as e:
            flash(f"❌ Error reading file: {e}", "danger")
            return redirect(request.url)

        # Normalize headers to lower_case_underscore
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Try synonyms mapping so user can upload common header names
        for target, alt in SYNONYMS.items():
            if target not in df.columns and alt in df.columns:
                df = df.rename(columns={alt: target})

        # Ensure required columns exist
        missing = [c for c in FEATURE_NAMES_LOWER + ["email"] if c not in df.columns]
        if missing:
            flash(f"Missing required columns: {missing}", "danger")
            return redirect(request.url)

        # Reorder df to feature order and cast floats
        X_df = df[FEATURE_NAMES_LOWER].astype(float).copy()
        # Rename to scaler original names expected by scaler
        X_df.columns = FEATURE_NAMES

        # Vectorized scale + assign clusters
        try:
            X_scaled = scaler.transform(X_df)  # shape (n, m)
            distances = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)  # (n_clusters)
            clusters = np.argmin(distances, axis=1)
        except Exception as e:
            flash(f"❌ Error during scaling/assignment: {e}", "danger")
            return redirect(request.url)

        # Attach cluster assignment to original df
        df["cluster_assigned"] = clusters

        results = []
        # Save customers, send emails and log
        for idx, row in df.iterrows():
            email = row.get("email", "unknown")
            features = [float(row[c]) for c in FEATURE_NAMES_LOWER]
            cluster_id = int(row["cluster_assigned"])

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

        # Render results page (bulk_result.html expects results list)
        return render_template("bulk_result.html", results=results)

    return render_template("upload.html")


# ---------------- ADS MANAGEMENT ----------------
@app.route("/ads", methods=["GET", "POST"])
def ads_management():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

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
    cursor = conn.cursor(cursor_factory=RealDictCursor)

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

# ---------------- LOGS VIEW ----------------
@app.route("/logs")
def view_logs():
    # require login (same check as dashboard)
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Join logs with customers and ads for friendly display
    cursor.execute("""
        SELECT
            l.id AS log_id,
            l.timestamp,
            l.email AS sent_to,
            l.customer_id,
            c.email AS customer_email,
            l.ad_id,
            a.cluster AS ad_cluster,
            a.ad_text
        FROM logs l
        LEFT JOIN customers c ON l.customer_id = c.id
        LEFT JOIN ads a ON l.ad_id = a.id
        ORDER BY l.timestamp DESC
        LIMIT 500
    """)
    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template("logs.html", logs=logs)


# ---------------- LOGS EXPORT CSV ----------------
@app.route("/logs/export")
def export_logs():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            l.id, l.timestamp, l.email, l.customer_id, c.email AS customer_email,
            l.ad_id, a.cluster AS ad_cluster, a.ad_text
        FROM logs l
        LEFT JOIN customers c ON l.customer_id = c.id
        LEFT JOIN ads a ON l.ad_id = a.id
        ORDER BY l.timestamp DESC
    """)
    rows = cursor.fetchall()
    colnames = [d[0] for d in cursor.description]
    cursor.close()
    conn.close()

    # Create CSV in-memory
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(colnames)
    for r in rows:
        writer.writerow(r)

    output = si.getvalue()
    si.close()

    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=logs_export.csv"}
    )


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
        cur = conn.cursor(cursor_factory=RealDictCursor)
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
