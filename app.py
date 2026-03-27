import os
import io
import csv
import logging
from datetime import timedelta
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    session,
    flash,
    Response,
    jsonify,
)
import joblib
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from utils.db_utils import get_ad_for_cluster, log_campaign
from utils.email_utils import send_email
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


# ---------------- DB CONFIG ----------------
# Use environment variables for database configuration (required for Vercel)
# Falls back to localhost for local development
load_dotenv()

APP_ENV = os.getenv("APP_ENV", os.getenv("FLASK_ENV", "development")).lower()
IS_PRODUCTION = APP_ENV == "production" or os.getenv("VERCEL") == "1"

logging.basicConfig(
    level=logging.INFO if IS_PRODUCTION else logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection using environment variables or fallback to local config"""
    required_in_production = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    if IS_PRODUCTION:
        missing = [key for key in required_in_production if not os.getenv(key)]
        if missing:
            raise RuntimeError(
                f"Missing required DB environment variables in production: {missing}"
            )

    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "dbname": os.getenv("DB_NAME", "customer_segmentation"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", 10)),
    }
    return psycopg2.connect(**db_config)


# ---------------- UPLOAD CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "images")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# Ensure upload folder exists (only if filesystem is writable)
# Note: Vercel's filesystem is read-only except /tmp, so file uploads won't persist
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except (OSError, PermissionError):
    # In serverless environments, we can't create directories
    # Consider using Vercel Blob Storage or similar for file uploads
    logger.warning(
        "Cannot create upload folder %s. File uploads may not work in serverless environment.",
        UPLOAD_FOLDER,
    )


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- FLASK APP ----------------
app = Flask(__name__, template_folder="Templates", static_folder="static")

secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    if IS_PRODUCTION:
        raise RuntimeError("SECRET_KEY is required in production")
    secret_key = "dev-only-secret-key-change-me"

app.config.update(
    SECRET_KEY=secret_key,
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
)


@app.after_request
def set_security_headers(response):
    """Apply baseline security headers for production deployments."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


@app.get("/healthz")
def healthz():
    """Simple health endpoint for uptime checks and deployment probes."""
    return jsonify({"status": "ok", "environment": APP_ENV}), 200


# Load model with error handling for serverless environments
# In Vercel, the working directory might be different, so we try multiple paths
def load_model_file(filename):
    """Try to load model file from multiple possible locations"""
    possible_paths = [
        os.path.join("Model", filename),  # Model subdirectory
        os.path.join(os.path.dirname(__file__), "Model", filename),  # Absolute path
        filename,  # Current directory fallback
        os.path.join("/var/task", "Model", filename),  # Vercel Lambda path
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                logger.info("Loading model file from %s", path)
                return joblib.load(path)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
                continue

    raise FileNotFoundError(
        f"Model file '{filename}' not found in any of these locations: {possible_paths}. "
        "Make sure the model files are included in your deployment."
    )


try:
    centroids = load_model_file("centroids.pkl")
    scaler = load_model_file("scaler.pkl")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error("Error loading models: %s", e)
    # Set to None so we can handle this gracefully in routes
    centroids = None
    scaler = None

# Get exact training feature names (in the order scaler expects)
if hasattr(scaler, "feature_names_in_"):
    FEATURE_NAMES = list(scaler.feature_names_in_)  # e.g. ['BALANCE','PURCHASES',...]
else:
    # fallback if scaler doesn't carry names (adjust if your training order differs)
    FEATURE_NAMES = [
        "BALANCE",
        "PURCHASES",
        "CASH_ADVANCE",
        "CREDIT_LIMIT",
        "PAYMENTS",
        "PRC_FULL_PAYMENT",
        "PURCHASES_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY",
    ]

# Normalized lowercase names that match form/CSV headers you use in the app
FEATURE_NAMES_LOWER = [n.strip().lower().replace(" ", "_") for n in FEATURE_NAMES]

# Small synonyms mapping (expand if your CSV/form uses different names)
SYNONYMS = {
    "prc_full_payment": "full_payment",
    "purchases_frequency": "purchases_freq",
    "cash_advance_frequency": "cash_adv_freq",
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
        raise ValueError(
            f"Expected {len(FEATURE_NAMES_LOWER)} features in order {FEATURE_NAMES_LOWER}"
        )

    # Build a DataFrame with normalized columns then rename to scaler's original column names
    df = pd.DataFrame([new_data], columns=FEATURE_NAMES_LOWER)
    df.columns = FEATURE_NAMES  # now matches scaler.feature_names_in_
    scaled = scaler.transform(
        df
    )  # scaler.transform accepts DataFrame with those column names
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
        except (TypeError, ValueError):
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

        # Predict cluster (check if models are loaded)
        if centroids is None or scaler is None:
            errors[
                "general"
            ] = "Model files not loaded. Please check server configuration."
            return render_template("single.html", errors=errors, values=request.form)

        features = [
            balance,
            purchases,
            cash_advance,
            credit_limit,
            payments,
            full_payment,
            purchases_freq,
            cash_adv_freq,
        ]
        cluster_id = assign_cluster(features)

        # Save customer
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO customers 
            (email, balance, purchases, cash_advance, credit_limit, payments, full_payment, purchases_freq, cash_adv_freq, cluster)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """,
            (email, *features, cluster_id),
        )
        customer_id = cursor.fetchone()[0]  # PostgreSQL uses RETURNING, not lastrowid
        conn.commit()
        cursor.close()
        conn.close()

        # Fetch ad
        ad = get_ad_for_cluster(cluster_id)
        if not ad:
            return render_template(
                "single_result.html",
                results=[
                    {"email": email, "status": "❌ No ad found", "cluster": cluster_id}
                ],
            )

        try:
            # Send email
            image_path = (
                ad.get("image_url").lstrip("/") if ad.get("image_url") else None
            )
            send_email(
                email,
                f"Ad for Cluster {cluster_id}",
                f"<h2>Special Offer for You!</h2><p>{ad['ad_text']}</p>",
                image_path=image_path,
            )

            # Log campaign
            log_campaign(customer_id, ad["id"], email)
            results = [
                {"email": email, "status": "✅ Email sent", "cluster": cluster_id}
            ]
        except Exception as e:
            logger.exception("Failed to send email or log campaign")
            results = [
                {
                    "email": email,
                    "status": "❌ Email failed",
                    "cluster": cluster_id,
                    "error": str(e),
                }
            ]

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

        # Vectorized scale + assign clusters (check if models are loaded)
        if centroids is None or scaler is None:
            flash(
                "❌ Model files not loaded. Please check server configuration.", "danger"
            )
            return redirect(request.url)

        try:
            X_scaled = scaler.transform(X_df)  # shape (n, m)
            distances = np.linalg.norm(
                X_scaled[:, None, :] - centroids[None, :, :], axis=2
            )  # (n_clusters)
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
                cursor.execute(
                    """
                    INSERT INTO customers
                    (email, balance, purchases, cash_advance, credit_limit, payments, full_payment, purchases_freq, cash_adv_freq, cluster)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id
                """,
                    (email, *features, cluster_id),
                )
                customer_id = cursor.fetchone()[
                    0
                ]  # PostgreSQL uses RETURNING, not lastrowid
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                results.append(
                    {"email": email, "status": "❌ DB error", "error": str(e)}
                )
                continue

            ad = get_ad_for_cluster(cluster_id)
            if not ad:
                results.append({"email": email, "status": "❌ No ad for cluster"})
                continue

            try:
                image_path = (
                    ad.get("image_url").lstrip("/") if ad.get("image_url") else None
                )
                send_email(
                    email,
                    f"Ad for Cluster {cluster_id}",
                    f"<h2>Special Offer for You!</h2><p>{ad['ad_text']}</p>",
                    image_path=image_path,
                )
                log_campaign(customer_id, ad["id"], email)
                results.append(
                    {"email": email, "status": "✅ Email sent", "cluster": cluster_id}
                )
            except Exception as e:
                results.append(
                    {"email": email, "status": "❌ Email failed", "error": str(e)}
                )

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
        if (
            image_file
            and image_file.filename != ""
            and allowed_file(image_file.filename)
        ):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image_file.save(filepath)
            image_url = f"/static/images/{filename}"  # store relative URL

        # Insert into database
        cursor.execute(
            "INSERT INTO ads (cluster, ad_text, image_url) VALUES (%s, %s, %s)",
            (cluster_id, ad_text, image_url),
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
        if (
            image_file
            and image_file.filename != ""
            and allowed_file(image_file.filename)
        ):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image_file.save(filepath)
            image_url = f"/static/images/{filename}"
            cursor.execute(
                "UPDATE ads SET cluster=%s, ad_text=%s, image_url=%s WHERE id=%s",
                (cluster_id, ad_text, image_url, ad_id),
            )
        else:
            # Update without changing image
            cursor.execute(
                "UPDATE ads SET cluster=%s, ad_text=%s WHERE id=%s",
                (cluster_id, ad_text, ad_id),
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
    cursor.execute(
        """
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
    """
    )
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
    cursor.execute(
        """
        SELECT
            l.id, l.timestamp, l.email, l.customer_id, c.email AS customer_email,
            l.ad_id, a.cluster AS ad_cluster, a.ad_text
        FROM logs l
        LEFT JOIN customers c ON l.customer_id = c.id
        LEFT JOIN ads a ON l.ad_id = a.id
        ORDER BY l.timestamp DESC
    """
    )
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
        headers={"Content-Disposition": "attachment;filename=logs_export.csv"},
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
                (username, email, hashed_pw),
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
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    logger.info("Starting Flask app on port %s (debug=%s)", port, debug_mode)
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
