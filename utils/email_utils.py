import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

SENDER_EMAIL = os.getenv("SMTP_USERNAME")
SENDER_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "1") == "1"

def send_email(receiver_email, subject, body_html, image_path=None):
    """
    Send an email with optional inline image.
    - body_html: the HTML content
    - image_path: if provided, attach the image and reference as cid:ad_image
    """
    msg = MIMEMultipart("related")
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        raise RuntimeError("Missing SMTP_USERNAME/SMTP_PASSWORD environment variables")

    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # Alternative part (HTML)
    msg_alternative = MIMEMultipart("alternative")
    msg.attach(msg_alternative)

    if image_path and os.path.exists(image_path):
        # Attach HTML with inline <img>
        body_with_img = body_html + '<br><img src="cid:ad_image">'
        msg_alternative.attach(MIMEText(body_with_img, "html"))

        # Attach the image
        with open(image_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<ad_image>")
            msg.attach(img)
    else:
        # Fallback: send only HTML body
        msg_alternative.attach(MIMEText(body_html, "html"))

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20)
        if SMTP_USE_TLS:
            server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        server.quit()
        print(f"✅ Email sent to {receiver_email}")
    except Exception as e:
        print(f"❌ Error sending email: {e}")
