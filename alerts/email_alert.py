import os
import smtplib
from email.mime.text import MIMEText

SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "survproj908@gmail.com")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "zqpv lffs atya pbky")

def send_alert(message):

    msg = MIMEText(message)

    msg["Subject"] = "AI Threat Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = SENDER_EMAIL

    server = smtplib.SMTP("smtp.gmail.com", 587)

    server.starttls()

    server.login(SENDER_EMAIL, APP_PASSWORD)

    server.send_message(msg)

    server.quit()

    print("Alert Email Sent!")