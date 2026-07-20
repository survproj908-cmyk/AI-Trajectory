from alerts.email_alert import send_alert
from logging_system.threat_logger import log_threat
from evaluation.threat_score import calculate_threat_score
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import sys
import os
from datetime import datetime
# Allow importing from project root
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from models.lstm_model import LSTMPredictor

# -----------------------------
# Load YOLO
# -----------------------------
yolo_model = YOLO("yolov8n.pt")

# -----------------------------
# Load LSTM
# -----------------------------
lstm_model = LSTMPredictor()

lstm_model.load_state_dict(
    torch.load(
        "models/lstm_model.pth",
        map_location=torch.device("cpu")
    )
)

lstm_model.eval()

# -----------------------------
# Open Video
# -----------------------------
cap = cv2.VideoCapture(
    "data/videos/test.mp4"
)

# -----------------------------
# Track History
# -----------------------------
track_history = {}
email_sent = set()
# -----------------------------
# Restricted Zone
# -----------------------------
ZONE_X1 = 900
ZONE_Y1 = 400

ZONE_X2 = 1700
ZONE_Y2 = 1400

# -----------------------------
# Dataset Normalization Range
# -----------------------------
MIN_VAL = 33
MAX_VAL = 3808

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    boxes = results[0].boxes

    annotated = results[0].plot()

    # -----------------------------
    # Draw Restricted Zone
    # -----------------------------
    cv2.rectangle(
        annotated,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        (0, 0, 255),
        3
    )

    cv2.putText(
        annotated,
        "RESTRICTED ZONE",
        (ZONE_X1, ZONE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    if boxes.id is not None:

        ids = boxes.id.int().cpu().tolist()
        coords = boxes.xyxy.cpu().tolist()

        for track_id, box in zip(ids, coords):

            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append(
                (cx, cy)
            )

            if len(track_history[track_id]) > 100:
                track_history[track_id].pop(0)

            points = track_history[track_id]

            # -----------------------------
            # Draw Past Path (GREEN)
            # -----------------------------
            for i in range(1, len(points)):

                cv2.line(
                    annotated,
                    points[i - 1],
                    points[i],
                    (0, 255, 0),
                    2
                )

            # -----------------------------
            # Future Prediction
            # -----------------------------
            if len(points) >= 8:

                recent = np.array(
                    points[-8:]
                )

                recent = (
                    recent - MIN_VAL
                ) / (
                    MAX_VAL - MIN_VAL
                )

                recent = torch.tensor(
                    recent,
                    dtype=torch.float32
                ).unsqueeze(0)

                with torch.no_grad():

                    future = lstm_model(
                        recent
                    )

                future = (
                    future.squeeze(0)
                    .cpu()
                    .numpy()
                )

                future = (
                    future *
                    (MAX_VAL - MIN_VAL)
                ) + MIN_VAL

                future_points = []
                threat = False

                for px, py in future:

                    px = int(px)
                    py = int(py)

                    future_points.append(
                        (px, py)
                    )

                    if (
                        ZONE_X1 <= px <= ZONE_X2
                        and
                        ZONE_Y1 <= py <= ZONE_Y2
                    ):
                        threat = True
                    
                zone = (
                    ZONE_X1,
                    ZONE_Y1,
                    ZONE_X2,
                    ZONE_Y2
                )

                object_class = yolo_model.names[
                    int(
                        boxes.cls.cpu().tolist()[
                            ids.index(track_id)
                        ]
                    )
                ]

                score = calculate_threat_score(
                    future_points,
                    object_class,
                    zone
                )  

                # -----------------------------
                # Connect current position
                # to future path
                # -----------------------------
                if len(future_points) > 0:

                    cv2.line(
                        annotated,
                        points[-1],
                        future_points[0],
                        (255, 0, 0),
                        3
                    )

                # -----------------------------
                # Draw Future Path (BLUE)
                # -----------------------------
                for i in range(
                    1,
                    len(future_points)
                ):

                    cv2.line(
                        annotated,
                        future_points[i - 1],
                        future_points[i],
                        (255, 0, 0),
                        3
                    )

                # -----------------------------
                # Threat Alert
                # -----------------------------
                if score >= 50:
                    print("Inside threat block")
                    if track_id not in email_sent:

                        send_alert(
                            f"""
THREAT DETECTED

Object ID: {track_id}
Object Type: {object_class}
Threat Score: {score}/100

Predicted path enters restricted zone.
"""
                        )

                        log_threat(
                            track_id,
                            object_class,
                            score
                        )

                        filename = os.path.join(
                            "results",
                            "screenshots",
                            f"threat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        )

                        success = cv2.imwrite(
                            filename,
                            annotated
                        )

                        print("Saving to:", filename)
                        print("Image saved:", success)

                        email_sent.add(track_id)

                    cv2.rectangle(
                        annotated,
                        (0, 0),
                        (annotated.shape[1], 120),
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        annotated,
                        f"THREAT SCORE: {score}/100",
                        (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        4
                    )

                    cv2.putText(
                        annotated,
                        f"Object ID: {track_id}",
                        (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2
                    )

                    cv2.rectangle(
                        annotated,
                        (ZONE_X1, ZONE_Y1),
                        (ZONE_X2, ZONE_Y2),
                        (0, 0, 255),
                        6
                    )

 

    resized = cv2.resize(
        annotated,
        (1280, 720)
    )

    cv2.imshow(
        "AI Threat Surveillance System",
        resized
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()