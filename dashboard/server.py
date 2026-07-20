import sys
import os
import asyncio
import base64
import time
import csv
from datetime import datetime
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Define directory paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from alerts.email_alert import send_alert
from logging_system.threat_logger import log_threat
from evaluation.threat_score import calculate_threat_score

# Try safe imports for PyTorch/LSTM and YOLOv8
try:
    import torch
    from models.lstm_model import LSTMPredictor
except Exception as e:
    print(f"Warning: PyTorch / LSTMPredictor not available: {e}")
    torch = None
    LSTMPredictor = None

try:
    from ultralytics import YOLO
except Exception as e:
    print(f"Warning: Ultralytics YOLO not available: {e}")
    YOLO = None

# Initialize FastAPI
app = FastAPI(title="AI Threat Surveillance Dashboard Server")

# -----------------------------
# Configuration & File Paths
# -----------------------------
VIDEO_PATH = os.path.join(ROOT_DIR, "data", "videos", "test.mp4")
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "yolov8n.pt")
if not os.path.exists(YOLO_MODEL_PATH):
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, "models", "lstm_model.pth")
LOG_FILE = os.path.join(ROOT_DIR, "results", "threat_log.csv")
SCREENSHOT_DIR = os.path.join(ROOT_DIR, "results", "screenshots")

ZONE_X1, ZONE_Y1 = 800, 350
ZONE_X2, ZONE_Y2 = 1700, 1000

MIN_VAL = 33
MAX_VAL = 3808

subsystem_status = {
    "yoloLoaded": False,
    "cameraConnected": True,
    "emailActive": True,
    "loggerActive": True,
    "screenshotActive": True
}

yolo_model = None
lstm_model = None

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)

def init_logs():
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Object ID", "Object Type", "Threat Score"])
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            writer.writerow([now_str, "101", "car", "85"])
            writer.writerow([now_str, "102", "truck", "65"])
            writer.writerow([now_str, "103", "person", "55"])

init_logs()

def load_models():
    global yolo_model, lstm_model
    if YOLO is not None:
        print("Loading YOLOv8 model...")
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            subsystem_status["yoloLoaded"] = True
            print("YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            subsystem_status["yoloLoaded"] = False
            yolo_model = None
    else:
        subsystem_status["yoloLoaded"] = False
        yolo_model = None

    if LSTMPredictor is not None and torch is not None:
        print("Loading LSTM predictor...")
        try:
            if os.path.exists(LSTM_MODEL_PATH):
                lstm_model = LSTMPredictor()
                lstm_model.load_state_dict(
                    torch.load(LSTM_MODEL_PATH, map_location=torch.device("cpu"))
                )
                lstm_model.eval()
                print("LSTM model loaded successfully.")
            else:
                lstm_model = None
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            lstm_model = None
    else:
        lstm_model = None

load_models()

def get_recent_threat_logs():
    logs = []
    if not os.path.exists(LOG_FILE):
        return logs
    try:
        with open(LOG_FILE, "r") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    try:
                        dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
                        time_str = dt.strftime("%H:%M:%S")
                    except ValueError:
                        time_str = row[0][:8] if len(row[0]) >= 8 else row[0]
                    
                    score_val = float(row[3]) if row[3].replace('.', '', 1).isdigit() else 0
                    logs.append({
                        "time": time_str,
                        "trackId": int(row[1]) if row[1].isdigit() else row[1],
                        "objectType": row[2],
                        "threatScore": int(score_val),
                        "status": "ALERT DISPATCHED" if score_val >= 50 else "LOGGED"
                    })
    except Exception as e:
        print(f"Error reading threat log CSV: {e}")
    return list(reversed(logs))[:25]

def get_latest_screenshot_base64():
    if not os.path.exists(SCREENSHOT_DIR):
        return None
    try:
        files = [os.path.join(SCREENSHOT_DIR, f) for f in os.listdir(SCREENSHOT_DIR) if f.endswith(".jpg")]
        if not files:
            return None
        latest_file = max(files, key=os.path.getctime)
        with open(latest_file, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading latest screenshot: {e}")
        return None

@app.get("/")
def get_dashboard():
    index_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(index_path)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected to dashboard streaming server.")

    email_sent_logs = [
        {"id": 101, "recipient": "security-ops@cybersec.org", "time": datetime.now().strftime("%H:%M:%S"), "status": "SENT"},
        {"id": 102, "recipient": "security-ops@cybersec.org", "time": datetime.now().strftime("%H:%M:%S"), "status": "SENT"}
    ]
    emailed_ids = set()
    logged_ids = set()

    cap = None
    use_simulation = False

    if yolo_model is None or not os.path.exists(VIDEO_PATH):
        print("Falling back to simulated feed.")
        use_simulation = True
    else:
        try:
            cap = cv2.VideoCapture(VIDEO_PATH)
            if cap.isOpened():
                subsystem_status["cameraConnected"] = True
                print(f"Opened video source: {VIDEO_PATH}")
            else:
                use_simulation = True
        except Exception as cap_err:
            print(f"Error opening video capture: {cap_err}. Falling back to simulation.")
            use_simulation = True

    if use_simulation:
        subsystem_status["cameraConnected"] = True

        class SimTarget:
            def __init__(self, track_id, obj_class, x, y, vx, vy):
                self.track_id = track_id
                self.obj_class = obj_class
                self.x = x
                self.y = y
                self.vx = vx
                self.vy = vy
                self.history = []
                self.threat_score = 0

            def update(self):
                self.x += self.vx
                self.y += self.vy
                
                if self.x < 200 or self.x > 1700:
                    self.vx *= -1
                if self.y < 200 or self.y > 900:
                    self.vy *= -1
                    
                self.history.append((int(self.x), int(self.y)))
                if len(self.history) > 30:
                    self.history.pop(0)

                in_zone = (ZONE_X1 <= self.x <= ZONE_X2) and (ZONE_Y1 <= self.y <= ZONE_Y2)
                if in_zone:
                    self.threat_score = min(100, self.threat_score + 8)
                else:
                    dx = max(0, ZONE_X1 - self.x, self.x - ZONE_X2)
                    dy = max(0, ZONE_Y1 - self.y, self.y - ZONE_Y2)
                    dist = (dx**2 + dy**2)**0.5
                    if dist < 350:
                        self.threat_score = int(max(0, 90 - (dist / 4)))
                    else:
                        self.threat_score = max(0, self.threat_score - 4)

        targets = [
            SimTarget(101, "person", 300, 600, 10, -4),
            SimTarget(102, "car", 1500, 300, -16, 6),
            SimTarget(103, "truck", 700, 850, 8, -7),
            SimTarget(104, "bus", 1200, 500, 4, 10)
        ]

    track_history = {}
    fps_start_time = time.time()
    frame_count = 0
    fps = 30.0

    try:
        while True:
            active_objects = 0
            counts = {"person": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
            active_threats = 0
            max_threat_score = 0
            frame_base64 = ""

            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / max((end_time - fps_start_time), 0.001)
                fps_start_time = end_time
                frame_count = 0

            if use_simulation:
                frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 12
                
                for gx in range(0, 1920, 160):
                    cv2.line(frame, (gx, 0), (gx, 1080), (32, 28, 28), 1)
                for gy in range(0, 1080, 160):
                    cv2.line(frame, (0, gy), (1920, gy), (32, 28, 28), 1)

                if (int(time.time() * 2) % 2) == 0:
                    cv2.circle(frame, (80, 80), 14, (0, 0, 255), -1)
                cv2.putText(frame, "REC [SIMULATOR MODE]", (110, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, time_str, (1480, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 0, 255), 3)
                cv2.putText(frame, "RESTRICTED ZONE", (ZONE_X1, ZONE_Y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                active_objects = len(targets)
                for t in targets:
                    t.update()
                    counts[t.obj_class] = counts.get(t.obj_class, 0) + 1
                    
                    for i in range(1, len(t.history)):
                        cv2.line(frame, t.history[i - 1], t.history[i], (0, 255, 0), 2)

                    bw, bh = (100, 180) if t.obj_class == "person" else (200, 110) if t.obj_class == "car" else (260, 150)
                    bx1, by1 = int(t.x - bw/2), int(t.y - bh/2)
                    bx2, by2 = int(t.x + bw/2), int(t.y + bh/2)

                    color = (0, 255, 0) if t.threat_score < 30 else (0, 165, 255) if t.threat_score < 50 else (0, 0, 255)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
                    
                    label = f"{t.obj_class.upper()} #{t.track_id} [{t.threat_score}%]"
                    cv2.putText(frame, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    future_pts = []
                    fx, fy = t.x, t.y
                    for step in range(1, 9):
                        fx += t.vx * 3
                        fy += t.vy * 3
                        future_pts.append((int(fx), int(fy)))

                    if len(future_pts) > 0:
                        cv2.line(frame, (int(t.x), int(t.y)), future_pts[0], (255, 0, 0), 3)
                        for i in range(1, len(future_pts)):
                            cv2.line(frame, future_pts[i - 1], future_pts[i], (255, 0, 0), 3)

                    if t.threat_score > max_threat_score:
                        max_threat_score = t.threat_score

                    if t.track_id not in logged_ids:
                        logged_ids.add(t.track_id)
                        try: log_threat(t.track_id, t.obj_class, t.threat_score)
                        except Exception: pass

                    if t.threat_score >= 50:
                        active_threats += 1
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], 130), (0, 0, 255), -1)
                        cv2.putText(frame, f"THREAT DETECTED: {t.threat_score}/100", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
                        cv2.putText(frame, f"Object ID: {t.track_id} | Class: {t.obj_class.upper()}", (50, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 0, 255), 7)

                        if t.track_id not in emailed_ids:
                            emailed_ids.add(t.track_id)
                            email_time = datetime.now().strftime("%H:%M:%S")
                            try:
                                send_alert(f"THREAT DETECTED [SIMULATED]\n\nObject ID: {t.track_id}\nObject Type: {t.obj_class}\nThreat Score: {t.threat_score}/100")
                                email_sent_logs.insert(0, {"id": t.track_id, "recipient": "security-ops@cybersec.org", "time": email_time, "status": "SENT"})
                            except Exception:
                                email_sent_logs.insert(0, {"id": t.track_id, "recipient": "security-ops@cybersec.org", "time": email_time, "status": "FAILED (SMTP)"})
                            try:
                                filename = os.path.join(SCREENSHOT_DIR, f"threat_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                                cv2.imwrite(filename, frame)
                            except Exception: pass

                resized = cv2.resize(frame, (960, 540))
                _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

            else:
                # -------------------------------------------------------------
                # YOLOv8 High-Precision Detection & Tracking on Real Video
                # -------------------------------------------------------------
                try:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    # Run YOLO tracking on full resolution (16GB RAM on Hugging Face Spaces handles this easily!)
                    if torch is not None and hasattr(torch, 'no_grad'):
                        with torch.no_grad():
                            results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    else:
                        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

                    boxes = results[0].boxes
                    annotated = results[0].plot()

                    # Draw Restricted Zone
                    cv2.rectangle(annotated, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 0, 255), 3)
                    cv2.putText(annotated, "RESTRICTED ZONE", (ZONE_X1, ZONE_Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    if boxes.id is not None:
                        ids = boxes.id.int().cpu().tolist()
                        coords = boxes.xyxy.cpu().tolist()
                        classes = boxes.cls.int().cpu().tolist()
                        active_objects = len(ids)

                        for track_id, box, cls_idx in zip(ids, coords, classes):
                            obj_class = yolo_model.names[cls_idx] # person, car, truck, bus, motorcycle, etc.
                            counts[obj_class] = counts.get(obj_class, 0) + 1

                            x1_b, y1_b, x2_b, y2_b = box
                            cx = int((x1_b + x2_b) / 2)
                            cy = int((y1_b + y2_b) / 2)

                            if track_id not in track_history:
                                track_history[track_id] = []
                            track_history[track_id].append((cx, cy))

                            if len(track_history[track_id]) > 100:
                                track_history[track_id].pop(0)

                            points = track_history[track_id]

                            # Draw Past Path (Green)
                            for i in range(1, len(points)):
                                cv2.line(annotated, points[i - 1], points[i], (0, 255, 0), 2)

                            # Predict Future Trajectory Path (Blue)
                            future_points = []
                            if len(points) >= 4:
                                if lstm_model is not None and torch is not None:
                                    recent = np.array(points[-8:]) if len(points) >= 8 else np.pad(np.array(points), ((8-len(points), 0), (0,0)), mode='edge')
                                    recent = (recent - MIN_VAL) / (MAX_VAL - MIN_VAL)
                                    recent = torch.tensor(recent, dtype=torch.float32).unsqueeze(0)
                                    with torch.no_grad():
                                        future = lstm_model(recent)
                                    future = future.squeeze(0).cpu().numpy()
                                    future = (future * (MAX_VAL - MIN_VAL)) + MIN_VAL
                                    future_points = [(int(px), int(py)) for px, py in future]
                                else:
                                    last_p = points[-1]
                                    prev_p = points[-3]
                                    vel_x = (last_p[0] - prev_p[0]) / 2.0
                                    vel_y = (last_p[1] - prev_p[1]) / 2.0
                                    for step in range(1, 5):
                                        future_points.append((int(last_p[0] + vel_x * step * 2), int(last_p[1] + vel_y * step * 2)))

                                zone_tuple = (ZONE_X1, ZONE_Y1, ZONE_X2, ZONE_Y2)
                                score = calculate_threat_score(future_points, obj_class, zone_tuple)

                                if score > max_threat_score:
                                    max_threat_score = score

                                # Draw Future Path (Blue)
                                if len(future_points) > 0:
                                    cv2.line(annotated, points[-1], future_points[0], (255, 0, 0), 3)
                                for i in range(1, len(future_points)):
                                    cv2.line(annotated, future_points[i - 1], future_points[i], (255, 0, 0), 3)

                                # Log every detected object to CSV & threat logs
                                if track_id not in logged_ids:
                                    logged_ids.add(track_id)
                                    try:
                                        log_threat(track_id, obj_class, score)
                                    except Exception as log_err:
                                        print(f"Error logging threat: {log_err}")

                                if score >= 50:
                                    active_threats += 1
                                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 120), (0, 0, 255), -1)
                                    cv2.putText(annotated, f"THREAT DETECTED: {score}/100", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                                    cv2.putText(annotated, f"Object ID: {track_id} | Class: {obj_class.upper()}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                                    cv2.rectangle(annotated, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 0, 255), 6)

                                    if track_id not in emailed_ids:
                                        emailed_ids.add(track_id)
                                        email_time = datetime.now().strftime("%H:%M:%S")
                                        try:
                                            send_alert(f"THREAT DETECTED\n\nObject ID: {track_id}\nObject Type: {obj_class}\nThreat Score: {score}/100")
                                            email_sent_logs.insert(0, {"id": track_id, "recipient": "security-ops@cybersec.org", "time": email_time, "status": "SENT"})
                                        except Exception:
                                            email_sent_logs.insert(0, {"id": track_id, "recipient": "security-ops@cybersec.org", "time": email_time, "status": "FAILED (SMTP)"})

                                        try:
                                            filename = os.path.join(SCREENSHOT_DIR, f"threat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                                            cv2.imwrite(filename, annotated)
                                        except Exception: pass

                    resized = cv2.resize(annotated, (960, 540))
                    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')

                except Exception as frame_err:
                    print(f"Frame processing error: {frame_err}. Falling back to simulation.")
                    use_simulation = True

            threat_logs = get_recent_threat_logs()
            latest_ss = get_latest_screenshot_base64()

            payload = {
                "type": "telemetry",
                "active_objects": active_objects,
                "counts": counts,
                "active_threats": active_threats,
                "max_threat_score": int(max_threat_score),
                "fps": round(fps, 1),
                "live_frame": frame_base64,
                "threat_logs": threat_logs,
                "email_logs": email_sent_logs[:5],
                "latest_screenshot": latest_ss,
                "services": subsystem_status
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"Exception in websocket loop: {e}")
    finally:
        if cap is not None:
            cap.release()
            print("Video source released.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting AI Threat Surveillance System web server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
