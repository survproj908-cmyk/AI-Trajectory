from ultralytics import YOLO
import cv2
import json

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("data/videos/test.mp4")

trajectories = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    boxes = results[0].boxes

    if boxes.id is not None:

        ids = boxes.id.int().cpu().tolist()

        coords = boxes.xyxy.cpu().tolist()

        for track_id, box in zip(ids, coords):

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if track_id not in trajectories:
                trajectories[track_id] = []

            trajectories[track_id].append(
                [center_x, center_y]
            )

cap.release()

with open(
    "data/trajectories/trajectories.json",
    "w"
) as f:

    json.dump(
        trajectories,
        f,
        indent=4
    )

print("Trajectories saved!")