from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("data/videos/test.mp4")

while True:
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Resize frame for display
    resized_frame = cv2.resize(annotated_frame, (1280, 720))

    # Show frame
    cv2.imshow("YOLO Detection", resized_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()