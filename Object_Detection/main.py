# pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2

# Load YOLOv11 model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the "access granted" image
access_img = cv2.imread("access.jpg")  # make sure access.png exists in your folder
if access_img is None:
    print("Error: access.png not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Get detected classes for this frame
    detected_classes = set()
    for box in results[0].boxes:
        cls_id = int(box.cls[0])  # class ID
        detected_classes.add(cls_id)

    # Annotate frame
    annotated_frame = results[0].plot()

    if 3 <= len(detected_classes) <= 4:
        resized_access = cv2.resize(access_img, (frame.shape[1], frame.shape[0]))
        cv2.imshow("YOLOv11 Webcam", resized_access)
    else:
        cv2.imshow("YOLOv11 Webcam", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
