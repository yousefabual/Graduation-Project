from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 nano model (smaller and faster)

# Load image or video stream (for drone, this would be real-time camera feed)
image_path = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/M1/M.jpg"
img = cv2.imread(image_path)

# Perform detection
results = model(img)

# Results
results.show()  # Display the results
results.save()  # Save the results (optional)

# Get class predictions and their confidences
boxes = results.xyxy[0]  # Bounding boxes
labels = results.names  # Class names
conf = results.conf  # Confidence scores

print("Detected classes:", labels)
print("Detection confidence:", conf)
