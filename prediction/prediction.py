import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

# Load YAML
with open('data.yaml', mode='r') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)

labels = data_yaml['names']
print(labels)

# Load YOLO model
yolo = cv2.dnn.readNetFromONNX('Model6/weights/best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load the image
img = cv2.imread('testimage.jpg')
image = img.copy()
row, col, d = image.shape

# Convert image into square image (array)
max_rc = max(row, col)
input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
input_image[0:row, 0:col] = image

# YOLO model expects a 640x640 input image
INPUT_WH_YOLO = 640
blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
yolo.setInput(blob)
preds = yolo.forward()  # Prediction from YOLO

# Non Maximum Suppression
detections = preds[0]
boxes = []
confidences = []
classes = []

image_w, image_h = input_image.shape[:2]
x_factor = image_w / INPUT_WH_YOLO
y_factor = image_h / INPUT_WH_YOLO

# Iterate through detections and filter based on confidence
for i in range(len(detections)):
    row = detections[i]
    confidence = row[4]
    if confidence > 0.4:
        class_score = row[5:].max()
        class_id = row[5:].argmax()

        if class_score > 0.25:
            cx, cy, w, h = row[0:4]

            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            box = np.array([left, top, width, height])
            confidences.append(confidence)
            boxes.append(box)
            classes.append(class_id)

# Clean up and apply Non Maximum Suppression (NMS)
boxes_np = np.array(boxes).tolist()
confidences_np = np.array(confidences).tolist()

if boxes_np:  # Check if there are any boxes to process
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

    # Draw the Bounding Boxes
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = int(confidences_np[ind] * 100)
        class_name = labels[classes[ind]]

        text = f'{class_name}: {bb_conf}%'
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label_background_top = max(0, y - 30)  # Ensure the label does not go off the image
        cv2.rectangle(image, (x, label_background_top), (x + w, y), (255, 255, 255), -1)

        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

    # Show or save the resulting image
    cv2.imwrite('testimage__model6_prediction.jpg', image)  # Save the image
else:
    print("No boxes detected.")
