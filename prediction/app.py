import cv2
import numpy as np
import pytesseract as py
import pandas as pd
import streamlit as st
from PIL import Image

def load_yolo_model():
    model = cv2.dnn.readNetFromONNX('Model6/weights/best.onnx')
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

def predict_yolo(model, image):
    INPUT_WH_YOLO = 640
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_image

def process_predictions(predictions, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes = []
    confidences = []
    detections = predictions[0]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / 640
    y_factor = image_height / 640

    for detection in detections:
        confidence = detection[4]
        if confidence > conf_threshold:
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            if class_score > score_threshold:
                center_x, center_y, width, height = detection[0:4]
                left = int((center_x - 0.5 * width) * x_factor)
                top = int((center_y - 0.5 * height) * y_factor)
                boxes.append([left, top, int(width * x_factor), int(height * y_factor)])
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes

def preprocess_image(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = cv2.bitwise_not(thresh)
    return roi

def perform_ocr_on_crops(image, boxes, indices, ocr_config='--oem 3 --psm 6'):
    # Initialize lists to hold data
    test_names = []
    values = []
    units = []
    reference_ranges = []

    box_mapping = {
        61: "Test Name",
        14: "Value",
        26: "Units",
        41: "Reference Range"
    }

    for i in indices.flatten():
        if i >= len(boxes):
            continue

        x, y, w, h = boxes[i]
        x = max(0, x)
        y = max(0, y)
        x_end = min(image.shape[1], x + w)
        y_end = min(image.shape[0], y + h)
        w = x_end - x
        h = y_end - y

        if w <= 0 or h <= 0:
            continue

        crop_img = image[y:y+h, x:x+w]
        roi = preprocess_image(crop_img)

        # Perform OCR
        text = py.image_to_string(roi, config=ocr_config).strip()
        print(f"OCR Result for crop {i}: {text}")  # Debugging output

        if i in box_mapping:
            field_name = box_mapping[i]
            lines = text.splitlines()
            for line in lines:
                line = line.strip()
                if field_name == "Test Name":
                    if line:
                        test_names.append(line)
                elif field_name == "Value":
                    if line:
                        values.append(line)
                elif field_name == "Units":
                    if line:
                        units.append(line)
                elif field_name == "Reference Range":
                    if line:
                        reference_ranges.append(line)

    # Ensure equal lengths
    max_length = max(len(test_names), len(values), len(units), len(reference_ranges))
    test_names.extend([''] * (max_length - len(test_names)))
    values.extend([''] * (max_length - len(values)))
    units.extend([''] * (max_length - len(units)))
    reference_ranges.extend([''] * (max_length - len(reference_ranges)))

    results = pd.DataFrame({
        "Test Name": test_names,
        "Value": values,
        "Units": units,
        "Reference Range": reference_ranges
    })

    return results

def draw_bounding_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def main():
    st.title("Object Detection and OCR for Medical Reports")

    uploaded_files = st.file_uploader("Upload JPG images", type="jpg", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Processing: {uploaded_file.name}")
            image = np.array(Image.open(uploaded_file))

            # Load YOLO model
            yolo_model = load_yolo_model()

            # Run YOLO predictions
            preds, input_image = predict_yolo(yolo_model, image)

            # Process predictions and filter boxes
            indices, boxes = process_predictions(preds, input_image)

            # Perform OCR on the cropped regions
            ocr_results = perform_ocr_on_crops(image, boxes, indices)

            # Display OCR results in a table
            st.write(f"OCR Results for {uploaded_file.name}:")
            st.dataframe(ocr_results)

            # Option to download CSV for each image
            csv = ocr_results.to_csv(index=False)
            st.download_button(label=f"Download CSV for {uploaded_file.name}", data=csv, file_name=f'{uploaded_file.name}_ocr_results.csv', mime='text/csv')

            # Draw bounding boxes and display the image
            annotated_image = draw_bounding_boxes(image, boxes, indices)
            st.image(annotated_image, caption=f"Annotated Image for {uploaded_file.name}", use_column_width=True)

if __name__ == "__main__":
    main()
