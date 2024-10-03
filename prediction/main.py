import cv2
import numpy as np
import pytesseract as py
import pandas as pd
import os
import argparse
import csv

def load_yolo_model():
    # Load the YOLO model
    model = cv2.dnn.readNetFromONNX('Model6/weights/best.onnx')
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Set to CPU for compatibility
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

def process_predictions(preds, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes = []
    confidences = []
    detections = preds[0]
    image_h, image_w = input_image.shape[:2]
    x_factor = image_w / 640
    y_factor = image_h / 640

    for detection in detections:
        confidence = detection[4]
        if confidence > conf_threshold:
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            if class_score > score_threshold:
                cx, cy, w, h = detection[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                boxes.append([left, top, int(w * x_factor), int(h * y_factor)])
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes

def perform_ocr_on_crops(image, boxes, indices, ocr_config='--oem 3'):
    results = []

    # Initialize a dictionary to hold data for the current test
    box_data = {"Test Name": [], "Value": [], "Units": [], "Reference Range": []}

    # Specify the mapping based on your description
    box_mapping = {
        36: "Test Name",          # Box 36 is for Test Name
        27: "Value",              # Box 27 is for Value
        29: "Units",              # Box 29 is for Units
        34: "Reference Range"     # Box 34 is for Reference Range
    }

    for i in indices.flatten():
        if i >= len(boxes):
            print(f"Index {i} is out of bounds for boxes list.")
            continue

        x, y, w, h = boxes[i]

        # Ensure the crop coordinates are within the image boundaries
        x = max(0, x)  # Clamp x to be at least 0
        y = max(0, y)  # Clamp y to be at least 0
        x_end = min(image.shape[1], x + w)  # Ensure x + w does not exceed image width
        y_end = min(image.shape[0], y + h)  # Ensure y + h does not exceed image height

        # Adjust width and height if the crop goes out of bounds
        w = x_end - x
        h = y_end - y

        if w <= 0 or h <= 0:
            print(f"Invalid crop dimensions for index {i}: Width: {w}, Height: {h}")
            continue

        crop_img = image[y:y+h, x:x+w]

        # Preprocess the crop for better OCR
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        roi = cv2.bitwise_not(thresh)

        # Perform OCR
        try:
            text = py.image_to_string(roi, config=ocr_config).strip()
        except Exception as e:
            print(f"OCR error for crop {i}: {e}")
            text = ""

        # Store all lines of text instead of just the first one
        if i in box_mapping:
            field_name = box_mapping[i]
            lines = text.splitlines()
            box_data[field_name].extend(lines)

    # Append the box_data to results as a new entry
    results.append(box_data)
    return results

def save_to_csv(data_list, output_file):
    # Convert the structured data into a DataFrame
    df = pd.DataFrame(data_list)

    # Create a new DataFrame to organize the data by columns
    output_df = pd.DataFrame()

    # Expand each column into separate rows
    for column in df.columns:
        temp_df = pd.DataFrame(df[column].explode()).reset_index(drop=True)
        temp_df.columns = [column]
        output_df = pd.concat([output_df, temp_df], axis=1)

    # Print the DataFrame to the console
    print("\nFinal DataFrame to be saved:")
    print(output_df)

    # Save the DataFrame into a CSV file
    output_df.to_csv(output_file, index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

def draw_bounding_boxes(image, boxes, indices):
    # Draw bounding boxes on the image
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def main():
    # Prompt the user to enter the path to the image
    image_path = input("Enter the path of the JPG image file: ").strip()
    if not os.path.exists(image_path) or not image_path.lower().endswith('.jpg'):
        print("Please provide a valid path to a JPG image.")
        return

    # Set the output directory and filenames
    output_dir = "output_csv"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_csv_file = os.path.join(output_dir, f"{base_name}_output.csv")
    output_image_file = os.path.join(output_dir, f"{base_name}_annotated.jpg")

    # Load YOLO model
    yolo_model = load_yolo_model()

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load the image. Please check the file path.")
        return

    image_copy = image.copy()

    # Run prediction with YOLO
    preds, input_image = predict_yolo(yolo_model, image_copy)

    # Process predictions and filter boxes
    indices, boxes = process_predictions(preds, input_image)

    # Perform OCR on the cropped images
    ocr_results = perform_ocr_on_crops(image_copy, boxes, indices)

    # Save results to CSV
    save_to_csv(ocr_results, output_csv_file)

    # Draw bounding boxes and save the annotated image
    annotated_image = draw_bounding_boxes(image_copy, boxes, indices)
    cv2.imwrite(output_image_file, annotated_image)
    print(f"Annotated image saved to {output_image_file}")

if __name__ == "__main__":
    main()


# to run the code: python main.py
#Enter the path of the JPG image file: testimage.jpg