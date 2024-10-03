# Project10-Custom-Object Character Recognition(OCR)

## Overview
This project implements a Streamlit application that utilizes a YOLOV5 model for object detection and Optical Character Recognition (OCR) to extract critical information from medical reports.

## Features
- **Upload Images**: Users can upload JPG images of medical reports.
- **Object Detection**: The application uses a YOLOV5 model to detect relevant fields in the images.
- **OCR Processing**: Extracts data such as Test Name, Value, Units, and Reference Range from the detected regions.
- **Results Display**: Shows extracted data in a structured format.
- **CSV Download**: Allows users to download the extracted information as a CSV file.

## Prerequisites
- Python 3.8+
## requirements.txt:
  - numpy
  - scipy
  - wget
  - seaborn
  -  opencv-python
  - tqdm
  - pandas
  - awscli
  - urllib3
  - mss
  - labelImg
  - ipykernel
  - matplotlib
  - pytesseract
  - streamlit
You can install the required libraries using the following command: pip install -r requirements.txt

### Getting Started
  - Clone the repository or download the code files.
  - Place the YOLO model weights (best.onnx) in the Model6/weights/ directory.
  - Navigate to the Directory :cd path/to/your/prediction
  - Run the Streamlit App : streamlit run app.py
### Usage
  - Upload your JPG images of medical reports.
  - Wait for the processing to complete.
  - Review the extracted OCR results displayed in the application.
  - Download the results as a CSV file if needed.
### Conclusion
  - This project provides an efficient solution for automating the extraction of vital information from medical reports, enhancing data accessibility and management.
